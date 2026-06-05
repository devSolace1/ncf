
# Instruksi GitHub Copilot Chat — NCF Codebase Repair & Hardening
### 5 Phase: Bug Fixes → Code Quality → Performance → Ecosystem → Produksi 
---

## PHASE 1 — Critical Bug Fixes (Blocker)
*Target: Perbaiki semua bug yang membuat API tidak bisa diandalkan*

---

**Prompt 1.1 — Fix HTTP Reader (bug paling kritis)**

```
@workspace Di file ncf-io/src/http_reader.rs, fungsi `open()` saat ini melakukan 2 HTTP request:
1. Fetch prefix 48 bytes
2. Fetch SEMUA sisa file dari byte 48 ke akhir

Ini membuat HTTP reader tidak berguna karena mengunduh seluruh file.

Perbaiki dengan arsitektur 3-step yang benar:
1. Fetch 48 bytes prefix untuk dapat `schema_offset`, `index_offset`
2. Fetch header block + schema block saja (dari byte 48 sampai `index_offset - footer_size`)
3. Fetch footer 16 bytes terakhir, decode `index_len`, lalu fetch index block saja

Untuk `fetch_tensor()`, gunakan HTTP Range Request per chunk berdasarkan `chunk_ref.byte_offset` dan `chunk_ref.byte_len` dari index.

Pastikan:
- Setiap range request menggunakan format header `bytes=start-end`
- Tambahkan validasi bahwa server mendukung range requests (header `Accept-Ranges: bytes` atau status 206)
- Error yang jelas jika server tidak support range request
- Tidak ada download seluruh file sama sekali
```

---

**Prompt 1.2 — Fix Streaming `is_last_chunk` dan tensor naming**

```
@workspace Di ncf-io/src/stream.rs, dalam impl `NcfStreamReader::next_chunk()`, ada 2 bug:

BUG 1: `is_last_chunk` selalu `false`
```rust
let is_last = false; // placeholder, tidak pernah diset
```

BUG 2: Tensor name tidak akurat
```rust
let name = format!("chunk_{}", chunk_header.chunk_id);
```
Seharusnya nama tensor diambil dari schema yang sudah di-parse di state `AwaitingSchema`.

Perbaiki:
1. Simpan `Vec<TensorSchema>` di struct `NcfStreamReader` setelah state `AwaitingSchema` selesai
2. Buat helper method `tensor_name_for_chunk(chunk_id: u64) -> Option<String>` yang lookup dari schema
3. Untuk `is_last_chunk`: sebuah chunk adalah "last" jika chunk_id yang sedang di-read adalah chunk_id terakhir dalam `schema.chunks` untuk tensor tersebut
4. Jika chunk_id tidak ditemukan di schema manapun, emit nama fallback tapi log warning

Juga perbaiki: `StreamChunk::Metadata` saat ini dikembalikan dengan struct kosong. Pindahkan decode `NcfHeader` yang benar ke state `AwaitingHeader`, simpan hasilnya, dan emit yang benar.
```

---

**Prompt 1.3 — Fix `find_schema()` O(n) → O(log n)**

```
@workspace Di ncf-io/src/reader.rs, method `find_schema()` melakukan linear scan:

```rust
pub fn find_schema(&self, name: &str) -> Result<Option<&TensorSchema>> {
    Ok(self.schemas()?.iter().find(|schema| schema.name == name))
}
```

Padahal `BorrowedNcfIndex.tensor_map: BTreeMap<String, u64>` sudah ada.

Perbaiki dengan:
1. Gunakan `tensor_map` untuk dapat `chunk_id` dari nama tensor — O(log n)
2. Gunakan `chunk_map` untuk dapat `IndexEntry` dari `chunk_id` — O(log n)
3. Kemudian lookup schema dari `Vec<TensorSchema>` menggunakan index position yang disimpan saat parse

Jika schema belum memiliki reverse-lookup map, tambahkan `schema_map: BTreeMap<String, usize>` di `NcfReaderData` yang diisi saat schema pertama kali di-parse di `OnceLock`. Key adalah `schema.name`, value adalah index di Vec.

Lakukan hal yang sama untuk `NcfMmap::find_schema()` jika ada.

Tulis unit test yang memverifikasi lookup benar untuk 1000 tensor.
```

---

**Prompt 1.4 — Fix Writer schema stabilization loop**

```
@workspace Di ncf-io/src/writer.rs, method `finalize()` memiliki loop 10 iterasi untuk menstabilkan ukuran CBOR schema:

```rust
for attempt in 0..10 {
    // encode schema, check if size stabilized
    if candidate_schema_len == schema_len { break; }
    if attempt == 9 { return Err(...) }
}
```

Ini adalah workaround untuk masalah: offset dalam ChunkRef mempengaruhi ukuran CBOR encoding, yang mempengaruhi offset.

Perbaiki dengan two-pass approach:
PASS 1 (dry-run):
- Encode schema dengan semua `ChunkRef.byte_offset = 0` sebagai placeholder
- Hitung ukuran schema CBOR → `schema_len`
- Dengan `schema_len` yang diketahui, hitung offset aktual setiap chunk secara deterministik
- Isi semua `ChunkRef` dengan offset yang benar

PASS 2 (final):
- Re-encode schema dengan offset yang sudah benar
- Assert bahwa ukuran encoding sama dengan `schema_len` dari pass 1
- Jika berbeda, return Err dengan pesan yang jelas (bukan retry loop)

Hapus loop dan semua variabel `attempt`, `candidate_*`. Pastikan logika offset calculation benar: `chunk_start = FILE_HEADER_PREFIX_SIZE + header_len + schema_len + sum(previous_chunk_total_lens)`.
```

---

**Prompt 1.5 — Fix Python binding F16 dan BF16**

```
@workspace Di ncf-io/src/lib.rs (atau ncf-py/src/lib.rs), fungsi `tensor_data_to_numpy()` tidak mengimplementasikan F16 dan BF16:

```rust
_ => { // error atau panic }
```

Implementasikan:

Untuk F16:
- Gunakan crate `half::f16` (sudah ada di dependencies)
- Convert bytes ke `Vec<f16>` menggunakan `half::f16::from_le_bytes()`
- Return sebagai `PyArray<f32>` setelah convert ke f32, ATAU gunakan numpy float16 via `numpy::PyArray1::<numpy::f16>`
- Jika numpy binding tidak support f16 langsung, return raw bytes dengan dtype annotation

Untuk BF16:
- Gunakan `half::bf16`
- Sama seperti F16, convert ke f32 untuk numpy compatibility
- Tambahkan doc comment yang menjelaskan bahwa konversi ke f32 adalah lossy-free (BF16 adalah subset presisi F32)

Tambahkan test Python-side (di doctest atau test module) yang:
1. Buat NCF file dengan tensor F16 dari Python
2. Read kembali
3. Verifikasi nilai mendekati original (dalam toleransi float16)

```

---

## PHASE 2 — Code Quality & Deduplication
*Target: Hapus duplikasi, perkuat error handling, enforce konsistensi*

---

**Prompt 2.1 — Unify Reader melalui trait `NcfSource`**

```
@workspace Saat ini ada duplikasi besar antara ncf-io/src/reader.rs (`NcfReader`) dan ncf-io/src/mmap.rs (`NcfMmap`). Logika parsing header, schema, index, dan footer identik di keduanya (~200 baris duplikat).

Refactor dengan langkah berikut:

STEP 1: Buat file baru `ncf-io/src/parse.rs` dengan fungsi-fungsi parsing yang shared:
```rust
pub(crate) fn parse_header_prefix(bytes: &[u8]) -> Result<FileHeaderPrefix>
pub(crate) fn parse_cbor_header(bytes: &[u8]) -> Result<NcfHeader>
pub(crate) fn parse_schema_block(bytes: &[u8]) -> Result<Vec<TensorSchema>>
pub(crate) fn parse_index_block(bytes: &[u8]) -> Result<NcfIndex>
pub(crate) fn validate_footer(bytes: &[u8], file_len: usize) -> Result<u64> // returns index_len
pub(crate) fn validate_bounds(file_len: usize, header_prefix: &FileHeaderPrefix) -> Result<()>
```

STEP 2: Refactor `NcfMmap::open()` untuk memanggil fungsi dari `parse.rs`
STEP 3: Refactor `NcfReader` (self_cell) untuk memanggil fungsi yang sama
STEP 4: Pastikan semua error message tetap identik (tidak breaking untuk user yang parse error string)
STEP 5: Tulis 1 test untuk setiap fungsi di `parse.rs`

Jangan hapus `NcfReader` atau `NcfMmap` — keduanya tetap ada sebagai public API, hanya internal implementation yang di-share.
```

---

**Prompt 2.2 — Perkuat error types**

```
@workspace Error handling di NCF saat ini menggunakan `NcfError` yang terlalu generic. Banyak error di-wrap sebagai `std::io::Error` dengan pesan string, yang membuat programmatic error handling tidak mungkin.

Perbaiki `ncf-core/src/header.rs` enum `NcfError`:

```rust
// Saat ini:
pub enum NcfError {
    Io(#[from] std::io::Error),
    Cbor(...),
    CborSer(...),
    Header(String),  // terlalu generic
}
```

Tambahkan variant yang spesifik:
```rust
pub enum NcfError {
    Io(#[from] std::io::Error),
    CborDeserialize(#[from] ciborium::de::Error<std::io::Error>),
    CborSerialize(#[from] ciborium::ser::Error<std::io::Error>),
    
    // File structure errors
    InvalidMagic { expected: [u8; 8], got: [u8; 8] },
    FileTooSmall { actual: usize, minimum: usize },
    BlockOutOfBounds { block: &'static str, start: usize, end: usize, file_size: usize },
    SizeExceedsMaximum { block: &'static str, actual: u64, maximum: u64 },
    IntegerOverflow { context: &'static str },
    
    // Data errors
    InvalidChunkMagic { chunk_id: u64 },
    ChecksumMismatch { chunk_id: u64, expected: [u8; 32], computed: [u8; 32] },
    TensorNotFound { name: String },
    CompressionMismatch { chunk_id: u64 },
    DecompressionFailed { algorithm: &'static str, reason: String },
    SchemaEncodingUnstable,
}
```

Update semua callsite di ncf-io untuk menggunakan variant baru. Pastikan `Display` impl memberikan pesan yang readable. Tambahkan `#[non_exhaustive]` agar future variants tidak breaking.
```

---

**Prompt 2.3 — Tambahkan SIGBUS protection untuk mmap**

```
@workspace Di ncf-io/src/mmap.rs dan reader.rs, penggunaan mmap tidak memiliki proteksi terhadap SIGBUS yang bisa terjadi jika file berubah atau di-truncate saat sedang di-read.

Implementasikan defensive mmap:

1. Setelah `Mmap::map()`, tambahkan length snapshot:
```rust
let mmap_len_at_open = mmap.len();
```

2. Buat wrapper struct `SafeMmapSlice` yang sebelum setiap access memverifikasi bahwa offset + len tidak melebihi `mmap_len_at_open` (bukan `mmap.len()` yang bisa berubah).

3. Di Linux, tambahkan `mmap.advise(memmap2::Advice::Sequential)` untuk tensor sequential read dan `mmap.advise(memmap2::Advice::Random)` untuk selective access. Gunakan `#[cfg(target_os = "linux")]` guard.

4. Tambahkan method `NcfMmap::prefetch_tensor(&self, name: &str)` yang calls `madvise(MADV_WILLNEED)` pada range tensor tersebut — hint ke OS untuk pre-fault pages ke RAM.

5. Dokumen `# Safety` section di semua fungsi yang menggunakan `unsafe { Mmap::map() }` menjelaskan invariant yang diasumsikan.

```

---

**Prompt 2.4 — Enforce compression ratio guard**

```
@workspace Di ncf-io/src/reader.rs, method `read_tensor()` dan `verify_chunk()` melakukan decompression tanpa memverifikasi rasio kompresi. Ini rentan terhadap zip bomb: payload 1 KiB yang terdekompresi menjadi 10 GiB.

Tambahkan konstanta di ncf-core/src/lib.rs:
```rust
pub mod constants {
    /// Maximum allowed decompression ratio (compressed:uncompressed).
    /// A payload cannot decompress to more than 1000x its compressed size.
    pub const MAX_DECOMPRESSION_RATIO: u64 = 1000;
    
    /// Maximum decompressed size per chunk in bytes (4 GiB).
    pub const MAX_DECOMPRESSED_CHUNK_SIZE: u64 = 4 * 1024 * 1024 * 1024;
}
```

Di setiap tempat decompression terjadi (reader.rs: `read_tensor()`, `verify_chunk()`):
1. Sebelum decompression, cek: `chunk.uncompressed_len <= chunk.compressed_len * MAX_DECOMPRESSION_RATIO`
2. Cek: `chunk.uncompressed_len <= MAX_DECOMPRESSED_CHUNK_SIZE`
3. Jika gagal, return `Err(NcfError::SuspiciousDecompressionRatio { chunk_id, ratio })`

Lakukan hal yang sama di http_reader.rs setelah fetch.

Tulis fuzz test tambahan `fuzz_decompression_bomb` yang mengirim payload dengan `uncompressed_len` sangat besar.
```

---

**Prompt 2.5 — Konsistensi API: unify `tensor_slice` semantics**

```
@workspace Saat ini ada inkonsistensi antara `NcfReader::tensor_slice()` dan `NcfMmap::tensor_slice()`:
- Keduanya return compressed bytes jika tensor compressed
- Doc comment mengatakan "raw compressed payload" tapi tidak ada cara mudah untuk tahu apakah tensor compressed atau tidak tanpa inspect schema

Perbaiki:

1. Tambahkan method `is_compressed(&self, name: &str) -> Option<bool>` di keduanya
2. Rename `tensor_slice()` menjadi `tensor_slice_raw()` dengan doc yang jelas: "returns raw payload bytes, which may be compressed"
3. Tambahkan `tensor_slice_uncompressed(&self, name: &str) -> Result<Option<Vec<u8>>>` yang selalu return decompressed bytes
4. Buat `tensor_slice()` sebagai alias ke `tensor_slice_raw()` dengan deprecation notice jika semanticnya membingungkan
5. Update semua callsite internal
6. Pastikan benchmark di `ncf-io/benches/ncf_performance.rs` menggunakan method yang benar

Tulis test yang memverifikasi bahwa `tensor_slice_raw()` return compressed bytes dan `tensor_slice_uncompressed()` return identical data untuk tensor compressed dan uncompressed.
```

---

## PHASE 3 — Performance Hardening
*Target: Optimasi hot path, parallel I/O, memory efficiency*

---

**Prompt 3.1 — Parallel tensor writing di Writer**

```
@workspace Di ncf-io/src/writer.rs, method `finalize()` mengompresi semua tensor secara sequential:

```rust
let prepared_chunks: Vec<PreparedChunk> = self.tensors.iter()
    .map(|(tensor, payload)| { compress_payload(...) })
    .collect::<Result<Vec<_>>>()?;
```

Untuk model besar (100+ tensor), kompresi sequential sangat lambat.

Implementasikan parallel compression:

1. Tambahkan optional dependency di `ncf-io/Cargo.toml`:
```toml
rayon = { version = "1.7", optional = true }

[features]
parallel = ["rayon"]
```

2. Dengan `#[cfg(feature = "parallel")]`, ganti `.iter()` dengan `.par_iter()` dari rayon
3. Karena kompresi bisa fail, gunakan `par_iter().map(...).collect::<Result<Vec<_>>>()?` — rayon mendukung ini
4. Tambahkan `NcfWriterConfig` struct:
```rust
pub struct NcfWriterConfig {
    pub parallel_compression: bool,
    pub compression_threads: Option<usize>, // None = use rayon default
}
```
5. Tambahkan `NcfWriter::with_config(metadata, flags, config)` constructor
6. Tulis benchmark `writer_parallel_vs_sequential` yang mengukur speedup untuk 32 tensor × 32 MiB

Default tetap sequential untuk `NcfWriter::new()` agar backward compatible.
```

---

**Prompt 3.2 — Schema cache dan lazy index rebuild**

```
@workspace `NcfIndex::build_from_schemas()` di ncf-core/src/index.rs saat ini hanya mengindeks chunk pertama dari setiap tensor. Tensor multi-chunk (tensor besar yang dipecah) tidak bisa di-lookup chunk-by-chunk.

Perbaiki:

1. Ubah `tensor_map: BTreeMap<String, u64>` menjadi `tensor_map: BTreeMap<String, Vec<u64>>` — map ke semua chunk_id untuk tensor tersebut

2. Update `find_chunk_id()` menjadi `find_chunk_ids()` yang return `Option<&[u64]>`

3. Tambahkan `NcfIndex::chunk_count_for_tensor(&self, name: &str) -> Option<usize>`

4. Di `NcfMmap` dan `NcfReader`, ketika membangun `chunk_map`, iterasi semua chunks dari setiap schema, bukan hanya `schema.chunks.first()`:
```rust
for schema in &schemas {
    for chunk_ref in &schema.chunks {
        // index semua chunks
    }
}
```

5. Tambahkan `NcfIndex::total_data_bytes(&self) -> u64` yang menjumlahkan semua `uncompressed_len`

6. Update `build_from_schemas()` di `index.rs` dengan logika yang sama

Tulis test dengan tensor multi-chunk dan verifikasi semua chunk bisa di-lookup.

```

---

**Prompt 3.3 — Prefetch API dan madvise hints**

```
@workspace Tambahkan API prefetching untuk use case inference yang sudah tahu layer mana yang akan diakses.

Di ncf-io/src/mmap.rs, tambahkan:

```rust
impl NcfMmap {
    /// Hint ke OS untuk preload tensor ke page cache sebelum dibutuhkan.
    /// Non-blocking: return segera, OS handle di background.
    pub fn prefetch_tensors(&self, names: &[&str]) -> Result<()>
    
    /// Prefetch semua tensor secara sequential (untuk full-model load).
    pub fn prefetch_all(&self) -> Result<()>
    
    /// Drop hint ke OS bahwa tensor sudah tidak dibutuhkan (free page cache).
    pub fn evict_tensor(&self, name: &str) -> Result<()>
}
```

Implementasi menggunakan:
- Linux: `madvise(MADV_WILLNEED)` untuk prefetch, `madvise(MADV_DONTNEED)` untuk evict
- macOS: `madvise(MADV_WILLNEED)` juga tersedia
- Windows: `PrefetchVirtualMemory()` via `windows-sys` crate
- Fallback: no-op dengan `Ok(())` untuk platform lain

Gunakan `#[cfg(target_os = "linux")]`, `#[cfg(target_os = "macos")]`, `#[cfg(target_os = "windows")]` guards.

Tulis benchmark `mmap_with_prefetch_vs_cold` yang membandingkan:
1. mmap tensor tanpa prefetch (cold)
2. mmap tensor setelah prefetch (warm)
```

---

**Prompt 3.4 — Optimasi `NcfWriter` untuk large payloads**

```
@workspace Di ncf-io/src/writer.rs, method `add_tensor()` menerima `Vec<u8>` yang meng-clone data:

```rust
pub fn add_tensor(&mut self, schema: TensorSchema, payload: Vec<u8>) {
    self.tensors.push((schema, payload));
}
```

Untuk model besar, ini berarti semua tensor data di-hold di memory sebelum write.

Implementasikan streaming writer mode:

1. Buat trait `TensorSource`:
```rust
pub trait TensorSource: Send {
    fn schema(&self) -> &TensorSchema;
    fn read_payload(&mut self) -> Result<Vec<u8>>;
    fn payload_len_hint(&self) -> Option<u64> { None }
}
```

2. Implementasi `VecTensorSource` (existing behavior) dan `FileTensorSource` (baca dari file saat write)

3. Tambahkan `NcfWriter::add_tensor_source<S: TensorSource>(&mut self, source: S)`

4. Di `finalize()`, panggil `source.read_payload()` per tensor saat sedang di-write, bukan hold semua di memory

5. Ganti `Vec<(TensorSchema, Vec<u8>)>` menjadi `Vec<Box<dyn TensorSource>>`

6. Pastikan existing `add_tensor(schema, payload: Vec<u8>)` tetap work sebagai convenience wrapper via `VecTensorSource`

Tulis test yang menulis model 1 GiB tanpa peak memory > 200 MiB menggunakan `FileTensorSource`.
```

---

**Prompt 3.5 — Benchmark suite expansion**

```
@workspace Benchmark di ncf-io/benches/ncf_performance.rs saat ini menggunakan data synthetic kecil. Perluas untuk coverage yang realistis.

Tambahkan benchmark groups berikut:

GROUP 1: "model_scale"
- `bench_16_layers_32mb` (existing: 512 MiB total)
- `bench_32_layers_32mb` (1 GiB total)
- `bench_64_layers_32mb` (2 GiB total)

GROUP 2: "compression_algorithms"
Untuk payload 32 MiB:
- `write_no_compression`
- `write_zstd_level_1`
- `write_zstd_level_3`
- `write_lz4`
- `write_snappy`
Ukur write throughput (GB/s)

GROUP 3: "access_patterns"
- `sequential_all_layers` (baca semua layer urut)
- `random_single_layer` (baca 1 layer random)
- `selective_first_half` (baca hanya layer 0..n/2)
- `concurrent_reads_4threads` (pakai rayon untuk 4 goroutine simultan)

GROUP 4: "vs_safetensors"
- `ncf_full_load_512mb` vs `safetensors_full_load_512mb`
- `ncf_selective_4_of_16` vs `safetensors_selective_4_of_16`

Gunakan `criterion::BenchmarkGroup::throughput(Throughput::Bytes(n))` untuk semua benchmark sehingga output dalam GB/s bukan ns/iter saja.

Output benchmark ke file `benchmark-output.txt` via `cargo bench -- --output-format bencher 2>&1 | tee benchmark-output.txt`.
```

---

## PHASE 4 — Ecosystem & Interoperability
*Target: Koneksi ke ekosistem AI yang ada, tooling yang usable*

---

**Prompt 4.1 — Lengkapi GGUF converter**

```
@workspace Di ncf-convert/src/from_gguf.rs, converter GGUF ke NCF sudah ada tapi belum handle semua tensor type GGUF.

Audit dan perbaiki:

1. GGUF memiliki tensor types: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, dll. Buat mapping lengkap ke NCF `DType`:
```rust
fn gguf_type_to_ncf_dtype(gguf_type: u32) -> Result<DType> {
    match gguf_type {
        0 => Ok(DType::F32),
        1 => Ok(DType::F16),
        6 => Ok(DType::Q4_0),
        7 => Ok(DType::Q4_1),
        // ... lengkapi semua
        unknown => Err(NcfError::UnsupportedDType(unknown))
    }
}
```

2. GGUF menyimpan metadata sebagai key-value pairs. Map metadata GGUF standar ke `NcfHeader.metadata`:
- `general.name` → `model_name`
- `general.architecture` → `architecture`
- `general.author` → `author`
- `general.license` → `license`
- `general.quantization_version` → `quantization`
- Semua key lain → `custom` BTreeMap

3. Tambahkan `--preserve-quantization` flag: jika set, copy raw quantized bytes langsung tanpa de-quantize. Jika tidak set, de-quantize ke F32.

4. Tambahkan progress callback:
```rust
pub fn gguf_to_ncf<P, F>(input: P, output: P, on_progress: F) -> Result<()>
where F: Fn(usize, usize, &str) // (current, total, tensor_name)
```

5. Tulis integration test dengan sample GGUF file (buat synthetic GGUF dengan known values).
```

---

**Prompt 4.2 — Python binding lengkap**

```

@workspace Lengkapi ncf-py/src/lib.rs dengan Python API yang production-ready.

Implementasikan class berikut via PyO3:

```python
class NcfFile:
    @staticmethod
    def open(path: str) -> "NcfFile": ...
    def tensor_names(self) -> list[str]: ...
    def tensor_shape(self, name: str) -> list[int]: ...
    def tensor_dtype(self, name: str) -> str: ...  # "F32", "F16", etc
    def read_tensor(self, name: str) -> np.ndarray: ...
    def metadata(self) -> dict: ...
    def verify(self, name: str = None) -> bool: ...  # None = verify all

class NcfWriter:
    def __init__(self, model_name: str, architecture: str): ...
    def add_tensor(self, name: str, array: np.ndarray, compression: str = "none"): ...
    # compression: "none" | "zstd" | "lz4" | "snappy"
    def save(self, path: str): ...

def convert_from_safetensors(input_path: str, output_path: str, **kwargs) -> None: ...
def convert_from_gguf(input_path: str, output_path: str, **kwargs) -> None: ...
```

Untuk `add_tensor(array: np.ndarray)`:
- Support numpy dtypes: float32, float16, bfloat16 (via `ml_dtypes`), int8, uint8, int16, int32, int64
- Gunakan `array.tobytes()` untuk ekstrak raw bytes, deteksi dtype dari `array.dtype`
- Validasi bahwa array C-contiguous, jika tidak: `np.ascontiguousarray(array)`

Tambahkan `__repr__` dan `__str__` yang informatif untuk semua class.

Tambahkan `setup.py` / `pyproject.toml` menggunakan maturin sebagai build backend.
```

---

**Prompt 4.3 — CLI: tambahkan subcommand `diff` dan `validate`**

```
@workspace Di ncf-cli/src/main.rs, tambahkan 2 subcommand baru ke CLI yang sudah ada.

SUBCOMMAND 1: `ncf validate <file>`
```
ncf validate model.ncf
ncf validate model.ncf --tensor layers.0.attn.q
ncf validate model.ncf --all --json
```

Output:
- Verifikasi magic bytes dan version
- Verifikasi setiap chunk checksum (Blake3)
- Report tensor mana yang pass/fail
- `--json` flag untuk machine-readable output
- Exit code 0 jika semua pass, 1 jika ada failure

Implementasi menggunakan `NcfReader::verify_all()` dan `verify_tensor()` yang sudah ada.

SUBCOMMAND 2: `ncf diff <file1> <file2>`
```
ncf diff model_v1.ncf model_v2.ncf
ncf diff model_v1.ncf model_v2.ncf --stats-only
```

Output:
- Tensor yang ada di file1 tapi tidak di file2 (removed)
- Tensor yang ada di file2 tapi tidak di file1 (added)
- Tensor yang ada di keduanya tapi berbeda shape atau dtype (changed)
- Tensor yang identik (unchanged) — hanya tampil dengan `--verbose`
- Summary: X added, Y removed, Z changed, W unchanged

Untuk "changed": bandingkan shape dan dtype dari schema saja, tidak baca payload.

Gunakan clap derive macros untuk argument parsing. Tambahkan `--color` flag untuk colored output menggunakan `anstream` atau `colored` crate.
```

---

**Prompt 4.4 — Tambahkan `ncf-compat` crate untuk backward compatibility**

```

@workspace Buat crate baru `ncf-compat` di workspace untuk handle format versioning dan migration.

Struktur:
```
ncf-compat/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── version.rs      # Version detection
    ├── migrate_v0.rs   # Migration dari format hypothetical v0
    └── detect.rs       # Format fingerprinting
```

Implementasikan:

1. `FormatDetector`: deteksi format file dari magic bytes
```rust
pub enum DetectedFormat {
    Ncf { version: (u8, u8, u8) },
    Safetensors,
    Gguf { version: u32 },
    Unknown,
}

pub fn detect_format(path: &Path) -> Result<DetectedFormat>
pub fn detect_format_bytes(header_bytes: &[u8]) -> DetectedFormat
```

2. `VersionMigrator`: framework untuk migration antar NCF versions
```rust
pub trait NcfMigration {
    fn from_version(&self) -> (u8, u8, u8);
    fn to_version(&self) -> (u8, u8, u8);
    fn migrate(&self, input: &Path, output: &Path) -> Result<()>;
}
```

3. Auto-migration: jika user membuka NCF v0.x dengan library v1.x, beri opsi migrate:
```rust
pub fn open_with_migration(path: &Path, target_version: (u8, u8, u8)) 
    -> Result<MigrationResult>

pub enum MigrationResult {
    AlreadyCurrent(NcfReader),
    MigrationAvailable { from: (u8,u8,u8), to: (u8,u8,u8) },
    MigrationRequired { from: (u8,u8,u8) },
}
```

4. Tambahkan crate ke workspace members di root `Cargo.toml`
5. Buat test yang verify magic byte detection bekerja untuk semua format
```

---

**Prompt 4.5 — Integrasi dengan HuggingFace Hub format**

```
@workspace Tambahkan di ncf-convert support untuk HuggingFace model repository format.

HuggingFace model terdiri dari:
- `config.json` — model architecture config
- `*.safetensors` atau `pytorch_model.bin` — weights (satu atau sharded)
- `tokenizer.json`, `tokenizer_config.json` — tokenizer
- `generation_config.json` — inference config

Implementasikan di ncf-convert/src/from_hf.rs:

```rust
pub struct HfModelConverter {
    pub model_dir: PathBuf,
    pub output_path: PathBuf,
    pub compression: Compression,
    pub include_metadata: bool,  // embed config.json ke NCF custom metadata
}

impl HfModelConverter {
    pub fn new(model_dir: &Path, output_path: &Path) -> Self
    pub fn convert(&self) -> Result<ConversionReport>
}

pub struct ConversionReport {
    pub tensors_converted: usize,
    pub total_bytes: u64,
    pub shards_processed: usize,
    pub skipped_files: Vec<String>,
}
```

Logic:
1. Baca `config.json`, map `model_type` ke `architecture` field di NCF header
2. Detect apakah weights sharded (`model.safetensors.index.json` exists) atau single file
3. Untuk sharded: baca index file, process shard dalam urutan, merge ke single NCF file
4. Embed content `config.json` dan `tokenizer_config.json` ke `NcfHeader.metadata.custom` sebagai CBOR values
5. Progress reporting via callback

Tambahkan ke CLI: `ncf convert --from hf-model-dir --to output.ncf`
```

---

## PHASE 5 — Production Hardening & Observability
*Target: Logging, metrics, concurrency safety, formal testing*

---

**Prompt 5.1 — Tambahkan tracing dan structured logging**

```
@workspace Tambahkan structured logging dan tracing ke ncf-io untuk production observability.

Tambahkan optional dependency di ncf-io/Cargo.toml:
```toml
tracing = { version = "0.1", optional = true }

[features]
tracing = ["dep:tracing"]
```

Dengan `#[cfg(feature = "tracing")]`, tambahkan spans dan events di:

1. `NcfReader::open()` — span dengan fields: path, file_size_bytes, header_parse_duration_us
2. `NcfMmap::open()` — sama
3. `read_tensor()` — span dengan: tensor_name, chunk_count, compressed_bytes, decompressed_bytes, duration_us
4. `verify_all()` — span dengan: tensor_count, pass_count, fail_count
5. `NcfWriter::finalize()` — span dengan: tensor_count, total_bytes, compression_ratio, duration_ms

Contoh:
```rust
#[cfg(feature = "tracing")]
let _span = tracing::info_span!("ncf::read_tensor", tensor.name = name).entered();

#[cfg(feature = "tracing")]
tracing::debug!(
    tensor.compressed_bytes = compressed_bytes,
    tensor.decompressed_bytes = decompressed_bytes,
    tensor.compression_ratio = ratio,
    "tensor decompressed"
);
```

Tambahkan juga `tracing::warn!` untuk:
- Tensor ditemukan tapi checksum tidak diverifikasi (jika called via `tensor_slice_raw`)
- Schema decode memakan >100ms

Tanpa feature flag, zero overhead (tidak ada tracing calls sama sekali).
```

---

**Prompt 5.2 — Concurrency: thread-safe multi-reader**

```
@workspace Verifikasi dan dokumentasikan thread-safety guarantees NCF.

Audit semua public types di ncf-io:

1. `NcfMmap`: apakah aman untuk di-share across threads?
   - `Mmap` dari memmap2 adalah `Send + Sync` ✅
   - `OnceLock<...>` adalah `Send + Sync` ✅
   - `BTreeMap` fields: read-only setelah `open()` ✅
   - Tambahkan `static_assertions::assert_impl_all!(NcfMmap: Send, Sync);` di test

2. `NcfReader` (self_cell based): audit apakah self_cell menjamin Send+Sync
   - Tambahkan assertions yang sama
   - Jika tidak Sync, dokumentasikan mengapa dan berikan `Arc<NcfMmap>` sebagai alternatif

3. Tambahkan contoh di dokumentasi untuk multi-threaded inference:

```rust
use std::sync::Arc;
use rayon::prelude::*;

let reader = Arc::new(NcfMmap::open("model.ncf")?);
let layer_names: Vec<String> = (0..32).map(|i| format!("layer_{}", i)).collect();

let tensors: Vec<_> = layer_names.par_iter().map(|name| {
    let reader = Arc::clone(&reader);
    reader.tensor_slice_raw(name)
}).collect();
```

4. Buat benchmark `concurrent_read_4_threads` yang verify tidak ada data race:
   - Gunakan `std::thread::spawn` bukan rayon (untuk kontrol eksplisit)
   - Read 32 tensor simultan dari 4 thread
   - Verifikasi semua hasil identik dengan sequential read
```

---

**Prompt 5.3 — Property-based testing dengan proptest**

```
@workspace Tambahkan property-based testing menggunakan `proptest` untuk memverifikasi invariant format NCF.

Tambahkan di ncf-io/Cargo.toml dev-dependencies:

```toml
proptest = "1.4"
```

Buat file `ncf-io/src/prop_tests.rs` dengan test berikut:

PROPERTY 1: Round-trip invariant
```rust
proptest! {
    #[test]
    fn prop_roundtrip_preserves_data(
        tensor_count in 1usize..=10,
        tensor_size_kb in 1usize..=512,
        compression in prop_compression(),
    ) {
        // Generate random tensors
        // Write ke NCF
        // Read kembali
        // Assert: data identik byte-for-byte
    }
}
```

PROPERTY 2: Checksum correctness
```rust
proptest! {
    fn prop_checksum_detects_corruption(data: Vec<u8>, corrupt_offset: usize) {
        // Write tensor
        // Corrupt 1 byte di payload
        // verify_tensor() harus return false atau Err
    }
}
```

PROPERTY 3: Size bounds
```rust
proptest! {
    fn prop_file_size_is_deterministic(tensors: Vec<(String, Vec<u8>)>) {
        // Write sama 2x ke 2 file berbeda
        // File size harus identik
        // File content harus identik (byte-for-byte)
    }
}
```

PROPERTY 4: Index consistency
```rust
proptest! {
    fn prop_index_matches_schema(tensor_names: Vec<String>, shapes: Vec<Vec<u64>>) {
        // Setiap nama di tensor_map harus ada di schema
        // Setiap chunk_id di index harus valid
        // entry_count harus == entries.len()
    }
}
```

Tambahkan `mod prop_tests;` di `ncf-io/src/lib.rs` dengan `#[cfg(test)]`.
```

---

**Prompt 5.4 — Formal documentation dan README**

```
@workspace Buat dokumentasi production-grade untuk proyek NCF.

TASK 1: Perluas README.md dari 5 baris menjadi README lengkap:
```markdown
# NCF — Neural Columnar Format

## Quick Start (30 detik)
[code snippet: open file, read tensor]

## Why NCF?
[benchmark table: vs safetensors, vs GGUF]

## Installation
[Cargo.toml snippet, Python pip install]

## Usage: Rust
[contoh: NcfMmap, NcfReader, NcfWriter, streaming]

## Usage: Python  
[contoh: NcfFile.open(), read_tensor(), convert_from_safetensors()]

## CLI
[ncf inspect, ncf convert, ncf validate, ncf diff]

## Format Specification
[link ke BLUEPRINT.md]

## Performance
[benchmark table dari benchmark-output.txt]

## Roadmap
[dari BLUEPRINT.md, update status tiap item]

## Contributing
[cara run tests, cara run benchmarks, cara run fuzzer]

## License
[MIT/Apache 2.0 dual]
```

TASK 2: Tambahkan `#[doc = include_str!("../README.md")]` di `ncf-core/src/lib.rs` agar README muncul di docs.rs.

TASK 3: Tambahkan rustdoc examples di semua public API:
```rust
/// # Example
/// ```
/// use ncf_io::NcfMmap;
/// let file = NcfMmap::open("model.ncf")?;
/// let data = file.tensor_slice_raw("layer_0").unwrap();
/// println!("tensor size: {} bytes", data.len());
/// # Ok::<(), ncf_core::NcfError>(())
/// ```
```

TASK 4: Pastikan `cargo doc --no-deps --all-features` build tanpa warning.
TASK 5: Tambahkan `CHANGELOG.md` dengan entry `[Unreleased]` mengikuti format keepachangelog.com.
```

---

**Prompt 5.5 — CI/CD hardening final**

```
@workspace Upgrade CI pipeline di .github/workflows/ untuk production-grade quality gate.

UPGRADE 1: ci.yml — tambahkan quality gates
```yaml
jobs:
  test:
    # existing...
    steps:
      # Setelah test, tambahkan:
      - name: Check formatting
        run: cargo fmt --all -- --check
      
      - name: Clippy (deny warnings)
        run: cargo clippy --all-targets --all-features -- -D warnings
      
      - name: Check documentation
        run: cargo doc --no-deps --all-features 2>&1 | grep -E "^error" && exit 1 || true
      
      - name: Check semver compatibility
        run: cargo install cargo-semver-checks && cargo semver-checks

  coverage:
    name: Code coverage
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install cargo-tarpaulin
        run: cargo install cargo-tarpaulin
      - name: Generate coverage
        run: cargo tarpaulin --all-features --out Xml --output-dir coverage/
      - name: Upload to codecov
        uses: codecov/codecov-action@v4
        with:
          files: coverage/cobertura.xml
          fail_ci_if_error: true
          threshold: 70  # minimum 70% coverage
```

UPGRADE 2: bench.yml — jalankan benchmark di PR dan comment hasilnya
```yaml
- name: Run benchmarks
  run: cargo bench --package ncf-io 2>&1 | tee bench-output.txt

- name: Comment benchmark results on PR
  uses: actions/github-script@v7
  with:
    script: |
      const fs = require('fs');
      const bench = fs.readFileSync('bench-output.txt', 'utf8');
      github.rest.issues.createComment({
        issue_number: context.issue.number,
        owner: context.repo.owner,
        repo: context.repo.repo,
        body: '## Benchmark Results\n```\n' + bench + '\n```'
      });
```

UPGRADE 3: Tambahkan workflow baru `publish.yml` untuk auto-publish ke crates.io saat tag `v*.*.*` di-push:
```yaml
on:
  push:
    tags: ['v*.*.*']
jobs:
  publish:
    steps:
      - cargo publish -p ncf-core
      - cargo publish -p ncf-io  # setelah ncf-core
      - cargo publish -p ncf-convert
      - cargo publish -p ncf-cli
```

Dengan `CARGO_REGISTRY_TOKEN` dari GitHub Secrets.
```

---

## Urutan Eksekusi yang Direkomendasikan

```
Phase 1 → Selesaikan semua 5 prompt → commit "fix: critical bug fixes"
Phase 2 → Selesaikan semua 5 prompt → commit "refactor: code quality"  
Phase 3 → Selesaikan semua 5 prompt → commit "perf: performance hardening"
Phase 4 → Selesaikan semua 5 prompt → commit "feat: ecosystem integration"
Phase 5 → Selesaikan semua 5 prompt → commit "chore: production hardening"
```

Jalankan `cargo test --workspace` dan `cargo clippy --all-targets -- -D warnings` setelah setiap phase sebelum lanjut ke phase berikutnya.
