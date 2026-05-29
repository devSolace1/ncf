# Neural Columnar Format (.ncf)
### Blueprint v0.1 — *"Engineered for the Next Century of AI"*

---

## Visi

> Format `.ncf` lahir dari satu keyakinan: model AI berhak mendapatkan format penyimpanan yang **secepat memori**, **seringan mungkin**, dan **siap untuk streaming** — tanpa kompromi. Bukan patch di atas format lama. Ini rancangan ulang dari nol.

| Format | Arsitektur | Zero-Copy | Streaming | Kolumnar | Bahasa |
|--------|-----------|-----------|-----------|----------|--------|
| safetensors | Row-based | Sebagian | ❌ | ❌ | Python/Rust |
| GGUF | Row-based | ❌ | ❌ | ❌ | C |
| ONNX | Protobuf | ❌ | ❌ | ❌ | C++ |
| **NCF** | **Columnar** | **✅ Penuh** | **✅ Native** | **✅** | **Rust murni** |

---

## Prinsip Desain

1. **Columnar-first** — setiap tensor disimpan per-dimensi, bukan per-elemen. Layer attention bisa dibaca tanpa menyentuh layer lain.
2. **Zero-copy** — data dibaca langsung dari `mmap`, tanpa alokasi heap tambahan. Pointer diberikan, bukan salinan.
3. **Chunk/streaming native** — format dirancang agar bisa dikirim dan dibaca secara parsial, cocok untuk edge device dan download progresif.
4. **Self-describing** — setiap file membawa skema lengkapnya. Tidak perlu file eksternal.
5. **Pure Rust** — tidak ada C FFI, tidak ada unsafe yang tidak perlu. `#[no_std]`-friendly untuk embedded.

---

## Struktur File

```
┌─────────────────────────────────────────────┐
│  MAGIC (8 bytes)  │  VERSION (4 bytes)       │
│  FLAGS  (4 bytes) │  HEADER_LEN (8 bytes)    │
├─────────────────────────────────────────────┤
│                                             │
│           HEADER BLOCK (JSON/CBOR)          │
│   - metadata model                          │
│   - daftar semua tensor & kolom             │
│   - offset setiap chunk                     │
│                                             │
├─────────────────────────────────────────────┤
│                                             │
│           SCHEMA BLOCK                      │
│   - dtype per kolom                         │
│   - shape, compression, encoding            │
│                                             │
├─────────────────────────────────────────────┤
│  CHUNK 0  │  CHUNK 1  │  CHUNK 2  │  ...    │
│  (kolom)  │  (kolom)  │  (kolom)  │         │
├─────────────────────────────────────────────┤
│           INDEX BLOCK                       │
│   - offset tabel untuk random access        │
│   - checksum per chunk (Blake3)             │
├─────────────────────────────────────────────┤
│  FOOTER MAGIC (8 bytes) │ FOOTER_LEN (8 b)  │
└─────────────────────────────────────────────┘
```

> **Mengapa Footer?** Sama seperti Parquet — file bisa dibaca dari ujung tanpa scan penuh. Efisien untuk streaming dan partial download.

---

## Header Block

Ditulis dalam **CBOR** (bukan JSON biasa) agar compact dan bisa di-parse tanpa alokasi string besar.

> CBOR parsing menambah biaya open file awal, karena NCF membaca dan memvalidasi header, skema, dan indeks.
> Ini trade-off yang sah untuk fitur richer metadata, extensibility, dan zero-copy access di jalur baca berikutnya.

```rust
struct NcfHeader {
    magic: [u8; 8],          // b"NCF\x00\xDE\xAD\xBE\xEF"
    version: u32,            // semver encoded: major<<16 | minor<<8 | patch
    flags: NcfFlags,         // bitfield: compressed, encrypted, streaming_safe
    header_len: u64,
    schema_offset: u64,
    index_offset: u64,
    chunk_count: u64,
    metadata: Metadata,
}

struct Metadata {
    model_name: String,
    architecture: String,    // "llama", "mistral", "bert", dst.
    created_at: u64,         // unix timestamp
    author: Option<String>,
    license: Option<String>,
    quantization: Option<String>, // "Q4_K_M", "F16", dst.
    custom: HashMap<String, CborValue>, // extensible tanpa breaking change
}
```

---

## Schema Block — Inti Kolumnar

Setiap **tensor** dipecah menjadi **kolom-kolom logis** berdasarkan dimensinya.

```rust
struct TensorSchema {
    name: String,             // "model.layers.0.self_attn.q_proj.weight"
    dtype: DType,             // F32, F16, BF16, I8, U8, Q4, Q8, ...
    shape: Vec<u64>,          // [4096, 4096]
    column_layout: Layout,    // RowMajor | ColMajor | Tiled(N)
    compression: Compression, // None | Zstd(level) | Lz4 | Snappy
    encoding: Encoding,       // Plain | DeltaRLE | BitPacked | DictionaryRLE
    chunks: Vec<ChunkRef>,    // daftar chunk yang menyusun tensor ini
}

struct ChunkRef {
    chunk_id: u64,
    byte_offset: u64,
    byte_len: u64,
    uncompressed_len: u64,
    checksum: [u8; 32],      // Blake3
}

enum DType {
    F64, F32, F16, BF16,
    I32, I16, I8, U8,
    Q4K, Q4_0, Q8_0,        // format quant populer
    Custom(u8),              // extensible
}
```

---

## Chunk Layout — Penyimpanan Aktual

Setiap chunk adalah unit independen yang bisa di-mmap atau di-stream sendiri.

```
┌──────────────────────────────────┐
│  CHUNK_MAGIC (4 bytes)           │  b"NCFK"
│  chunk_id (8 bytes)              │
│  flags (2 bytes)                 │  compressed? encrypted? last?
│  uncompressed_len (8 bytes)      │
│  compressed_len (8 bytes)        │
├──────────────────────────────────┤
│                                  │
│         PAYLOAD                  │
│   (data kolom, bisa compressed)  │
│                                  │
├──────────────────────────────────┤
│  CHECKSUM Blake3 (32 bytes)      │
└──────────────────────────────────┘
```

**Keunggulan chunk independen:**
- Bisa diunduh sebagian (HTTP Range Request)
- Bisa diverifikasi sendiri
- Bisa diproses paralel (multi-thread atau multi-core)
- Bisa di-skip jika tidak dibutuhkan (selective layer loading)

---

## Zero-Copy dengan `mmap`

```rust
// Contoh pseudocode Rust
pub struct NcfMmap {
    mmap: memmap2::Mmap,
    index: &'static NcfIndex, // pointer langsung ke region mmap
}

impl NcfMmap {
    /// Kembalikan slice data tensor TANPA alokasi baru
    pub fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        let chunk_ref = self.index.find(name)?;
        // Pointer aritmetik langsung ke mmap — zero copy
        Some(&self.mmap[chunk_ref.byte_offset as usize
            ..chunk_ref.byte_offset as usize + chunk_ref.byte_len as usize])
    }
}
```

- Tidak ada `Vec::clone()`, tidak ada `memcpy` tersembunyi.
- Data hidup di address space OS, dibaca on-demand oleh page fault.
- Cocok untuk model yang lebih besar dari RAM (sparse loading).

---

## Streaming API

NCF mendukung dua mode baca:

### Mode 1: Random Access (lokal, mmap)
```rust
let file = NcfReader::open("model.ncf")?;
let weights = file.get_tensor("layers.0.attn.q")?; // O(1), zero-copy
```

### Mode 2: Streaming (jaringan / edge)
```rust
let mut stream = NcfStream::from_url("https://cdn.example.com/model.ncf");
stream.request_tensors(&["layers.0", "layers.1"]); // deklaratif

while let Some(chunk) = stream.next_chunk().await? {
    match chunk {
        NcfChunk::TensorData { name, data, is_last } => {
            engine.load_layer(&name, data);
        }
        NcfChunk::Metadata(meta) => { /* skema diterima duluan */ }
    }
}
```

**Fitur streaming:**
- Header + Schema dikirim pertama — receiver tahu struktur sebelum data tiba.
- Tensor prioritas bisa di-request lebih awal (selective prefetch).
- Setiap chunk diverifikasi Blake3 sebelum digunakan.

---

## Kompresi & Encoding

NCF tidak memaksa satu algoritma. Setiap kolom/tensor punya strategi sendiri:

| Tipe Data | Encoding Rekomendasi | Kompresi |
|-----------|---------------------|----------|
| Weight FP16/BF16 | Plain | Zstd level 3 |
| Weight INT4/INT8 | BitPacked | Lz4 (speed) |
| Bias & embedding | DeltaRLE | Zstd |
| Sparse tensor | DictionaryRLE | Zstd level 6 |
| Metadata/string | Plain | Zstd |

> Kompresi bersifat **opsional per-chunk** — untuk deployment inference, bisa pilih `None` agar latency minimum.

---

## Index Block

Diletakkan di akhir file (sebelum footer) untuk mendukung baca-dari-ujung:

```rust
struct NcfIndex {
    entry_count: u64,
    entries: Vec<IndexEntry>,
    tensor_map: HashMap<String, u64>, // nama → chunk_id pertama
}

struct IndexEntry {
    chunk_id: u64,
    byte_offset: u64,
    byte_len: u64,
    tensor_name_hash: u64, // xxHash3 untuk pencarian cepat
}
```

---

## Rencana Implementasi Rust

### Crate Structure
```
ncf/
├── ncf-core/        # tipe data, schema, serialisasi
│   └── src/
│       ├── header.rs
│       ├── schema.rs
│       ├── chunk.rs
│       └── index.rs
├── ncf-io/          # reader/writer, mmap, streaming
│   └── src/
│       ├── reader.rs
│       ├── writer.rs
│       ├── mmap.rs
│       └── stream.rs
├── ncf-convert/     # konverter dari safetensors, GGUF, ONNX
│   └── src/
│       ├── from_safetensors.rs
│       ├── from_gguf.rs
│       └── from_onnx.rs
└── ncf-cli/         # tool CLI: inspect, convert, benchmark
    └── src/
        └── main.rs
```

### Dependencies Minimal
```toml
[dependencies]
memmap2   = "0.9"       # zero-copy mmap
zstd      = "0.13"      # kompresi
lz4_flex  = "0.11"      # kompresi cepat
ciborium  = "0.2"       # CBOR encode/decode
blake3    = "1.5"        # checksum cepat
xxhash-rust = "0.8"     # hash index
tokio     = { version = "1", optional = true }  # async streaming
```

---

## Roadmap

### Fase 1 — Fondasi (v0.1)
- [ ] Spesifikasi format final & magic bytes
- [ ] `ncf-core`: tipe data, schema, CBOR encode/decode
- [ ] `ncf-io`: writer & reader dasar
- [ ] Zero-copy mmap reader
- [ ] CLI `ncf inspect` & `ncf info`

### Fase 2 — Konversi & Ekosistem (v0.2)
- [ ] Konverter dari safetensors → NCF
- [ ] Konverter dari GGUF → NCF
- [ ] Python binding via PyO3
- [ ] Benchmark vs safetensors & GGUF
- [ ] Dokumentasi publik

### Fase 3 — Streaming & Edge (v0.3)
- [ ] Async streaming API (tokio)
- [ ] HTTP Range Request support
- [ ] Selective tensor loading
- [ ] WASM target (ncf-wasm)
- [ ] Enkripsi opsional (AES-256-GCM per chunk)

### Fase 4 — Produksi (v1.0)
- [ ] Fuzzing & formal verification bagian kritis
- [ ] Integrasi referensi dengan candle / burn
- [ ] Spesifikasi publik (RFC-style document)
- [ ] `no_std` support untuk embedded/firmware

---

## Keunggulan vs Kompetitor

| Fitur | safetensors | GGUF | ONNX | **NCF** |
|-------|------------|------|------|---------|
| Bahasa impl. utama | Rust/Python | C | C++ | **Rust murni** |
| Zero-copy mmap | Sebagian | ❌ | ❌ | **✅ Penuh** |
| Streaming native | ❌ | ❌ | ❌ | **✅** |
| Columnar storage | ❌ | ❌ | ❌ | **✅** |
| Selective layer load | ❌ | ❌ | ❌ | **✅** |
| Checksum per chunk | ❌ | ❌ | ❌ | **✅ Blake3** |
| Self-describing | ✅ | ✅ | ✅ | **✅** |
| Ekstensi custom | Terbatas | ✅ | Terbatas | **✅ CBOR** |
| Parallel load | ❌ | ❌ | ❌ | **✅** |
| Edge/embedded friendly | ❌ | Sebagian | ❌ | **✅** |

---

## Penutup

NCF bukan sekadar format baru — ini **infrastruktur generasi berikutnya** untuk AI deployment. Ketika model makin besar dan harus berjalan di mana saja — dari server farm hingga perangkat edge — kita butuh format yang dirancang dengan filosofi yang benar sejak awal.

> *"Format yang baik bukan yang paling kompleks, tapi yang paling jujur terhadap cara data benar-benar digunakan."*

---

*Blueprint ini adalah dokumen hidup. Setiap keputusan desain didokumentasikan di sini beserta alasannya, bukan hanya hasilnya.*

**Lisensi Spesifikasi:** Apache 2.0 / MIT dual-license (untuk adopsi seluas mungkin)
