use crate::error::KvcacheError;
use crate::header::{self, KVCACHE_HEADER_SIZE, KVCACHE_TRAILER_MAGIC, KvCacheConfig, KvCacheHeader, KvCacheMetadata};
use crate::index::{ChunkIndexEntry, IndexBlock, IndexTrailer};
use crate::reader::KvcacheReader;
use crate::Result;
use ciborium::ser::into_writer;
use crossbeam_channel::{self, Receiver, Sender};
use memmap2::MmapMut;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

struct ColumnBuffer {
    layer: u32,
    head: u32,
    block_idx: u64,
    token_count: u32,
    existing_tokens: u32,
    data: Vec<u8>,
}

impl ColumnBuffer {
    fn new(layer: u32, head: u32, element_bytes: usize) -> Self {
        Self {
            layer,
            head,
            block_idx: 0,
            token_count: 0,
            existing_tokens: 0,
            data: Vec::with_capacity(header::BLOCK_TOKEN_COUNT * element_bytes),
        }
    }

    fn append_token(&mut self, token: &[u8]) {
        self.data.extend_from_slice(token);
        self.token_count += 1;
    }

    fn reset(&mut self, element_bytes: usize) {
        self.data.clear();
        self.data.reserve(header::BLOCK_TOKEN_COUNT * element_bytes);
        self.token_count = 0;
        self.existing_tokens = 0;
        self.block_idx = self.block_idx.wrapping_add(1);
    }
}

struct FlushBatch {
    block_idx: u64,
    batch_epoch: u64,
    token_count: u32,
    existing_tokens: u32,
    entries: Vec<ChunkIndexEntry>,
    payloads: Vec<Vec<u8>>,
    prev_index_offset: u64,
}

enum FlushCommand {
    Flush(FlushBatch),
    Stop,
}

/// Asynchronous writer for append-only ncf-kvcache files.
pub struct KvCacheWriter {
    config: KvCacheConfig,
    buffers: Vec<ColumnBuffer>,
    sender: Sender<FlushCommand>,
    thread: Option<JoinHandle<()>>,
    header_mmap: MmapMut,
    current_token_count: u64,
    next_chunk_id: u64,
    pending_block_idx: u64,
    flush_error: Arc<Mutex<Option<KvcacheError>>>,
}

impl KvCacheWriter {
    /// Create a new writer and reserve the header block.
    pub fn create<P: AsRef<Path>>(
        path: P,
        config: KvCacheConfig,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let metadata = KvCacheMetadata {
            model_name: "ncf-kvcache".into(),
            architecture: Some("columnar-kv".into()),
            custom: Default::default(),
        };
        let mut metadata_bytes = Vec::new();
        into_writer(&metadata, &mut metadata_bytes)?;
        let metadata_len = metadata_bytes.len() as u32;

        let header = KvCacheHeader::new(&config, metadata_len);
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)?;
        file.write_all(&header.encode())?;
        file.write_all(&metadata_bytes)?;
        file.flush()?;

        let header_file = OpenOptions::new().read(true).write(true).open(&path)?;
        let header_mmap = unsafe { MmapMut::map_mut(&header_file)? };

        let (sender, receiver) = crossbeam_channel::unbounded();
        let flush_error = Arc::new(Mutex::new(None));
        let thread_path = path.clone();
        let error_clone = flush_error.clone();
        let handle = thread::Builder::new()
            .name("ncf-kvcache-flush".into())
            .spawn(move || Self::flush_loop(thread_path, receiver, error_clone))?;

        let mut buffers = Vec::new();
        for layer in 0..config.layers {
            for head in 0..config.heads {
                buffers.push(ColumnBuffer::new(layer, head, config.element_bytes as usize));
            }
        }

        Ok(Self {
            config,
            buffers,
            sender,
            thread: Some(handle),
            header_mmap,
            current_token_count: 0,
            next_chunk_id: 0,
            pending_block_idx: 0,
            flush_error,
        })
    }

    /// Open an existing cache file for appending.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let reader = KvcacheReader::open(&path)?;
        let config = KvCacheConfig {
            layers: reader.header().layers,
            heads: reader.header().heads,
            element_bytes: reader.header().element_bytes,
        };
        let committed = reader.visible_token_count();
        let last_block_idx = if committed == 0 {
            0
        } else {
            (committed - 1) / header::BLOCK_TOKEN_COUNT as u64
        };
        let partial_count = (committed % header::BLOCK_TOKEN_COUNT as u64) as u32;
        let pending_block_idx = if partial_count == 0 {
            committed / header::BLOCK_TOKEN_COUNT as u64
        } else {
            last_block_idx
        };
        let next_chunk_id = reader.index().next_chunk_id();

        let header_file = OpenOptions::new().read(true).write(true).open(&path)?;
        let header_mmap = unsafe { MmapMut::map_mut(&header_file)? };

        let (sender, receiver) = crossbeam_channel::unbounded();
        let flush_error = Arc::new(Mutex::new(None));
        let thread_path = path.clone();
        let error_clone = flush_error.clone();
        let handle = thread::Builder::new()
            .name("ncf-kvcache-flush".into())
            .spawn(move || Self::flush_loop(thread_path, receiver, error_clone))?;

        let mut buffers = Vec::new();
        for layer in 0..config.layers {
            for head in 0..config.heads {
                let mut buffer = ColumnBuffer::new(layer, head, config.element_bytes as usize);
                buffer.block_idx = pending_block_idx;
                if partial_count > 0 {
                    if let Some(existing_bytes) = reader.block_bytes(layer, head, last_block_idx) {
                        buffer.data = existing_bytes.to_vec();
                        buffer.token_count = partial_count;
                        buffer.existing_tokens = partial_count;
                    } else {
                        return Err(KvcacheError::Layout(
                            "missing partial block payload when reopening cache".into(),
                        ));
                    }
                }
                buffers.push(buffer);
            }
        }

        Ok(Self {
            config,
            buffers,
            sender,
            thread: Some(handle),
            header_mmap,
            current_token_count: committed,
            next_chunk_id,
            pending_block_idx,
            flush_error,
        })
    }

    fn get_header_epoch(&self) -> &AtomicU64 {
        let ptr = self.header_mmap.as_ptr();
        let atomic_ptr = unsafe { ptr.add(32) as *const AtomicU64 };
        unsafe { &*atomic_ptr }
    }

    fn get_header_atomic(&self) -> &AtomicU64 {
        let ptr = self.header_mmap.as_ptr();
        let atomic_ptr = unsafe { ptr.add(40) as *const AtomicU64 };
        unsafe { &*atomic_ptr }
    }

    fn update_valid_token_count(&self, count: u64) -> Result<()> {
        let atomic = self.get_header_atomic();
        atomic.store(count, Ordering::Release);
        Ok(())
    }

    fn update_commit_epoch(&self, next_epoch: u64) -> Result<()> {
        let atomic = self.get_header_epoch();
        atomic.store(next_epoch, Ordering::Release);
        Ok(())
    }

    fn flush_loop(path: PathBuf, receiver: Receiver<FlushCommand>, error: Arc<Mutex<Option<KvcacheError>>>) {
        let file = match OpenOptions::new().read(true).write(true).open(&path) {
            Ok(file) => file,
            Err(err) => {
                let mut lock = error.lock().unwrap();
                *lock = Some(KvcacheError::Io(err));
                return;
            }
        };

        for command in receiver {
            match command {
                FlushCommand::Stop => break,
                FlushCommand::Flush(batch) => {
                    if let Err(err) = Self::write_batch(&file, &path, batch) {
                        let mut lock = error.lock().unwrap();
                        *lock = Some(err);
                        break;
                    }
                }
            }
        }
    }

    fn write_batch(file: &File, path: &Path, batch: FlushBatch) -> Result<()> {
        let mut file = file.try_clone()?;
        let mut entries = Vec::with_capacity(batch.entries.len());

        file.seek(SeekFrom::End(0))?;
        for (payload, mut entry) in batch.payloads.into_iter().zip(batch.entries.into_iter()) {
            let offset = file.stream_position()?;
            file.write_all(&payload)?;
            entry.byte_offset = offset;
            entry.byte_len = payload.len() as u64;
            entries.push(entry);
        }

        let index_block = IndexBlock {
            prev_index_offset: batch.prev_index_offset,
            block_idx: batch.block_idx,
            entries,
        };

        let mut cbor = Vec::new();
        into_writer(&index_block, &mut cbor)?;
        let trailer = IndexTrailer {
            magic: *KVCACHE_TRAILER_MAGIC,
            cbor_len: cbor.len() as u64,
            prev_index_offset: batch.prev_index_offset,
        };

        file.write_all(&cbor)?;
        let index_offset = file.stream_position()?;
        file.write_all(&trailer.encode())?;
        file.flush()?;
        let header_file = OpenOptions::new().read(true).write(true).open(path)?;
        let header_map = unsafe { MmapMut::map_mut(&header_file)? };
        let current_epoch = unsafe { &*(header_map.as_ptr().add(32) as *const AtomicU64) }
            .load(Ordering::Acquire);

        if batch.batch_epoch == current_epoch {
            let valid_atomic = unsafe { &*(header_map.as_ptr().add(40) as *const AtomicU64) };
            let index_atomic = unsafe { &*(header_map.as_ptr().add(48) as *const AtomicU64) };
            let current_valid = valid_atomic.load(Ordering::Acquire);
            let increment = batch
                .token_count
                .saturating_sub(batch.existing_tokens) as u64;
            valid_atomic.store(
                current_valid.saturating_add(increment),
                Ordering::Release,
            );
            index_atomic.store(index_offset, Ordering::Release);
            header_map.flush_async()?;
        }

        Ok(())
    }

    /// Append a full token frame for every layer and head.
    pub fn append_frame(&mut self, frame: &[u8]) -> Result<()> {
        if frame.len() != self.config.frame_stride() {
            return Err(KvcacheError::Layout(format!(
                "expected frame size {} but got {}",
                self.config.frame_stride(),
                frame.len()
            )));
        }

        for (index, buffer) in self.buffers.iter_mut().enumerate() {
            let token_bytes = &frame[index * self.config.element_bytes as usize
                ..(index + 1) * self.config.element_bytes as usize];
            buffer.append_token(token_bytes);
        }

        self.current_token_count = self.current_token_count.saturating_add(1);

        if self.buffers[0].token_count as usize == header::BLOCK_TOKEN_COUNT {
            self.flush_block(false)?;
        }

        Ok(())
    }

    /// Flush any partially filled block to disk without stopping the worker.
    pub fn flush_pending(&mut self) -> Result<()> {
        self.flush_block(false)
    }

    /// Read the current visible token count from the mmap header.
    pub fn visible_token_count(&self) -> u64 {
        self.get_header_atomic().load(Ordering::Acquire)
    }

    /// Read the current commit epoch from the mmap header.
    pub fn commit_epoch(&self) -> u64 {
        self.get_header_epoch().load(Ordering::Acquire)
    }

    fn flush_block(&mut self, final_block: bool) -> Result<()> {
        let token_count = self.buffers[0].token_count;
        if token_count == 0 {
            return Ok(());
        }

        let prev_index_offset = KvCacheHeader::decode(&self.header_mmap[..KVCACHE_HEADER_SIZE])?
            .index_head_offset;
        let mut entries = Vec::with_capacity(self.buffers.len());
        let mut payloads = Vec::with_capacity(self.buffers.len());

        for buffer in &mut self.buffers {
            let chunk_id = self.next_chunk_id;
            self.next_chunk_id = self.next_chunk_id.wrapping_add(1);
            let payload = std::mem::take(&mut buffer.data);
            entries.push(ChunkIndexEntry {
                chunk_id,
                layer: buffer.layer,
                head: buffer.head,
                block_idx: buffer.block_idx,
                byte_offset: 0,
                byte_len: 0,
                token_count,
            });
            payloads.push(payload);
        }

        let batch = FlushBatch {
            block_idx: self.pending_block_idx,
            batch_epoch: self.get_header_epoch().load(Ordering::Acquire),
            token_count,
            existing_tokens: self.buffers[0].existing_tokens,
            entries,
            payloads,
            prev_index_offset,
        };

        self.sender
            .send(FlushCommand::Flush(batch))
            .map_err(|err| KvcacheError::Flush(err.to_string()))?;

        self.pending_block_idx = self.pending_block_idx.wrapping_add(1);
        for buffer in &mut self.buffers {
            buffer.reset(self.config.element_bytes as usize);
        }

        if final_block {
            self.drain_flush_worker()?;
        }

        Ok(())
    }

    fn drain_flush_worker(&mut self) -> Result<()> {
        self.sender
            .send(FlushCommand::Stop)
            .map_err(|err| KvcacheError::Flush(err.to_string()))?;
        if let Some(handle) = self.thread.take() {
            handle
                .join()
                .map_err(|_| KvcacheError::Flush("flush thread panicked".into()))?;
        }
        if let Some(err) = self.flush_error.lock().unwrap().take() {
            return Err(err);
        }
        Ok(())
    }

    /// Truncate the visible token count without modifying backing storage.
    pub fn truncate(&mut self, count: u64) -> Result<()> {
        if count > self.current_token_count {
            return Err(KvcacheError::Layout(format!(
                "truncate count {} exceeds current token count {}",
                count, self.current_token_count
            )));
        }

        let next_epoch = self
            .get_header_epoch()
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);
        self.update_commit_epoch(next_epoch)?;

        let committed = self.get_header_atomic().load(Ordering::Acquire);
        if count <= committed {
            self.current_token_count = count;
            return self.update_valid_token_count(count);
        }

        let token_bytes = self.config.element_bytes as usize;
        let pending_count = self.current_token_count.checked_sub(committed).ok_or_else(|| {
            KvcacheError::Overflow("pending token count underflow".into())
        })?;
        let keep_pending = count.checked_sub(committed).ok_or_else(|| {
            KvcacheError::Overflow("keep pending count underflow".into())
        })?;
        let drop_tokens = pending_count.checked_sub(keep_pending).ok_or_else(|| {
            KvcacheError::Overflow("drop token count underflow".into())
        })?;

        let new_buffer_count = self.buffers[0]
            .token_count
            .checked_sub(drop_tokens as u32)
            .ok_or_else(|| KvcacheError::Overflow("buffer truncate underflow".into()))?;

        for buffer in &mut self.buffers {
            buffer.token_count = new_buffer_count;
            buffer.existing_tokens = buffer.existing_tokens.min(new_buffer_count);
            buffer.data.truncate(new_buffer_count as usize * token_bytes);
        }

        self.current_token_count = count;
        Ok(())
    }

    /// Return the current local token count that has been appended.
    pub fn local_token_count(&self) -> u64 {
        self.current_token_count
    }

    /// Return the committed token count visible to readers.
    pub fn committed_token_count(&self) -> u64 {
        self.get_header_atomic().load(Ordering::Acquire)
    }
}

impl Drop for KvCacheWriter {
    fn drop(&mut self) {
        let _ = self.flush_block(true);
        if let Some(handle) = self.thread.take() {
            let _ = self.sender.send(FlushCommand::Stop);
            let _ = handle.join();
        }
    }
}
