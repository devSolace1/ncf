use crate::error::KvcacheError;
use crate::header::{KVCACHE_HEADER_SIZE, KVCACHE_MAGIC, KVCACHE_TRAILER_MAGIC, KvCacheHeader, KvCacheMetadata};
use crate::index::{IndexBlock, IndexTrailer, KvcacheIndex};
use crate::Result;
use ciborium::de::from_reader;
use memmap2::Mmap;
use self_cell::self_cell;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

self_cell! {
    /// Reader that owns a memory map and exposes borrowed cache views.
    pub struct KvcacheReader {
        owner: Mmap,
        #[covariant]
        dependent: KvcacheReaderData,
    }
}

#[derive(Debug)]
/// Owned dependent data for a mapped reader.
pub struct KvcacheReaderData<'this> {
    /// Cached header metadata from the mapped file.
    pub header: KvCacheHeader,
    /// Optional CBOR metadata stored after the header.
    pub metadata: KvCacheMetadata,
    /// In-memory index reconstructed from the chained footers.
    pub index: KvcacheIndex,
    /// File offset where payload bytes begin.
    pub payload_offset: usize,
    /// Zero-copy payload bytes available for block reads.
    pub payload: &'this [u8],
}

impl KvcacheReader {
    /// Open an existing ncf-kvcache file for zero-copy access.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < KVCACHE_HEADER_SIZE {
            return Err(KvcacheError::Layout("file too small to contain header".into()));
        }

        Self::try_new(mmap, |mmap| {
            let header = KvCacheHeader::decode(&mmap[..KVCACHE_HEADER_SIZE])?;
            if &header.magic != KVCACHE_MAGIC {
                return Err(KvcacheError::Layout("invalid kvcache magic".into()));
            }
            let metadata_len = header.metadata_len as usize;
            let metadata_end = KVCACHE_HEADER_SIZE
                .checked_add(metadata_len)
                .ok_or_else(|| KvcacheError::Overflow("metadata length overflow".into()))?;
            if metadata_end > mmap.len() {
                return Err(KvcacheError::Layout(
                    "metadata block extends past end of file".into(),
                ));
            }

            let metadata: KvCacheMetadata = from_reader(Cursor::new(&mmap[KVCACHE_HEADER_SIZE..metadata_end]))
                .map_err(KvcacheError::Cbor)?;

            let payload_offset = metadata_end;
            let index = Self::reconstruct_index(&mmap, header.index_head_offset)?;

            let payload = &mmap[payload_offset..];
            Ok(KvcacheReaderData {
                header,
                metadata,
                index,
                payload_offset,
                payload,
            })
        })
    }

    fn reconstruct_index(mmap: &Mmap, mut trailer_offset: u64) -> Result<KvcacheIndex> {
        let mut index = KvcacheIndex::default();
        while trailer_offset != 0 {
            let trailer_offset_usize = trailer_offset as usize;
            if trailer_offset_usize.checked_add(24).is_none()
                || trailer_offset_usize + 24 > mmap.len()
            {
                return Err(KvcacheError::Layout("invalid index trailer offset".into()));
            }
            let trailer = IndexTrailer::decode(&mmap[trailer_offset_usize..trailer_offset_usize + 24])
                .map_err(|err| KvcacheError::Layout(err))?;
            if &trailer.magic != KVCACHE_TRAILER_MAGIC {
                return Err(KvcacheError::Layout("invalid index trailer magic".into()));
            }

            let index_start = trailer_offset_usize
                .checked_sub(trailer.cbor_len as usize)
                .ok_or_else(|| KvcacheError::Overflow("index block underflow".into()))?;
            if index_start > mmap.len() {
                return Err(KvcacheError::Layout("index block extends past file".into()));
            }

            let block: IndexBlock = from_reader(Cursor::new(&mmap[index_start..trailer_offset_usize]))
                .map_err(KvcacheError::Cbor)?;
            index.insert_entries(block.entries);
            trailer_offset = trailer.prev_index_offset;
        }
        Ok(index)
    }

    fn header_atomic_ptr(&self) -> &AtomicU64 {
        let ptr = self.borrow_owner().as_ptr();
        let atomic_ptr = unsafe { ptr.add(40) as *const AtomicU64 };
        unsafe { &*atomic_ptr }
    }

    fn commit_epoch_ptr(&self) -> &AtomicU64 {
        let ptr = self.borrow_owner().as_ptr();
        let atomic_ptr = unsafe { ptr.add(32) as *const AtomicU64 };
        unsafe { &*atomic_ptr }
    }

    /// Return the currently visible valid token count.
    pub fn valid_token_count(&self) -> u64 {
        self.header_atomic_ptr().load(Ordering::Acquire)
    }

    /// Return the configured element width in bytes.
    pub fn element_bytes(&self) -> usize {
        self.borrow_dependent().header.element_bytes as usize
    }

    /// Return a read-only reference to the parsed cache header.
    pub fn header(&self) -> &KvCacheHeader {
        &self.borrow_dependent().header
    }

    /// Return the current commit epoch embedded in the header.
    pub fn commit_epoch(&self) -> u64 {
        self.commit_epoch_ptr().load(Ordering::Acquire)
    }

    /// Return the parsed CBOR metadata block.
    pub fn metadata(&self) -> &KvCacheMetadata {
        &self.borrow_dependent().metadata
    }

    /// Return the number of committed tokens visible to readers.
    pub fn visible_token_count(&self) -> u64 {
        self.valid_token_count()
    }

    /// Return the number of valid tokens in the given block entry.
    pub fn block_token_count(&self, layer: u32, head: u32, block_idx: u64) -> Option<u32> {
        self.borrow_dependent()
            .index
            .get(layer, head, block_idx)
            .map(|entry| entry.token_count)
    }

    /// Read a whole per-column block as a zero-copy byte slice.
    pub fn block_bytes(&self, layer: u32, head: u32, block_idx: u64) -> Option<&[u8]> {
        let entry = self.borrow_dependent().index.get(layer, head, block_idx)?;
        let start = entry
            .byte_offset
            .checked_sub(self.borrow_dependent().payload_offset as u64)? as usize;
        let end = start.checked_add(entry.byte_len as usize)?;
        Some(&self.borrow_dependent().payload[start..end])
    }

    /// Read a single token slice for a given layer/head and token index.
    pub fn token_bytes(&self, layer: u32, head: u32, token_index: u64) -> Option<&[u8]> {
        if token_index >= self.valid_token_count() {
            return None;
        }

        let block_idx = token_index >> 6;
        let inner_offset = (token_index & 63) as usize;
        let entry = self.borrow_dependent().index.get(layer, head, block_idx)?;
        let start = entry
            .byte_offset
            .checked_sub(self.borrow_dependent().payload_offset as u64)
            .and_then(|base| base.checked_add((inner_offset as u64).saturating_mul(self.element_bytes() as u64)))?
            as usize;
        let length = self.element_bytes();
        let end = start.checked_add(length)?;
        Some(&self.borrow_dependent().payload[start..end])
    }
}
