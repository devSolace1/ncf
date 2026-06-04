//! Zero-copy, append-capable columnar KV cache built on top of NCF primitives.
#![deny(missing_docs)]

/// Error types returned by the cache engine.
pub mod error;

/// File header and metadata definitions for ncf-kvcache files.
pub mod header;

/// Chained index and payload entry definitions.
pub mod index;

/// Zero-copy reader for mapped cache files.
pub mod reader;

/// Append-only writer with async block flushing.
pub mod writer;

pub use crate::error::KvcacheError;
pub use crate::header::{BLOCK_TOKEN_COUNT, KVCACHE_HEADER_SIZE, KvCacheConfig, KvCacheHeader};
pub use crate::index::{ChunkIndexEntry, KvcacheIndex};
pub use crate::reader::KvcacheReader;
pub use crate::writer::KvCacheWriter;

/// Result type returned by ncf-kvcache operations.
pub type Result<T> = std::result::Result<T, KvcacheError>;
