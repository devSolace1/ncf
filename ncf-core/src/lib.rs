//! Core types and utilities for the NCF file format.
#![deny(missing_docs)]

/// Chunk-level structures and helpers.
pub mod chunk;
/// Header encoding/decoding utilities and types.
pub mod header;
/// Index block representation and helpers.
pub mod index;
/// Tensor schemas and related enums.
pub mod schema;

pub use chunk::*;
pub use header::*;
pub use index::*;
pub use schema::*;

/// NCF Format Structure Constants
///
/// These constants define the exact binary layout of the NCF format.
/// Changing these values would break backward compatibility.
pub mod constants {
    /// Size of FileHeaderPrefix structure in bytes
    /// magic (8) + version (4) + flags (4) + header_len (8) + schema_offset (8) + index_offset (8) + chunk_count (8)
    pub const FILE_HEADER_PREFIX_SIZE: u64 = 48;

    /// Size of ChunkHeader structure in bytes
    /// magic (4) + chunk_id (8) + flags (2) + uncompressed_len (8) + compressed_len (8)
    pub const CHUNK_HEADER_SIZE: u64 = 30;

    /// Size of Blake3 checksum in bytes
    pub const CHUNK_CHECKSUM_SIZE: u64 = 32;

    /// Total overhead per chunk (header + checksum)
    pub const CHUNK_OVERHEAD: u64 = CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE;
}

/// Common result type returned by NCF core APIs.
pub type Result<T> = std::result::Result<T, NcfError>;
