//! Utilities for reading and writing NCF files (I/O helpers).
#![deny(missing_docs)]

/// Memory-mapped reader implementation.
pub mod mmap;
/// Borrowed reader API for safe zero-copy access.
pub mod reader;
/// File writer utilities to create NCF files.
pub mod writer;
/// Streaming API placeholder (hidden until implemented.
pub mod stream;

#[cfg(feature = "http")]
/// HTTP-backed access for remote NCF files.
pub mod http_reader;

pub use mmap::NcfMmap;
pub use reader::NcfReader;
pub use writer::NcfWriter;
#[doc(hidden)]
pub use stream::NcfStream;
#[cfg(feature = "http")]
pub use http_reader::NcfHttpReader;

#[cfg(test)]
mod roundtrip_tests;
