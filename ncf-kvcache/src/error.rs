use thiserror::Error;

/// Error type returned from the KV cache crate.
#[derive(Debug, Error)]
pub enum KvcacheError {
    /// IO or mmap error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// CBOR deserialization error.
    #[error("CBOR deserialize error: {0}")]
    Cbor(#[from] ciborium::de::Error<std::io::Error>),

    /// CBOR serialization error.
    #[error("CBOR serialize error: {0}")]
    CborSer(#[from] ciborium::ser::Error<std::io::Error>),

    /// Invalid file layout or index corruption.
    #[error("layout error: {0}")]
    Layout(String),

    /// Overflow seen during offset arithmetic.
    #[error("overflow error: {0}")]
    Overflow(String),

    /// Flush worker failed.
    #[error("flush worker error: {0}")]
    Flush(String),
}
