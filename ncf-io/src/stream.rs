use std::fmt;

/// Errors related to streaming APIs (not implemented yet).
#[derive(Debug)]
pub enum NcfStreamError {
    /// Streaming not supported in this release.
    Unsupported,
}

impl std::error::Error for NcfStreamError {}
impl fmt::Display for NcfStreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "streaming is not implemented in this release")
    }
}

/// Placeholder streaming API type. Implementation is pending.
pub struct NcfStream;

impl NcfStream {
    /// Create a new placeholder `NcfStream`.
    pub fn new() -> Self {
        NcfStream
    }
}
