use std::fmt;

#[derive(Debug)]
pub enum NcfStreamError {
    Unsupported,
}

impl std::error::Error for NcfStreamError {}
impl fmt::Display for NcfStreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "streaming is not implemented in this release")
    }
}

pub struct NcfStream;

impl NcfStream {
    pub fn new() -> Self {
        NcfStream
    }
}
