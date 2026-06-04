use std::path::Path;

use crate::writer::NcfWriter;
use ncf_core::header::{NcfFlags, NcfHeader};
use ncf_core::schema::TensorSchema;
use ncf_core::Result;

/// Streaming writer for NCF files.
#[derive(Debug)]
pub struct NcfStream {
    writer: NcfWriter,
}

impl NcfStream {
    /// Create a new streaming NCF writer.
    pub fn new(metadata: NcfHeader, flags: NcfFlags) -> Self {
        Self {
            writer: NcfWriter::new(metadata, flags),
        }
    }

    /// Add a tensor to the streaming writer.
    pub fn add_tensor(&mut self, schema: TensorSchema, payload: Vec<u8>) {
        self.writer.add_tensor(schema, payload);
    }

    /// Finalize the stream and write the NCF file.
    pub fn finalize<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.writer.finalize(path)
    }

    /// Convenience helper to write a stream directly to a file path.
    pub fn write_to_path<P: AsRef<Path>>(metadata: NcfHeader, flags: NcfFlags, tensors: Vec<(TensorSchema, Vec<u8>)>, path: P) -> Result<()> {
        let mut stream = Self::new(metadata, flags);
        for (schema, payload) in tensors {
            stream.add_tensor(schema, payload);
        }
        stream.finalize(path)
    }
}
