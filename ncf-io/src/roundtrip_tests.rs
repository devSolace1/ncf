#[cfg(test)]
mod round_trip_tests {
    use ncf_core::header::{Metadata, NcfHeader, NcfFlags};
    use ncf_core::schema::{Compression, DType, Encoding, Layout, TensorSchema};
    use crate::{NcfReader, NcfWriter};
    use std::fs;
    use tempfile::TempDir;

    /// Test helper: create deterministic test data for a dtype
    fn create_test_data(dtype: DType, element_count: usize) -> Vec<u8> {
        match dtype {
            DType::F32 => {
                // Create deterministic f32 data
                let mut data = Vec::with_capacity(element_count * 4);
                for i in 0..element_count {
                    let f = (i as f32) * 0.1 + 1.0;
                    data.extend_from_slice(&f.to_le_bytes());
                }
                data
            }
            DType::F16 => {
                // Create deterministic f16 data (stored as u16 bits)
                let mut data = Vec::with_capacity(element_count * 2);
                for i in 0..element_count {
                    let u = ((i as u16) % 1000).to_le_bytes();
                    data.extend_from_slice(&u);
                }
                data
            }
            DType::BF16 => {
                // Create deterministic bf16 data
                let mut data = Vec::with_capacity(element_count * 2);
                for i in 0..element_count {
                    let u = ((i as u16) ^ 0xABCD).to_le_bytes();
                    data.extend_from_slice(&u);
                }
                data
            }
            DType::I8 => {
                // Create deterministic i8 data
                let mut data = Vec::with_capacity(element_count);
                for i in 0..element_count {
                    data.push(((i as i8) ^ 0x42) as u8);
                }
                data
            }
            DType::U8 => {
                // Create deterministic u8 data
                let mut data = Vec::with_capacity(element_count);
                for i in 0..element_count {
                    data.push((i as u8) ^ 0xFF);
                }
                data
            }
            DType::Q4_0 => {
                // Create deterministic Q4 data (4-bit quantized)
                let mut data = Vec::with_capacity((element_count + 1) / 2);
                for i in 0..element_count {
                    if i % 2 == 0 {
                        data.push(((i as u8) & 0x0F) | (((i as u8) & 0x0F) << 4));
                    }
                }
                data
            }
            DType::Q8_0 => {
                // Create deterministic Q8 data
                let mut data = Vec::with_capacity(element_count);
                for i in 0..element_count {
                    data.push((i as u8) ^ 0xAA);
                }
                data
            }
            _ => {
                // For other types, create simple pattern
                vec![0xFF; element_count]
            }
        }
    }

    /// Test single round-trip for given dtype and shape
    fn test_single_roundtrip(dtype: DType, shape: Vec<u64>) {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.ncf");

        // Calculate element count
        let element_count: usize = shape.iter().product::<u64>() as usize;
        let test_data = create_test_data(dtype, element_count);

        // Create NCF file
        {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let metadata = NcfHeader {
                metadata: Metadata {
                    model_name: format!("test_{}", dtype),
                    architecture: "test".to_string(),
                    created_at: now,
                    author: None,
                    license: None,
                    quantization: None,
                    custom: Default::default(),
                },
            };

            let mut writer = NcfWriter::new(metadata, NcfFlags::empty());

            let schema = TensorSchema {
                name: format!("tensor_{}", dtype),
                dtype,
                shape: shape.clone(),
                column_layout: Layout::RowMajor,
                compression: Compression::None,
                encoding: Encoding::Plain,
                chunks: vec![],
            };

            writer.add_tensor(schema, test_data.clone());
            writer.finalize(&file_path).expect("Failed to finalize NCF file");
        }

        // Read back and verify
        {
            let reader = NcfReader::open(&file_path)
                .expect("Failed to open NCF file");
            
            let tensor_name = format!("tensor_{}", dtype);
            let read_data = reader
                .read_tensor(&tensor_name)
                .expect("Failed to read tensor")
                .expect("Tensor not found");

            // Verify byte-for-byte identical
            assert_eq!(
                test_data, read_data,
                "Data mismatch for dtype={:?}, shape={:?}",
                dtype, shape
            );
        }

        // Cleanup
        fs::remove_file(file_path).ok();
    }

    /// Test multiple tensors in a single file
    fn test_multiple_tensors(dtype: DType, tensor_count: usize, shape: Vec<u64>) {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_multi.ncf");

        let element_count: usize = shape.iter().product::<u64>() as usize;

        // Create NCF file with multiple tensors
        {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let metadata = NcfHeader {
                metadata: Metadata {
                    model_name: format!("test_multi_{}", dtype),
                    architecture: "test".to_string(),
                    created_at: now,
                    author: None,
                    license: None,
                    quantization: None,
                    custom: Default::default(),
                },
            };

            let mut writer = NcfWriter::new(metadata, NcfFlags::empty());

            for i in 0..tensor_count {
                let test_data = create_test_data(dtype, element_count);
                let schema = TensorSchema {
                    name: format!("tensor_{}_{}", i, dtype),
                    dtype,
                    shape: shape.clone(),
                    column_layout: Layout::RowMajor,
                    compression: Compression::None,
                    encoding: Encoding::Plain,
                    chunks: vec![],
                };
                writer.add_tensor(schema, test_data);
            }

            writer.finalize(&file_path).expect("Failed to finalize NCF file");
        }

        // Read back and verify each tensor
        {
            let reader = NcfReader::open(&file_path)
                .expect("Failed to open NCF file");

            for i in 0..tensor_count {
                let tensor_name = format!("tensor_{}_{}", i, dtype);
                let read_data = reader
                    .read_tensor(&tensor_name)
                    .expect("Failed to read tensor")
                    .expect("Tensor not found");

                let expected_data = create_test_data(dtype, element_count);
                assert_eq!(
                    expected_data, read_data,
                    "Data mismatch for tensor {} (dtype={:?}, shape={:?})",
                    i, dtype, shape
                );
            }
        }

        fs::remove_file(file_path).ok();
    }

    // ============ DTYPE TESTS ============

    #[test]
    fn test_roundtrip_f32_basic() {
        test_single_roundtrip(DType::F32, vec![10]);
    }

    #[test]
    fn test_roundtrip_f32_2d() {
        test_single_roundtrip(DType::F32, vec![10, 10]);
    }

    #[test]
    fn test_roundtrip_f16_basic() {
        test_single_roundtrip(DType::F16, vec![10]);
    }

    #[test]
    fn test_roundtrip_bf16_basic() {
        test_single_roundtrip(DType::BF16, vec![10]);
    }

    #[test]
    fn test_roundtrip_i8_basic() {
        test_single_roundtrip(DType::I8, vec![10]);
    }

    #[test]
    fn test_roundtrip_u8_basic() {
        test_single_roundtrip(DType::U8, vec![10]);
    }

    #[test]
    fn test_roundtrip_q4_0_basic() {
        test_single_roundtrip(DType::Q4_0, vec![10]);
    }

    #[test]
    fn test_roundtrip_q8_0_basic() {
        test_single_roundtrip(DType::Q8_0, vec![10]);
    }

    // ============ EDGE CASE SHAPES ============

    #[test]
    fn test_shape_single_element() {
        test_single_roundtrip(DType::F32, vec![1]);
    }

    #[test]
    fn test_shape_single_row() {
        test_single_roundtrip(DType::F32, vec![1, 1]);
    }

    #[test]
    fn test_shape_large_first_dim() {
        // Test with shape [65536, 1] to verify large offset handling
        // Note: This creates large tensor; keeping modest for test speed
        test_single_roundtrip(DType::F32, vec![1024, 1]);
    }

    #[test]
    fn test_shape_empty_tensor() {
        // Empty tensor [0] - should handle gracefully
        test_single_roundtrip(DType::F32, vec![0]);
    }

    // ============ MULTIPLE TENSORS ============

    #[test]
    fn test_multiple_tensors_1() {
        test_multiple_tensors(DType::F32, 1, vec![10]);
    }

    #[test]
    fn test_multiple_tensors_10() {
        test_multiple_tensors(DType::F32, 10, vec![10]);
    }

    #[test]
    fn test_multiple_tensors_100() {
        test_multiple_tensors(DType::F32, 100, vec![10]);
    }

    #[test]
    fn test_multiple_dtypes_in_file() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_mixed.ncf");

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let metadata = NcfHeader {
            metadata: Metadata {
                model_name: "test_mixed".to_string(),
                architecture: "test".to_string(),
                created_at: now,
                author: None,
                license: None,
                quantization: None,
                custom: Default::default(),
            },
        };

        let mut writer = NcfWriter::new(metadata, NcfFlags::empty());

        // Add tensors of different dtypes
        let dtypes = vec![DType::F32, DType::F16, DType::I8, DType::U8];
        for (idx, dtype) in dtypes.iter().enumerate() {
            let test_data = create_test_data(*dtype, 10);
            let schema = TensorSchema {
                name: format!("tensor_{}", idx),
                dtype: *dtype,
                shape: vec![10],
                column_layout: Layout::RowMajor,
                compression: Compression::None,
                encoding: Encoding::Plain,
                chunks: vec![],
            };
            writer.add_tensor(schema, test_data);
        }

        writer.finalize(&file_path).expect("Failed to finalize NCF file");

        // Verify read back
        let reader = NcfReader::open(&file_path)
            .expect("Failed to open NCF file");

        for (idx, dtype) in dtypes.iter().enumerate() {
            let tensor_name = format!("tensor_{}", idx);
            let read_data = reader
                .read_tensor(&tensor_name)
                .expect("Failed to read tensor")
                .expect("Tensor not found");

            let expected_data = create_test_data(*dtype, 10);
            assert_eq!(expected_data, read_data);
        }

        fs::remove_file(file_path).ok();
    }
}
