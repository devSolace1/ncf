use clap::{Parser, Subcommand};
use ncf_convert::{gguf_to_ncf, safetensors_to_ncf};
use ncf_core::header::{Metadata, NcfHeader, NcfFlags};
use ncf_core::schema::{Compression, DType, Encoding, Layout, TensorSchema};
use ncf_io::NcfWriter;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about = "NCF CLI tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Inspect {
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },
    Info {
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },
    Create {
        #[arg(value_name = "INPUT")]
        input: PathBuf,
        #[arg(value_name = "OUTPUT")]
        output: PathBuf,
        #[arg(long, default_value = "tensor")]
        name: String,
    },
    ConvertSafetensors {
        #[arg(value_name = "INPUT")]
        input: PathBuf,
        #[arg(value_name = "OUTPUT")]
        output: PathBuf,
    },
    ConvertGguf {
        #[arg(value_name = "INPUT")]
        input: PathBuf,
        #[arg(value_name = "OUTPUT")]
        output: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Inspect { file } => {
            let reader = ncf_io::NcfReader::open(file)?;
            reader.inspect();
        }
        Commands::Info { file } => {
            let reader = ncf_io::NcfReader::open(file)?;
            let prefix = reader.header_prefix();
            println!("NCF v{}", prefix.version);
            println!("Flags: {}", prefix.flags);
            println!("Header length: {}", prefix.header_len);
            println!("Schema offset: {}", prefix.schema_offset);
            println!("Index offset: {}", prefix.index_offset);
            println!("Chunk count: {}", prefix.chunk_count);
        }
        Commands::Create { input, output, name } => {
            let bytes = fs::read(&input)?;
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();
            let metadata = NcfHeader {
                metadata: Metadata {
                    model_name: name.clone(),
                    architecture: "generic".into(),
                    created_at: now,
                    author: None,
                    license: None,
                    quantization: None,
                    custom: BTreeMap::new(),
                },
            };
            let tensor_schema = TensorSchema {
                name: name.clone(),
                dtype: DType::U8,
                shape: vec![bytes.len() as u64],
                column_layout: Layout::RowMajor,
                compression: Compression::None,
                encoding: Encoding::Plain,
                chunks: Vec::new(),
            };
            let mut writer = NcfWriter::new(metadata, NcfFlags::empty());
            writer.add_tensor(tensor_schema, bytes);
            writer.finalize(output)?;
            println!("Created NCF file from {}", input.display());
        }
        Commands::ConvertSafetensors { input, output } => {
            safetensors_to_ncf(input, output)?;
            println!("Converted safetensors to NCF.");
        }
        Commands::ConvertGguf { input, output } => {
            gguf_to_ncf(input, output)?;
            println!("Converted GGUF to NCF.");
        }
    }
    Ok(())
}
