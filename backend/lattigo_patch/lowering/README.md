# Lattigo FHE Lowering

This directory contains tools for running FHE (Fully Homomorphic Encryption) operations using the Lattigo library, supporting both MLIR files and instruction files.

## Setup

### Prerequisites
- Lattigo dependencies

### Initial Setup
1. Navigate to the lattigo root directory and ensure dependencies are installed:
   ```bash
   cd /path/to/lattigo
   go mod tidy
   ```

2. Build the FHE binary:
   ```bash
   cd lowering
   go build -o fhe_binary ./fhe
   ```

## Usage

### Basic Usage
```bash
./fhe_binary [options]
```

### Development Usage
If you're actively developing and want to run without building each time:
```bash
go run ./fhe [options]
```

### Command Line Options

- `-mlir <file>`: Path to MLIR file (required for MLIR mode)
- `-i <file>`: Path to instructions file (required for instruction mode)
- `-n <number>`: Polynomial modulus degree (default: 4096)
- `-maxLevel <number>`: Maximum level of the FHE scheme (default: 29)
- `-bootstrapMinLevel <number>`: Minimum bootstrap level (default: 3)
- `-bootstrapMaxLevel <number>`: Maximum bootstrap level (default: 16)
- `-cons <directory>`: Path to constants directory
- `-input <file>`: Path to input file
- `-output [filename]`: Obtain output [filename] in outputs/ directory
- `-getLog [filename]`: Obtain debug log [filename] in logs/ directory (default: precision_debug.txt)

### Examples

#### Running MLIR Files
```bash
# Basic MLIR execution
./fhe_binary -mlir mlirs/orbit_ResNetReLU.mlir -n 16384

# With constants and inputs
./fhe_binary -mlir mlirs/orbit_ResNetReLU.mlir -n 16384 \
  -cons /path/to/constants \
  -input /path/to/input.txt \
  -output result.txt

# With debug logging
./fhe_binary -mlir mlirs/orbit_ResNetReLU.mlir -n 16384 \
  -getLog debug_output.txt
```

## Memory Management

For large computations, you may want to limit memory usage:

### Using systemd-run (Linux)
```bash
# Limit to 200GB of memory
systemd-run --scope --user -p MemoryMax=200G ./fhe_binary \
  -mlir mlirs/orbit_ResNetReLU.mlir -n 16384 \
  -cons /path/to/constants \
  -input /path/to/input.txt \
  -getLog orbit_ResNetReLU.txt \
  -output output.txt
```

### Using ulimit
```bash
# Set virtual memory limit to 200GB (209715200 KB)
ulimit -v 209715200
./fhe_binary [options]
```

### Monitoring Memory Usage

#### Check Running Process Memory
```bash
# Real-time monitoring with htop (shows VIRT, RES, %MEM columns)
htop

# Alternative with top
top

# Check specific process memory usage
ps aux | grep fhe_binary
```

#### Verify systemd Memory Limits
```bash
# Check if systemd memory limit is active
systemctl --user status run-r*.scope

# Monitor systemd service memory usage
systemctl --user show run-r*.scope | grep Memory
```

#### Understanding Memory Columns
- **VIRT**: Virtual memory (total memory space used by process)
- **RES**: Resident memory (actual physical RAM being used)
- **%MEM**: Percentage of total system memory being used

## Directory Structure

```
lowering/
├── fhe/                    # Source code for FHE operations
├── tests/                  # Test files and utilities
│   └── bootstrap_test     # Built test binary (optional)
├── logs/                   # Debug output logs (created automatically)
├── outputs/                # Output files (created automatically)
├── fhe_binary             # Built binary (after running go build)
└── README.md              # This file
```

### Output

- **Console Output**: Progress information and final results
- **Log Files**: Detailed debug information (when `-getLog` is used)
  - Only operations with accuracy < 99.99% are logged to reduce file size
  - Logs are stored in the `logs/` directory
- **Output Files**: Final computation results (when `-output` is used)
  - Stored in the `outputs/` directory

## Troubleshooting

### Common Issues

1. **"go: command not found" with systemd-run**
   - Always use the built binary (`./fhe_binary`) with systemd-run
   - The binary doesn't require the Go toolchain at runtime

2. **Memory Issues**
   - Use memory limiting with systemd-run or ulimit
   - Consider reducing `-n` parameter for smaller polynomial degree

3. **File Not Found Errors**
   - Ensure all input files and directories exist
   - Use absolute paths for external files (constants, inputs)

### Debug Mode

Enable debug logging to troubleshoot accuracy issues:
```bash
./fhe_binary -mlir your_file.mlir -getLog debug.txt
```

Debug logs will show detailed information only for operations with accuracy below 99.99%.

## Performance Tips

1. **Use the built binary** for production runs - it's faster and more reliable
2. **Set appropriate polynomial degree** (`-n`) based on your security/performance needs
3. **Use memory limits** for large computations to prevent system instability
4. **Monitor log files** to identify operations causing accuracy issues

## Development

To modify the FHE operations, edit files in the `fhe/` directory:
- `main.go`: Entry point and command-line parsing
- `scheme.go`: Core FHE scheme implementation
- `parser.go`: MLIR and instruction file parsing
- `evaluator.go`: FHE operation implementations
- `auxiliary.go`: Utility functions and precision statistics
- `encoder.go`: Encoding/decoding operations 

## (Optional) Testing

### Bootstrap Test
Test bootstrap accuracy and functionality:
```bash
# Run bootstrap test directly
go run ./tests/bootstrap.go [options]

# Or build and run the test binary
go build -o tests/bootstrap_test ./tests/bootstrap.go
./tests/bootstrap_test [options]
```

#### Bootstrap Test Options
The bootstrap test supports the same basic options as the main program:
- `-n <number>`: Polynomial modulus degree
- `-maxLevel <number>`: Maximum FHE level
- `-bootstrapMinLevel <number>`: Minimum bootstrap level  
- `-bootstrapMaxLevel <number>`: Maximum bootstrap level
