# Cost Calculator

This script calculates the total cost of MLIR operations based on a cost model JSON file.

## Usage

```bash
go run cost_calculator.go <cost_model.json> <mlir_file.mlir> <max_level>
```

## Example

```bash
go run cost_calculator.go ../../Orbit-ILP/cost_models/lattigo_config.json circuit.mlir 29
```

## Parameters

- `cost_model.json`: Path to the cost model JSON file containing operation latencies
- `mlir_file.mlir`: Path to the MLIR file to analyze
- `max_level`: Maximum level for the FHE scheme (typically 29)

## Cost Model Format

The cost model JSON should have the following structure:

```json
{
    "latencyTable": {
        "earth.add_single": [cost_level_0, cost_level_1, ...],
        "earth.add_double": [cost_level_0, cost_level_1, ...],
        "earth.mul_single": [cost_level_0, cost_level_1, ...],
        "earth.mul_double": [cost_level_0, cost_level_1, ...],
        "earth.rotate_single": [cost_level_0, cost_level_1, ...],
        "earth.modswitch_single": [cost_level_0, cost_level_1, ...],
        "earth.negate_single": [cost_level_0, cost_level_1, ...],
        "earth.bootstrap_single": [cost_level_0, cost_level_1, ...],
        "earth.rescale_single": [cost_level_0, cost_level_1, ...]
    }
}
```

## Level Calculation

The script converts MLIR levels to cost model levels using:
```
cost_model_level = max_level - mlir_level
```

For example, if max_level=29 and MLIR shows `earth.ci<90 * 13>`, the cost model level is 29-13=16.

## Operation Types

- **Single operations**: Operations between a ciphertext and a plaintext/constant
- **Double operations**: Operations between two ciphertexts

The script automatically determines if an operation is single or double based on the number of ciphertext operands.

## Supported Operations

- `earth.add` (single/double)
- `earth.mul` (single/double)
- `earth.rotate` (single only)
- `earth.modswitch` (single only)
- `earth.negate` (single only)
- `earth.bootstrap` (single only)
- `earth.rescale` (single only)

## Output

The script outputs:
1. Each operation with its level and cost
2. Total number of operations processed
3. Total estimated cost in microseconds
4. Total estimated cost in seconds 