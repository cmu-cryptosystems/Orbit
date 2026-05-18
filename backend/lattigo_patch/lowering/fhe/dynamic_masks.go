package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type DynamicMaskTable struct {
	SchemaVersion string              `json:"schema_version"`
	Records       []DynamicMaskRecord `json:"records"`
}

type DynamicMaskRecord struct {
	ConstantID  int     `json:"constant_id"`
	Kind        string  `json:"kind"`
	I           int     `json:"i"`
	RotateSteps int     `json:"rotate_steps"`
	N           int     `json:"n"`
	TileDim     int     `json:"tile_dim"`
	ScaleBits   int     `json:"scale_bits"`
	Factor      float64 `json:"factor"`
}

func (lattigo *LattigoFHE) dynamicMaskPathForInput(inputPath string) string {
	if lattigo.dynamicMaskPath != "" {
		return lattigo.dynamicMaskPath
	}
	if inputPath == "" {
		return ""
	}
	ext := filepath.Ext(inputPath)
	base := strings.TrimSuffix(inputPath, ext)
	return base + "_mask" + ext
}

func (lattigo *LattigoFHE) applyDynamicPlaintextMasks(maskPath string) error {
	if lattigo.dynamicMaskTablePath == "" {
		return nil
	}
	if maskPath == "" {
		return fmt.Errorf("dynamic mask table was supplied but no mask input path was provided")
	}
	mask, err := readDynamicMaskValues(maskPath)
	if err != nil {
		return err
	}
	tableBytes, err := os.ReadFile(lattigo.dynamicMaskTablePath)
	if err != nil {
		return fmt.Errorf("failed to read dynamic mask table %s: %v", lattigo.dynamicMaskTablePath, err)
	}
	var table DynamicMaskTable
	if err := json.Unmarshal(tableBytes, &table); err != nil {
		return fmt.Errorf("failed to parse dynamic mask table %s: %v", lattigo.dynamicMaskTablePath, err)
	}
	for _, record := range table.Records {
		if record.Kind != "qk_output_mask" {
			return fmt.Errorf("unsupported dynamic mask kind %q for constant %d", record.Kind, record.ConstantID)
		}
		lattigo.constants[record.ConstantID] = buildQKOutputMask(record, mask, lattigo.n)
	}
	return nil
}

func readDynamicMaskValues(path string) ([]float64, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read dynamic mask %s: %v", path, err)
	}
	fields := strings.Fields(string(content))
	if len(fields) == 0 {
		return nil, fmt.Errorf("dynamic mask %s is empty", path)
	}
	length, err := strconv.Atoi(fields[0])
	if err != nil {
		return nil, fmt.Errorf("dynamic mask %s has invalid length: %v", path, err)
	}
	if len(fields)-1 < length {
		return nil, fmt.Errorf("dynamic mask %s has %d values, expected %d", path, len(fields)-1, length)
	}
	values := make([]float64, length)
	for i := 0; i < length; i++ {
		value, err := strconv.ParseFloat(fields[i+1], 64)
		if err != nil {
			return nil, fmt.Errorf("dynamic mask %s value %d is invalid: %v", path, i, err)
		}
		values[i] = value
	}
	return values, nil
}

func buildQKOutputMask(record DynamicMaskRecord, attentionMask []float64, n int) []float64 {
	recordN := record.N
	if recordN <= 0 {
		recordN = n
	}
	tileDim := record.TileDim
	if tileDim <= 0 {
		tileDim = 128
	}
	factor := record.Factor
	if factor == 0 {
		factor = math.Ceil(math.Pow(2, float64(record.ScaleBits)) / 1024.0)
	}
	values := make([]float64, recordN)
	for half := 0; half < 2; half++ {
		halfBase := tileDim * tileDim * half
		for token := 0; token < tileDim; token++ {
			maskIndex := (record.I + token) % tileDim
			maskValue := 0.0
			if maskIndex >= 0 && maskIndex < len(attentionMask) {
				maskValue = attentionMask[maskIndex]
			}
			slot := halfBase + tileDim*token
			if slot >= 0 && slot < len(values) {
				values[slot] = maskValue * factor
			}
		}
	}
	return rotateFheBertMask(values, record.RotateSteps)
}

func rotateFheBertMask(values []float64, steps int) []float64 {
	n := len(values)
	if n == 0 {
		return values
	}
	out := make([]float64, n)
	for i, value := range values {
		target := ((i-steps)%n + n) % n
		out[target] = value
	}
	return out
}
