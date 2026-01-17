package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type ConstantMap map[int][]float64

func loadConstants(filename string) (ConstantMap, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file %s: %v", filename, err)
	}
	defer file.Close()

	var length int64
	err = binary.Read(file, binary.LittleEndian, &length)
	if err != nil {
		return nil, fmt.Errorf("failed to read length: %v", err)
	}

	constants := make(ConstantMap)

	for i := int64(0); i < length; i++ {
		var veclen int64
		err = binary.Read(file, binary.LittleEndian, &veclen)
		if err != nil {
			return nil, fmt.Errorf("failed to read vector length at index %d: %v", i, err)
		}

		values := make([]float64, veclen)
		err = binary.Read(file, binary.LittleEndian, &values)
		if err != nil {
			return nil, fmt.Errorf("failed to read vector values at index %d: %v", i, err)
		}

		constants[int(i)] = values
	}

	return constants, nil
}

func loadConstantFile(filename string) (int, int, []float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return 0, 0, nil, fmt.Errorf("failed to open file %s: %v", filename, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	// Read length (first line)
	if !scanner.Scan() {
		return 0, 0, nil, fmt.Errorf("failed to read length from %s", filename)
	}
	length, err := strconv.Atoi(strings.TrimSpace(scanner.Text()))
	if err != nil {
		return 0, 0, nil, fmt.Errorf("failed to parse length in %s: %v", filename, err)
	}

	// Read id (second line)
	if !scanner.Scan() {
		return 0, 0, nil, fmt.Errorf("failed to read id from %s", filename)
	}
	id, err := strconv.Atoi(strings.TrimSpace(scanner.Text()))
	if err != nil {
		return 0, 0, nil, fmt.Errorf("failed to parse id in %s: %v", filename, err)
	}

	// Read values
	values := make([]float64, 0, length)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		value, err := strconv.ParseFloat(line, 64)
		if err != nil {
			return 0, 0, nil, fmt.Errorf("failed to parse value '%s' in %s: %v", line, filename, err)
		}
		values = append(values, value)
	}

	if err := scanner.Err(); err != nil {
		return 0, 0, nil, fmt.Errorf("error reading file %s: %v", filename, err)
	}

	if len(values) != length {
		return 0, 0, nil, fmt.Errorf("expected %d values but got %d in file %s", length, len(values), filename)
	}

	return length, id, values, nil
}

func compareConstants(constantMap ConstantMap, constantsFolder string) error {
	files, err := filepath.Glob(filepath.Join(constantsFolder, "*"))
	if err != nil {
		return fmt.Errorf("failed to list files in %s: %v", constantsFolder, err)
	}

	if len(files) == 0 {
		return fmt.Errorf("no constant files found in %s", constantsFolder)
	}

	foundErrors := false

	for _, file := range files {
		if info, err := os.Stat(file); err != nil || info.IsDir() {
			continue
		}

		fmt.Printf("Checking constant file: %s\n", file)

		length, id, expectedValues, err := loadConstantFile(file)
		if err != nil {
			fmt.Printf("ERROR: Failed to load constant file %s: %v\n", file, err)
			foundErrors = true
			continue
		}

		// Check if this ID exists in the constant map
		actualValues, exists := constantMap[id]
		if !exists {
			fmt.Printf("ERROR: Constant ID %d from file %s not found in cst file\n", id, file)
			foundErrors = true
			continue
		}

		// Check length
		if len(actualValues) != length {
			fmt.Printf("ERROR: Length mismatch for ID %d. Expected: %d, Got: %d\n", id, length, len(actualValues))
			foundErrors = true
			continue
		}

		// Check values
		const tolerance = 1e-10
		for i, expected := range expectedValues {
			actual := actualValues[i]
			diff := actual - expected
			if diff < 0 {
				diff = -diff
			}
			if diff > tolerance {
				fmt.Printf("ERROR: Value mismatch for ID %d at index %d. Expected: %f, Got: %f, Diff: %e\n",
					id, i, expected, actual, diff)
				foundErrors = true
			}
		}
	}

	if foundErrors {
		return fmt.Errorf("FAILED: Found mismatches")
	}

	fmt.Printf("SUCCESS: All constants match!\n")
	return nil
}

func main() {
	if len(os.Args) < 2 {
		fmt.Printf("Usage: %s <cst_file_path> [constants_folder_path]\n", os.Args[0])
		fmt.Printf("  If only cst_file_path is provided, will print constant count and identify all-zero constants\n")
		fmt.Printf("  If both arguments provided, will compare cst file with constants folder\n")
		os.Exit(1)
	}

	cstFilePath := os.Args[1]

	fmt.Printf("Loading cst file: %s\n", cstFilePath)
	constantMap, err := loadConstants(cstFilePath)
	if err != nil {
		fmt.Printf("ERROR: Failed to load cst file: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Loaded %d constants from cst file\n", len(constantMap))

	// If only one argument, just analyze the cst file
	if len(os.Args) == 2 {
		fmt.Printf("\nAnalyzing constants:\n")
		allZeroCount := 0
		maxID := -1

		for id, values := range constantMap {
			// Track max ID
			if id > maxID {
				maxID = id
			}

			allZero := true
			for _, val := range values {
				if val != 0 {
					allZero = false
					break
				}
			}

			if allZero {
				fmt.Printf("Constant ID %d is all zeros (length: %d)\n", id, len(values))
				allZeroCount++
			}
		}

		fmt.Printf("Max constant ID: %d\n", maxID)

		if allZeroCount == 0 {
			fmt.Printf("No all-zero constants found\n")
		} else {
			fmt.Printf("Found %d all-zero constants\n", allZeroCount)
		}
		return
	}

	// If two arguments, do the comparison
	constantsFolderPath := os.Args[2]
	fmt.Printf("Comparing with constants folder: %s\n", constantsFolderPath)
	err = compareConstants(constantMap, constantsFolderPath)
	if err != nil {
		fmt.Printf("ERROR: %v\n", err)
		os.Exit(1)
	}
}
