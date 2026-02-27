package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// BenchmarkResult stores timing information for each level
type BenchmarkResult struct {
	Level   int
	AvgTime time.Duration
	Times   []time.Duration
}

// readConstantFile reads the constant file in the specified format
func readConstantFile(filePath string) ([]float64, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	// Read first line - length
	if !scanner.Scan() {
		return nil, fmt.Errorf("error reading length from file")
	}
	length, err := strconv.Atoi(strings.TrimSpace(scanner.Text()))
	if err != nil {
		return nil, fmt.Errorf("error parsing length: %v", err)
	}

	// Read second line - index (ignore as per user request)
	if !scanner.Scan() {
		return nil, fmt.Errorf("error reading index from file")
	}

	// Read the values
	values := make([]float64, length)
	for i := 0; i < length; i++ {
		if !scanner.Scan() {
			return nil, fmt.Errorf("error reading value %d", i)
		}
		value, err := strconv.ParseFloat(strings.TrimSpace(scanner.Text()), 64)
		if err != nil {
			return nil, fmt.Errorf("error parsing value %d: %v", i, err)
		}
		values[i] = value
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %v", err)
	}

	return values, nil
}

// setupLattigo initializes the lattigo parameters and encoder
func setupLattigo(n int, maxLevel int) (*ckks.Parameters, *ckks.Encoder, *rlwe.Encryptor, error) {
	logQ := append([]int{55}, make([]int, maxLevel)...)
	for i := 1; i <= maxLevel; i++ {
		logQ[i] = 51
	}
	logN := int(math.Log2(float64(n * 2)))

	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            logN,
		LogQ:            logQ,
		LogP:            []int{61, 61, 61},
		LogDefaultScale: 51,
	})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("error creating parameters: %v", err)
	}

	encoder := ckks.NewEncoder(params)

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	encryptor := rlwe.NewEncryptor(params, sk)

	return &params, encoder, encryptor, nil
}

// benchmarkEncoding runs encoding benchmark at a specific level
func benchmarkEncoding(values []float64, n, level int, params *ckks.Parameters, encoder *ckks.Encoder, encryptor *rlwe.Encryptor, runs int) (time.Duration, []time.Duration, error) {
	encodingValues := make([]float64, n)

	if len(values) >= n {
		copy(encodingValues, values[:n])
	} else {
		for i := 0; i < n; i++ {
			encodingValues[i] = values[i%len(values)]
		}
	}

	var times []time.Duration
	var totalTime time.Duration

	for run := 0; run < runs; run++ {
		// Time the encoding operation
		scale := rlwe.NewScale(math.Pow(2, 40))
		start := time.Now()

		plaintext := ckks.NewPlaintext(*params, level)
		plaintext.Scale = scale

		encoder.Encode(encodingValues, plaintext)
		_, err := encryptor.EncryptNew(plaintext)
		if err != nil {
			return 0, nil, fmt.Errorf("error encrypting: %v", err)
		}

		elapsed := time.Since(start)
		times = append(times, elapsed)
		totalTime += elapsed
	}

	avgTime := totalTime / time.Duration(runs)
	return avgTime, times, nil
}

// writeResults writes the benchmark results to output file
func writeResults(results []BenchmarkResult, outputPath string, inputFile string) error {
	// Ensure logs directory exists
	logsDir := filepath.Dir(outputPath)
	if err := os.MkdirAll(logsDir, 0755); err != nil {
		return fmt.Errorf("error creating logs directory: %v", err)
	}

	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("error creating output file: %v", err)
	}
	defer file.Close()

	// Header line
	fmt.Fprintf(file, "Level Avg_Time\n")

	// Data lines: level and average time in microseconds
	for _, result := range results {
		avgMicroseconds := float64(result.AvgTime.Nanoseconds()) / 1000.0
		fmt.Fprintf(file, "%d %.2f\n", result.Level, avgMicroseconds)
	}

	return nil
}

func main() {
	if len(os.Args) != 2 {
		fmt.Println("Usage: go run encode_benchmark.go <constant_file.txt>")
		fmt.Println("Example: go run encode_benchmark.go constants/constant_0.txt")
		os.Exit(1)
	}

	n := 65536
	maxLevel := 29
	constantFile := os.Args[1]

	// Read constant values
	fmt.Printf("Reading constant file: %s\n", constantFile)
	values, err := readConstantFile(constantFile)
	if err != nil {
		log.Fatalf("Error reading constant file: %v", err)
	}
	params, encoder, encryptor, err := setupLattigo(n, maxLevel)
	if err != nil {
		log.Fatalf("Error setting up lattigo: %v", err)
	}

	var results []BenchmarkResult

	for level := 1; level <= maxLevel; level++ {
		avgTime, times, err := benchmarkEncoding(values, n, level, params, encoder, encryptor, 20)
		if err != nil {
			log.Printf("Error benchmarking level %d: %v", level, err)
			continue
		}

		results = append(results, BenchmarkResult{
			Level:   level,
			AvgTime: avgTime,
			Times:   times,
		})

		fmt.Printf("  Level %d average: %.3f ms\n", level, float64(avgTime.Nanoseconds())/1000000.0)
	}

	// Generate output filename
	inputBasename := filepath.Base(constantFile)
	inputName := strings.TrimSuffix(inputBasename, ".txt")
	logsDir := filepath.Join("tests", "logs")
	outputPath := filepath.Join(logsDir, "encode_benchmark_"+inputName+".txt")

	// Write results
	fmt.Printf("Writing results to: %s\n", outputPath)
	err = writeResults(results, outputPath, constantFile)
	if err != nil {
		log.Fatalf("Error writing results: %v", err)
	}
}
