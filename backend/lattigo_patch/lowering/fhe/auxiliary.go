package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

var reInputFileNumber = regexp.MustCompile(`input(\d+)\.txt`)

func (lattigo *LattigoFHE) calculateAccuracy(want []float64, ct *rlwe.Ciphertext) bool {
	decrypted := lattigo.decode(ct)

	if len(want) != len(decrypted) {
		fmt.Printf("Length mismatch: want %d, decrypted %d\n", len(want), len(decrypted))
		return false
	}

	for i := 0; i < len(want); i++ {
		tolerance := 0.00005
		if math.Abs(want[i]-decrypted[i]) > tolerance {
			fmt.Printf("Value mismatch: want %f, decrypted %f on index %d\n", want[i], decrypted[i], i)
			return false
		}
	}

	return true
}

func (lattigo *LattigoFHE) validateResult(result []float64, trueLabel int) (bool, int) {
	if len(result) < 10 {
		return false, -1
	}

	maxIndex := 0
	maxValue := result[0]

	for i := 1; i < 10; i++ {
		if result[i] > maxValue {
			maxValue = result[i]
			maxIndex = i
		}
	}

	return maxIndex == trueLabel, maxIndex
}

func mergeMaps(src, dst map[int]*rlwe.Ciphertext) map[int]*rlwe.Ciphertext {
	for key, value := range src {
		dst[key] = value
	}
	return dst
}

func getKeys(m map[int]map[int]bool) []int {
	if len(m) == 0 {
		return []int{}
	}
	keys := make([]int, len(m))
	i := 0
	for k := range m {
		keys[i] = k
		i++
	}
	return keys
}

func (lattigo *LattigoFHE) doPrecisionStats(lineNum int, term *Term) []float64 {
	want := make([]float64, lattigo.n)
	md := term.Metadata
	if _, ok := lattigo.ptEnv[lineNum]; !ok {
		switch term.Op {
		case PACK:
			want = md.PackedValue
		case MASK:
			want = md.MaskedValue
		case CONST:
			for i := 0; i < lattigo.n; i++ {
				want[i] = float64(md.Value)
			}
		case ADD:
			a := lattigo.ptEnv[term.Children[0]]
			b := lattigo.ptEnv[term.Children[1]]
			for i := 0; i < min(len(a), len(b)); i++ {
				want[i] = a[i] + b[i]
			}
		case MUL:
			a := lattigo.ptEnv[term.Children[0]]
			b := lattigo.ptEnv[term.Children[1]]
			for i := 0; i < min(len(a), len(b)); i++ {
				want[i] = a[i] * b[i]
			}
		case ROT:
			rot := md.Offset
			a := lattigo.ptEnv[term.Children[0]]
			for i := 0; i < lattigo.n; i++ {
				index := ((i+rot)%lattigo.n + lattigo.n) % lattigo.n
				want[i] = a[index]
			}
		case NEGATE:
			a := lattigo.ptEnv[term.Children[0]]
			for i := 0; i < lattigo.n; i++ {
				want[i] = -a[i]
			}
		case BOOTSTRAP, MODSWITCH, UPSCALE, RESCALE:
			want = lattigo.ptEnv[term.Children[0]]
		}
		lattigo.ptEnv[lineNum] = want
	} else {
		want = lattigo.ptEnv[lineNum]
	}

	// For CONST operations, use ptEnv directly instead of decrypting (constant lazy evaluation)
	var decrypted []float64
	var accuracyStats bool

	if term.Op == CONST {
		decrypted = lattigo.ptEnv[lineNum]
		accuracyStats = true // Constants should always be accurate
	} else {
		decrypted = lattigo.decode(lattigo.env[lineNum])
		accuracyStats = lattigo.calculateAccuracy(want, lattigo.env[lineNum])
	}

	if 0 == 0 {
		// Create logs directory if it doesn't exist
		err := os.MkdirAll("logs", 0755)
		if err != nil {
			fmt.Printf("Error creating logs directory: %v\n", err)
			return nil
		}

		// Open file in logs directory
		logPath := filepath.Join("logs", lattigo.logFile)
		logFile, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			fmt.Printf("Error opening output file: %v\n", err)
			return nil
		}
		defer logFile.Close()

		writer := bufio.NewWriter(logFile)

		fmt.Fprintf(writer, "==============================\n")
		fmt.Fprintf(writer, "Line Number: %d\n", lineNum)
		fmt.Fprintf(writer, "Scale: %f, Level: %v\n", math.Log2(term.Scale.Float64()), term.Level)
		fmt.Fprintf(writer, "Operation: %v\n", term.Op)
		fmt.Fprintf(writer, "Children: %v\n", term.Children)
		// Print first 10 values
		var indicesToPrint []int

		// Create indices for first 10 values (or until end of data)
		for i := 0; i < 10 && i < len(want); i++ {
			indicesToPrint = append(indicesToPrint, i)
		}

		fmt.Fprintf(writer, "Want:      [")
		for i, idx := range indicesToPrint {
			if i > 0 {
				fmt.Fprintf(writer, ", ")
			}
			fmt.Fprintf(writer, "%.6f", want[idx])
		}
		if len(indicesToPrint) < len(want) {
			fmt.Fprintf(writer, ", ...")
		}
		fmt.Fprintf(writer, "]\n")
		fmt.Fprintf(writer, "Decrypted: [")
		for i, idx := range indicesToPrint {
			if i > 0 {
				fmt.Fprintf(writer, ", ")
			}
			fmt.Fprintf(writer, "%.6f", decrypted[idx])
		}
		if len(indicesToPrint) < len(decrypted) {
			fmt.Fprintf(writer, ", ...")
		}
		fmt.Fprintf(writer, "]\n")
		fmt.Fprintf(writer, "Accuracy: %t\n", accuracyStats)
		fmt.Fprintf(writer, "==============================\n\n")
		writer.Flush()
	}

	return want
}

func (lattigo *LattigoFHE) findInputFiles() ([]string, error) {
	files, err := filepath.Glob(filepath.Join(lattigo.inputPath, "input*.txt"))
	if err != nil {
		return nil, err
	}

	// Sort files to ensure consistent processing order
	sort.Strings(files)
	return files, nil
}

func (lattigo *LattigoFHE) createOutputDirectory() (string, error) {
	// Extract model, benchmark, and waterline from MLIR filename (similar to process_fhe_inputs.sh)
	var outputDirName string
	if lattigo.fileType == MLIR {
		mlirBase := filepath.Base(lattigo.mlirPath)
		// Remove .mlir extension
		if strings.HasSuffix(mlirBase, ".mlir") {
			outputDirName = strings.TrimSuffix(mlirBase, ".mlir")
		} else {
			outputDirName = mlirBase
		}
	} else {
		// For instruction files, use a generic name
		outputDirName = "batch_output"
	}

	outputDir := filepath.Join("outputs", outputDirName)
	err := os.MkdirAll(outputDir, 0755)
	if err != nil {
		return "", err
	}

	return outputDir, nil
}

func (lattigo *LattigoFHE) resetForNewInput() {
	lattigo.env = make(map[int]*rlwe.Ciphertext)
	lattigo.ptEnv = make(map[int][]float64)
	lattigo.terms = make(map[int]*Term)
	lattigo.refCounts = make(map[int]int)
}

func (lattigo *LattigoFHE) generateOutputFileName(inputFile string) string {
	inputBase := filepath.Base(inputFile)
	matches := reInputFileNumber.FindStringSubmatch(inputBase)

	if len(matches) >= 2 {
		number := matches[1]
		if lattigo.fileType == MLIR {
			mlirBase := filepath.Base(lattigo.mlirPath)
			if strings.HasSuffix(mlirBase, ".mlir") {
				baseName := strings.TrimSuffix(mlirBase, ".mlir")
				return fmt.Sprintf("%s_output%s.txt", baseName, number)
			}
		}
		return fmt.Sprintf("output%s.txt", number)
	}

	return fmt.Sprintf("output_%s", inputBase)
}

func (lattigo *LattigoFHE) writeOutputFile(outputPath string, results []float64) error {
	content := fmt.Sprintf("%v\n", len(results))
	for _, v := range results {
		content += fmt.Sprintf("%v\n", v)
	}

	return os.WriteFile(outputPath, []byte(content), 0644)
}

func (lattigo *LattigoFHE) writeRuntimesFile(outputDir string, runtimeInfos []RuntimeInfo, avgDuration time.Duration) error {
	runtimesPath := filepath.Join(outputDir, "runtimes.txt")

	// Build the content string with average runtime at the top
	content := fmt.Sprintf("Average runtime: %v\n\n", avgDuration)

	// Add individual file runtimes with prediction information
	for _, info := range runtimeInfos {
		if info.HasValidation {
			status := "FAIL"
			if info.PredictedClass == info.TrueClass {
				status = "PASS"
			}
			content += fmt.Sprintf("%s: %v (predicted: %d, true: %d, %s)\n",
				info.OutputFileName, info.Runtime, info.PredictedClass, info.TrueClass, status)
		} else {
			content += fmt.Sprintf("%s: %v\n", info.OutputFileName, info.Runtime)
		}
	}

	return os.WriteFile(runtimesPath, []byte(content), 0644)
}

func (lattigo *LattigoFHE) writeTimingReport() error {
	if !lattigo.enableTiming || lattigo.timingStats == nil {
		return nil
	}

	// Create logs directory if it doesn't exist
	err := os.MkdirAll("logs", 0755)
	if err != nil {
		return fmt.Errorf("error creating logs directory: %v", err)
	}

	// Generate timestamp for unique filename
	timestamp := time.Now().Format("20060102_150405")
	filename := filepath.Join("logs", fmt.Sprintf("timing_report_%s.txt", timestamp))

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("error creating timing report file: %v", err)
	}
	defer file.Close()

	// Write header
	fmt.Fprintf(file, "FHE Operations Timing Analysis Report\n")
	fmt.Fprintf(file, "=====================================\n")
	fmt.Fprintf(file, "Generated: %s\n", time.Now().Format("2006-01-02 15:04:05"))
	fmt.Fprintf(file, "Total Execution Time: %v\n\n", lattigo.timingStats.TotalTime)

	type OpLevel struct {
		Op    op
		Level int
	}
	var opLevels []OpLevel

	for operation, levelMap := range lattigo.timingStats.OperationStats {
		for level, stats := range levelMap {
			if stats.Count > 0 { // Only include non-zero counts
				opLevels = append(opLevels, OpLevel{Op: operation, Level: level})
			}
		}
	}

	sort.Slice(opLevels, func(i, j int) bool {
		if opLevels[i].Op != opLevels[j].Op {
			return opLevels[i].Op < opLevels[j].Op
		}
		return opLevels[i].Level < opLevels[j].Level
	})

	fmt.Fprintf(file, "Detailed Operations Statistics:\n")
	fmt.Fprintf(file, "%-12s %-6s %-8s %-15s %-15s %-10s\n",
		"Operation", "Level", "Count", "Total Time", "Avg Time", "% of Total")
	fmt.Fprintf(file, "%-12s %-6s %-8s %-15s %-15s %-10s\n",
		"----------", "-----", "-----", "----------", "--------", "----------")

	for _, ol := range opLevels {
		stats := lattigo.timingStats.OperationStats[ol.Op][ol.Level]
		avgTime := stats.TotalTime / time.Duration(stats.Count)
		percentage := float64(stats.TotalTime) / float64(lattigo.timingStats.TotalTime) * 100

		fmt.Fprintf(file, "%-12s %-6d %-8d %-15v %-15v %-9.2f%%\n",
			getOpName(ol.Op), ol.Level, stats.Count,
			stats.TotalTime, avgTime, percentage)
	}

	// Write summary by operation (across all levels)
	opSummary := make(map[op]*LevelStats)
	for operation, levelMap := range lattigo.timingStats.OperationStats {
		opSummary[operation] = &LevelStats{}
		for _, stats := range levelMap {
			if stats.Count > 0 {
				opSummary[operation].Count += stats.Count
				opSummary[operation].TotalTime += stats.TotalTime
			}
		}
	}

	fmt.Fprintf(file, "\nSummary by Operation Type:\n")
	fmt.Fprintf(file, "%-12s %-8s %-15s %-15s %-10s\n",
		"Operation", "Count", "Total Time", "Avg Time", "% of Total")
	fmt.Fprintf(file, "%-12s %-8s %-15s %-15s %-10s\n",
		"----------", "-----", "----------", "--------", "----------")

	var operations []op
	for op, stats := range opSummary {
		if stats.Count > 0 {
			operations = append(operations, op)
		}
	}
	sort.Slice(operations, func(i, j int) bool {
		return operations[i] < operations[j]
	})

	for _, operation := range operations {
		stats := opSummary[operation]
		avgTime := stats.TotalTime / time.Duration(stats.Count)
		percentage := float64(stats.TotalTime) / float64(lattigo.timingStats.TotalTime) * 100

		fmt.Fprintf(file, "%-12s %-8d %-15v %-15v %-9.2f%%\n",
			getOpName(operation), stats.Count, stats.TotalTime, avgTime, percentage)
	}

	fmt.Printf("Timing report written to: %s\n", filename)
	return nil
}
