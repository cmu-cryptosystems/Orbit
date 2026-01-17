package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// CostModel represents the structure of the cost model JSON
type CostModel struct {
	BootstrapLevelLowerBound int                  `json:"bootstrapLevelLowerBound"`
	BootstrapLevelUpperBound int                  `json:"bootstrapLevelUpperBound"`
	LatencyTable             map[string][]float64 `json:"latencyTable"`
	LevelLowerBound          int                  `json:"levelLowerBound"`
	LevelUpperBound          int                  `json:"levelUpperBound"`
	PolynomialDegree         int                  `json:"polynomialDegree"`
	RescalingFactor          int                  `json:"rescalingFactor"`
	Runtime                  string               `json:"runtime"`
}

// ConstantCosts stores encoding costs by level (in microseconds)
type ConstantCosts struct {
	CostsByLevel map[int]float64 // [level] = cost in microseconds
}

// Operation represents a parsed MLIR operation
type Operation struct {
	OpType   string
	Level    int
	IsDouble bool
	Metadata string
}

// OperationStats tracks detailed statistics for analysis
type OperationStats struct {
	Count      map[string]map[int]int     // [operation][level] = count
	TotalCost  map[string]map[int]float64 // [operation][level] = total cost
	Operations []Operation                // all operations for detailed analysis
}

// NewOperationStats creates a new OperationStats instance
func NewOperationStats() *OperationStats {
	return &OperationStats{
		Count:      make(map[string]map[int]int),
		TotalCost:  make(map[string]map[int]float64),
		Operations: make([]Operation, 0),
	}
}

// getShortOpName converts full operation names to short-handed versions for analysis
func getShortOpName(opType string) string {
	switch opType {
	case "earth.add_single":
		return "add_s"
	case "earth.add_double":
		return "add_d"
	case "earth.mul_single":
		return "mul_s"
	case "earth.mul_double":
		return "mul_d"
	case "earth.rotate_single":
		return "rot"
	case "earth.upscale_single":
		return "upscale"
	case "earth.rescale_single":
		return "rescale"
	case "earth.bootstrap_single":
		return "bootstrap"
	case "earth.modswitch_single":
		return "modswitch"
	case "earth.negate_single":
		return "negate"
	case "earth.constant_single":
		return "constant"
	default:
		return opType
	}
}

// decomposeRotation decomposes a rotation using Non-Adjacent Form (NAF) for efficiency
func decomposeRotation(k int) []int {
	if k == 0 {
		return []int{0}
	}

	var decomposition []int

	// Record the sign of the original value and compute abs
	sign := k < 0
	value := k
	if value < 0 {
		value = -value
	}

	// Transform to non-adjacent form (NAF)
	for i := 0; value > 0; i++ {
		zi := 0
		if value&1 == 1 { // if value is odd
			zi = 2 - (value & 3) // zi = 2 - (value mod 4)
		}
		value = (value - zi) >> 1 // value = (value - zi) / 2

		if zi != 0 {
			term := zi * (1 << i) // zi * 2^i
			if sign {
				term = -term
			}
			decomposition = append(decomposition, term)
		}
	}

	return decomposition
}

// readConstantCosts reads the encoding benchmark results file
func readConstantCosts(filePath string) (*ConstantCosts, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening constant costs file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	// Skip header line
	if !scanner.Scan() {
		return nil, fmt.Errorf("error reading header from constant costs file")
	}

	constantCosts := &ConstantCosts{
		CostsByLevel: make(map[int]float64),
	}

	// Read data lines
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		parts := strings.Fields(line)
		if len(parts) != 2 {
			continue
		}

		level, err := strconv.Atoi(parts[0])
		if err != nil {
			continue
		}

		cost, err := strconv.ParseFloat(parts[1], 64)
		if err != nil {
			continue
		}

		constantCosts.CostsByLevel[level] = cost
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading constant costs file: %v", err)
	}

	return constantCosts, nil
}

// parseMLIROperation parses a single MLIR operation line
func parseMLIROperation(line string, maxLevel int) (*Operation, error) {
	line = strings.TrimSpace(line)
	if line == "" || !strings.HasPrefix(line, "%") {
		return nil, nil
	}

	// Split at '='
	parts := strings.SplitN(line, "=", 2)
	if len(parts) < 2 {
		return nil, nil
	}

	rest := strings.TrimSpace(parts[1])

	// Get op name
	opStart := strings.Index(rest, "\"")
	opEnd := strings.Index(rest[opStart+1:], "\"")
	if opStart == -1 || opEnd == -1 {
		return nil, nil
	}
	op := rest[opStart+1 : opStart+1+opEnd]

	// Get metadata (the stuff inside <{ ... }>)
	metadata := ""
	metaStart := strings.Index(rest, "<{")
	metaEnd := strings.Index(rest, "}>")
	if metaStart != -1 && metaEnd != -1 && metaEnd > metaStart {
		metadata = rest[metaStart+2 : metaEnd]
	}

	// Extract input types from the operation signature to determine level and operation type
	level := 0
	isDouble := false
	ciphertextCount := 0

	// Find the input types in the signature (between : and ->)
	colonIdx := strings.Index(rest, ":")
	arrowIdx := strings.Index(rest, "->")
	if colonIdx != -1 && arrowIdx != -1 && arrowIdx > colonIdx {
		inputTypes := rest[colonIdx+1 : arrowIdx]

		// Parse each input type from the type signature (separated by commas within the signature parentheses)
		parenStart := strings.Index(inputTypes, "(")
		parenEnd := strings.LastIndex(inputTypes, ")")
		if parenStart != -1 && parenEnd != -1 && parenEnd > parenStart {
			typesStr := inputTypes[parenStart+1 : parenEnd]
			types := strings.Split(typesStr, ",")

			var levels []int

			for _, typeStr := range types {
				typeStr = strings.TrimSpace(typeStr)

				// Check if it's a ciphertext (.ci) or plaintext (.pl)
				if strings.Contains(typeStr, "earth.ci<") {
					ciphertextCount++
					// Extract level from earth.ci<scale * level>
					ciIdx := strings.Index(typeStr, "earth.ci<")
					if ciIdx != -1 {
						typeIdx := ciIdx + len("earth.ci<")
						endIdx := strings.Index(typeStr[typeIdx:], ">")
						if endIdx != -1 {
							nums := typeStr[typeIdx : typeIdx+endIdx]
							numParts := strings.Split(nums, "*")
							if len(numParts) == 2 {
								levelStr := strings.TrimSpace(numParts[1])
								if levelInt, err := strconv.Atoi(levelStr); err == nil {
									// Convert to maxLevel - level as specified
									levels = append(levels, maxLevel-levelInt)
								}
							}
						}
					}
				} else if strings.Contains(typeStr, "earth.pl<") {
					// Extract level from earth.pl<scale * level>
					plIdx := strings.Index(typeStr, "earth.pl<")
					if plIdx != -1 {
						typeIdx := plIdx + len("earth.pl<")
						endIdx := strings.Index(typeStr[typeIdx:], ">")
						if endIdx != -1 {
							nums := typeStr[typeIdx : typeIdx+endIdx]
							numParts := strings.Split(nums, "*")
							if len(numParts) == 2 {
								levelStr := strings.TrimSpace(numParts[1])
								if levelInt, err := strconv.Atoi(levelStr); err == nil {
									// Convert to maxLevel - level as specified
									levels = append(levels, maxLevel-levelInt)
								}
							}
						}
					}
				}
			}

			// Check if levels are consistent
			if len(levels) > 0 {
				level = levels[0]
				for _, l := range levels {
					if l != level {
						fmt.Printf("Warning: Operation %s has inconsistent input levels: %v\n", op, levels)
						break
					}
				}
			}
		}
	}

	// Special handling for earth.constant - level is in output type after ->
	if op == "earth.constant" {
		arrowIdx := strings.Index(rest, "->")
		if arrowIdx != -1 {
			outputType := rest[arrowIdx+2:]
			// Look for earth.pl<scale * level> in output
			if strings.Contains(outputType, "earth.pl<") {
				plIdx := strings.Index(outputType, "earth.pl<")
				if plIdx != -1 {
					typeIdx := plIdx + len("earth.pl<")
					endIdx := strings.Index(outputType[typeIdx:], ">")
					if endIdx != -1 {
						nums := outputType[typeIdx : typeIdx+endIdx]
						numParts := strings.Split(nums, "*")
						if len(numParts) == 2 {
							levelStr := strings.TrimSpace(numParts[1])
							if levelInt, err := strconv.Atoi(levelStr); err == nil {
								// Convert to maxLevel - level as specified
								level = maxLevel - levelInt
							}
						}
					}
				}
			}
		}
	}

	// Determine if operation is double (between two ciphertexts)
	isDouble = ciphertextCount >= 2

	return &Operation{
		OpType:   op,
		Level:    level,
		IsDouble: isDouble,
		Metadata: metadata,
	}, nil
}

// getSingleOperationCost handles standard single/double operation cost calculation
func getSingleOperationCost(costModel *CostModel, overrideCostKey string, op *Operation, stats *OperationStats) float64 {
	// Map MLIR operation names to cost model keys
	var costKey string
	if overrideCostKey != "" {
		costKey = overrideCostKey
	} else {
		switch op.OpType {
		case "earth.add":
			if op.IsDouble {
				costKey = "earth.add_double"
			} else {
				costKey = "earth.add_single"
			}
		case "earth.mul":
			if op.IsDouble {
				costKey = "earth.mul_double"
			} else {
				costKey = "earth.mul_single"
			}
		case "earth.rotate":
			costKey = "earth.rotate_single"
		case "earth.modswitch":
			costKey = "earth.modswitch_single"
		case "earth.negate":
			costKey = "earth.negate_single"
		case "earth.bootstrap":
			costKey = "earth.bootstrap_single"
		case "earth.rescale":
			costKey = "earth.rescale_single"
		default:
			// Skip operations not in cost model
			return 0
		}
	}

	// Get the cost array for this operation
	costs, exists := costModel.LatencyTable[costKey]
	if !exists {
		return 0
	}

	// Convert level to array index (level 1 = index 0, level 2 = index 1, etc.)
	costIndex := op.Level - 1

	// Check if level is within bounds
	if costIndex < 0 || costIndex >= len(costs) {
		return 0
	}

	cost := costs[costIndex]
	trackOperation(stats, costKey, op.Level, 1, cost)

	return cost
}

// getOperationCost returns the cost for a given operation at a given level
func getOperationCost(costModel *CostModel, constantCosts *ConstantCosts, op *Operation, stats *OperationStats) float64 {
	totalCost := 0.0

	switch op.OpType {
	case "earth.constant":
		// Use encoding costs from benchmark file
		if constantCosts != nil {
			if cost, exists := constantCosts.CostsByLevel[op.Level]; exists {
				trackOperation(stats, "earth.constant_single", op.Level, 1, cost)
				return cost
			}
		}
		// If no constant costs available, return 0
		return 0
	case "earth.rotate":
		// Parse rotation offset from metadata
		re := regexp.MustCompile(`offset\s*=\s*array<i64:\s*(-?\d+)>`)
		match := re.FindStringSubmatch(op.Metadata)
		if len(match) == 2 {
			if offset, err := strconv.Atoi(match[1]); err == nil {
				decomposition := decomposeRotation(offset)
				rotationCount := len(decomposition)

				costs, exists := costModel.LatencyTable["earth.rotate_single"]
				if exists {
					// Convert level to array index (level 1 = index 0, level 2 = index 1, etc.)
					costIndex := op.Level - 1
					if costIndex >= 0 && costIndex < len(costs) {
						singleRotCost := costs[costIndex]
						totalCost = float64(rotationCount) * singleRotCost

						// Track statistics for each individual rotation
						trackOperation(stats, "earth.rotate_single", op.Level, rotationCount, totalCost)
					}
				}
				return totalCost
			}
		}
		// Fallback to single rotation if parsing fails
		return getSingleOperationCost(costModel, "earth.rotate_single", op, stats)

	case "earth.upscale":
		// Upscale is treated as a mul_single operation for cost calculation
		// but tracked separately for analysis
		costs, exists := costModel.LatencyTable["earth.mul_single"]
		if !exists {
			return 0
		}

		costIndex := op.Level - 1
		if costIndex < 0 || costIndex >= len(costs) {
			return 0
		}

		cost := costs[costIndex]
		trackOperation(stats, "earth.upscale_single", op.Level, 1, cost)
		return cost

	case "earth.bootstrap":
		// Bootstrap includes both bootstrap and modswitch operations
		// Calculate combined cost but track as single bootstrap operation
		bootstrapCosts, bootstrapExists := costModel.LatencyTable["earth.bootstrap_single"]
		modswitchCosts, modswitchExists := costModel.LatencyTable["earth.modswitch_single"]

		if !bootstrapExists || !modswitchExists {
			return 0
		}

		costIndex := op.Level - 1
		if costIndex < 0 || costIndex >= len(bootstrapCosts) || costIndex >= len(modswitchCosts) {
			return 0
		}

		totalCost = bootstrapCosts[costIndex] + modswitchCosts[costIndex]
		trackOperation(stats, "earth.bootstrap_single", op.Level, 1, totalCost)
		return totalCost

	default:
		return getSingleOperationCost(costModel, "", op, stats)
	}
}

// trackOperation updates the statistics for an operation
func trackOperation(stats *OperationStats, opType string, level int, count int, cost float64) {
	if stats.Count[opType] == nil {
		stats.Count[opType] = make(map[int]int)
		stats.TotalCost[opType] = make(map[int]float64)
	}
	stats.Count[opType][level] += count
	stats.TotalCost[opType][level] += cost
}

// calculateTotalCost calculates the total cost of all operations in the MLIR file
func calculateTotalCost(costModel *CostModel, constantCosts *ConstantCosts, mlirPath string, maxLevel int) (float64, *OperationStats, int, error) {
	file, err := os.Open(mlirPath)
	if err != nil {
		return 0, nil, 0, fmt.Errorf("error opening MLIR file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	totalCost := 0.0
	operationCount := 0
	stats := NewOperationStats()

	for scanner.Scan() {
		line := scanner.Text()
		op, err := parseMLIROperation(line, maxLevel)
		if err != nil {
			continue
		}
		if op == nil {
			continue
		}

		cost := getOperationCost(costModel, constantCosts, op, stats)
		totalCost += cost
		operationCount++
		stats.Operations = append(stats.Operations, *op)
	}

	if err := scanner.Err(); err != nil {
		return 0, nil, 0, fmt.Errorf("error reading MLIR file: %v", err)
	}

	return totalCost, stats, operationCount, nil
}

// writeDetailedAnalysis creates a detailed analysis report
func writeDetailedAnalysis(stats *OperationStats, totalCost float64, outputPath string) error {
	// Ensure logs directory exists (relative to lowering directory)
	logsDir := filepath.Join("tests", "logs")
	if err := os.MkdirAll(logsDir, 0755); err != nil {
		return fmt.Errorf("error creating logs directory: %v", err)
	}

	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("error creating analysis file: %v", err)
	}
	defer file.Close()

	fmt.Fprintf(file, "DETAILED COST ANALYSIS REPORT\n")
	fmt.Fprintf(file, "==============================\n\n")

	// Calculate total cost without constants
	constantCost := 0.0
	if constantCosts, exists := stats.TotalCost["earth.constant_single"]; exists {
		for _, cost := range constantCosts {
			constantCost += cost
		}
	}
	totalCostWithoutConstants := totalCost - constantCost

	fmt.Fprintf(file, "SUMMARY\n")
	fmt.Fprintf(file, "-------\n")
	fmt.Fprintf(file, "Total estimated cost (with constants): %.2f microseconds\n", totalCost)
	fmt.Fprintf(file, "Total estimated cost (with constants): %.6f seconds\n", totalCost/1000000)
	fmt.Fprintf(file, "Total estimated cost (without constants): %.2f microseconds\n", totalCostWithoutConstants)
	fmt.Fprintf(file, "Total estimated cost (without constants): %.6f seconds\n", totalCostWithoutConstants/1000000)
	fmt.Fprintf(file, "Constant operations cost: %.2f microseconds (%.2f%% of total)\n\n", constantCost, (constantCost/totalCost)*100)

	// Operation counts and costs by level
	fmt.Fprintf(file, "OPERATION BREAKDOWN BY LEVEL\n")
	fmt.Fprintf(file, "----------------------------\n")

	// Get all operation types and levels
	allOps := make(map[string]bool)
	allLevels := make(map[int]bool)
	for opType := range stats.Count {
		allOps[opType] = true
		for level := range stats.Count[opType] {
			allLevels[level] = true
		}
	}

	// Sort levels for consistent output
	var sortedLevels []int
	for level := range allLevels {
		sortedLevels = append(sortedLevels, level)
	}
	for i := 0; i < len(sortedLevels)-1; i++ {
		for j := i + 1; j < len(sortedLevels); j++ {
			if sortedLevels[i] > sortedLevels[j] {
				sortedLevels[i], sortedLevels[j] = sortedLevels[j], sortedLevels[i]
			}
		}
	}

	for opType := range allOps {
		shortName := getShortOpName(opType)
		fmt.Fprintf(file, "\n%s:\n", shortName)
		opTotal := 0.0
		opCount := 0

		for _, level := range sortedLevels {
			if count, exists := stats.Count[opType][level]; exists {
				cost := stats.TotalCost[opType][level]
				percentage := (cost / totalCost) * 100
				fmt.Fprintf(file, "  Level %d: %d operations, %.2f cost (%.2f%% of total)\n",
					level, count, cost, percentage)
				opTotal += cost
				opCount += count
			}
		}

		opPercentage := (opTotal / totalCost) * 100
		fmt.Fprintf(file, "  Total: %d operations, %.2f cost (%.2f%% of total)\n",
			opCount, opTotal, opPercentage)
	}

	// Level distribution analysis
	fmt.Fprintf(file, "\nLEVEL DISTRIBUTION ANALYSIS\n")
	fmt.Fprintf(file, "---------------------------\n")

	levelTotals := make(map[int]float64)
	levelCounts := make(map[int]int)

	for opType := range stats.Count {
		for level, count := range stats.Count[opType] {
			cost := stats.TotalCost[opType][level]
			levelTotals[level] += cost
			levelCounts[level] += count
		}
	}

	for _, level := range sortedLevels {
		if cost, exists := levelTotals[level]; exists {
			count := levelCounts[level]
			percentage := (cost / totalCost) * 100
			fmt.Fprintf(file, "Level %d: %d operations, %.2f cost (%.2f%% of total)\n",
				level, count, cost, percentage)
		}
	}

	// Operation type analysis
	fmt.Fprintf(file, "\nOPERATION TYPE ANALYSIS\n")
	fmt.Fprintf(file, "-----------------------\n")

	for opType := range allOps {
		shortName := getShortOpName(opType)
		opTotal := 0.0
		opCount := 0
		levelBreakdown := make(map[int]float64)

		for level, count := range stats.Count[opType] {
			cost := stats.TotalCost[opType][level]
			opTotal += cost
			opCount += count
			levelBreakdown[level] = cost
		}

		opPercentage := (opTotal / totalCost) * 100
		fmt.Fprintf(file, "\n%s: %d operations, %.2f cost (%.2f%% of total)\n",
			shortName, opCount, opTotal, opPercentage)

		// Show level distribution for this operation
		for _, level := range sortedLevels {
			if cost, exists := levelBreakdown[level]; exists {
				levelPercentage := (cost / opTotal) * 100
				fmt.Fprintf(file, "  %.2f%% at level %d\n", levelPercentage, level)
			}
		}
	}

	return nil
}

func main() {
	if len(os.Args) < 3 || len(os.Args) > 4 {
		fmt.Println("Usage: go run cost_calculator.go <cost_model.json> <mlir_file.mlir> [constant_costs.txt]")
		fmt.Println("Example: go run cost_calculator.go lattigo_config.json circuit.mlir")
		fmt.Println("Example with constants: go run cost_calculator.go lattigo_config.json circuit.mlir encode_benchmark_constant_0.txt")
		os.Exit(1)
	}

	costModelPath := os.Args[1]
	mlirPath := os.Args[2]
	var constantCostsPath string
	if len(os.Args) == 4 {
		constantCostsPath = os.Args[3]
	}

	// Read and parse cost model JSON
	costModelData, err := os.ReadFile(costModelPath)
	if err != nil {
		log.Fatalf("Error reading cost model file: %v", err)
	}

	var costModel CostModel
	err = json.Unmarshal(costModelData, &costModel)
	if err != nil {
		log.Fatalf("Error parsing cost model JSON: %v", err)
	}

	// Read constant costs if provided
	var constantCosts *ConstantCosts
	if constantCostsPath != "" {
		fmt.Printf("Reading constant costs from: %s\n", constantCostsPath)
		constantCosts, err = readConstantCosts(constantCostsPath)
		if err != nil {
			log.Fatalf("Error reading constant costs file: %v", err)
		}
	}

	// Use max level from cost model
	maxLevel := costModel.LevelUpperBound

	fmt.Printf("Processing MLIR file: %s\n", mlirPath)

	// Calculate total cost
	totalCost, stats, operationCount, err := calculateTotalCost(&costModel, constantCosts, mlirPath, maxLevel)
	if err != nil {
		log.Fatalf("Error calculating cost: %v", err)
	}

	// Generate detailed analysis with proper naming convention (relative to lowering directory)
	mlirFilename := filepath.Base(mlirPath)
	mlirName := strings.TrimSuffix(mlirFilename, ".mlir")
	analysisPath := filepath.Join("tests", "logs", "cost_"+mlirName+".txt")

	err = writeDetailedAnalysis(stats, totalCost, analysisPath)
	if err != nil {
		log.Fatalf("Error writing detailed analysis: %v", err)
	}

	fmt.Printf("Total operations processed: %d\n", operationCount)
	fmt.Printf("Total estimated cost: %.2f microseconds (%.6f seconds)\n", totalCost, totalCost/1000000)
	fmt.Printf("Detailed analysis written to: %s\n", analysisPath)
}
