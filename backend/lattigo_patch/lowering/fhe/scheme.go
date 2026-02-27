package main

import (
	"fmt"
	"math"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"

	// "github.com/schollz/progressbar/v3"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/utils"
)

type op int

const (
	PACK op = iota
	CONST
	MASK
	ADD
	MUL
	ROT
	MODSWITCH
	NEGATE
	BOOTSTRAP
	RESCALE
	UPSCALE
)

type Metadata struct {
	RMSVar      float64   // Const
	Value       int       // Const
	PackedValue []float64 // Pack
	MaskedValue []float64 // Mask
	Offset      int       // Rot
	UpFactor    int       // Upscale
	DownFactor  int       // Modswitch
	TargetLevel int       // Bootstrap
}

type Term struct {
	Op       op
	Children []int
	Secret   bool
	Metadata Metadata
	Scale    rlwe.Scale
	Level    int
}

type RuntimeInfo struct {
	OutputFileName string
	Runtime        time.Duration
	PredictedClass int
	TrueClass      int
	HasValidation  bool
}

type LevelStats struct {
	Count     int
	TotalTime time.Duration
}

type TimingStats struct {
	OperationStats map[op]map[int]*LevelStats // op -> level -> stats
	TotalTime      time.Duration
}

type LattigoFHE struct {
	params            *ckks.Parameters
	btpParams         *bootstrapping.Parameters
	terms             map[int]*Term                    // stores term info
	env               map[int]*rlwe.Ciphertext         // stores ciphertexts
	ptEnv             map[int][]float64                // stores plaintexts
	constants         map[int][]float64                // stores constants by value
	refCounts         map[int]int                      // stores reference counts for memory management
	hoistedRots       map[int]map[int]*rlwe.Ciphertext // hoisted rotations: childlinenum -> offset -> ciphertext
	rotCount          map[int]int                      // count of rotation uses: childlinenum -> count
	n                 int
	maxLevel          int
	bootstrapMinLevel int
	bootstrapMaxLevel int
	eval              *ckks.Evaluator
	btpEval           *bootstrapping.Evaluator
	enc               *rlwe.Encryptor
	ecd               *ckks.Encoder
	dec               *rlwe.Decryptor
	instructionsPath  string
	mlirPath          string
	constantsPath     string
	inputPath         string
	outputFile        string
	trueLabelsPath    string
	fileType          FileType
	getStats          bool
	logFile           string
	enableTiming      bool
	heMode            bool
	timingStats       *TimingStats
}

func NewLattigoFHE(n int, instructionsPath string, mlirPath string, constantsPath string, inputPath string, outputFile string, trueLabelsPath string, fileType FileType, maxLevel int, bootstrapMinLevel int, bootstrapMaxLevel int, logFile string, enableTiming bool, heMode bool) *LattigoFHE {
	var timingStats *TimingStats
	if enableTiming {
		timingStats = &TimingStats{
			OperationStats: make(map[op]map[int]*LevelStats),
			TotalTime:      0,
		}
	}

	return &LattigoFHE{
		terms:             make(map[int]*Term),
		env:               make(map[int]*rlwe.Ciphertext),
		ptEnv:             make(map[int][]float64),
		constants:         make(map[int][]float64),
		refCounts:         make(map[int]int),
		hoistedRots:       make(map[int]map[int]*rlwe.Ciphertext),
		rotCount:          make(map[int]int),
		n:                 n,
		maxLevel:          maxLevel,
		bootstrapMinLevel: bootstrapMinLevel,
		bootstrapMaxLevel: bootstrapMaxLevel,
		instructionsPath:  instructionsPath,
		mlirPath:          mlirPath,
		constantsPath:     constantsPath,
		inputPath:         inputPath,
		outputFile:        outputFile,
		trueLabelsPath:    trueLabelsPath,
		fileType:          fileType,
		getStats:          logFile != "",
		logFile:           logFile,
		enableTiming:      enableTiming,
		timingStats:       timingStats,
		heMode:            heMode,
	}
}

func (lattigo *LattigoFHE) findUniqueRots(operations []string) []int {
	uniqueRots := make(map[int]struct{})

	for _, operation := range operations {
		var rot int
		var found bool

		if strings.Contains(operation, "ROT") {
			parts := strings.Split(operation, " ")
			if len(parts) > 4 {
				rot, _ = strconv.Atoi(parts[4])
				found = true
			}
		} else if strings.Contains(operation, "rotate") {
			rot, found = extractRotateOffsetFromMLIRLine(operation, lattigo.n)
		}

		if found {
			uniqueRots[rot] = struct{}{}
		}
	}

	result := make([]int, 0, len(uniqueRots))
	for rot := range uniqueRots {
		result = append(result, rot)
	}
	return result
}

func (lattigo *LattigoFHE) findUniqueRotsPow2(operations []string) []int {
	maxRot := 0
	minRot := 0

	for _, operation := range operations {
		var rot int
		var found bool

		if strings.Contains(operation, "ROT") {
			parts := strings.Split(operation, " ")
			if len(parts) > 4 {
				rot, _ = strconv.Atoi(parts[4])
				found = true
			}
		} else if strings.Contains(operation, "rotate") {
			rot, found = extractRotateOffsetFromMLIRLine(operation, lattigo.n)
		}

		if found {
			if rot > maxRot {
				maxRot = rot
			}
			if rot < minRot {
				minRot = rot
			}
		}
	}

	capacity := 0
	for power := 1; power <= maxRot; power *= 2 {
		capacity++
	}
	if minRot < 0 {
		for power := 1; power <= -minRot; power *= 2 {
			capacity++
		}
	}

	result := make([]int, 0, capacity)
	for power := 1; power <= maxRot; power *= 2 {
		result = append(result, power)
	}
	if minRot < 0 {
		for power := 1; power <= -minRot; power *= 2 {
			result = append(result, -power)
		}
	}

	return result
}

func (lattigo *LattigoFHE) createContext(depth int, rots []int) {
	logQ := append([]int{55}, make([]int, depth)...)
	for i := 1; i <= lattigo.maxLevel; i++ {
		logQ[i] = 51
	}
	logN := int(math.Log2(float64(lattigo.n * 2)))
	params, _ := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            logN,
		LogQ:            logQ,
		LogP:            []int{61, 61, 61},
		LogDefaultScale: 51,
	})
	btpParams, _ := bootstrapping.NewParametersFromLiteral(params, bootstrapping.ParametersLiteral{
		LogN: utils.Pointy(logN),
		LogP: []int{61, 61, 61, 61},
	})

	lattigo.params = &params
	lattigo.btpParams = &btpParams

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)

	evk := rlwe.NewMemEvaluationKeySet(rlk)
	lattigo.enc = rlwe.NewEncryptor(params, pk)
	lattigo.ecd = ckks.NewEncoder(params)
	lattigo.dec = rlwe.NewDecryptor(params, sk)
	eval := ckks.NewEvaluator(params, evk)

	if lattigo.heMode {
		fmt.Println("Doing bootstrapping keys...")
		btpEvk, _, _ := btpParams.GenEvaluationKeys(sk)
		btpEval, _ := bootstrapping.NewEvaluator(btpParams, btpEvk)
		lattigo.btpEval = btpEval
		fmt.Println("Bootstrapping circuit depth:", btpEval.Depth())
		fmt.Println("Bootstrapping Output level:", btpEval.OutputLevel())
		fmt.Println("Bootstrapping min input level:", btpEval.MinimumInputLevel())
	} else {
		fmt.Println("Skipping bootstrapping keys for plaintext execution...")
		lattigo.btpEval = nil
	}

	if lattigo.heMode {
		fmt.Println("Doing rotation keys...")
		galEls := make([]uint64, len(rots))
		for i, rot := range rots {
			galEls[i] = params.GaloisElement(rot)
		}
		lattigo.eval = eval.WithKey(rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(galEls, sk)...))
	} else {
		fmt.Println("Skipping rotation keys for plaintext execution...")
	}

	if lattigo.maxLevel != params.MaxLevel() {
		fmt.Printf("Warning: maxLevel mismatch. Expected: %d, Actual: %d\n", params.MaxLevel(), lattigo.maxLevel)
	}
}

func (lattigo *LattigoFHE) deleteFromEnvironments(lineNum int) {
	delete(lattigo.terms, lineNum)
	delete(lattigo.ptEnv, lineNum)
	delete(lattigo.env, lineNum)
}

func (lattigo *LattigoFHE) preprocess(operations []string) {
	for _, line := range operations {
		lineNum, term := lattigo.parseOperation(line)

		for _, child := range term.Children {
			lattigo.refCounts[child]++
		}

		md := term.Metadata
		switch term.Op {
		/* case PACK:
			pt := md.PackedValue
			if !term.Secret {
				lattigo.ptEnv[lineNum] = pt
			}
			lattigo.env[lineNum] = lattigo.encode(pt, nil, lattigo.params.MaxLevel())
		case MASK:
			pt := md.MaskedValue
			lattigo.ptEnv[lineNum] = pt
			lattigo.env[lineNum] = lattigo.encode(pt, nil, lattigo.params.MaxLevel()) */
		case CONST:
			var pt []float64
			if lattigo.constantsPath != "" {
				pt = lattigo.constants[md.Value]
			} else {
				pt = make([]float64, lattigo.n)
				for i := 0; i < lattigo.n; i++ {
					pt[i] = float64(md.Value)
				}
			}
			lattigo.ptEnv[lineNum] = pt
		// case ADD:
		// 	if a, oka := lattigo.ptEnv[term.Children[0]]; oka && !lattigo.terms[term.Children[0]].Secret {
		// 		if b, okb := lattigo.ptEnv[term.Children[1]]; okb && !lattigo.terms[term.Children[1]].Secret {
		// 			pt := make([]float64, lattigo.n)
		// 			for i := 0; i < lattigo.n; i++ {
		// 				pt[i] = a[i] + b[i]
		// 			}
		// 			lattigo.ptEnv[lineNum] = pt
		// 			if lattigo.fileType == MLIR {
		// 				lattigo.env[lineNum] = lattigo.encode(pt, &term.Scale, term.Level)
		// 			} else {
		// 				lattigo.env[lineNum] = lattigo.encode(pt, nil, lattigo.params.MaxLevel())
		// 			}
		// 		}
		// 	}
		case MUL:
			if a, oka := lattigo.ptEnv[term.Children[0]]; oka && !lattigo.terms[term.Children[0]].Secret {
				if b, okb := lattigo.ptEnv[term.Children[1]]; okb && !lattigo.terms[term.Children[1]].Secret {
					pt := make([]float64, lattigo.n)
					for i := 0; i < lattigo.n; i++ {
						pt[i] = a[i] * b[i]
					}
					lattigo.ptEnv[lineNum] = pt
				}
			}
		case ROT:
			childLineNum := term.Children[0]
			lattigo.rotCount[childLineNum]++
		// if a, oka := lattigo.ptEnv[term.Children[0]]; oka && !lattigo.terms[term.Children[0]].Secret {
		// 	rot := md.Offset
		// 	pt := make([]float64, lattigo.n)
		// 	for i := 0; i < lattigo.n; i++ {
		// 		index := ((i+rot)%lattigo.n + lattigo.n) % lattigo.n
		// 		pt[i] = a[index]
		// 	}
		// 	lattigo.ptEnv[lineNum] = pt
		// 	lattigo.env[lineNum] = lattigo.encode(pt, &term.Scale, term.Level)
		// }
		// case NEGATE:
		// 	if a, oka := lattigo.ptEnv[term.Children[0]]; oka && !lattigo.terms[term.Children[0]].Secret {
		// 		pt := make([]float64, lattigo.n)
		// 		for i := 0; i < lattigo.n; i++ {
		// 			pt[i] = -a[i]
		// 		}
		// 		lattigo.ptEnv[lineNum] = pt
		// 	}
		case MODSWITCH, UPSCALE:
			if !lattigo.terms[term.Children[0]].Secret {
				lattigo.ptEnv[lineNum] = lattigo.ptEnv[term.Children[0]]
			}
		}
	}

	// Initialize hoistedRots by tracking unique rotation offsets per childlinenum
	for _, line := range operations {
		_, term := lattigo.parseOperation(line)
		if term.Op == ROT {
			childLineNum := term.Children[0]
			offset := term.Metadata.Offset

			// Initialize the map for this childLineNum if it doesn't exist
			if lattigo.hoistedRots[childLineNum] == nil {
				lattigo.hoistedRots[childLineNum] = make(map[int]*rlwe.Ciphertext)
			}

			// Mark that this offset is needed for this childLineNum (no ciphertext yet)
			lattigo.hoistedRots[childLineNum][offset] = nil
		}
	}
	// clear lattigo.constants
	for i := range lattigo.constants {
		delete(lattigo.constants, i)
	}
}

func (lattigo *LattigoFHE) runInstructions(numOps int) ([]float64, time.Duration, error) {
	want := make([]float64, lattigo.n)
	// var f *os.File

	// f, _ = os.Create(filepath.Join("outputs", "profile.prof"))
	// pprof.StartCPUProfile(f)

	order_list, _, _ := lattigo.getOptimalRunOrder(numOps)
	naive_list := make([]int, numOps)
	for i := range numOps {
		naive_list[i] = i
	}
	working_size, err := lattigo.getMaxWorkingPoolSize(numOps, order_list)
	if err != nil {
		return nil, 0, err
	}
	fmt.Printf("Optimized execution order working pool size: %v\n", working_size)

	working_size_naive, err := lattigo.getMaxWorkingPoolSize(numOps, naive_list)
	if err != nil {
		return nil, 0, err
	}
	fmt.Printf("Naive execution order working pool size: %v\n", working_size_naive)

	// fmt.Println("Optimized execution order: ")
	// for i := range numOps {
	// 	fmt.Printf("Line %d, Op %v, Inputs %v\n", order_list[i], lattigo.terms[order_list[i]].Op, lattigo.terms[order_list[i]].Children)
	// }

	startTime := time.Now()
	fmt.Println("numOps: ", numOps)
	for i := range numOps {
		lineNum := order_list[i]
		term := lattigo.terms[lineNum]

		if lattigo.heMode {
			if _, ok := lattigo.env[lineNum]; !ok {
				lattigo.env[lineNum] = lattigo.evalOp(term)
			}
		} else {
			if _, ok := lattigo.ptEnv[lineNum]; !ok {
				lattigo.ptEnv[lineNum] = lattigo.evalOpPlain(term)
			}
		}

		if lattigo.getStats {
			want = lattigo.doPrecisionStats(lineNum, term)
		}

		for _, child := range term.Children {
			lattigo.refCounts[child]--
			if lattigo.refCounts[child] <= 0 {
				lattigo.deleteFromEnvironments(child)
				delete(lattigo.refCounts, child)
			}
		}
		if i%1000 == 0 {
			fmt.Println("lineNum: ", i)
		}
	}
	runtime := time.Since(startTime)
	// pprof.StopCPUProfile()
	// f.Close()
	fmt.Println()

	return want, runtime, nil
}

func (lattigo *LattigoFHE) testOptimalOrder(operations []string) {
	numOps := len(operations)
	for _, line := range operations {
		lattigo.parseOperation(line)
	}

	order_list, depth, back_depth := lattigo.getOptimalRunOrder(numOps)
	naive_list := make([]int, numOps)
	for i := range numOps {
		naive_list[i] = i
	}
	working_size, err := lattigo.getMaxWorkingPoolSize(numOps, order_list)
	if err != nil {
		fmt.Printf("Error calculating working pool size: %v\n", err)
	}
	fmt.Printf("Optimized execution order working pool size: %v\n", working_size)

	working_size_naive, err := lattigo.getMaxWorkingPoolSize(numOps, naive_list)
	if err != nil {
		fmt.Printf("Error calculating working pool size: %v\n", err)
	}
	fmt.Printf("Naive execution order working pool size: %v\n", working_size_naive)

	fmt.Println("Optimized execution order: ")
	for i := range numOps {
		fmt.Printf("Line %d, Op %v, Depth %v, Back-Depth %v, Inputs %v\n", order_list[i], lattigo.terms[order_list[i]].Op, depth[order_list[i]], back_depth[order_list[i]], lattigo.terms[order_list[i]].Children)
	}
}

func (lattigo *LattigoFHE) Run() (pt_results []float64, _err error) {
	var file string
	if lattigo.fileType == MLIR {
		file = lattigo.mlirPath
	} else {
		file = lattigo.instructionsPath
	}
	fmt.Println("Instructions file: ", file)
	if lattigo.logFile != "" {
		fmt.Println("Debug log: ", filepath.Join("logs", lattigo.logFile))
	}
	if lattigo.outputFile != "" {
		fmt.Println("Output file: ", filepath.Join("outputs", lattigo.outputFile))
	}
	expected_str, operations, inputs, outputId, err := lattigo.ReadFile(file)
	if err != nil {
		return nil, fmt.Errorf("error reading file: %v", err)
	}
	var expected []float64

	// lattigo.testOptimalOrder(operations)

	// return nil, nil

	fmt.Println("\nFinding unique rots...")
	rots := lattigo.findUniqueRotsPow2(operations)
	fmt.Println("Creating context...")
	lattigo.createContext(lattigo.maxLevel, rots)
	if len(inputs) > 0 {
		fmt.Println("Processing inputs from file", lattigo.inputPath, "...")
		lattigo.processInputs(inputs)
	}
	if lattigo.constantsPath != "" {
		fmt.Println("Loading constants from file", lattigo.constantsPath, "...")
		err := lattigo.loadConstants(lattigo.constantsPath)
		if err != nil {
			return nil, fmt.Errorf("error loading constants: %v", err)
		}
	}

	fmt.Println("Preprocessing...")
	lattigo.preprocess(operations)
	lattigo.refCounts[outputId]++ // Ensure output is not deleted during execution

	fmt.Println("Running instructions...")
	want, runtime, err := lattigo.runInstructions(len(operations))
	if err != nil {
		return nil, fmt.Errorf("error running instructions: %v", err)
	}

	var lastResult *rlwe.Ciphertext
	if lattigo.heMode {
		lastResult = lattigo.env[outputId]
		pt_results = lattigo.decode(lastResult)
	} else {
		lastResult = nil
		pt_results = lattigo.ptEnv[outputId]
	}
	if expected_str != "" {
		fmt.Printf("\nOverall Statistics:\n")
		expected = parseFloatArray(expected_str)
		accuracy := lattigo.calculateAccuracy(expected, lastResult)
		if accuracy {
			fmt.Println("Passed! ")
		} else {
			fmt.Println("Failed... ")
			for i := 0; i < len(expected); i++ {
				fmt.Printf("Difference: %v\n", expected[i]-pt_results[i])
			}
		}
		fmt.Printf("\nFinal Result Stats:\n")
		finalStats := ckks.GetPrecisionStats(*lattigo.params, lattigo.ecd, lattigo.dec, expected, lastResult, 0, false)
		fmt.Printf("Final Result Precision: %.2f bits\n", finalStats.AVGLog2Prec.Real)
		fmt.Printf("Final Result Std Deviation: %.2f bits\n", finalStats.STDLog2Prec.Real)
	}
	if lattigo.fileType == MLIR {
		fmt.Printf("\nMLIR Result Stats:\n")
		fmt.Printf("Decrypted Result: %v...\n", pt_results[:20])
		if lattigo.heMode {
			fmt.Printf("Result Scale: %f\n", math.Log2(lastResult.Scale.Float64()))
			fmt.Printf("Result Level (following lattigo): %v\n", lastResult.Level())
		} else {
			fmt.Printf("Result Scale: N/A (plaintext execution)\n")
			fmt.Printf("Result Level (following lattigo): N/A (plaintext execution)\n")
		}
	}
	if lattigo.getStats && want != nil {
		accuracy := lattigo.calculateAccuracy(want, lastResult)
		fmt.Printf("Final Result Accuracy: %.2f%%\n", accuracy)
	}
	fmt.Printf("Runtime: %.3f sec\n", runtime.Seconds())

	if lattigo.enableTiming {
		err := lattigo.writeTimingReport()
		if err != nil {
			fmt.Printf("Warning: Failed to write timing report: %v\n", err)
		}
	}

	return pt_results, nil
}

func (lattigo *LattigoFHE) RunBatch() error {
	var file string
	if lattigo.fileType == MLIR {
		file = lattigo.mlirPath
	} else {
		file = lattigo.instructionsPath
	}
	fmt.Println("Instructions file: ", file)
	if lattigo.logFile != "" {
		fmt.Println("Debug log: ", filepath.Join("logs", lattigo.logFile))
	}

	// Parse true labels if provided
	trueLabels, err := lattigo.parseTrueLabels()
	if err != nil {
		return fmt.Errorf("error parsing true labels: %v", err)
	}
	if trueLabels != nil {
		fmt.Printf("True labels loaded for %d files\n", len(trueLabels))
	}

	// Read operations and inputs from instruction/MLIR file
	expected_str, operations, inputs, outputId, err := lattigo.ReadFile(file)
	if err != nil {
		return fmt.Errorf("error reading file: %v", err)
	}

	// Initialize context once for all inputs
	fmt.Println("\nFinding unique rots...")
	rots := lattigo.findUniqueRotsPow2(operations)
	fmt.Println("Creating context...")
	lattigo.createContext(lattigo.maxLevel, rots)

	// Process constants once for all inputs
	if lattigo.constantsPath != "" {
		fmt.Println("Loading constants...")
		err := lattigo.loadConstants(lattigo.constantsPath)
		if err != nil {
			return fmt.Errorf("error loading constants: %v", err)
		}
	}

	inputFiles, err := lattigo.findInputFiles()
	if err != nil {
		return fmt.Errorf("error finding input files: %v", err)
	}

	if len(inputFiles) == 0 {
		return fmt.Errorf("no input files found matching pattern input*.txt")
	}

	outputDir, err := lattigo.createOutputDirectory()
	if err != nil {
		return fmt.Errorf("error creating output directory: %v", err)
	}

	// Track runtimes and corresponding output file names
	var runtimeInfos []RuntimeInfo
	var expected []float64
	if expected_str != "" {
		expected = parseFloatArray(expected_str)
	}

	// Track validation results
	var passedCount, failedCount int

	fmt.Printf("\nProcessing %d input files...\n", len(inputFiles))

	// Process each input file
	for _, inputFile := range inputFiles {
		fmt.Printf("Processing %s...\n", filepath.Base(inputFile))

		lattigo.resetForNewInput()

		originalInputPath := lattigo.inputPath
		lattigo.inputPath = inputFile
		lattigo.processInputs(inputs)
		lattigo.inputPath = originalInputPath

		fmt.Println("Preprocessing...")
		lattigo.preprocess(operations)

		fmt.Println("Running instructions...")
		_, runtime, err := lattigo.runInstructions(len(operations))
		if err != nil {
			fmt.Printf("Error running instructions for %s: %v\n", inputFile, err)
			continue
		}

		finalResult := lattigo.env[outputId]
		pt_results := lattigo.decode(finalResult)

		// Print first 10 values from the output
		fmt.Printf("  First 10 output values: [")
		for i := 0; i < 10 && i < len(pt_results); i++ {
			if i > 0 {
				fmt.Printf(", ")
			}
			fmt.Printf("%.6f", pt_results[i])
		}
		fmt.Printf("]\n")

		// Validate results if true labels are available
		var predictedClass, trueClass int
		var hasValidation bool
		if trueLabels != nil {
			inputFileName := filepath.Base(inputFile)
			if trueLabel, exists := trueLabels[inputFileName]; exists {
				isCorrect, predicted := lattigo.validateResult(pt_results, trueLabel)
				predictedClass = predicted
				trueClass = trueLabel
				hasValidation = true
				if isCorrect {
					fmt.Printf("  PASSED (predicted: %d, true: %d)\n", predictedClass, trueLabel)
					passedCount++
				} else {
					fmt.Printf("  FAILED (predicted: %d, true: %d)\n", predictedClass, trueLabel)
					failedCount++
				}
			}
		}

		// Write output file
		outputFileName := lattigo.generateOutputFileName(inputFile)
		outputPath := filepath.Join(outputDir, outputFileName)
		err = lattigo.writeOutputFile(outputPath, pt_results)
		if err != nil {
			fmt.Printf("Error writing output file %s: %v\n", outputPath, err)
			continue
		}

		runtimeInfos = append(runtimeInfos, RuntimeInfo{
			OutputFileName: outputFileName,
			Runtime:        runtime,
			PredictedClass: predictedClass,
			TrueClass:      trueClass,
			HasValidation:  hasValidation,
		})

		if len(expected) > 0 {
			accuracy := lattigo.calculateAccuracy(expected, finalResult)
			fmt.Printf("  Accuracy: %.2f%%\n", accuracy)
		}

		fmt.Printf("Runtime: %.3f sec\n\n", runtime.Seconds())
	}

	if len(runtimeInfos) > 0 {
		var totalDuration time.Duration
		for _, info := range runtimeInfos {
			totalDuration += info.Runtime
		}
		avgDuration := totalDuration / time.Duration(len(runtimeInfos))
		fmt.Printf("\nBatch processing completed!")
		fmt.Printf("\nProcessed %d files", len(runtimeInfos))
		fmt.Printf("\nTotal runtime: %v", totalDuration)
		fmt.Printf("\nAverage runtime per file: %v\n", avgDuration)
		fmt.Printf("Output files written to: %s\n", outputDir)

		if trueLabels != nil && (passedCount+failedCount) > 0 {
			totalValidated := passedCount + failedCount
			accuracy := float64(passedCount) / float64(totalValidated) * 100
			fmt.Printf("\nValidation Summary:")
			fmt.Printf("\nTotal validated: %d", totalValidated)
			fmt.Printf("\nPassed: %d", passedCount)
			fmt.Printf("\nFailed: %d", failedCount)
			fmt.Printf("\nAccuracy: %.2f%%\n", accuracy)
		}

		err = lattigo.writeRuntimesFile(outputDir, runtimeInfos, avgDuration)
		if err != nil {
			fmt.Printf("Warning: Error writing runtimes file: %v\n", err)
		} else {
			fmt.Printf("Runtimes written to: %s\n\n", filepath.Join(outputDir, "runtimes.txt"))
		}
	}

	if lattigo.enableTiming {
		err := lattigo.writeTimingReport()
		if err != nil {
			fmt.Printf("Warning: Failed to write timing report: %v\n", err)
		}
	}

	return nil
}
