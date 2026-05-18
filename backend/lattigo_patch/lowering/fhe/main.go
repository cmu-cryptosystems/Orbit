package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
)

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage of %s:\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  Runs and executes Lattigo FHE operations from Orbit MLIR file\n\n")
		flag.PrintDefaults()
	}

	var n int
	var outFile string
	var instructionsPath string
	var mlirPath string
	var constantsPath string
	var dynamicMaskTablePath string
	var dynamicMaskPath string
	var inputsPath string
	var outputPath string
	var trueLabelsPath string
	var maxLevel int
	var bootstrapMinLevel int
	var bootstrapMaxLevel int
	var enableTiming bool
	var heMode bool
	flag.IntVar(&n, "n", 4096, "The polynomial modulus degree")
	flag.IntVar(&maxLevel, "maxLevel", 29, "The maximum level of the FHE scheme")
	flag.IntVar(&bootstrapMinLevel, "bootstrapMinLevel", 3, "The minimum bootstrap level of the FHE scheme")
	flag.IntVar(&bootstrapMaxLevel, "bootstrapMaxLevel", 16, "The maximum bootstrap level of the FHE scheme")
	flag.BoolVar(&enableTiming, "time", false, "Enable detailed timing analysis and generate timing report")
	flag.StringVar(&outFile, "getLog", "", "Enable debug log. Optionally specify output file (default: precision_debug.txt)")
	flag.StringVar(&instructionsPath, "i", "", "Path to instructions file")
	flag.StringVar(&constantsPath, "cons", "", "Path to constants cst file")
	flag.StringVar(&dynamicMaskTablePath, "dynamicMaskTable", "", "Path to dynamic plaintext mask metadata JSON")
	flag.StringVar(&dynamicMaskPath, "dynamicMask", "", "Path to per-sample dynamic plaintext mask values")
	flag.StringVar(&inputsPath, "input", "", "Path to inputs directory")
	flag.StringVar(&outputPath, "output", "", "Path to output file")
	flag.StringVar(&trueLabelsPath, "true", "", "Path to true labels file (for batch processing validation)")
	flag.StringVar(&mlirPath, "mlir", "", "Path to MLIR file")
	flag.BoolVar(&heMode, "heMode", true, "Enable FHE mode (set to false for plaintext execution)")
	flag.Parse()

	if outFile == "true" || outFile == "1" {
		outFile = "precision_debug.txt"
	}

	var fileType FileType
	if mlirPath != "" {
		fileType = MLIR
	} else {
		fileType = Instructions
	}

	if _, err := os.Stat(filepath.Join("logs", outFile)); err == nil {
		os.Remove(filepath.Join("logs", outFile))
	}

	fhe := NewLattigoFHE(n, instructionsPath, mlirPath, constantsPath, dynamicMaskTablePath, dynamicMaskPath, inputsPath, outputPath, trueLabelsPath, fileType, maxLevel, bootstrapMinLevel, bootstrapMaxLevel, outFile, enableTiming, heMode)

	// Batch processing mode
	if inputsPath != "" {
		if info, err := os.Stat(inputsPath); err == nil && info.IsDir() {
			err := fhe.RunBatch()
			if err != nil {
				fmt.Println(err)
				return
			}
			return
		}
	}

	// Single file processing mode
	decrypted, err := fhe.Run()
	if err != nil {
		fmt.Println(err)
		return
	}

	// Handle output file writing
	if outputPath != "" {
		outputFile := "outputs/" + outputPath
		os.MkdirAll("outputs", 0755)
		err = fhe.writeOutputFile(outputFile, decrypted)
		if err != nil {
			fmt.Printf("Error writing output file: %v\n", err)
		} else {
			fmt.Println("Output written to: ", outputFile)
		}
	}
}
