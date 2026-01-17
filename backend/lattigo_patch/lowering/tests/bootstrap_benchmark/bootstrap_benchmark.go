package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"time"
)

type TestResult struct {
	Value    int
	Run      int
	Accuracy float64
	Success  bool
	ErrorMsg string
}

func generateMLIR(value int, filename string) error {
	mlirTemplate := `"builtin.module"() <{sym_name = "test.mlir"}> ({
  "func.func"() <{function_type = (tensor<1x!earth.ci<90 * 13>>) -> tensor<1x!earth.ci<90 * 0>>, sym_name = "_hecate_"}> ({
  ^bb0(%%arg0: tensor<1x!earth.ci<90 * 13>> loc(unknown)):
    %%0 = "earth.constant"() <{1x rms_var = 0.1, value = %d : i64}> : () -> tensor<1x!earth.pl<51 * 13>> loc(unknown)
    %%1 = "earth.bootstrap"(%%0) <{1x targetLevel = 0 : i64}> : (tensor<1x!earth.ci<51 * 13>>) -> tensor<1x!earth.ci<51 * 0>> loc(unknown)

    "func.return"(%%1) : (tensor<1x!earth.ci<51 * 0>>) -> () loc(unknown)
  }) : () -> () loc(unknown)
}) : () -> () loc(unknown)`

	content := fmt.Sprintf(mlirTemplate, value)
	return os.WriteFile(filename, []byte(content), 0644)
}

func runTest(mlirFile string) (float64, bool, string) {
	cmd := exec.Command("go", "run", "../fhe", "-mlir", mlirFile, "-getLog", "temp.txt")

	output, err := cmd.CombinedOutput()
	if err != nil {
		return 0.0, false, fmt.Sprintf("Execution error: %v\nOutput: %s", err, string(output))
	}

	// Parse the accuracy from output
	outputStr := string(output)

	re := regexp.MustCompile(`Final Result Accuracy: ([\d.]+)%`)
	matches := re.FindStringSubmatch(outputStr)

	if len(matches) < 2 {
		return 0.0, false, "Could not parse accuracy from output"
	}

	accuracy, err := strconv.ParseFloat(matches[1], 64)
	if err != nil {
		return 0.0, false, fmt.Sprintf("Could not convert accuracy to float: %v", err)
	}

	return accuracy, true, ""
}

func saveResults(results []TestResult, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	// Write CSV header
	fmt.Fprintf(writer, "Value,Run,Accuracy,Success,ErrorMsg\n")

	// Write data
	for _, result := range results {
		fmt.Fprintf(writer, "%d,%d,%.2f,%t,\"%s\"\n",
			result.Value, result.Run, result.Accuracy, result.Success, result.ErrorMsg)
	}

	return nil
}

func generateSummaryReport(results []TestResult, filename string, minVal, maxVal, numRuns int) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	fmt.Fprintf(writer, "Bootstrap Accuracy Test Summary\n")
	fmt.Fprintf(writer, "Generated: %s\n", time.Now().Format("2006-01-02 15:04:05"))
	fmt.Fprintf(writer, "Test Range: %d to %d (%d run(s) each)\n", minVal, maxVal, numRuns)
	fmt.Fprintf(writer, "Input Scale: 51 bits\n\n")

	// Group results by value
	valueStats := make(map[int][]float64)
	for _, result := range results {
		if result.Success {
			valueStats[result.Value] = append(valueStats[result.Value], result.Accuracy)
		}
	}

	fmt.Fprintf(writer, "%-6s %-12s %-12s %-12s %-8s\n", "Value", "Avg Accuracy", "Min Accuracy", "Max Accuracy", "Runs")
	fmt.Fprintf(writer, "%-6s %-12s %-12s %-12s %-8s\n", "-----", "------------", "------------", "------------", "----")

	for value := minVal; value <= maxVal; value++ {
		accuracies := valueStats[value]
		if len(accuracies) == 0 {
			fmt.Fprintf(writer, "%-6d %-12s %-12s %-12s %-8d\n", value, "FAILED", "FAILED", "FAILED", 0)
			continue
		}

		var sum, min, max float64
		min = accuracies[0]
		max = accuracies[0]

		for _, acc := range accuracies {
			sum += acc
			if acc < min {
				min = acc
			}
			if acc > max {
				max = acc
			}
		}

		avg := sum / float64(len(accuracies))
		fmt.Fprintf(writer, "%-6d %-12.2f %-12.2f %-12.2f %-8d\n", value, avg, min, max, len(accuracies))
	}

	// Failed tests summary
	fmt.Fprintf(writer, "\nFailed Tests:\n")
	for _, result := range results {
		if !result.Success {
			fmt.Fprintf(writer, "Value %d, Run %d: %s\n", result.Value, result.Run, result.ErrorMsg)
		}
	}

	return nil
}

func main() {
	var values []int
	for i := -32; i <= 32; i++ {
		values = append(values, i)
	}
	numTimes := 1

	fmt.Println("Starting Bootstrap Accuracy Test...")
	fmt.Println("Testing values\n", values, "\nwith", numTimes, "run each")

	// Create logs directory if it doesn't exist
	logsDir := "logs"
	err := os.MkdirAll(logsDir, 0755)
	if err != nil {
		fmt.Printf("Error creating logs directory: %v\n", err)
		return
	}

	var results []TestResult
	totalTests := len(values) * numTimes
	currentTest := 0

	// Test each value
	for _, value := range values {
		for run := 1; run <= numTimes; run++ {
			currentTest++
			fmt.Printf("\nTesting value %d, run %d (%d/%d)\n", value, run, currentTest, totalTests)

			// Generate MLIR file for this test in logs directory
			mlirFile := fmt.Sprintf("%s/test_value_%d_run_%d.mlir", logsDir, value, run)
			err := generateMLIR(value, mlirFile)
			if err != nil {
				fmt.Printf("    Error generating MLIR: %v\n", err)
				results = append(results, TestResult{
					Value:    value,
					Run:      run,
					Success:  false,
					ErrorMsg: fmt.Sprintf("MLIR generation error: %v", err),
				})
				continue
			}

			// Run the test
			accuracy, success, errorMsg := runTest(mlirFile)

			result := TestResult{
				Value:    value,
				Run:      run,
				Accuracy: accuracy,
				Success:  success,
				ErrorMsg: errorMsg,
			}
			results = append(results, result)

			if success {
				fmt.Printf("    Accuracy: %.2f%%\n", accuracy)
			} else {
				fmt.Printf("    FAILED: %s\n", errorMsg)
			}

			os.Remove(mlirFile)
		}
	}

	fmt.Println("\nSaving results...")

	// Save detailed CSV
	csvFile := fmt.Sprintf("%s/bootstrap_accuracy_results.csv", logsDir)
	err = saveResults(results, csvFile)
	if err != nil {
		fmt.Printf("Error saving CSV results: %v\n", err)
	} else {
		fmt.Printf("Detailed results saved to: %s\n", csvFile)
	}

	// Save summary report
	summaryFile := fmt.Sprintf("%s/bootstrap_accuracy_summary.txt", logsDir)
	err = generateSummaryReport(results, summaryFile, values[0], values[len(values)-1], numTimes)
	if err != nil {
		fmt.Printf("Error saving summary report: %v\n", err)
	} else {
		fmt.Printf("Summary report saved to: %s\n", summaryFile)
	}

	// Print quick summary to console
	fmt.Println("\nQuick Summary:")
	successCount := 0
	totalAccuracy := 0.0
	for _, result := range results {
		if result.Success {
			successCount++
			totalAccuracy += result.Accuracy
		}
	}

	fmt.Printf("Total tests: %d\n", len(results))
	fmt.Printf("Successful tests: %d\n", successCount)
	fmt.Printf("Failed tests: %d\n", len(results)-successCount)
	if successCount > 0 {
		fmt.Printf("Average accuracy: %.2f%%\n", totalAccuracy/float64(successCount))
	}

	fmt.Println("\nTest complete! Check the generated files for detailed results.")
}
