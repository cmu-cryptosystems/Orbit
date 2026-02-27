package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

var (
	reRotateOffset = regexp.MustCompile(`offset\s*=\s*array<i64:\s*(-?\d+)>`)
	reMLIRCi       = regexp.MustCompile(`earth\.ci<\s*([0-9]+)\s*\*\s*([0-9]+)\s*>`)
	reRMSVar       = regexp.MustCompile(`rms_var\s*=\s*([0-9eE\.\-]+)\s*:\s*f64`)
	reValue        = regexp.MustCompile(`value\s*=\s*([-+]?[0-9\.]+)(?:\s*:\s*\w+)?`)
	reUpFactor     = regexp.MustCompile(`upFactor\s*=\s*([0-9]+)(?:\s*:\s*\w+)?`)
	reDownFactor   = regexp.MustCompile(`downFactor\s*=\s*([0-9]+)(?:\s*:\s*\w+)?`)
	reTargetLevel  = regexp.MustCompile(`targetLevel\s*=\s*([0-9]+)(?:\s*:\s*\w+)?`)
	reOutputVar    = regexp.MustCompile(`(?:\"func\.return\"\s*\(\s*%|return\s+%)((?:arg)?-?\d+)`)
)

type FileType int

const (
	Instructions FileType = iota
	MLIR
)

func instructionOptoOp(op string) op {
	switch op {
	case "FHEOp.PACK":
		return PACK
	case "FHEOp.MASK":
		return MASK
	case "FHEOp.ADD":
		return ADD
	case "FHEOp.MUL":
		return MUL
	case "FHEOp.ROT":
		return ROT
	}
	return -1
}

func mlirOpToOp(op string) op {
	switch op {
	case "earth.constant":
		return CONST
	case "earth.add":
		return ADD
	case "earth.mul":
		return MUL
	case "earth.rotate":
		return ROT
	case "earth.modswitch":
		return MODSWITCH
	case "earth.negate":
		return NEGATE
	case "earth.bootstrap":
		return BOOTSTRAP
	case "earth.rescale":
		return RESCALE
	case "earth.upscale":
		return UPSCALE
	}
	return -1
}

func getOpName(operation op) string {
	switch operation {
	case PACK:
		return "PACK"
	case CONST:
		return "CONST"
	case MASK:
		return "MASK"
	case ADD:
		return "ADD"
	case MUL:
		return "MUL"
	case ROT:
		return "ROT"
	case MODSWITCH:
		return "MODSWITCH"
	case NEGATE:
		return "NEGATE"
	case BOOTSTRAP:
		return "BOOTSTRAP"
	case RESCALE:
		return "RESCALE"
	case UPSCALE:
		return "UPSCALE"
	default:
		return "UNKNOWN"
	}
}

func (lattigo *LattigoFHE) ReadFile(path string) (expected string, operations []string, inputs []Term, outputId int, err error) {
	file, err := os.Open(path)
	if err != nil {
		return "", nil, nil, -1, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var section string
	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		if lattigo.fileType == Instructions {
			switch line {
			case "# Expected":
				section = "expected"
				continue
			case "# Operations":
				section = "operations"
				continue
			case "":
				continue
			}

			switch section {
			case "expected":
				expected = line
			case "operations":
				operations = append(operations, line)
			}
		} else if lattigo.fileType == MLIR {
			if strings.HasPrefix(trimmed, "%") {
				operations = append(operations, line)
			} else if strings.HasPrefix(trimmed, "^bb0(") || strings.HasPrefix(trimmed, "func.func @") {
				matches := reMLIRCi.FindStringSubmatch(trimmed)
				if len(matches) == 3 {
					scaleInt, _ := strconv.Atoi(matches[1])
					level, _ := strconv.Atoi(matches[2])
					scale := rlwe.NewScale(math.Pow(2, float64(scaleInt)))

					term := &Term{
						Secret: strings.Contains(trimmed, "earth.ci"),
						Scale:  scale,
						Level:  lattigo.bootstrapMaxLevel - level,
					}
					inputs = append(inputs, *term)
				}
			} else if strings.HasPrefix(trimmed, "\"func.return\"") || strings.HasPrefix(trimmed, "return") {
				matches := reOutputVar.FindStringSubmatch(trimmed)
				if len(matches) == 2 {
					outputStr := matches[1]
					if strings.HasPrefix(outputStr, "arg") {
						outputId, _ = strconv.Atoi(strings.TrimPrefix(outputStr, "arg"))
						outputId = -1 - outputId
					} else {
						outputId, _ = strconv.Atoi(outputStr)
					}
				}
			}
		}
	}

	return expected, operations, inputs, outputId, scanner.Err()
}

func (lattigo *LattigoFHE) processInputs(inputs []Term) {
	// line num is -1 - index of input
	for i, input := range inputs {
		// read from lattigo.inputPath (path to a file)
		readFile, err := os.ReadFile(lattigo.inputPath)
		if err != nil {
			log.Fatal(err)
		}
		lines := strings.Split(string(readFile), "\n")
		numValues, _ := strconv.Atoi(lines[0])
		pt := make([]float64, numValues)
		for j := 0; j < numValues; j++ {
			pt[j], _ = strconv.ParseFloat(lines[j+1], 64)
		}
		lattigo.terms[-1-i] = &input
		lattigo.ptEnv[-1-i] = pt
	}
}

func parseFloatArray(s string) []float64 {
	s = strings.Trim(s, "[]")
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	result := make([]float64, len(parts))
	for i, p := range parts {
		result[i], _ = strconv.ParseFloat(strings.TrimSpace(p), 64)
	}
	return result
}

func parseIntArray(s string) []int {
	s = strings.Trim(s, "[]")
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	result := make([]int, len(parts))
	for i, p := range parts {
		result[i], _ = strconv.Atoi(strings.TrimSpace(p))
	}
	return result
}

/*
Parses a line of instructions and creates a term.
*/
func (lattigo *LattigoFHE) parseOperation(line string) (lineNum int, term *Term) {
	switch lattigo.fileType {
	case Instructions:
		return lattigo.parseInstructionOperation(line)
	case MLIR:
		return lattigo.parseMLIROperation(line)
	default:
		return -1, nil
	}
}

func (lattigo *LattigoFHE) parseInstructionOperation(line string) (int, *Term) {
	if line == "" || strings.HasPrefix(line, "#") {
		return -1, nil
	}

	parts := strings.Split(line, " ")
	lineNum, _ := strconv.Atoi(strings.TrimSuffix(parts[0], ":"))
	op := instructionOptoOp(parts[1])
	cs := parseIntArray(parts[2])
	isSecret, _ := strconv.ParseBool(parts[3])
	metadataStr := parts[4]

	term := &Term{
		Op:       op,
		Children: cs,
		Secret:   isSecret,
		Metadata: parseInstructionsMetadata(metadataStr, op),
	}
	if _, ok := lattigo.terms[lineNum]; !ok {
		lattigo.terms[lineNum] = term
	}
	return lineNum, term
}

func (lattigo *LattigoFHE) parseMLIROperation(line string) (int, *Term) {
	line = strings.TrimSpace(line)
	if line == "" || !strings.HasPrefix(line, "%") {
		return -1, nil
	}

	// Split at '='
	parts := strings.SplitN(line, "=", 2)
	if len(parts) < 2 {
		return -1, nil
	}
	lineNumStr := strings.TrimPrefix(strings.TrimSpace(parts[0]), "%")
	lineNum, err := strconv.Atoi(lineNumStr)
	if err != nil {
		return -1, nil
	}
	rest := strings.TrimSpace(parts[1])

	// Get op name
	opStart := strings.Index(rest, "\"")
	opEnd := strings.Index(rest[opStart+1:], "\"")
	op := rest[opStart+1 : opStart+1+opEnd]

	// Get children (arguments in parentheses)
	children := []int{}
	argStart := strings.Index(rest, "(")
	argEnd := strings.Index(rest, ")")
	if argStart != -1 && argEnd != -1 && argEnd > argStart+1 {
		args := rest[argStart+1 : argEnd]
		for _, a := range strings.Split(args, ",") {
			a = strings.TrimSpace(a)
			if strings.HasPrefix(a, "%arg") {
				argIdxStr := strings.TrimPrefix(a, "%arg")
				if b, err := strconv.Atoi(argIdxStr); err == nil {
					children = append(children, -1-b)
				}
			} else if strings.HasPrefix(a, "%") {
				if n, err := strconv.Atoi(strings.TrimPrefix(a, "%")); err == nil {
					children = append(children, n)
				}
			}
		}
	}

	// For MUL operation with single parameter, duplicate the first child
	if (mlirOpToOp(op) == MUL || mlirOpToOp(op) == ADD) && len(children) == 1 {
		children = append(children, children[0])
	}

	// Get metadata (the stuff inside <{ ... }>)
	metadata := ""
	metaStart := strings.Index(rest, "<{")
	metaEnd := strings.Index(rest, "}>")
	if metaStart != -1 && metaEnd != -1 && metaEnd > metaStart {
		metadata = rest[metaStart+2 : metaEnd]
	}

	// Extract output type after '->'
	secret := false
	var scale rlwe.Scale
	var level int
	arrowIdx := strings.Index(rest, "->")
	if arrowIdx != -1 {
		afterArrow := rest[arrowIdx+2:]
		ciIdx := strings.Index(afterArrow, "earth.ci<")
		plIdx := strings.Index(afterArrow, "earth.pl<")
		var typeIdx int
		if ciIdx != -1 {
			typeIdx = ciIdx + len("earth.ci<")
			secret = true
		} else if plIdx != -1 {
			typeIdx = plIdx + len("earth.pl<")
			secret = false
		}
		if ciIdx != -1 || plIdx != -1 {
			// Extract the numbers before '>'
			endIdx := strings.Index(afterArrow[typeIdx:], ">")
			if endIdx != -1 {
				nums := afterArrow[typeIdx : typeIdx+endIdx]
				numParts := strings.Split(nums, "*")
				if len(numParts) == 2 {
					scaleStr := strings.TrimSpace(numParts[0])
					levelStr := strings.TrimSpace(numParts[1])
					scaleInt, _ := strconv.Atoi(scaleStr)
					level, _ = strconv.Atoi(levelStr)
					scale = rlwe.NewScale(math.Pow(2, float64(scaleInt)))
				}
			}
		}
	}

	term := &Term{
		Op:       mlirOpToOp(op),
		Children: children,
		Secret:   secret,
		Scale:    scale,
		Level:    lattigo.bootstrapMaxLevel - level,
		Metadata: lattigo.parseMetadata(metadata, mlirOpToOp(op)),
	}
	if _, ok := lattigo.terms[lineNum]; !ok {
		lattigo.terms[lineNum] = term
	}
	return lineNum, term
}

func extractRotateOffsetFromMLIRLine(line string, n int) (int, bool) {
	match := reRotateOffset.FindStringSubmatch(line)
	if len(match) == 2 {
		value, err := strconv.Atoi(match[1])
		if err == nil {
			return value % n, true
		}
	}
	return 0, false
}

func (lattigo *LattigoFHE) parseMetadata(metadata string, op op) Metadata {
	switch lattigo.fileType {
	case Instructions:
		return parseInstructionsMetadata(metadata, op)
	case MLIR:
		return parseMLIRMetadata(metadata, op, lattigo.n)
	default:
		return Metadata{}
	}
}

func parseInstructionsMetadata(metadata string, op op) Metadata {
	md := Metadata{}

	switch op {
	case PACK:
		md.PackedValue = parseFloatArray(metadata)
	case MASK:
		md.MaskedValue = parseFloatArray(metadata)
	case ROT:
		md.Offset, _ = strconv.Atoi(metadata)
	}
	return md
}

func parseMLIRMetadata(metadata string, op op, n int) Metadata {
	md := Metadata{}
	switch op {
	case CONST:
		if match := reRMSVar.FindStringSubmatch(metadata); len(match) == 2 {
			if v, err := strconv.ParseFloat(match[1], 64); err == nil {
				md.RMSVar = v
			}
		}
		if match := reValue.FindStringSubmatch(metadata); len(match) == 2 {
			if v, err := strconv.Atoi(match[1]); err == nil {
				md.Value = v
			}
		}
	case ROT:
		if match := reRotateOffset.FindStringSubmatch(metadata); len(match) == 2 {
			if v, err := strconv.Atoi(match[1]); err == nil {
				md.Offset = v % n
			}
		}
	case UPSCALE:
		if match := reUpFactor.FindStringSubmatch(metadata); len(match) == 2 {
			if v, err := strconv.Atoi(match[1]); err == nil {
				md.UpFactor = v
			}
		}
	case MODSWITCH:
		if match := reDownFactor.FindStringSubmatch(metadata); len(match) == 2 {
			if v, err := strconv.Atoi(match[1]); err == nil {
				md.DownFactor = v
			}
		}
	case BOOTSTRAP:
		if match := reTargetLevel.FindStringSubmatch(metadata); len(match) == 2 {
			if v, err := strconv.Atoi(match[1]); err == nil {
				md.TargetLevel = v
			}
		}
	}
	return md
}

// parseTrueLabels parses the true_labels.txt file and returns a map of filename -> true label
func (lattigo *LattigoFHE) parseTrueLabels() (map[string]int, error) {
	if lattigo.trueLabelsPath == "" {
		return nil, nil
	}

	file, err := os.Open(lattigo.trueLabelsPath)
	if err != nil {
		return nil, fmt.Errorf("error opening true labels file: %v", err)
	}
	defer file.Close()

	trueLabels := make(map[string]int)
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		// Parse format: "input0.txt: 3"
		parts := strings.Split(line, ":")
		if len(parts) != 2 {
			continue
		}

		filename := strings.TrimSpace(parts[0])
		labelStr := strings.TrimSpace(parts[1])

		label, err := strconv.Atoi(labelStr)
		if err != nil {
			fmt.Printf("Warning: invalid label '%s' for file '%s'\n", labelStr, filename)
			continue
		}

		trueLabels[filename] = label
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading true labels file: %v", err)
	}

	return trueLabels, nil
}
