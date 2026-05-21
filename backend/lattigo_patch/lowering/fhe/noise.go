package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
)

type NoiseOptions struct {
	Mode        string
	ProfilePath string
	Seed        int64
	ReportPath  string
}

type noiseProfile struct {
	LogScale               int                `json:"log_scale"`
	OperationPrecisionBits map[string]float64 `json:"operation_precision_bits"`
}

type NoiseReportEntry struct {
	Count       int     `json:"count"`
	SumStdDev   float64 `json:"sum_std_dev"`
	MaxStdDev   float64 `json:"max_std_dev"`
	MaxAbsNoise float64 `json:"max_abs_noise"`
}

type NoiseSimulator struct {
	mode          string
	seed          int64
	reportPath    string
	referenceLogS int
	precisionBits map[string]float64
	rng           *rand.Rand
	report        map[string]*NoiseReportEntry
}

var fallbackPrecisionBits = map[string]float64{
	"constant":   60,
	"input":      60,
	"add":        46,
	"mul_plain":  42,
	"mul_cipher": 36,
	"rotate":     42,
	"rescale":    38,
	"upscale":    52,
	"bootstrap":  24,
	"negate":     60,
	"modswitch":  42,
}

func NewNoiseSimulator(options NoiseOptions) (*NoiseSimulator, error) {
	if options.Mode == "" || options.Mode == "off" {
		return nil, nil
	}
	if options.Mode != "simulate" {
		return nil, fmt.Errorf("unsupported noiseMode %q: expected off or simulate", options.Mode)
	}

	precision := make(map[string]float64)
	for key, value := range fallbackPrecisionBits {
		precision[key] = value
	}
	referenceLogS := 40

	if options.ProfilePath != "" {
		raw, err := os.ReadFile(options.ProfilePath)
		if err != nil {
			return nil, fmt.Errorf("error reading noise profile %s: %w", options.ProfilePath, err)
		}
		var profile noiseProfile
		if err := json.Unmarshal(raw, &profile); err != nil {
			return nil, fmt.Errorf("error decoding noise profile %s: %w", options.ProfilePath, err)
		}
		if profile.LogScale > 0 {
			referenceLogS = profile.LogScale
		}
		for key, value := range profile.OperationPrecisionBits {
			if value > 0 {
				precision[key] = value
			}
		}
	}

	seed := options.Seed
	if seed == 0 {
		seed = 20260519
	}

	return &NoiseSimulator{
		mode:          options.Mode,
		seed:          seed,
		reportPath:    options.ReportPath,
		referenceLogS: referenceLogS,
		precisionBits: precision,
		rng:           rand.New(rand.NewSource(seed)),
		report:        make(map[string]*NoiseReportEntry),
	}, nil
}

func (noise *NoiseSimulator) Enabled() bool {
	return noise != nil && noise.mode == "simulate"
}

func (noise *NoiseSimulator) operationKey(lattigo *LattigoFHE, term *Term) string {
	switch term.Op {
	case ADD:
		return "add"
	case MUL:
		if len(term.Children) == 2 {
			left, leftOK := lattigo.terms[term.Children[0]]
			right, rightOK := lattigo.terms[term.Children[1]]
			if leftOK && rightOK && left.Secret && right.Secret {
				return "mul_cipher"
			}
		}
		return "mul_plain"
	case ROT:
		return "rotate"
	case MODSWITCH:
		return "modswitch"
	case NEGATE:
		return "negate"
	case BOOTSTRAP:
		return "bootstrap"
	case RESCALE:
		return "rescale"
	case UPSCALE:
		return "upscale"
	default:
		return "unknown"
	}
}

func (noise *NoiseSimulator) stddevFor(lattigo *LattigoFHE, term *Term, values []float64) (string, float64) {
	key := noise.operationKey(lattigo, term)
	bits := noise.precisionBits[key]
	if bits <= 0 {
		bits = 45
	}

	scaleLog := noise.referenceLogS
	scaleFloat := term.Scale.Float64()
	if scaleFloat > 0 {
		scaleLog = int(math.Round(math.Log2(scaleFloat)))
	}
	scalePenalty := math.Pow(2, float64(maxInt(0, noise.referenceLogS-scaleLog)))
	levelPenalty := math.Pow(2, 0.25*float64(maxInt(0, lattigo.bootstrapMinLevel-term.Level)))

	var sumSq float64
	for _, value := range values {
		sumSq += value * value
	}
	rms := math.Sqrt(sumSq / float64(maxInt(1, len(values))))
	if rms < 1e-12 {
		rms = 1e-12
	}

	return key, rms * math.Pow(2, -bits) * scalePenalty * levelPenalty
}

func (noise *NoiseSimulator) Inject(lattigo *LattigoFHE, term *Term, values []float64) []float64 {
	if !noise.Enabled() || values == nil || !term.Secret {
		return values
	}

	key, stddev := noise.stddevFor(lattigo, term, values)
	if stddev <= 0 {
		return values
	}
	out := append([]float64(nil), values...)

	entry := noise.report[key]
	if entry == nil {
		entry = &NoiseReportEntry{}
		noise.report[key] = entry
	}
	entry.Count++
	entry.SumStdDev += stddev
	if stddev > entry.MaxStdDev {
		entry.MaxStdDev = stddev
	}

	for index := range out {
		draw := noise.rng.NormFloat64() * stddev
		out[index] += draw
		absDraw := math.Abs(draw)
		if absDraw > entry.MaxAbsNoise {
			entry.MaxAbsNoise = absDraw
		}
	}
	return out
}

func (noise *NoiseSimulator) WriteReport() error {
	if !noise.Enabled() || noise.reportPath == "" {
		return nil
	}
	payload := map[string]any{
		"mode":                noise.mode,
		"seed":                noise.seed,
		"reference_log_scale": noise.referenceLogS,
		"operations":          noise.report,
	}
	raw, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	if dir := filepath.Dir(noise.reportPath); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(noise.reportPath, raw, 0o644)
}

func maxInt(left int, right int) int {
	if left > right {
		return left
	}
	return right
}
