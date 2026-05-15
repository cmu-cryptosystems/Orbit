package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"

	tiestimator "github.com/tuneinsight/ckks-noise-estimator"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type Request struct {
	SchemaVersion string `json:"schema_version"`
	CKKS          struct {
		PolyDegree int `json:"poly_degree"`
		Sw         int `json:"Sw"`
		Sf         int `json:"Sf"`
		LvlUB      int `json:"lvl_ub"`
	} `json:"ckks"`
	Candidate struct {
		BootstrapCount          int     `json:"bootstrap_count"`
		RescaleCount            int     `json:"rescale_count"`
		FallbackSelectedBudgets int     `json:"fallback_selected_budgets"`
		ProfileRisk             float64 `json:"profile_risk"`
		OutputScaleBits         int     `json:"output_scale_bits"`
	} `json:"candidate"`
}

type Response struct {
	SchemaVersion          string        `json:"schema_version"`
	Source                 string        `json:"source"`
	Paper                  string        `json:"paper"`
	Model                  string        `json:"model"`
	Version                string        `json:"version"`
	Mode                   string        `json:"mode"`
	Valid                  bool          `json:"valid"`
	Fallback               bool          `json:"fallback"`
	EstimatedNoiseBits     float64       `json:"estimated_noise_bits"`
	EstimatedPrecisionBits float64       `json:"estimated_precision_bits"`
	OutputMarginBits       float64       `json:"output_margin_bits"`
	UnsupportedOps         []string      `json:"unsupported_ops"`
	Hotspots               []interface{} `json:"hotspots"`
}

func main() {
	var req Request
	if err := json.NewDecoder(os.Stdin).Decode(&req); err != nil {
		fmt.Fprintf(os.Stderr, "decode request: %v\n", err)
		os.Exit(2)
	}
	polyDegree := req.CKKS.PolyDegree
	if polyDegree <= 0 {
		polyDegree = 32768
	}
	scaleBits := req.Candidate.OutputScaleBits
	if scaleBits <= 0 {
		scaleBits = req.CKKS.Sw
	}
	if scaleBits <= 0 {
		scaleBits = req.CKKS.Sf
	}
	if scaleBits <= 0 {
		scaleBits = 40
	}

	noiseBits, precisionBits, mode, unsupported := tuneInsightEstimate(req, polyDegree, scaleBits)
	if req.Candidate.FallbackSelectedBudgets > 0 {
		unsupported = append(unsupported, "seed_fallback_selected")
	}
	resp := Response{
		SchemaVersion:          "orbit-noise-estimator-result-v1",
		Source:                 "tuneinsight/ckks-noise-estimator",
		Paper:                  "https://eprint.iacr.org/2024/853",
		Model:                  "tuneinsight-lattigo-v6-heuristic",
		Version:                "orbit-estimator-sidecar-v1",
		Mode:                   mode,
		Valid:                  len(unsupported) == 0,
		Fallback:               false,
		EstimatedNoiseBits:     noiseBits,
		EstimatedPrecisionBits: precisionBits,
		OutputMarginBits:       precisionBits - 2.0,
		UnsupportedOps:         unsupported,
		Hotspots:               []interface{}{},
	}
	if err := json.NewEncoder(os.Stdout).Encode(resp); err != nil {
		fmt.Fprintf(os.Stderr, "encode response: %v\n", err)
		os.Exit(3)
	}
}

func tuneInsightEstimate(req Request, polyDegree, scaleBits int) (float64, float64, string, []string) {
	heurNoiseBits, heurPrecisionBits := heuristicEstimate(req, polyDegree, scaleBits)
	level := req.CKKS.LvlUB
	if level <= 0 {
		level = 16
	}
	logN := int(math.Round(math.Log2(float64(polyDegree))))
	if logN < 2 {
		return heurNoiseBits, heurPrecisionBits, "go_sidecar_heuristic_invalid_params", []string{"invalid_logN"}
	}
	sf := req.CKKS.Sf
	if sf <= 0 {
		sf = scaleBits
	}
	if sf <= 0 {
		sf = 40
	}
	logQ := append([]int{55}, make([]int, level)...)
	for i := 1; i <= level; i++ {
		logQ[i] = min(max(sf, 20), 60)
	}
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            logN,
		LogQ:            logQ,
		LogP:            []int{61, 61, 61},
		LogDefaultScale: scaleBits,
	})
	if err != nil {
		return heurNoiseBits, heurPrecisionBits, "go_sidecar_heuristic_param_fallback", []string{fmt.Sprintf("params: %v", err)}
	}
	est := tiestimator.NewEstimator(params)
	slots := min(params.MaxSlots(), 8)
	values := make([]float64, slots)
	for i := range values {
		values[i] = 1.0
	}
	el := est.NewElement(values, 1, params.MaxLevel(), params.DefaultScale())
	est.AddEncryptionNoiseSk(el)
	unsupported := []string{}
	rescales := max(req.Candidate.RescaleCount, 0)
	if rescales > el.Level {
		return heurNoiseBits, heurPrecisionBits, "go_sidecar_tuneinsight_aggregate_heuristic", unsupported
	}
	for i := 0; i < rescales; i++ {
		if err := est.Rescale(el, el); err != nil {
			unsupported = append(unsupported, fmt.Sprintf("rescale: %v", err))
			break
		}
	}
	decrypted := est.Decrypt(el)
	maxErr := 0.0
	for i := 0; i < slots && i < len(decrypted); i++ {
		realPart, _ := decrypted[i][0].Float64()
		imagPart, _ := decrypted[i][1].Float64()
		maxErr = math.Max(maxErr, math.Hypot(realPart-1.0, imagPart))
	}
	if maxErr <= 0 || math.IsNaN(maxErr) || math.IsInf(maxErr, 0) {
		return heurNoiseBits, heurPrecisionBits, "go_sidecar_heuristic_zero_error_fallback", unsupported
	}
	simPrecisionBits := -math.Log2(maxErr)
	bootstrapPenalty := math.Log2(1.0+float64(max(req.Candidate.BootstrapCount, 0))) * 0.75
	profilePenalty := 0.0
	if req.Candidate.ProfileRisk > 0.0 {
		profilePenalty = math.Log2(1.0 + req.Candidate.ProfileRisk)
	}
	precisionBits := math.Max(0.0, math.Min(heurPrecisionBits, simPrecisionBits-bootstrapPenalty-profilePenalty))
	noiseBits := math.Max(0.0, float64(scaleBits)-precisionBits)
	return noiseBits, precisionBits, "go_sidecar_tuneinsight_estimator", unsupported
}

func heuristicEstimate(req Request, polyDegree, scaleBits int) (float64, float64) {
	logN := math.Log2(float64(polyDegree))
	canonicalSigmaBits := 0.5 * math.Max(0.0, logN-1.0)
	roundingCost := math.Sqrt(math.Max(1.0, float64(req.Candidate.RescaleCount))) * math.Pow(2.0, canonicalSigmaBits)
	bootstrapCost := float64(req.Candidate.BootstrapCount) * math.Pow(2.0, math.Max(0.0, canonicalSigmaBits-2.0))
	profileCost := 0.0
	if req.Candidate.ProfileRisk > 0.0 {
		profileCost = math.Log2(1.0 + req.Candidate.ProfileRisk)
	}
	aggregateCost := 1.0 + roundingCost + bootstrapCost + profileCost
	noiseBits := math.Log2(math.Max(aggregateCost, 1.0))
	precisionBits := math.Max(0.0, float64(scaleBits)-noiseBits)
	return noiseBits, precisionBits
}
