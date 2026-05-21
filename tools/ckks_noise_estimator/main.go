package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"

	estimator "github.com/tuneinsight/ckks-noise-estimator"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

type opStats struct {
	RMSAbsError   float64 `json:"rms_abs_error"`
	MaxAbsError   float64 `json:"max_abs_error"`
	PrecisionBits float64 `json:"precision_bits"`
}

type output struct {
	Source                 string             `json:"source"`
	LogN                   int                `json:"log_n"`
	LogScale               int                `json:"log_scale"`
	Iterations             int                `json:"iterations"`
	OperationStats         map[string]opStats `json:"operation_stats"`
	OperationPrecisionBits map[string]float64 `json:"operation_precision_bits"`
}

func complex128FromBig(z *bignum.Complex) complex128 {
	r, _ := z[0].Float64()
	i, _ := z[1].Float64()
	return complex(r, i)
}

func cloneValues(values []*bignum.Complex) []complex128 {
	out := make([]complex128, len(values))
	for i := range values {
		out[i] = complex128FromBig(values[i])
	}
	return out
}

func errorStats(want []complex128, have []*bignum.Complex) opStats {
	n := min(len(want), len(have))
	var sumSq float64
	var maxErr float64
	for i := 0; i < n; i++ {
		diff := want[i] - complex128FromBig(have[i])
		err := math.Hypot(real(diff), imag(diff))
		sumSq += err * err
		if err > maxErr {
			maxErr = err
		}
	}
	rms := math.Sqrt(sumSq / float64(max(1, n)))
	bits := 80.0
	if rms > 0 {
		bits = -math.Log2(rms)
	}
	return opStats{RMSAbsError: rms, MaxAbsError: maxErr, PrecisionBits: bits}
}

func mergeStats(values []opStats) opStats {
	var sumRMS float64
	var maxErr float64
	for _, value := range values {
		sumRMS += value.RMSAbsError
		if value.MaxAbsError > maxErr {
			maxErr = value.MaxAbsError
		}
	}
	rms := sumRMS / float64(max(1, len(values)))
	bits := 80.0
	if rms > 0 {
		bits = -math.Log2(rms)
	}
	return opStats{RMSAbsError: rms, MaxAbsError: maxErr, PrecisionBits: bits}
}

func main() {
	logN := flag.Int("logN", 16, "CKKS log polynomial degree")
	logScale := flag.Int("logScale", 40, "CKKS default log scale")
	iters := flag.Int("iters", 3, "estimator samples per operation")
	outPath := flag.String("out", "", "optional JSON output path")
	flag.Parse()

	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            *logN,
		LogQ:            []int{55, *logScale, *logScale, *logScale, *logScale},
		LogP:            []int{60},
		LogDefaultScale: *logScale,
	})
	if err != nil {
		panic(err)
	}

	ecd := ckks.NewEncoder(params)
	kgen := ckks.NewKeyGenerator(params)
	_, pk := kgen.GenKeyPairNew()
	est := estimator.NewEstimator(params)

	results := map[string][]opStats{
		"add":        {},
		"mul_cipher": {},
		"mul_plain":  {},
		"rotate":     {},
		"rescale":    {},
	}

	for i := 0; i < *iters; i++ {
		values0, el0, _, _ := est.NewTestVector(ecd, pk, -1-1i, 1+1i)
		values1, el1, _, _ := est.NewTestVector(ecd, pk, -1-1i, 1+1i)

		wantAdd := cloneValues(values0)
		values1c := cloneValues(values1)
		for j := range wantAdd {
			wantAdd[j] += values1c[j]
		}
		addEl := el0.CopyNew()
		if err := est.Add(el0, el1, addEl); err != nil {
			panic(err)
		}
		results["add"] = append(results["add"], errorStats(wantAdd, est.Decrypt(addEl)))

		wantMul := cloneValues(values0)
		for j := range wantMul {
			wantMul[j] *= values1c[j]
		}
		mulEl := el0.CopyNew()
		if err := est.MulRelin(el0, el1, mulEl); err != nil {
			panic(err)
		}
		results["mul_cipher"] = append(results["mul_cipher"], errorStats(wantMul, est.Decrypt(mulEl)))

		wantMulPlain := cloneValues(values0)
		for j := range wantMulPlain {
			wantMulPlain[j] *= 2
		}
		mulPlainEl := el0.CopyNew()
		if err := est.Mul(el0, 2.0, mulPlainEl); err != nil {
			panic(err)
		}
		results["mul_plain"] = append(results["mul_plain"], errorStats(wantMulPlain, est.Decrypt(mulPlainEl)))

		wantRot := utils.RotateSlice(cloneValues(values0), 1)
		rotEl := el0.CopyNew()
		if err := est.Rotate(el0, 1, rotEl); err != nil {
			panic(err)
		}
		results["rotate"] = append(results["rotate"], errorStats(wantRot, est.Decrypt(rotEl)))

		wantRescale := cloneValues(values0)
		rescaleEl := el0.CopyNew()
		if err := est.Rescale(el0, rescaleEl); err != nil {
			panic(err)
		}
		results["rescale"] = append(results["rescale"], errorStats(wantRescale, est.Decrypt(rescaleEl)))
	}

	merged := map[string]opStats{}
	precision := map[string]float64{
		"constant":  60,
		"input":     60,
		"rescale":   38,
		"upscale":   52,
		"bootstrap": 24,
		"negate":    60,
		"modswitch": 42,
	}
	for op, stats := range results {
		merged[op] = mergeStats(stats)
		if merged[op].PrecisionBits > 0 {
			precision[op] = merged[op].PrecisionBits
		}
	}

	payload := output{
		Source:                 "github.com/tuneinsight/ckks-noise-estimator",
		LogN:                   *logN,
		LogScale:               *logScale,
		Iterations:             *iters,
		OperationStats:         merged,
		OperationPrecisionBits: precision,
	}

	raw, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		panic(err)
	}
	raw = append(raw, '\n')
	if *outPath != "" {
		if err := os.WriteFile(*outPath, raw, 0o644); err != nil {
			panic(err)
		}
		return
	}
	fmt.Print(string(raw))
}
