package main

import (
	"fmt"
	"math"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func (lattigo *LattigoFHE) recordTiming(operation op, level int, duration time.Duration) {
	if !lattigo.enableTiming || lattigo.timingStats == nil {
		return
	}

	if lattigo.timingStats.OperationStats[operation] == nil {
		lattigo.timingStats.OperationStats[operation] = make(map[int]*LevelStats)
	}
	if lattigo.timingStats.OperationStats[operation][level] == nil {
		lattigo.timingStats.OperationStats[operation][level] = &LevelStats{}
	}

	stats := lattigo.timingStats.OperationStats[operation][level]
	stats.Count++
	stats.TotalTime += duration
	lattigo.timingStats.TotalTime += duration
}

func (lattigo *LattigoFHE) evalDoubleAdd(ct1, ct2 *rlwe.Ciphertext) *rlwe.Ciphertext {
	ct, _ := lattigo.eval.AddNew(ct1, ct2)
	return ct
}

func (lattigo *LattigoFHE) evalSingleAdd(ct1 *rlwe.Ciphertext, pt1 []float64, ptLevel int, ptScale *rlwe.Scale) *rlwe.Ciphertext {
	pt := ckks.NewPlaintext(*lattigo.params, ptLevel)
	pt.Scale = *ptScale
	lattigo.ecd.Encode(pt1, pt)
	ct, _ := lattigo.eval.AddNew(ct1, pt)
	return ct
}

func (lattigo *LattigoFHE) evalAddPlain(pt1, pt2 []float64) []float64 {
	result := make([]float64, len(pt1))
	for i := range pt1 {
		result[i] = pt1[i] + pt2[i]
	}
	return result
}

func (lattigo *LattigoFHE) evalDoubleMul(ct1, ct2 *rlwe.Ciphertext) *rlwe.Ciphertext {
	ct, _ := lattigo.eval.MulRelinNew(ct1, ct2)
	if lattigo.fileType == Instructions {
		lattigo.eval.Rescale(ct, ct)
	}
	return ct
}

func (lattigo *LattigoFHE) evalSingleMul(ct1 *rlwe.Ciphertext, pt1 []float64, ptLevel int, ptScale *rlwe.Scale) *rlwe.Ciphertext {
	pt := ckks.NewPlaintext(*lattigo.params, ptLevel)
	pt.Scale = *ptScale
	lattigo.ecd.Encode(pt1, pt)
	ct, _ := lattigo.eval.MulRelinNew(ct1, pt)
	return ct
}

func (lattigo *LattigoFHE) evalMulPlain(pt1, pt2 []float64) []float64 {
	result := make([]float64, len(pt1))
	for i := range pt1 {
		result[i] = pt1[i] * pt2[i]
	}
	return result
}

func (lattigo *LattigoFHE) evalRot(ct1 *rlwe.Ciphertext, k int) *rlwe.Ciphertext {
	ct, _ := lattigo.eval.RotateNew(ct1, k)
	return ct
}

func (lattigo *LattigoFHE) evalRotPlain(pt1 []float64, k int) []float64 {
	n := len(pt1)
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		newIndex := (i - k + n) % n
		if newIndex < 0 {
			newIndex += n
		}
		result[newIndex] = pt1[i]
	}
	return result
}

func (lattigo *LattigoFHE) evalRotPow2(ct1 *rlwe.Ciphertext, k int) *rlwe.Ciphertext {
	absK := k
	if absK < 0 {
		absK = -absK
	}

	isPowerOfTwo := (absK & (absK - 1)) == 0

	if isPowerOfTwo || k == 0 {
		ct, err := lattigo.eval.RotateNew(ct1, k)
		if err != nil {
			fmt.Printf("Error rotating by %d: %v\n", k, err)
		}
		return ct
	} else {
		decomposition := lattigo.decomposeRotation(k)

		ct, _ := lattigo.eval.RotateNew(ct1, decomposition[0])
		for i := 1; i < len(decomposition); i++ {
			ct, _ = lattigo.eval.RotateNew(ct, decomposition[i])
		}
		return ct
	}
}

func (lattigo *LattigoFHE) evalUpscale(ct1 *rlwe.Ciphertext, upFactor int) *rlwe.Ciphertext {

	scaleup := rlwe.NewScale(math.Pow(2, float64(upFactor)))

	ctout := ckks.NewCiphertext(*lattigo.params, ct1.Degree(), ct1.Level())
	lattigo.eval.Mul(ct1, scaleup.Float64(), ctout)
	ctout.Scale = ct1.Scale.Mul(scaleup)

	return ctout

	// pt := ckks.NewPlaintext(*lattigo.params, ct1.Level())
	// pt.Scale = rlwe.NewScale(math.Pow(2, float64(upFactor)))
	// ones := make([]float64, lattigo.n)
	// for i := range ones {
	// 	ones[i] = 1
	// }
	// lattigo.ecd.Encode(ones, pt)
	// ct, _ := lattigo.eval.MulRelinNew(ct1, pt)
	// return ct
}

func (lattigo *LattigoFHE) evalRescale(ct1 *rlwe.Ciphertext) *rlwe.Ciphertext {
	ct := ckks.NewCiphertext(*lattigo.params, ct1.Degree(), ct1.Level()-1)
	lattigo.eval.Rescale(ct1, ct)
	return ct
}

func (lattigo *LattigoFHE) evalModswitch(ct1 *rlwe.Ciphertext, downFactor int) *rlwe.Ciphertext {
	ct := lattigo.eval.DropLevelNew(ct1, downFactor)
	return ct
}

func (lattigo *LattigoFHE) evalNegate(ct1 *rlwe.Ciphertext) *rlwe.Ciphertext {
	ct, _ := lattigo.eval.MulRelinNew(ct1, -1)
	return ct
}

func (lattigo *LattigoFHE) evalNegatePlain(pt1 []float64) []float64 {
	result := make([]float64, len(pt1))
	for i := range pt1 {
		result[i] = -pt1[i]
	}
	return result
}

func (lattigo *LattigoFHE) evalBootstrap(ct1 *rlwe.Ciphertext, targetLevel int) *rlwe.Ciphertext {
	ct_i := ct1.CopyNew()
	ct, err := lattigo.btpEval.Bootstrap(ct1)
	if err != nil {
		fmt.Println("initial Q: ", math.Log2(float64(lattigo.params.Q()[ct_i.Level()])))
		fmt.Println("initial Scale: ", math.Log2(ct_i.Scale.Float64()))
		fmt.Println("initial Q/Scale: ", float64(lattigo.params.Q()[ct_i.Level()])/ct_i.Scale.Float64())
		panic(fmt.Sprintf("Bootstrap failed: %v", err))
	}
	ct = lattigo.evalModswitch(ct, lattigo.maxLevel-(lattigo.bootstrapMaxLevel-targetLevel))
	return ct
}

func (lattigo *LattigoFHE) evalNullPlain(pt1 []float64) []float64 {
	result := make([]float64, len(pt1))
	for i := range pt1 {
		result[i] = pt1[i]
	}
	return result
}

func (lattigo *LattigoFHE) ensureEncoded(childID int) {
	if lattigo.terms[childID].Secret {
		if _, exists := lattigo.env[childID]; !exists {
			if lattigo.enableTiming {
				start := time.Now()
				lattigo.env[childID] = lattigo.encode(lattigo.ptEnv[childID], &lattigo.terms[childID].Scale, lattigo.terms[childID].Level)
				duration := time.Since(start)
				lattigo.recordTiming(lattigo.terms[childID].Op, lattigo.terms[childID].Level, duration)
			} else {
				lattigo.env[childID] = lattigo.encode(lattigo.ptEnv[childID], &lattigo.terms[childID].Scale, lattigo.terms[childID].Level)
			}
		}
	}
}

func (lattigo *LattigoFHE) evalOp(term *Term) *rlwe.Ciphertext {
	md := term.Metadata

	switch term.Op {
	case ADD, MUL:
		if !lattigo.terms[term.Children[0]].Secret && !lattigo.terms[term.Children[1]].Secret {
			return nil
		}
		lattigo.ensureEncoded(term.Children[0])
		lattigo.ensureEncoded(term.Children[1])
	case ROT, MODSWITCH, NEGATE, BOOTSTRAP, RESCALE, UPSCALE:
		if !lattigo.terms[term.Children[0]].Secret {
			return nil
		}
		lattigo.ensureEncoded(term.Children[0])
	case CONST:
		return nil
	}

	var result *rlwe.Ciphertext
	var start time.Time

	if lattigo.enableTiming {
		start = time.Now()
	}

	switch term.Op {
	/* case PACK:
		result = lattigo.encode(md.PackedValue, nil, lattigo.params.MaxLevel())
	case MASK:
		result = lattigo.encode(md.MaskedValue, nil, lattigo.params.MaxLevel()) */
	case ADD:
		a := term.Children[0]
		b := term.Children[1]
		if !lattigo.terms[a].Secret {
			result = lattigo.evalSingleAdd(lattigo.env[b], lattigo.ptEnv[a], lattigo.terms[a].Level, &lattigo.terms[a].Scale)
		} else if !lattigo.terms[b].Secret {
			result = lattigo.evalSingleAdd(lattigo.env[a], lattigo.ptEnv[b], lattigo.terms[b].Level, &lattigo.terms[b].Scale)
		} else {
			result = lattigo.evalDoubleAdd(lattigo.env[a], lattigo.env[b])
		}
	case MUL:
		a := term.Children[0]
		b := term.Children[1]
		if !lattigo.terms[a].Secret {
			result = lattigo.evalSingleMul(lattigo.env[b], lattigo.ptEnv[a], lattigo.terms[a].Level, &lattigo.terms[a].Scale)
		} else if !lattigo.terms[b].Secret {
			result = lattigo.evalSingleMul(lattigo.env[a], lattigo.ptEnv[b], lattigo.terms[b].Level, &lattigo.terms[b].Scale)
		} else {
			result = lattigo.evalDoubleMul(lattigo.env[a], lattigo.env[b])
		}
	case ROT:
		childLineNum := term.Children[0]
		offset := md.Offset

		// Check if hoisted rotation exists
		if hoistedMap, exists := lattigo.hoistedRots[childLineNum]; exists {
			if hoistedCt, hoistedExists := hoistedMap[offset]; hoistedExists && hoistedCt != nil {
				// Use existing hoisted rotation
				result = hoistedCt
			} else {
				// Compute hoisted rotations for this childLineNum
				lattigo.notdoHoisted(childLineNum)
				result = lattigo.hoistedRots[childLineNum][offset]
			}
		} else {
			panic(fmt.Sprintf("Hoisted rotation not found for childLineNum: %d, offset: %d", childLineNum, offset))
		}

		// Decrement rotation count
		lattigo.rotCount[childLineNum]--
		if lattigo.rotCount[childLineNum] <= 0 {
			// Remove from rotCount and hoistedRots when no longer needed
			delete(lattigo.rotCount, childLineNum)
			delete(lattigo.hoistedRots, childLineNum)
		}
	case MODSWITCH:
		result = lattigo.evalModswitch(lattigo.env[term.Children[0]], md.DownFactor)
	case NEGATE:
		result = lattigo.evalNegate(lattigo.env[term.Children[0]])
	case BOOTSTRAP:
		result = lattigo.evalBootstrap(lattigo.env[term.Children[0]], md.TargetLevel)
	case RESCALE:
		result = lattigo.evalRescale(lattigo.env[term.Children[0]])
	case UPSCALE:
		result = lattigo.evalUpscale(lattigo.env[term.Children[0]], md.UpFactor)
	default:
		fmt.Printf("Unknown op: %v\n", term.Op)
		return nil
	}

	if lattigo.enableTiming && result != nil {
		duration := time.Since(start)
		lattigo.recordTiming(term.Op, term.Level, duration)
	}

	return result
}

func (lattigo *LattigoFHE) evalOpPlain(term *Term) []float64 {
	md := term.Metadata

	switch term.Op {
	case CONST:
		return nil
	}

	var result []float64
	var start time.Time

	if lattigo.enableTiming {
		start = time.Now()
	}

	switch term.Op {
	/* case PACK:
		result = lattigo.encode(md.PackedValue, nil, lattigo.params.MaxLevel())
	case MASK:
		result = lattigo.encode(md.MaskedValue, nil, lattigo.params.MaxLevel()) */
	case ADD:
		a := term.Children[0]
		b := term.Children[1]
		result = lattigo.evalAddPlain(lattigo.ptEnv[a], lattigo.ptEnv[b])
	case MUL:
		a := term.Children[0]
		b := term.Children[1]
		result = lattigo.evalMulPlain(lattigo.ptEnv[a], lattigo.ptEnv[b])
	case ROT:
		childLineNum := term.Children[0]
		offset := md.Offset
		result = lattigo.evalRotPlain(lattigo.ptEnv[childLineNum], offset)
	case MODSWITCH:
		result = lattigo.evalNullPlain(lattigo.ptEnv[term.Children[0]])
	case NEGATE:
		result = lattigo.evalNegatePlain(lattigo.ptEnv[term.Children[0]])
	case BOOTSTRAP:
		result = lattigo.evalNullPlain(lattigo.ptEnv[term.Children[0]])
	case RESCALE:
		result = lattigo.evalNullPlain(lattigo.ptEnv[term.Children[0]])
	case UPSCALE:
		result = lattigo.evalNullPlain(lattigo.ptEnv[term.Children[0]])
	default:
		fmt.Printf("Unknown op: %v\n", term.Op)
		return nil
	}

	if lattigo.enableTiming && result != nil {
		duration := time.Since(start)
		lattigo.recordTiming(term.Op, term.Level, duration)
	}

	return result
}
