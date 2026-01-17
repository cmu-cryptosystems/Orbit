package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"time"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

type Context struct {
	params       *ckks.Parameters
	btpParams    *bootstrapping.Parameters
	ecd          *ckks.Encoder
	enc          *rlwe.Encryptor
	dec          *rlwe.Decryptor
	eval         *ckks.Evaluator
	btpEval      *bootstrapping.Evaluator
	defaultScale *rlwe.Scale

	btsMinLevel   int
	btsMaxLevel   int
	maxLevel      int
	polyDegree    int
	rescaleFactor int
}

type BenchmarkArgs struct {
	ctx     *Context
	Level   int
	ct1     *rlwe.Ciphertext
	ct2     *rlwe.Ciphertext
	pt1     []float64
	ptScale *rlwe.Scale
	eval    func(*BenchmarkArgs)
}

type Benchmark struct {
	OpName    string
	Level     int
	GoalRound int
	args      *BenchmarkArgs

	Times []time.Duration
}

func (b *Benchmark) AddDuration(duration time.Duration) {
	// if b.OpName == "earth.rotate_single" {
	// 	duration /= time.Duration(4)
	// }
	b.Times = append(b.Times, duration)
}
func (b *Benchmark) GetAvgTime() time.Duration {
	var sum time.Duration
	for _, t := range b.Times {
		sum += t
	}
	return sum / time.Duration(len(b.Times))
}

func (b *Benchmark) GetStdDev() time.Duration {
	var sum time.Duration
	for _, t := range b.Times {
		sum += t
	}

	avgTime := sum / time.Duration(len(b.Times))

	var sqDiffSum float64
	avgNano := float64(avgTime.Nanoseconds())
	for _, t := range b.Times {
		diff := float64(t.Nanoseconds()) - avgNano
		sqDiffSum += diff * diff
	}

	stdev := time.Duration(math.Sqrt(sqDiffSum / float64(len(b.Times))))
	return stdev
}

func getUniqueRots(n int) []int {
	rots := make([]int, 0)
	logN := int(math.Log2(float64(n)))
	for i := 0; i < logN-1; i++ {
		rots = append(rots, 1<<i)
		rots = append(rots, -1<<i)
	}
	return rots
}

func createContext(n int, maxLevel int, btsMinLevel int, btsMaxLevel int, Sf int) *Context {
	fmt.Println("Creating context...")
	logQ := append([]int{55}, make([]int, maxLevel)...)
	for i := 1; i <= maxLevel; i++ {
		logQ[i] = Sf
	}
	logN := int(math.Log2(float64(n * 2)))
	params, _ := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            logN,
		LogQ:            logQ,
		LogP:            []int{61, 61, 61},
		LogDefaultScale: Sf,
	})
	btpParams, _ := bootstrapping.NewParametersFromLiteral(params, bootstrapping.ParametersLiteral{
		LogN: utils.Pointy(logN),
		LogP: []int{61, 61, 61, 61},
	})
	defaultScale := rlwe.NewScale(math.Pow(2, float64(Sf)))

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)

	evk := rlwe.NewMemEvaluationKeySet(rlk)
	enc := rlwe.NewEncryptor(params, pk)
	ecd := ckks.NewEncoder(params)
	dec := rlwe.NewDecryptor(params, sk)
	eval := ckks.NewEvaluator(params, evk)

	fmt.Println("    Generating bootstrapping keys...")
	btpEvk, _, _ := btpParams.GenEvaluationKeys(sk)
	btpEval, _ := bootstrapping.NewEvaluator(btpParams, btpEvk)

	fmt.Println("    Generating rotation keys...")
	rots := getUniqueRots(n)
	galEls := make([]uint64, len(rots))
	for i, rot := range rots {
		galEls[i] = params.GaloisElement(rot)
	}

	new_eval := eval.WithKey(rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(galEls, sk)...))

	return &Context{
		params:        &params,
		btpParams:     &btpParams,
		ecd:           ecd,
		enc:           enc,
		dec:           dec,
		eval:          new_eval,
		btpEval:       btpEval,
		btsMinLevel:   btsMinLevel,
		btsMaxLevel:   btsMaxLevel,
		maxLevel:      maxLevel,
		polyDegree:    1 << logN,
		rescaleFactor: Sf,
		defaultScale:  &defaultScale,
	}
}

func eval_time(op func()) time.Duration {
	// runtime.GC()
	start := time.Now()
	op()
	elapsed := time.Since(start)
	return elapsed
}

func sample_pt(n int) []float64 {
	pt := make([]float64, n)
	for i := range pt {
		pt[i] = sampling.RandFloat64(-1, 1)
	}
	return pt
}
func sample_ct(ctx *Context, level int) *rlwe.Ciphertext {
	params := *ctx.params
	value := sample_pt(params.MaxSlots())
	pt := ckks.NewPlaintext(params, level)
	pt.Scale = *ctx.defaultScale
	ctx.ecd.Encode(value, pt)
	ct, _ := ctx.enc.EncryptNew(pt)
	return ct
}

func gen_all_benchmarks(ctx *Context, run_time int) []Benchmark {
	benchmarks := make([]Benchmark, 0)

	params := *ctx.params

	for level := params.MaxLevel(); level > 0; level-- {
		fmt.Println("Generating benchmarks for level", level)

		fmt.Println("    add_single")
		benchmarks = append(benchmarks, Benchmark{
			OpName:    "earth.add_single",
			Level:     level,
			GoalRound: run_time,
			args: &BenchmarkArgs{
				ctx:     ctx,
				Level:   level,
				ct1:     sample_ct(ctx, level),
				pt1:     sample_pt(params.MaxSlots()),
				ptScale: ctx.defaultScale,
				eval: func(args *BenchmarkArgs) {
					pt := ckks.NewPlaintext(*args.ctx.params, args.Level)
					pt.Scale = *args.ptScale
					args.ctx.ecd.Encode(args.pt1, pt)
					_, _ = args.ctx.eval.AddNew(args.ct1, pt)
				},
			},
		})

		fmt.Println("    add_double")
		benchmarks = append(benchmarks, Benchmark{
			OpName:    "earth.add_double",
			Level:     level,
			GoalRound: run_time,
			args: &BenchmarkArgs{
				ctx:   ctx,
				Level: level,
				ct1:   sample_ct(ctx, level),
				ct2:   sample_ct(ctx, level),
				eval: func(args *BenchmarkArgs) {
					_, _ = args.ctx.eval.AddNew(args.ct1, args.ct2)
				},
			},
		})

		fmt.Println("    mul_single")
		benchmarks = append(benchmarks, Benchmark{
			OpName:    "earth.mul_single",
			Level:     level,
			GoalRound: run_time,
			args: &BenchmarkArgs{
				ctx:     ctx,
				Level:   level,
				ct1:     sample_ct(ctx, level),
				pt1:     sample_pt(params.MaxSlots()),
				ptScale: ctx.defaultScale,
				eval: func(args *BenchmarkArgs) {
					pt := ckks.NewPlaintext(*args.ctx.params, args.Level)
					pt.Scale = *args.ptScale
					args.ctx.ecd.Encode(args.pt1, pt)
					_, _ = args.ctx.eval.MulRelinNew(args.ct1, pt)
				},
			},
		})

		fmt.Println("    mul_double")
		benchmarks = append(benchmarks, Benchmark{
			OpName:    "earth.mul_double",
			Level:     level,
			GoalRound: run_time,
			args: &BenchmarkArgs{
				ctx:   ctx,
				Level: level,
				ct1:   sample_ct(ctx, level),
				ct2:   sample_ct(ctx, level),
				eval: func(args *BenchmarkArgs) {
					_, _ = args.ctx.eval.MulRelinNew(args.ct1, args.ct2)
				},
			},
		})

		fmt.Println("    negate_single")
		benchmarks = append(benchmarks, Benchmark{
			OpName:    "earth.negate_single",
			Level:     level,
			GoalRound: run_time,
			args: &BenchmarkArgs{
				ctx:   ctx,
				Level: level,
				ct1:   sample_ct(ctx, level),
				eval: func(args *BenchmarkArgs) {
					_, _ = args.ctx.eval.MulRelinNew(args.ct1, -1)
				},
			},
		})

		fmt.Println("    rotate_single")
		benchmarks = append(benchmarks, Benchmark{
			OpName:    "earth.rotate_single",
			Level:     level,
			GoalRound: run_time,
			args: &BenchmarkArgs{
				ctx:   ctx,
				Level: level,
				ct1:   sample_ct(ctx, level),
				eval: func(args *BenchmarkArgs) {
					_, _ = args.ctx.eval.RotateNew(args.ct1, 4)
				},
			},
		})

		if level <= params.MaxLevel()-1 {
			fmt.Println("    rescale_single")
			benchmarks = append(benchmarks, Benchmark{
				OpName:    "earth.rescale_single",
				Level:     level,
				GoalRound: run_time,
				args: &BenchmarkArgs{
					ctx:   ctx,
					Level: level,
					ct1:   sample_ct(ctx, level+1),
					eval: func(args *BenchmarkArgs) {
						ctout := ckks.NewCiphertext(*args.ctx.params, args.ct1.Degree(), args.ct1.Level()-1)
						args.ctx.eval.Rescale(args.ct1, ctout)
					},
				},
			})
		}

		fmt.Println("    modswitch_single")
		benchmarks = append(benchmarks, Benchmark{
			OpName:    "earth.modswitch_single",
			Level:     level,
			GoalRound: run_time,
			args: &BenchmarkArgs{
				ctx:   ctx,
				Level: level,
				ct1:   sample_ct(ctx, level),
				eval: func(args *BenchmarkArgs) {
					args.ctx.eval.DropLevelNew(args.ct1, 1)
				},
			},
		})

		fmt.Println("    upscale_single")
		benchmarks = append(benchmarks, Benchmark{
			OpName:    "earth.upscale_single",
			Level:     level,
			GoalRound: run_time,
			args: &BenchmarkArgs{
				ctx:   ctx,
				Level: level,
				ct1:   sample_ct(ctx, level),
				eval: func(args *BenchmarkArgs) {
					scaleup := rlwe.NewScale(math.Pow(2, float64(args.ctx.rescaleFactor)))
					ctout := ckks.NewCiphertext(*args.ctx.params, args.ct1.Degree(), args.ct1.Level())
					args.ctx.eval.Mul(args.ct1, scaleup.Float64(), ctout)
					ctout.Scale = args.ct1.Scale.Mul(scaleup)
				},
			},
		})

		if level > ctx.btsMinLevel && level <= ctx.btsMaxLevel {
			fmt.Println("    bootstrap_single")
			benchmarks = append(benchmarks, Benchmark{
				OpName:    "earth.bootstrap_single",
				Level:     level,
				GoalRound: run_time / 10,
				args: &BenchmarkArgs{
					ctx:   ctx,
					Level: level,
					ct1:   sample_ct(ctx, ctx.btsMinLevel),
					eval: func(args *BenchmarkArgs) {
						ct_i := args.ct1.CopyNew()
						ct, _ := args.ctx.btpEval.Bootstrap(ct_i)
						args.ctx.eval.DropLevel(ct, args.ctx.maxLevel-args.Level)
					},
				},
			})
		}
		runtime.GC()
	}
	return benchmarks
}

func simulate_load(ctx *Context, num int, plnum int) ([]*rlwe.Ciphertext, [][]float64) {
	runtime.GC()
	// Simulate some load to make sure that the system is not idling during the benchmarks
	fmt.Println("Simulating load...")
	var cts []*rlwe.Ciphertext
	var pls [][]float64
	for i := 0; i < num; i++ {
		cts = append(cts, sample_ct(ctx, ctx.maxLevel))
		if i%100 == 0 {
			fmt.Println("    Simulated", i, "ciphertexts")
			runtime.GC()
		}
	}
	for i := 0; i < plnum; i++ {
		pl := make([]float64, ctx.params.MaxSlots())
		for j := range pl {
			pl[j] = sampling.RandFloat64(-1, 1)
		}
		pls = append(pls, pl)
	}
	runtime.GC()
	fmt.Println("    Simulated load with", num, "ciphertexts and", plnum, "unencoded plaintexts")
	return cts, pls
}
func consume_load(ctx *Context, cts []*rlwe.Ciphertext, pls [][]float64) {
	fmt.Println("Consuming load...")
	rots := getUniqueRots(ctx.params.MaxSlots())
	_, _ = ctx.eval.RotateHoistedNew(cts[0], rots)
	for i := range cts {
		pt := ckks.NewPlaintext(*ctx.params, ctx.maxLevel)
		ctx.dec.Decrypt(cts[i], pt)
	}
	for i := range pls {
		_ = pls[i]
	}
	fmt.Println("    Consumed load with", len(cts), "ciphertexts and", len(pls), "unencoded plaintexts")
	runtime.GC()
}

func eval_all_benchmarks(benchmarks []Benchmark) {

	sim_cts, sim_pls := simulate_load(benchmarks[0].args.ctx, 800, 0)

	maxRound := 0
	for i := range benchmarks {
		if benchmarks[i].GoalRound > maxRound {
			maxRound = benchmarks[i].GoalRound
		}
	}

	for round := maxRound - 1; round >= 0; round-- {
		if round%5 == 0 {
			runtime.GC()
		}
		fmt.Println("Starting round", maxRound-round, "of", maxRound)
		round_start_time := time.Now()
		for i := range benchmarks {
			if round < benchmarks[i].GoalRound {
				// if round%(maxRound/benchmarks[i].GoalRound) == 0 {
				duration := eval_time(func() { benchmarks[i].args.eval(benchmarks[i].args) })
				benchmarks[i].AddDuration(duration)
				fmt.Printf("    Finished %-30s at level %3d (%3d/%3d)  AvgTime %.2f us,  Stdev %.2f us\n",
					benchmarks[i].OpName, benchmarks[i].Level, len(benchmarks[i].Times), benchmarks[i].GoalRound,
					float64(benchmarks[i].GetAvgTime().Nanoseconds()/1000.0),
					float64(benchmarks[i].GetStdDev().Nanoseconds()/1000.0))
			}
		}
		round_elapsed := time.Since(round_start_time)
		fmt.Printf("Round %d done in %s\n", maxRound-round, round_elapsed.String())
	}

	consume_load(benchmarks[0].args.ctx, sim_cts, sim_pls)
}

func gen_cost_model_json(ctx *Context, benchmarks []Benchmark, n int, btsLb int, btsUb int) {
	latencyTableRaw := make(map[string]map[int]float64)
	stdevTableRaw := make(map[string]map[int]float64)
	latencyTable := make(map[string][]float64)
	stdevTable := make(map[string][]float64)
	for _, bench := range benchmarks {
		if _, exists := latencyTableRaw[bench.OpName]; !exists {
			latencyTableRaw[bench.OpName] = make(map[int]float64)
			stdevTableRaw[bench.OpName] = make(map[int]float64)
		}
		latencyTableRaw[bench.OpName][bench.Level] = float64(bench.GetAvgTime().Nanoseconds() / 1000.0)
		stdevTableRaw[bench.OpName][bench.Level] = float64(bench.GetStdDev().Nanoseconds() / 1000.0)
	}
	for opName, levelMap := range latencyTableRaw {
		latencyTable[opName] = make([]float64, 0)
		stdevTable[opName] = make([]float64, 0)
		this_max_level := 0
		for level := range levelMap {
			if level > this_max_level {
				this_max_level = level
			}
		}
		for level := 1; level <= this_max_level; level++ {
			if val, exists := levelMap[level]; exists {
				latencyTable[opName] = append(latencyTable[opName], val)
				stdevTable[opName] = append(stdevTable[opName], stdevTableRaw[opName][level])
			} else {
				latencyTable[opName] = append(latencyTable[opName], 0)
				stdevTable[opName] = append(stdevTable[opName], 0)
			}
		}
	}

	config := map[string]interface{}{
		"bootstrapLevelLowerBound": ctx.btsMinLevel,
		"bootstrapLevelUpperBound": ctx.btsMaxLevel,
		"latencyTable":             latencyTable,
		"levelLowerBound":          1,
		"levelUpperBound":          ctx.maxLevel,
		"polynomialDegree":         ctx.polyDegree,
		"rescalingFactor":          ctx.rescaleFactor,
		"runtime":                  "Lattigo",
		"stdevTable":               stdevTable,
	}

	jsonFile, err := os.Create(fmt.Sprintf("profiled_LATTIGONEW_CPU%dk_%d_%d.json", n, btsLb, btsUb))
	if err != nil {
		panic(err)
	}
	defer jsonFile.Close()

	encoder := json.NewEncoder(jsonFile)
	encoder.SetIndent("", "    ")
	if err := encoder.Encode(config); err != nil {
		panic(err)
	}

	fmt.Println("Cost model generated successfully.")
}

func main() {
	var n int
	var btsLb int
	var btsUb int

	flag.IntVar(&n, "n", 64, "The polynomial modulus degree")
	flag.IntVar(&btsLb, "btsLb", 3, "The bootstrap level lower bound")
	flag.IntVar(&btsUb, "btsUb", 16, "The bootstrap level upper bound")
	flag.Parse()

	pow2_of_n := n * 1024

	ctx := createContext(pow2_of_n, btsUb, btsLb, btsUb, 51)
	benchmarks := gen_all_benchmarks(ctx, 100)
	eval_all_benchmarks(benchmarks)
	gen_cost_model_json(ctx, benchmarks, n, btsLb, btsUb)
}
