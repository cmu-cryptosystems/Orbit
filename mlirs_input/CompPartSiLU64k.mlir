"builtin.module"() <{sym_name = "traced/CompPartSiLU64k.mlir"}> ({
  "func.func"() <{function_type = (tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>, sym_name = "_hecate_CompPartSiLU64k"}> ({
  ^bb0(%arg0: tensor<1x!earth.ci<0 * 0>> loc("/home/ubuntu/dacaporion/dacapo/examples/benchmarks/CompPartSiLU64k.py":39:0)):
    %0 = "earth.rotate"(%arg0) <{offset = array<i64: -33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %1 = "earth.rotate"(%arg0) <{offset = array<i64: -32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %2 = "earth.rotate"(%arg0) <{offset = array<i64: -31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %3 = "earth.rotate"(%arg0) <{offset = array<i64: -1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %4 = "earth.rotate"(%arg0) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %5 = "earth.rotate"(%arg0) <{offset = array<i64: 1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %6 = "earth.rotate"(%arg0) <{offset = array<i64: 31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %7 = "earth.rotate"(%arg0) <{offset = array<i64: 32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %8 = "earth.rotate"(%arg0) <{offset = array<i64: 33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %9 = "earth.constant"() <{rms_var = 0.21502784075237871 : f64, value = 0 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %10 = "earth.mul"(%0, %9) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %11 = "earth.constant"() <{rms_var = 0.29697466770158382 : f64, value = 1 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %12 = "earth.mul"(%1, %11) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %13 = "earth.add"(%10, %12) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %14 = "earth.constant"() <{rms_var = 0.2110101313789351 : f64, value = 2 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %15 = "earth.mul"(%2, %14) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %16 = "earth.add"(%13, %15) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %17 = "earth.constant"() <{rms_var = 0.31354217076661867 : f64, value = 3 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %18 = "earth.mul"(%3, %17) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %19 = "earth.add"(%16, %18) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %20 = "earth.constant"() <{rms_var = 0.56763220321587471 : f64, value = 4 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %21 = "earth.mul"(%4, %20) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %22 = "earth.add"(%19, %21) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %23 = "earth.constant"() <{rms_var = 0.36113259470213294 : f64, value = 5 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %24 = "earth.mul"(%5, %23) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %25 = "earth.add"(%22, %24) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %26 = "earth.constant"() <{rms_var = 0.2075197548584761 : f64, value = 6 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %27 = "earth.mul"(%6, %26) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %28 = "earth.add"(%25, %27) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %29 = "earth.constant"() <{rms_var = 0.31459453726516612 : f64, value = 7 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %30 = "earth.mul"(%7, %29) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %31 = "earth.add"(%28, %30) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %32 = "earth.constant"() <{rms_var = 0.19783543037357637 : f64, value = 8 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %33 = "earth.mul"(%8, %32) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %34 = "earth.add"(%31, %33) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %35 = "earth.rotate"(%34) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %36 = "earth.add"(%34, %35) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %37 = "earth.rotate"(%34) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %38 = "earth.add"(%36, %37) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %39 = "earth.rotate"(%38) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %40 = "earth.constant"() <{rms_var = 0.078300157290233222 : f64, value = 9 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %41 = "earth.mul"(%39, %40) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %42 = "earth.rotate"(%38) <{offset = array<i64: 3072>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %43 = "earth.constant"() <{rms_var = 0.047382078608036476 : f64, value = 10 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %44 = "earth.mul"(%42, %43) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %45 = "earth.add"(%41, %44) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %46 = "earth.rotate"(%38) <{offset = array<i64: 6144>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %47 = "earth.constant"() <{rms_var = 0.041945195295435618 : f64, value = 11 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %48 = "earth.mul"(%46, %47) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %49 = "earth.add"(%45, %48) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %50 = "earth.rotate"(%38) <{offset = array<i64: 9216>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %51 = "earth.constant"() <{rms_var = 0.030831893360619277 : f64, value = 12 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %52 = "earth.mul"(%50, %51) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %53 = "earth.add"(%49, %52) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %54 = "earth.rotate"(%38) <{offset = array<i64: 12288>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %55 = "earth.constant"() <{rms_var = 0.051812405636793707 : f64, value = 13 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %56 = "earth.mul"(%54, %55) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %57 = "earth.add"(%53, %56) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %58 = "earth.rotate"(%38) <{offset = array<i64: 15360>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %59 = "earth.constant"() <{rms_var = 0.044693416533488468 : f64, value = 14 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %60 = "earth.mul"(%58, %59) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %61 = "earth.add"(%57, %60) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %62 = "earth.rotate"(%38) <{offset = array<i64: 18432>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %63 = "earth.constant"() <{rms_var = 0.027800927584845578 : f64, value = 15 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %64 = "earth.mul"(%62, %63) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %65 = "earth.add"(%61, %64) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %66 = "earth.rotate"(%38) <{offset = array<i64: 21504>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %67 = "earth.constant"() <{rms_var = 0.056827530937879317 : f64, value = 16 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %68 = "earth.mul"(%66, %67) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %69 = "earth.add"(%65, %68) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %70 = "earth.rotate"(%38) <{offset = array<i64: 24576>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %71 = "earth.constant"() <{rms_var = 0.048481919368764191 : f64, value = 17 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %72 = "earth.mul"(%70, %71) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %73 = "earth.add"(%69, %72) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %74 = "earth.rotate"(%38) <{offset = array<i64: 27648>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %75 = "earth.constant"() <{rms_var = 0.043442944118082513 : f64, value = 18 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %76 = "earth.mul"(%74, %75) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %77 = "earth.add"(%73, %76) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %78 = "earth.rotate"(%38) <{offset = array<i64: 30720>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %79 = "earth.constant"() <{rms_var = 0.042358848962660997 : f64, value = 19 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %80 = "earth.mul"(%78, %79) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %81 = "earth.add"(%77, %80) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %82 = "earth.rotate"(%38) <{offset = array<i64: 33792>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %83 = "earth.constant"() <{rms_var = 0.048272564606738864 : f64, value = 20 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %84 = "earth.mul"(%82, %83) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %85 = "earth.add"(%81, %84) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %86 = "earth.rotate"(%38) <{offset = array<i64: 36864>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %87 = "earth.constant"() <{rms_var = 0.05489142138545456 : f64, value = 21 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %88 = "earth.mul"(%86, %87) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %89 = "earth.add"(%85, %88) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %90 = "earth.rotate"(%38) <{offset = array<i64: 39936>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %91 = "earth.constant"() <{rms_var = 0.012773214082388179 : f64, value = 22 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %92 = "earth.mul"(%90, %91) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %93 = "earth.add"(%89, %92) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %94 = "earth.rotate"(%38) <{offset = array<i64: 43008>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %95 = "earth.constant"() <{rms_var = 0.080126044793653344 : f64, value = 23 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %96 = "earth.mul"(%94, %95) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %97 = "earth.add"(%93, %96) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %98 = "earth.rotate"(%38) <{offset = array<i64: 46080>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %99 = "earth.constant"() <{rms_var = 0.059325468080234228 : f64, value = 24 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %100 = "earth.mul"(%98, %99) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %101 = "earth.add"(%97, %100) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %102 = "earth.rotate"(%101) <{offset = array<i64: -16384>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %103 = "earth.add"(%101, %102) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %104 = "earth.rotate"(%103) <{offset = array<i64: -32768>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %105 = "earth.add"(%103, %104) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %106 = "earth.constant"() <{rms_var = 0.017340254745366906 : f64, value = 25 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %107 = "earth.add"(%105, %106) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]convbn1-0
    %108 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %109 = "earth.mul"(%108, %107) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %110 = "earth.mul"(%109, %107) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %111 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %112 = "earth.add"(%110, %111) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %113 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %114 = "earth.mul"(%113, %112) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %115 = "earth.mul"(%114, %112) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %116 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %117 = "earth.add"(%115, %116) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %118 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %119 = "earth.mul"(%118, %117) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %120 = "earth.mul"(%119, %117) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %121 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %122 = "earth.add"(%120, %121) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %123 = "earth.mul"(%109, %112) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %124 = "earth.negate"(%107) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %125 = "earth.add"(%123, %124) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %126 = "earth.mul"(%109, %117) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %127 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %128 = "earth.mul"(%127, %125) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %129 = "earth.mul"(%128, %117) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %130 = "earth.negate"(%125) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %131 = "earth.add"(%126, %130) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %132 = "earth.add"(%129, %124) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %133 = "earth.mul"(%109, %122) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %134 = "earth.mul"(%128, %122) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %135 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %136 = "earth.mul"(%135, %131) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %137 = "earth.mul"(%136, %122) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %138 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %139 = "earth.mul"(%138, %132) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %140 = "earth.mul"(%139, %122) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %141 = "earth.negate"(%132) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %142 = "earth.add"(%133, %141) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %143 = "earth.negate"(%131) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %144 = "earth.add"(%134, %143) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %145 = "earth.add"(%137, %130) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %146 = "earth.add"(%140, %124) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %147 = "earth.constant"() <{rms_var = 0.051379788302527769 : f64, value = 28 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %148 = "earth.mul"(%147, %107) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %149 = "earth.constant"() <{rms_var = 0.04272842452341858 : f64, value = 29 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %150 = "earth.mul"(%149, %125) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %151 = "earth.add"(%148, %150) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %152 = "earth.constant"() <{rms_var = 0.036049430000943392 : f64, value = 30 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %153 = "earth.mul"(%152, %131) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %154 = "earth.add"(%151, %153) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %155 = "earth.constant"() <{rms_var = 0.030937245302699062 : f64, value = 31 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %156 = "earth.mul"(%155, %132) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %157 = "earth.add"(%154, %156) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %158 = "earth.constant"() <{rms_var = 0.027115258485962086 : f64, value = 32 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %159 = "earth.mul"(%158, %142) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %160 = "earth.add"(%157, %159) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %161 = "earth.constant"() <{rms_var = 0.024393077542268389 : f64, value = 33 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %162 = "earth.mul"(%161, %144) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %163 = "earth.add"(%160, %162) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %164 = "earth.constant"() <{rms_var = 0.022642601074970584 : f64, value = 34 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %165 = "earth.mul"(%164, %145) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %166 = "earth.add"(%163, %165) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %167 = "earth.constant"() <{rms_var = 0.021831260875010087 : f64, value = 35 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %168 = "earth.mul"(%167, %146) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %169 = "earth.add"(%166, %168) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %170 = "earth.constant"() <{rms_var = 0.021652089836507502 : f64, value = 36 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %171 = "earth.mul"(%170, %107) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %172 = "earth.constant"() <{rms_var = 0.018138222134916806 : f64, value = 37 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %173 = "earth.mul"(%172, %125) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %174 = "earth.add"(%171, %173) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %175 = "earth.constant"() <{rms_var = 0.01542303411540253 : f64, value = 38 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %176 = "earth.mul"(%175, %131) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %177 = "earth.add"(%174, %176) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %178 = "earth.constant"() <{rms_var = 0.013305654693862069 : f64, value = 39 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %179 = "earth.mul"(%178, %132) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %180 = "earth.add"(%177, %179) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %181 = "earth.constant"() <{rms_var = 0.011703046220094507 : f64, value = 40 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %182 = "earth.mul"(%181, %142) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %183 = "earth.add"(%180, %182) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %184 = "earth.constant"() <{rms_var = 0.010552642814751455 : f64, value = 41 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %185 = "earth.mul"(%184, %144) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %186 = "earth.add"(%183, %185) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %187 = "earth.constant"() <{rms_var = 0.0098096659804816355 : f64, value = 42 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %188 = "earth.mul"(%187, %145) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %189 = "earth.add"(%186, %188) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %190 = "earth.constant"() <{rms_var = 0.0094452495559895089 : f64, value = 43 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %191 = "earth.mul"(%190, %146) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %192 = "earth.add"(%189, %191) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %193 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %194 = "earth.mul"(%193, %122) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %195 = "earth.mul"(%194, %122) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %196 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %197 = "earth.add"(%195, %196) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %198 = "earth.mul"(%192, %197) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %199 = "earth.add"(%198, %169) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %200 = "earth.constant"() <{rms_var = 0.0042995278292336028 : f64, value = 44 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %201 = "earth.mul"(%200, %107) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %202 = "earth.constant"() <{rms_var = 0.0036191919476548503 : f64, value = 45 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %203 = "earth.mul"(%202, %125) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %204 = "earth.add"(%201, %203) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %205 = "earth.constant"() <{rms_var = 0.0030784419617177587 : f64, value = 46 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %206 = "earth.mul"(%205, %131) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %207 = "earth.add"(%204, %206) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %208 = "earth.constant"() <{rms_var = 0.0026564062051904268 : f64, value = 47 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %209 = "earth.mul"(%208, %132) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %210 = "earth.add"(%207, %209) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %211 = "earth.constant"() <{rms_var = 0.0023368004457586414 : f64, value = 48 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %212 = "earth.mul"(%211, %142) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %213 = "earth.add"(%210, %212) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %214 = "earth.constant"() <{rms_var = 0.002107295415524122 : f64, value = 49 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %215 = "earth.mul"(%214, %144) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %216 = "earth.add"(%213, %215) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %217 = "earth.constant"() <{rms_var = 0.0019590388891824136 : f64, value = 50 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %218 = "earth.mul"(%217, %145) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %219 = "earth.add"(%216, %218) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %220 = "earth.constant"() <{rms_var = 0.0018863129851277056 : f64, value = 51 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %221 = "earth.mul"(%220, %146) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %222 = "earth.add"(%219, %221) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %223 = "earth.constant"() <{rms_var = 0.63615477792235875 : f64, value = 52 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %224 = "earth.mul"(%223, %107) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %225 = "earth.constant"() <{rms_var = 0.21189795368271022 : f64, value = 53 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %226 = "earth.mul"(%225, %125) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %227 = "earth.add"(%224, %226) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %228 = "earth.constant"() <{rms_var = 0.12734215758896472 : f64, value = 54 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %229 = "earth.mul"(%228, %131) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %230 = "earth.add"(%227, %229) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %231 = "earth.constant"() <{rms_var = 0.091626347972833505 : f64, value = 55 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %232 = "earth.mul"(%231, %132) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %233 = "earth.add"(%230, %232) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %234 = "earth.constant"() <{rms_var = 0.07255941147658998 : f64, value = 56 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %235 = "earth.mul"(%234, %142) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %236 = "earth.add"(%233, %235) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %237 = "earth.constant"() <{rms_var = 0.061477909207391525 : f64, value = 57 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %238 = "earth.mul"(%237, %144) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %239 = "earth.add"(%236, %238) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %240 = "earth.constant"() <{rms_var = 0.055169030070493508 : f64, value = 58 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %241 = "earth.mul"(%240, %145) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %242 = "earth.add"(%239, %241) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %243 = "earth.constant"() <{rms_var = 0.052275954916496656 : f64, value = 59 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %244 = "earth.mul"(%243, %146) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %245 = "earth.add"(%242, %244) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %246 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %247 = "earth.mul"(%246, %197) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %248 = "earth.mul"(%247, %197) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %249 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %250 = "earth.add"(%248, %249) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %251 = "earth.mul"(%222, %250) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %252 = "earth.add"(%251, %199) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %253 = "earth.constant"() <{rms_var = 4.9481895575581374E-4 : f64, value = 60 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %254 = "earth.mul"(%253, %107) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %255 = "earth.constant"() <{rms_var = 3.771067298313324E-4 : f64, value = 61 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %256 = "earth.mul"(%255, %125) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %257 = "earth.add"(%254, %256) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %258 = "earth.constant"() <{rms_var = 3.2076765303085136E-4 : f64, value = 62 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %259 = "earth.mul"(%258, %131) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %260 = "earth.add"(%257, %259) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %261 = "earth.constant"() <{rms_var = 2.7679532400224407E-4 : f64, value = 63 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %262 = "earth.mul"(%261, %132) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %263 = "earth.add"(%260, %262) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %264 = "earth.constant"() <{rms_var = 2.4349440690240933E-4 : f64, value = 64 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %265 = "earth.mul"(%264, %142) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %266 = "earth.add"(%263, %265) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %267 = "earth.constant"() <{rms_var = 2.1958101095650637E-4 : f64, value = 65 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %268 = "earth.mul"(%267, %144) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %269 = "earth.add"(%266, %268) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %270 = "earth.constant"() <{rms_var = 2.0413318001830463E-4 : f64, value = 66 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %271 = "earth.mul"(%270, %145) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %272 = "earth.add"(%269, %271) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %273 = "earth.constant"() <{rms_var = 1.9655534146569248E-4 : f64, value = 67 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %274 = "earth.mul"(%273, %146) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %275 = "earth.add"(%272, %274) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %276 = "earth.constant"() <{rms_var = 1.7735088548126505E-4 : f64, value = 68 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %277 = "earth.mul"(%276, %107) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %278 = "earth.constant"() <{rms_var = 1.457794164045128E-4 : f64, value = 69 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %279 = "earth.mul"(%278, %125) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %280 = "earth.add"(%277, %279) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %281 = "earth.constant"() <{rms_var = 1.1982820692495149E-4 : f64, value = 70 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %282 = "earth.mul"(%281, %131) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %283 = "earth.add"(%280, %282) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %284 = "earth.constant"() <{rms_var = 9.8496754159654756E-5 : f64, value = 71 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %285 = "earth.mul"(%284, %132) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %286 = "earth.add"(%283, %285) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %287 = "earth.constant"() <{rms_var = 8.0962662339626343E-5 : f64, value = 72 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %288 = "earth.mul"(%287, %142) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %289 = "earth.add"(%286, %288) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %290 = "earth.constant"() <{rms_var = 6.6549936439435966E-5 : f64, value = 73 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %291 = "earth.mul"(%290, %144) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %292 = "earth.add"(%289, %291) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %293 = "earth.constant"() <{rms_var = 5.4702920205367267E-5 : f64, value = 74 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %294 = "earth.mul"(%293, %145) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %295 = "earth.add"(%292, %294) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %296 = "earth.constant"() <{rms_var = 1.3863334887734771E-4 : f64, value = 75 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %297 = "earth.mul"(%296, %146) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %298 = "earth.add"(%295, %297) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %299 = "earth.mul"(%252, %197) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %300 = "earth.add"(%299, %245) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %301 = "earth.mul"(%298, %197) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %302 = "earth.add"(%301, %275) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %303 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %304 = "earth.mul"(%303, %250) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %305 = "earth.mul"(%304, %250) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %306 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %307 = "earth.add"(%305, %306) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %308 = "earth.mul"(%302, %307) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %309 = "earth.add"(%308, %300) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]act1_SiLU_poly
    %310 = "earth.constant"() <{rms_var = 5.000000e-01 : f64, value = 76 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // add[]act1_SiLU_add
    %311 = "earth.add"(%309, %310) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // add[]act1_SiLU_add
    %312 = "earth.mul"(%107, %311) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // mul[]act1_SiLU_mul
    %313 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 77 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // mul[]act1_SiLU_mul
    %314 = "earth.add"(%312, %313) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // mul[]act1_SiLU_mul
    %315 = "earth.rotate"(%314) <{offset = array<i64: -33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %316 = "earth.rotate"(%314) <{offset = array<i64: -32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %317 = "earth.rotate"(%314) <{offset = array<i64: -31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %318 = "earth.rotate"(%314) <{offset = array<i64: -1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %319 = "earth.rotate"(%314) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %320 = "earth.rotate"(%314) <{offset = array<i64: 1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %321 = "earth.rotate"(%314) <{offset = array<i64: 31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %322 = "earth.rotate"(%314) <{offset = array<i64: 32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %323 = "earth.rotate"(%314) <{offset = array<i64: 33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %324 = "earth.constant"() <{rms_var = 0.096025948135613062 : f64, value = 78 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %325 = "earth.mul"(%315, %324) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %326 = "earth.constant"() <{rms_var = 0.11164571811099519 : f64, value = 79 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %327 = "earth.mul"(%316, %326) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %328 = "earth.add"(%325, %327) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %329 = "earth.constant"() <{rms_var = 0.13017961240207196 : f64, value = 80 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %330 = "earth.mul"(%317, %329) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %331 = "earth.add"(%328, %330) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %332 = "earth.constant"() <{rms_var = 0.16329299179200105 : f64, value = 81 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %333 = "earth.mul"(%318, %332) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %334 = "earth.add"(%331, %333) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %335 = "earth.constant"() <{rms_var = 0.17161395650663383 : f64, value = 82 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %336 = "earth.mul"(%319, %335) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %337 = "earth.add"(%334, %336) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %338 = "earth.constant"() <{rms_var = 0.14327661978876549 : f64, value = 83 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %339 = "earth.mul"(%320, %338) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %340 = "earth.add"(%337, %339) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %341 = "earth.constant"() <{rms_var = 0.12473295548036838 : f64, value = 84 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %342 = "earth.mul"(%321, %341) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %343 = "earth.add"(%340, %342) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %344 = "earth.constant"() <{rms_var = 0.12216228874268155 : f64, value = 85 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %345 = "earth.mul"(%322, %344) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %346 = "earth.add"(%343, %345) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %347 = "earth.constant"() <{rms_var = 0.11762778832214148 : f64, value = 86 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %348 = "earth.mul"(%323, %347) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %349 = "earth.add"(%346, %348) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %350 = "earth.rotate"(%349) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %351 = "earth.add"(%349, %350) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %352 = "earth.rotate"(%351) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %353 = "earth.add"(%351, %352) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %354 = "earth.rotate"(%353) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %355 = "earth.add"(%353, %354) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %356 = "earth.rotate"(%355) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %357 = "earth.add"(%355, %356) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %358 = "earth.rotate"(%357) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %359 = "earth.constant"() <{rms_var = 0.087056217584534953 : f64, value = 87 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %360 = "earth.mul"(%358, %359) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %361 = "earth.rotate"(%357) <{offset = array<i64: 15360>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %362 = "earth.constant"() <{rms_var = 0.12502985663775679 : f64, value = 88 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %363 = "earth.mul"(%361, %362) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %364 = "earth.add"(%360, %363) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %365 = "earth.rotate"(%357) <{offset = array<i64: 30720>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %366 = "earth.constant"() <{rms_var = 0.096192100162823937 : f64, value = 89 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %367 = "earth.mul"(%365, %366) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %368 = "earth.add"(%364, %367) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %369 = "earth.rotate"(%357) <{offset = array<i64: 46080>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %370 = "earth.constant"() <{rms_var = 0.10804870438693016 : f64, value = 90 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %371 = "earth.mul"(%369, %370) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %372 = "earth.add"(%368, %371) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %373 = "earth.constant"() <{rms_var = 0.11069324581445816 : f64, value = 91 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %374 = "earth.mul"(%315, %373) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %375 = "earth.constant"() <{rms_var = 0.10212811710108596 : f64, value = 92 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %376 = "earth.mul"(%316, %375) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %377 = "earth.add"(%374, %376) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %378 = "earth.constant"() <{rms_var = 0.12515967247344598 : f64, value = 93 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %379 = "earth.mul"(%317, %378) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %380 = "earth.add"(%377, %379) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %381 = "earth.constant"() <{rms_var = 0.14403503206128876 : f64, value = 94 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %382 = "earth.mul"(%318, %381) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %383 = "earth.add"(%380, %382) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %384 = "earth.constant"() <{rms_var = 0.1667235282544858 : f64, value = 95 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %385 = "earth.mul"(%319, %384) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %386 = "earth.add"(%383, %385) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %387 = "earth.constant"() <{rms_var = 0.17100829475650517 : f64, value = 96 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %388 = "earth.mul"(%320, %387) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %389 = "earth.add"(%386, %388) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %390 = "earth.constant"() <{rms_var = 0.13467979907864649 : f64, value = 97 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %391 = "earth.mul"(%321, %390) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %392 = "earth.add"(%389, %391) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %393 = "earth.constant"() <{rms_var = 0.15241337917045406 : f64, value = 98 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %394 = "earth.mul"(%322, %393) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %395 = "earth.add"(%392, %394) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %396 = "earth.constant"() <{rms_var = 0.14871851405931008 : f64, value = 99 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %397 = "earth.mul"(%323, %396) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %398 = "earth.add"(%395, %397) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %399 = "earth.rotate"(%398) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %400 = "earth.add"(%398, %399) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %401 = "earth.rotate"(%400) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %402 = "earth.add"(%400, %401) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %403 = "earth.rotate"(%402) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %404 = "earth.add"(%402, %403) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %405 = "earth.rotate"(%404) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %406 = "earth.add"(%404, %405) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %407 = "earth.rotate"(%406) <{offset = array<i64: -4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %408 = "earth.constant"() <{rms_var = 0.077111966010965025 : f64, value = 100 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %409 = "earth.mul"(%407, %408) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %410 = "earth.add"(%372, %409) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %411 = "earth.rotate"(%406) <{offset = array<i64: 11264>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %412 = "earth.constant"() <{rms_var = 0.083031619072431326 : f64, value = 101 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %413 = "earth.mul"(%411, %412) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %414 = "earth.add"(%410, %413) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %415 = "earth.rotate"(%406) <{offset = array<i64: 26624>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %416 = "earth.constant"() <{rms_var = 0.10179747430070245 : f64, value = 102 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %417 = "earth.mul"(%415, %416) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %418 = "earth.add"(%414, %417) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %419 = "earth.rotate"(%406) <{offset = array<i64: 41984>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %420 = "earth.constant"() <{rms_var = 0.094465346892113236 : f64, value = 103 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %421 = "earth.mul"(%419, %420) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %422 = "earth.add"(%418, %421) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %423 = "earth.constant"() <{rms_var = 0.15024445145871693 : f64, value = 104 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %424 = "earth.mul"(%315, %423) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %425 = "earth.constant"() <{rms_var = 0.14453231007460829 : f64, value = 105 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %426 = "earth.mul"(%316, %425) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %427 = "earth.add"(%424, %426) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %428 = "earth.constant"() <{rms_var = 0.12849137584132345 : f64, value = 106 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %429 = "earth.mul"(%317, %428) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %430 = "earth.add"(%427, %429) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %431 = "earth.constant"() <{rms_var = 0.14749882672058187 : f64, value = 107 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %432 = "earth.mul"(%318, %431) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %433 = "earth.add"(%430, %432) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %434 = "earth.constant"() <{rms_var = 0.19246860855744236 : f64, value = 108 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %435 = "earth.mul"(%319, %434) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %436 = "earth.add"(%433, %435) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %437 = "earth.constant"() <{rms_var = 0.13722066607129441 : f64, value = 109 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %438 = "earth.mul"(%320, %437) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %439 = "earth.add"(%436, %438) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %440 = "earth.constant"() <{rms_var = 0.12879954968209698 : f64, value = 110 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %441 = "earth.mul"(%321, %440) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %442 = "earth.add"(%439, %441) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %443 = "earth.constant"() <{rms_var = 0.17954346371926341 : f64, value = 111 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %444 = "earth.mul"(%322, %443) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %445 = "earth.add"(%442, %444) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %446 = "earth.constant"() <{rms_var = 0.15146363844901589 : f64, value = 112 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %447 = "earth.mul"(%323, %446) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %448 = "earth.add"(%445, %447) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %449 = "earth.rotate"(%448) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %450 = "earth.add"(%448, %449) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %451 = "earth.rotate"(%450) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %452 = "earth.add"(%450, %451) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %453 = "earth.rotate"(%452) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %454 = "earth.add"(%452, %453) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %455 = "earth.rotate"(%454) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %456 = "earth.add"(%454, %455) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %457 = "earth.rotate"(%456) <{offset = array<i64: -8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %458 = "earth.constant"() <{rms_var = 0.075950720919777073 : f64, value = 113 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %459 = "earth.mul"(%457, %458) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %460 = "earth.add"(%422, %459) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %461 = "earth.rotate"(%456) <{offset = array<i64: 7168>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %462 = "earth.constant"() <{rms_var = 0.056143433178170839 : f64, value = 114 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %463 = "earth.mul"(%461, %462) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %464 = "earth.add"(%460, %463) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %465 = "earth.rotate"(%456) <{offset = array<i64: 22528>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %466 = "earth.constant"() <{rms_var = 0.080830505320278037 : f64, value = 115 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %467 = "earth.mul"(%465, %466) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %468 = "earth.add"(%464, %467) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %469 = "earth.rotate"(%456) <{offset = array<i64: 37888>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %470 = "earth.constant"() <{rms_var = 0.075817629919315718 : f64, value = 116 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %471 = "earth.mul"(%469, %470) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %472 = "earth.add"(%468, %471) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %473 = "earth.constant"() <{rms_var = 0.16712501424218545 : f64, value = 117 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %474 = "earth.mul"(%315, %473) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %475 = "earth.constant"() <{rms_var = 0.18306558946978971 : f64, value = 118 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %476 = "earth.mul"(%316, %475) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %477 = "earth.add"(%474, %476) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %478 = "earth.constant"() <{rms_var = 0.15007764827437847 : f64, value = 119 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %479 = "earth.mul"(%317, %478) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %480 = "earth.add"(%477, %479) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %481 = "earth.constant"() <{rms_var = 0.20876102955069348 : f64, value = 120 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %482 = "earth.mul"(%318, %481) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %483 = "earth.add"(%480, %482) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %484 = "earth.constant"() <{rms_var = 0.21195263510998669 : f64, value = 121 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %485 = "earth.mul"(%319, %484) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %486 = "earth.add"(%483, %485) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %487 = "earth.constant"() <{rms_var = 0.1668770265076803 : f64, value = 122 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %488 = "earth.mul"(%320, %487) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %489 = "earth.add"(%486, %488) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %490 = "earth.constant"() <{rms_var = 0.14513823006078308 : f64, value = 123 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %491 = "earth.mul"(%321, %490) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %492 = "earth.add"(%489, %491) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %493 = "earth.constant"() <{rms_var = 0.13183801037256285 : f64, value = 124 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %494 = "earth.mul"(%322, %493) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %495 = "earth.add"(%492, %494) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %496 = "earth.constant"() <{rms_var = 0.15085163994978376 : f64, value = 125 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %497 = "earth.mul"(%323, %496) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %498 = "earth.add"(%495, %497) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %499 = "earth.rotate"(%498) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %500 = "earth.add"(%498, %499) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %501 = "earth.rotate"(%500) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %502 = "earth.add"(%500, %501) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %503 = "earth.rotate"(%502) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %504 = "earth.add"(%502, %503) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %505 = "earth.rotate"(%504) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %506 = "earth.add"(%504, %505) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %507 = "earth.rotate"(%506) <{offset = array<i64: -12288>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %508 = "earth.constant"() <{rms_var = 0.08598052716615745 : f64, value = 126 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %509 = "earth.mul"(%507, %508) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %510 = "earth.add"(%472, %509) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %511 = "earth.rotate"(%506) <{offset = array<i64: 3072>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %512 = "earth.constant"() <{rms_var = 0.082683080257550639 : f64, value = 127 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %513 = "earth.mul"(%511, %512) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %514 = "earth.add"(%510, %513) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %515 = "earth.rotate"(%506) <{offset = array<i64: 18432>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %516 = "earth.constant"() <{rms_var = 0.092891974580739542 : f64, value = 128 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %517 = "earth.mul"(%515, %516) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %518 = "earth.add"(%514, %517) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %519 = "earth.rotate"(%506) <{offset = array<i64: 33792>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %520 = "earth.constant"() <{rms_var = 0.0677699642845844 : f64, value = 129 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %521 = "earth.mul"(%519, %520) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %522 = "earth.add"(%518, %521) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %523 = "earth.rotate"(%522) <{offset = array<i64: -16384>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %524 = "earth.add"(%522, %523) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %525 = "earth.rotate"(%524) <{offset = array<i64: -32768>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %526 = "earth.add"(%524, %525) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %527 = "earth.constant"() <{rms_var = 0.035226288337298316 : f64, value = 130 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %528 = "earth.add"(%526, %527) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %529 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 77 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %530 = "earth.add"(%314, %529) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-0
    %531 = "earth.rotate"(%530) <{offset = array<i64: -33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %532 = "earth.rotate"(%530) <{offset = array<i64: -32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %533 = "earth.rotate"(%530) <{offset = array<i64: -31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %534 = "earth.rotate"(%530) <{offset = array<i64: -1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %535 = "earth.rotate"(%530) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %536 = "earth.rotate"(%530) <{offset = array<i64: 1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %537 = "earth.rotate"(%530) <{offset = array<i64: 31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %538 = "earth.rotate"(%530) <{offset = array<i64: 32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %539 = "earth.rotate"(%530) <{offset = array<i64: 33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %540 = "earth.constant"() <{rms_var = 0.096025948135613062 : f64, value = 78 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %541 = "earth.mul"(%531, %540) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %542 = "earth.constant"() <{rms_var = 0.11164571811099519 : f64, value = 79 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %543 = "earth.mul"(%532, %542) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %544 = "earth.add"(%541, %543) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %545 = "earth.constant"() <{rms_var = 0.13017961240207196 : f64, value = 80 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %546 = "earth.mul"(%533, %545) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %547 = "earth.add"(%544, %546) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %548 = "earth.constant"() <{rms_var = 0.16329299179200105 : f64, value = 81 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %549 = "earth.mul"(%534, %548) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %550 = "earth.add"(%547, %549) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %551 = "earth.constant"() <{rms_var = 0.17161395650663383 : f64, value = 82 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %552 = "earth.mul"(%535, %551) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %553 = "earth.add"(%550, %552) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %554 = "earth.constant"() <{rms_var = 0.14327661978876549 : f64, value = 83 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %555 = "earth.mul"(%536, %554) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %556 = "earth.add"(%553, %555) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %557 = "earth.constant"() <{rms_var = 0.12473295548036838 : f64, value = 84 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %558 = "earth.mul"(%537, %557) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %559 = "earth.add"(%556, %558) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %560 = "earth.constant"() <{rms_var = 0.12216228874268155 : f64, value = 85 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %561 = "earth.mul"(%538, %560) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %562 = "earth.add"(%559, %561) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %563 = "earth.constant"() <{rms_var = 0.11762778832214148 : f64, value = 86 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %564 = "earth.mul"(%539, %563) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %565 = "earth.add"(%562, %564) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %566 = "earth.rotate"(%565) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %567 = "earth.add"(%565, %566) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %568 = "earth.rotate"(%567) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %569 = "earth.add"(%567, %568) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %570 = "earth.rotate"(%569) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %571 = "earth.add"(%569, %570) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %572 = "earth.rotate"(%571) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %573 = "earth.add"(%571, %572) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %574 = "earth.rotate"(%573) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %575 = "earth.constant"() <{rms_var = 0.087056217584534953 : f64, value = 87 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %576 = "earth.mul"(%574, %575) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %577 = "earth.rotate"(%573) <{offset = array<i64: 15360>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %578 = "earth.constant"() <{rms_var = 0.12502985663775679 : f64, value = 88 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %579 = "earth.mul"(%577, %578) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %580 = "earth.add"(%576, %579) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %581 = "earth.rotate"(%573) <{offset = array<i64: 30720>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %582 = "earth.constant"() <{rms_var = 0.096192100162823937 : f64, value = 89 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %583 = "earth.mul"(%581, %582) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %584 = "earth.add"(%580, %583) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %585 = "earth.rotate"(%573) <{offset = array<i64: 46080>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %586 = "earth.constant"() <{rms_var = 0.10804870438693016 : f64, value = 90 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %587 = "earth.mul"(%585, %586) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %588 = "earth.add"(%584, %587) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %589 = "earth.constant"() <{rms_var = 0.11069324581445816 : f64, value = 91 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %590 = "earth.mul"(%531, %589) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %591 = "earth.constant"() <{rms_var = 0.10212811710108596 : f64, value = 92 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %592 = "earth.mul"(%532, %591) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %593 = "earth.add"(%590, %592) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %594 = "earth.constant"() <{rms_var = 0.12515967247344598 : f64, value = 93 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %595 = "earth.mul"(%533, %594) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %596 = "earth.add"(%593, %595) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %597 = "earth.constant"() <{rms_var = 0.14403503206128876 : f64, value = 94 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %598 = "earth.mul"(%534, %597) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %599 = "earth.add"(%596, %598) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %600 = "earth.constant"() <{rms_var = 0.1667235282544858 : f64, value = 95 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %601 = "earth.mul"(%535, %600) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %602 = "earth.add"(%599, %601) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %603 = "earth.constant"() <{rms_var = 0.17100829475650517 : f64, value = 96 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %604 = "earth.mul"(%536, %603) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %605 = "earth.add"(%602, %604) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %606 = "earth.constant"() <{rms_var = 0.13467979907864649 : f64, value = 97 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %607 = "earth.mul"(%537, %606) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %608 = "earth.add"(%605, %607) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %609 = "earth.constant"() <{rms_var = 0.15241337917045406 : f64, value = 98 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %610 = "earth.mul"(%538, %609) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %611 = "earth.add"(%608, %610) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %612 = "earth.constant"() <{rms_var = 0.14871851405931008 : f64, value = 99 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %613 = "earth.mul"(%539, %612) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %614 = "earth.add"(%611, %613) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %615 = "earth.rotate"(%614) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %616 = "earth.add"(%614, %615) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %617 = "earth.rotate"(%616) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %618 = "earth.add"(%616, %617) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %619 = "earth.rotate"(%618) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %620 = "earth.add"(%618, %619) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %621 = "earth.rotate"(%620) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %622 = "earth.add"(%620, %621) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %623 = "earth.rotate"(%622) <{offset = array<i64: -4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %624 = "earth.constant"() <{rms_var = 0.077111966010965025 : f64, value = 100 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %625 = "earth.mul"(%623, %624) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %626 = "earth.add"(%588, %625) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %627 = "earth.rotate"(%622) <{offset = array<i64: 11264>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %628 = "earth.constant"() <{rms_var = 0.083031619072431326 : f64, value = 101 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %629 = "earth.mul"(%627, %628) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %630 = "earth.add"(%626, %629) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %631 = "earth.rotate"(%622) <{offset = array<i64: 26624>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %632 = "earth.constant"() <{rms_var = 0.10179747430070245 : f64, value = 102 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %633 = "earth.mul"(%631, %632) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %634 = "earth.add"(%630, %633) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %635 = "earth.rotate"(%622) <{offset = array<i64: 41984>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %636 = "earth.constant"() <{rms_var = 0.094465346892113236 : f64, value = 103 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %637 = "earth.mul"(%635, %636) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %638 = "earth.add"(%634, %637) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %639 = "earth.constant"() <{rms_var = 0.15024445145871693 : f64, value = 104 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %640 = "earth.mul"(%531, %639) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %641 = "earth.constant"() <{rms_var = 0.14453231007460829 : f64, value = 105 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %642 = "earth.mul"(%532, %641) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %643 = "earth.add"(%640, %642) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %644 = "earth.constant"() <{rms_var = 0.12849137584132345 : f64, value = 106 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %645 = "earth.mul"(%533, %644) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %646 = "earth.add"(%643, %645) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %647 = "earth.constant"() <{rms_var = 0.14749882672058187 : f64, value = 107 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %648 = "earth.mul"(%534, %647) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %649 = "earth.add"(%646, %648) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %650 = "earth.constant"() <{rms_var = 0.19246860855744236 : f64, value = 108 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %651 = "earth.mul"(%535, %650) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %652 = "earth.add"(%649, %651) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %653 = "earth.constant"() <{rms_var = 0.13722066607129441 : f64, value = 109 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %654 = "earth.mul"(%536, %653) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %655 = "earth.add"(%652, %654) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %656 = "earth.constant"() <{rms_var = 0.12879954968209698 : f64, value = 110 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %657 = "earth.mul"(%537, %656) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %658 = "earth.add"(%655, %657) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %659 = "earth.constant"() <{rms_var = 0.17954346371926341 : f64, value = 111 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %660 = "earth.mul"(%538, %659) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %661 = "earth.add"(%658, %660) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %662 = "earth.constant"() <{rms_var = 0.15146363844901589 : f64, value = 112 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %663 = "earth.mul"(%539, %662) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %664 = "earth.add"(%661, %663) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %665 = "earth.rotate"(%664) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %666 = "earth.add"(%664, %665) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %667 = "earth.rotate"(%666) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %668 = "earth.add"(%666, %667) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %669 = "earth.rotate"(%668) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %670 = "earth.add"(%668, %669) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %671 = "earth.rotate"(%670) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %672 = "earth.add"(%670, %671) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %673 = "earth.rotate"(%672) <{offset = array<i64: -8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %674 = "earth.constant"() <{rms_var = 0.075950720919777073 : f64, value = 113 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %675 = "earth.mul"(%673, %674) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %676 = "earth.add"(%638, %675) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %677 = "earth.rotate"(%672) <{offset = array<i64: 7168>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %678 = "earth.constant"() <{rms_var = 0.056143433178170839 : f64, value = 114 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %679 = "earth.mul"(%677, %678) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %680 = "earth.add"(%676, %679) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %681 = "earth.rotate"(%672) <{offset = array<i64: 22528>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %682 = "earth.constant"() <{rms_var = 0.080830505320278037 : f64, value = 115 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %683 = "earth.mul"(%681, %682) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %684 = "earth.add"(%680, %683) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %685 = "earth.rotate"(%672) <{offset = array<i64: 37888>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %686 = "earth.constant"() <{rms_var = 0.075817629919315718 : f64, value = 116 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %687 = "earth.mul"(%685, %686) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %688 = "earth.add"(%684, %687) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %689 = "earth.constant"() <{rms_var = 0.16712501424218545 : f64, value = 117 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %690 = "earth.mul"(%531, %689) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %691 = "earth.constant"() <{rms_var = 0.18306558946978971 : f64, value = 118 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %692 = "earth.mul"(%532, %691) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %693 = "earth.add"(%690, %692) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %694 = "earth.constant"() <{rms_var = 0.15007764827437847 : f64, value = 119 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %695 = "earth.mul"(%533, %694) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %696 = "earth.add"(%693, %695) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %697 = "earth.constant"() <{rms_var = 0.20876102955069348 : f64, value = 120 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %698 = "earth.mul"(%534, %697) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %699 = "earth.add"(%696, %698) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %700 = "earth.constant"() <{rms_var = 0.21195263510998669 : f64, value = 121 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %701 = "earth.mul"(%535, %700) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %702 = "earth.add"(%699, %701) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %703 = "earth.constant"() <{rms_var = 0.1668770265076803 : f64, value = 122 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %704 = "earth.mul"(%536, %703) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %705 = "earth.add"(%702, %704) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %706 = "earth.constant"() <{rms_var = 0.14513823006078308 : f64, value = 123 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %707 = "earth.mul"(%537, %706) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %708 = "earth.add"(%705, %707) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %709 = "earth.constant"() <{rms_var = 0.13183801037256285 : f64, value = 124 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %710 = "earth.mul"(%538, %709) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %711 = "earth.add"(%708, %710) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %712 = "earth.constant"() <{rms_var = 0.15085163994978376 : f64, value = 125 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %713 = "earth.mul"(%539, %712) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %714 = "earth.add"(%711, %713) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %715 = "earth.rotate"(%714) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %716 = "earth.add"(%714, %715) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %717 = "earth.rotate"(%716) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %718 = "earth.add"(%716, %717) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %719 = "earth.rotate"(%718) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %720 = "earth.add"(%718, %719) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %721 = "earth.rotate"(%720) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %722 = "earth.add"(%720, %721) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %723 = "earth.rotate"(%722) <{offset = array<i64: -12288>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %724 = "earth.constant"() <{rms_var = 0.08598052716615745 : f64, value = 126 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %725 = "earth.mul"(%723, %724) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %726 = "earth.add"(%688, %725) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %727 = "earth.rotate"(%722) <{offset = array<i64: 3072>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %728 = "earth.constant"() <{rms_var = 0.082683080257550639 : f64, value = 127 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %729 = "earth.mul"(%727, %728) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %730 = "earth.add"(%726, %729) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %731 = "earth.rotate"(%722) <{offset = array<i64: 18432>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %732 = "earth.constant"() <{rms_var = 0.092891974580739542 : f64, value = 128 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %733 = "earth.mul"(%731, %732) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %734 = "earth.add"(%730, %733) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %735 = "earth.rotate"(%722) <{offset = array<i64: 33792>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %736 = "earth.constant"() <{rms_var = 0.0677699642845844 : f64, value = 129 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %737 = "earth.mul"(%735, %736) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %738 = "earth.add"(%734, %737) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %739 = "earth.rotate"(%738) <{offset = array<i64: -16384>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %740 = "earth.add"(%738, %739) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %741 = "earth.rotate"(%740) <{offset = array<i64: -32768>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %742 = "earth.add"(%740, %741) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %743 = "earth.constant"() <{rms_var = 0.035226288337298316 : f64, value = 130 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %744 = "earth.add"(%742, %743) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %745 = "earth.add"(%528, %744) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %746 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %747 = "earth.add"(%312, %746) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %748 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 77 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %749 = "earth.add"(%747, %748) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-1
    %750 = "earth.rotate"(%749) <{offset = array<i64: -33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %751 = "earth.rotate"(%749) <{offset = array<i64: -32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %752 = "earth.rotate"(%749) <{offset = array<i64: -31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %753 = "earth.rotate"(%749) <{offset = array<i64: -1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %754 = "earth.rotate"(%749) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %755 = "earth.rotate"(%749) <{offset = array<i64: 1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %756 = "earth.rotate"(%749) <{offset = array<i64: 31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %757 = "earth.rotate"(%749) <{offset = array<i64: 32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %758 = "earth.rotate"(%749) <{offset = array<i64: 33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %759 = "earth.constant"() <{rms_var = 0.096025948135613062 : f64, value = 78 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %760 = "earth.mul"(%750, %759) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %761 = "earth.constant"() <{rms_var = 0.11164571811099519 : f64, value = 79 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %762 = "earth.mul"(%751, %761) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %763 = "earth.add"(%760, %762) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %764 = "earth.constant"() <{rms_var = 0.13017961240207196 : f64, value = 80 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %765 = "earth.mul"(%752, %764) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %766 = "earth.add"(%763, %765) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %767 = "earth.constant"() <{rms_var = 0.16329299179200105 : f64, value = 81 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %768 = "earth.mul"(%753, %767) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %769 = "earth.add"(%766, %768) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %770 = "earth.constant"() <{rms_var = 0.17161395650663383 : f64, value = 82 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %771 = "earth.mul"(%754, %770) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %772 = "earth.add"(%769, %771) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %773 = "earth.constant"() <{rms_var = 0.14327661978876549 : f64, value = 83 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %774 = "earth.mul"(%755, %773) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %775 = "earth.add"(%772, %774) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %776 = "earth.constant"() <{rms_var = 0.12473295548036838 : f64, value = 84 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %777 = "earth.mul"(%756, %776) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %778 = "earth.add"(%775, %777) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %779 = "earth.constant"() <{rms_var = 0.12216228874268155 : f64, value = 85 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %780 = "earth.mul"(%757, %779) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %781 = "earth.add"(%778, %780) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %782 = "earth.constant"() <{rms_var = 0.11762778832214148 : f64, value = 86 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %783 = "earth.mul"(%758, %782) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %784 = "earth.add"(%781, %783) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %785 = "earth.rotate"(%784) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %786 = "earth.add"(%784, %785) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %787 = "earth.rotate"(%786) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %788 = "earth.add"(%786, %787) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %789 = "earth.rotate"(%788) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %790 = "earth.add"(%788, %789) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %791 = "earth.rotate"(%790) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %792 = "earth.add"(%790, %791) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %793 = "earth.rotate"(%792) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %794 = "earth.constant"() <{rms_var = 0.087056217584534953 : f64, value = 87 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %795 = "earth.mul"(%793, %794) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %796 = "earth.rotate"(%792) <{offset = array<i64: 15360>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %797 = "earth.constant"() <{rms_var = 0.12502985663775679 : f64, value = 88 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %798 = "earth.mul"(%796, %797) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %799 = "earth.add"(%795, %798) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %800 = "earth.rotate"(%792) <{offset = array<i64: 30720>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %801 = "earth.constant"() <{rms_var = 0.096192100162823937 : f64, value = 89 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %802 = "earth.mul"(%800, %801) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %803 = "earth.add"(%799, %802) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %804 = "earth.rotate"(%792) <{offset = array<i64: 46080>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %805 = "earth.constant"() <{rms_var = 0.10804870438693016 : f64, value = 90 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %806 = "earth.mul"(%804, %805) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %807 = "earth.add"(%803, %806) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %808 = "earth.constant"() <{rms_var = 0.11069324581445816 : f64, value = 91 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %809 = "earth.mul"(%750, %808) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %810 = "earth.constant"() <{rms_var = 0.10212811710108596 : f64, value = 92 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %811 = "earth.mul"(%751, %810) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %812 = "earth.add"(%809, %811) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %813 = "earth.constant"() <{rms_var = 0.12515967247344598 : f64, value = 93 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %814 = "earth.mul"(%752, %813) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %815 = "earth.add"(%812, %814) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %816 = "earth.constant"() <{rms_var = 0.14403503206128876 : f64, value = 94 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %817 = "earth.mul"(%753, %816) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %818 = "earth.add"(%815, %817) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %819 = "earth.constant"() <{rms_var = 0.1667235282544858 : f64, value = 95 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %820 = "earth.mul"(%754, %819) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %821 = "earth.add"(%818, %820) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %822 = "earth.constant"() <{rms_var = 0.17100829475650517 : f64, value = 96 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %823 = "earth.mul"(%755, %822) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %824 = "earth.add"(%821, %823) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %825 = "earth.constant"() <{rms_var = 0.13467979907864649 : f64, value = 97 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %826 = "earth.mul"(%756, %825) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %827 = "earth.add"(%824, %826) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %828 = "earth.constant"() <{rms_var = 0.15241337917045406 : f64, value = 98 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %829 = "earth.mul"(%757, %828) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %830 = "earth.add"(%827, %829) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %831 = "earth.constant"() <{rms_var = 0.14871851405931008 : f64, value = 99 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %832 = "earth.mul"(%758, %831) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %833 = "earth.add"(%830, %832) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %834 = "earth.rotate"(%833) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %835 = "earth.add"(%833, %834) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %836 = "earth.rotate"(%835) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %837 = "earth.add"(%835, %836) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %838 = "earth.rotate"(%837) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %839 = "earth.add"(%837, %838) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %840 = "earth.rotate"(%839) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %841 = "earth.add"(%839, %840) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %842 = "earth.rotate"(%841) <{offset = array<i64: -4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %843 = "earth.constant"() <{rms_var = 0.077111966010965025 : f64, value = 100 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %844 = "earth.mul"(%842, %843) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %845 = "earth.add"(%807, %844) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %846 = "earth.rotate"(%841) <{offset = array<i64: 11264>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %847 = "earth.constant"() <{rms_var = 0.083031619072431326 : f64, value = 101 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %848 = "earth.mul"(%846, %847) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %849 = "earth.add"(%845, %848) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %850 = "earth.rotate"(%841) <{offset = array<i64: 26624>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %851 = "earth.constant"() <{rms_var = 0.10179747430070245 : f64, value = 102 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %852 = "earth.mul"(%850, %851) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %853 = "earth.add"(%849, %852) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %854 = "earth.rotate"(%841) <{offset = array<i64: 41984>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %855 = "earth.constant"() <{rms_var = 0.094465346892113236 : f64, value = 103 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %856 = "earth.mul"(%854, %855) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %857 = "earth.add"(%853, %856) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %858 = "earth.constant"() <{rms_var = 0.15024445145871693 : f64, value = 104 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %859 = "earth.mul"(%750, %858) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %860 = "earth.constant"() <{rms_var = 0.14453231007460829 : f64, value = 105 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %861 = "earth.mul"(%751, %860) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %862 = "earth.add"(%859, %861) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %863 = "earth.constant"() <{rms_var = 0.12849137584132345 : f64, value = 106 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %864 = "earth.mul"(%752, %863) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %865 = "earth.add"(%862, %864) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %866 = "earth.constant"() <{rms_var = 0.14749882672058187 : f64, value = 107 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %867 = "earth.mul"(%753, %866) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %868 = "earth.add"(%865, %867) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %869 = "earth.constant"() <{rms_var = 0.19246860855744236 : f64, value = 108 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %870 = "earth.mul"(%754, %869) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %871 = "earth.add"(%868, %870) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %872 = "earth.constant"() <{rms_var = 0.13722066607129441 : f64, value = 109 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %873 = "earth.mul"(%755, %872) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %874 = "earth.add"(%871, %873) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %875 = "earth.constant"() <{rms_var = 0.12879954968209698 : f64, value = 110 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %876 = "earth.mul"(%756, %875) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %877 = "earth.add"(%874, %876) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %878 = "earth.constant"() <{rms_var = 0.17954346371926341 : f64, value = 111 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %879 = "earth.mul"(%757, %878) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %880 = "earth.add"(%877, %879) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %881 = "earth.constant"() <{rms_var = 0.15146363844901589 : f64, value = 112 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %882 = "earth.mul"(%758, %881) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %883 = "earth.add"(%880, %882) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %884 = "earth.rotate"(%883) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %885 = "earth.add"(%883, %884) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %886 = "earth.rotate"(%885) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %887 = "earth.add"(%885, %886) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %888 = "earth.rotate"(%887) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %889 = "earth.add"(%887, %888) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %890 = "earth.rotate"(%889) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %891 = "earth.add"(%889, %890) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %892 = "earth.rotate"(%891) <{offset = array<i64: -8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %893 = "earth.constant"() <{rms_var = 0.075950720919777073 : f64, value = 113 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %894 = "earth.mul"(%892, %893) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %895 = "earth.add"(%857, %894) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %896 = "earth.rotate"(%891) <{offset = array<i64: 7168>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %897 = "earth.constant"() <{rms_var = 0.056143433178170839 : f64, value = 114 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %898 = "earth.mul"(%896, %897) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %899 = "earth.add"(%895, %898) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %900 = "earth.rotate"(%891) <{offset = array<i64: 22528>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %901 = "earth.constant"() <{rms_var = 0.080830505320278037 : f64, value = 115 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %902 = "earth.mul"(%900, %901) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %903 = "earth.add"(%899, %902) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %904 = "earth.rotate"(%891) <{offset = array<i64: 37888>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %905 = "earth.constant"() <{rms_var = 0.075817629919315718 : f64, value = 116 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %906 = "earth.mul"(%904, %905) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %907 = "earth.add"(%903, %906) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %908 = "earth.constant"() <{rms_var = 0.16712501424218545 : f64, value = 117 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %909 = "earth.mul"(%750, %908) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %910 = "earth.constant"() <{rms_var = 0.18306558946978971 : f64, value = 118 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %911 = "earth.mul"(%751, %910) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %912 = "earth.add"(%909, %911) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %913 = "earth.constant"() <{rms_var = 0.15007764827437847 : f64, value = 119 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %914 = "earth.mul"(%752, %913) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %915 = "earth.add"(%912, %914) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %916 = "earth.constant"() <{rms_var = 0.20876102955069348 : f64, value = 120 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %917 = "earth.mul"(%753, %916) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %918 = "earth.add"(%915, %917) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %919 = "earth.constant"() <{rms_var = 0.21195263510998669 : f64, value = 121 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %920 = "earth.mul"(%754, %919) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %921 = "earth.add"(%918, %920) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %922 = "earth.constant"() <{rms_var = 0.1668770265076803 : f64, value = 122 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %923 = "earth.mul"(%755, %922) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %924 = "earth.add"(%921, %923) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %925 = "earth.constant"() <{rms_var = 0.14513823006078308 : f64, value = 123 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %926 = "earth.mul"(%756, %925) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %927 = "earth.add"(%924, %926) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %928 = "earth.constant"() <{rms_var = 0.13183801037256285 : f64, value = 124 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %929 = "earth.mul"(%757, %928) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %930 = "earth.add"(%927, %929) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %931 = "earth.constant"() <{rms_var = 0.15085163994978376 : f64, value = 125 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %932 = "earth.mul"(%758, %931) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %933 = "earth.add"(%930, %932) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %934 = "earth.rotate"(%933) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %935 = "earth.add"(%933, %934) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %936 = "earth.rotate"(%935) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %937 = "earth.add"(%935, %936) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %938 = "earth.rotate"(%937) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %939 = "earth.add"(%937, %938) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %940 = "earth.rotate"(%939) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %941 = "earth.add"(%939, %940) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %942 = "earth.rotate"(%941) <{offset = array<i64: -12288>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %943 = "earth.constant"() <{rms_var = 0.08598052716615745 : f64, value = 126 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %944 = "earth.mul"(%942, %943) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %945 = "earth.add"(%907, %944) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %946 = "earth.rotate"(%941) <{offset = array<i64: 3072>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %947 = "earth.constant"() <{rms_var = 0.082683080257550639 : f64, value = 127 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %948 = "earth.mul"(%946, %947) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %949 = "earth.add"(%945, %948) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %950 = "earth.rotate"(%941) <{offset = array<i64: 18432>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %951 = "earth.constant"() <{rms_var = 0.092891974580739542 : f64, value = 128 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %952 = "earth.mul"(%950, %951) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %953 = "earth.add"(%949, %952) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %954 = "earth.rotate"(%941) <{offset = array<i64: 33792>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %955 = "earth.constant"() <{rms_var = 0.0677699642845844 : f64, value = 129 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %956 = "earth.mul"(%954, %955) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %957 = "earth.add"(%953, %956) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %958 = "earth.rotate"(%957) <{offset = array<i64: -16384>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %959 = "earth.add"(%957, %958) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %960 = "earth.rotate"(%959) <{offset = array<i64: -32768>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %961 = "earth.add"(%959, %960) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %962 = "earth.constant"() <{rms_var = 0.035226288337298316 : f64, value = 130 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %963 = "earth.add"(%961, %962) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %964 = "earth.add"(%745, %963) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %965 = "earth.constant"() <{rms_var = 3.000000e+00 : f64, value = 131 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %966 = "earth.add"(%312, %965) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %967 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 77 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %968 = "earth.add"(%966, %967) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-2
    %969 = "earth.rotate"(%968) <{offset = array<i64: -33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %970 = "earth.rotate"(%968) <{offset = array<i64: -32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %971 = "earth.rotate"(%968) <{offset = array<i64: -31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %972 = "earth.rotate"(%968) <{offset = array<i64: -1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %973 = "earth.rotate"(%968) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %974 = "earth.rotate"(%968) <{offset = array<i64: 1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %975 = "earth.rotate"(%968) <{offset = array<i64: 31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %976 = "earth.rotate"(%968) <{offset = array<i64: 32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %977 = "earth.rotate"(%968) <{offset = array<i64: 33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %978 = "earth.constant"() <{rms_var = 0.096025948135613062 : f64, value = 78 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %979 = "earth.mul"(%969, %978) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %980 = "earth.constant"() <{rms_var = 0.11164571811099519 : f64, value = 79 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %981 = "earth.mul"(%970, %980) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %982 = "earth.add"(%979, %981) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %983 = "earth.constant"() <{rms_var = 0.13017961240207196 : f64, value = 80 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %984 = "earth.mul"(%971, %983) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %985 = "earth.add"(%982, %984) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %986 = "earth.constant"() <{rms_var = 0.16329299179200105 : f64, value = 81 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %987 = "earth.mul"(%972, %986) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %988 = "earth.add"(%985, %987) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %989 = "earth.constant"() <{rms_var = 0.17161395650663383 : f64, value = 82 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %990 = "earth.mul"(%973, %989) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %991 = "earth.add"(%988, %990) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %992 = "earth.constant"() <{rms_var = 0.14327661978876549 : f64, value = 83 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %993 = "earth.mul"(%974, %992) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %994 = "earth.add"(%991, %993) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %995 = "earth.constant"() <{rms_var = 0.12473295548036838 : f64, value = 84 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %996 = "earth.mul"(%975, %995) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %997 = "earth.add"(%994, %996) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %998 = "earth.constant"() <{rms_var = 0.12216228874268155 : f64, value = 85 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %999 = "earth.mul"(%976, %998) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1000 = "earth.add"(%997, %999) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1001 = "earth.constant"() <{rms_var = 0.11762778832214148 : f64, value = 86 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1002 = "earth.mul"(%977, %1001) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1003 = "earth.add"(%1000, %1002) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1004 = "earth.rotate"(%1003) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1005 = "earth.add"(%1003, %1004) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1006 = "earth.rotate"(%1005) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1007 = "earth.add"(%1005, %1006) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1008 = "earth.rotate"(%1007) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1009 = "earth.add"(%1007, %1008) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1010 = "earth.rotate"(%1009) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1011 = "earth.add"(%1009, %1010) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1012 = "earth.rotate"(%1011) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1013 = "earth.constant"() <{rms_var = 0.087056217584534953 : f64, value = 87 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1014 = "earth.mul"(%1012, %1013) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1015 = "earth.rotate"(%1011) <{offset = array<i64: 15360>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1016 = "earth.constant"() <{rms_var = 0.12502985663775679 : f64, value = 88 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1017 = "earth.mul"(%1015, %1016) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1018 = "earth.add"(%1014, %1017) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1019 = "earth.rotate"(%1011) <{offset = array<i64: 30720>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1020 = "earth.constant"() <{rms_var = 0.096192100162823937 : f64, value = 89 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1021 = "earth.mul"(%1019, %1020) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1022 = "earth.add"(%1018, %1021) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1023 = "earth.rotate"(%1011) <{offset = array<i64: 46080>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1024 = "earth.constant"() <{rms_var = 0.10804870438693016 : f64, value = 90 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1025 = "earth.mul"(%1023, %1024) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1026 = "earth.add"(%1022, %1025) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1027 = "earth.constant"() <{rms_var = 0.11069324581445816 : f64, value = 91 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1028 = "earth.mul"(%969, %1027) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1029 = "earth.constant"() <{rms_var = 0.10212811710108596 : f64, value = 92 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1030 = "earth.mul"(%970, %1029) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1031 = "earth.add"(%1028, %1030) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1032 = "earth.constant"() <{rms_var = 0.12515967247344598 : f64, value = 93 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1033 = "earth.mul"(%971, %1032) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1034 = "earth.add"(%1031, %1033) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1035 = "earth.constant"() <{rms_var = 0.14403503206128876 : f64, value = 94 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1036 = "earth.mul"(%972, %1035) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1037 = "earth.add"(%1034, %1036) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1038 = "earth.constant"() <{rms_var = 0.1667235282544858 : f64, value = 95 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1039 = "earth.mul"(%973, %1038) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1040 = "earth.add"(%1037, %1039) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1041 = "earth.constant"() <{rms_var = 0.17100829475650517 : f64, value = 96 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1042 = "earth.mul"(%974, %1041) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1043 = "earth.add"(%1040, %1042) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1044 = "earth.constant"() <{rms_var = 0.13467979907864649 : f64, value = 97 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1045 = "earth.mul"(%975, %1044) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1046 = "earth.add"(%1043, %1045) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1047 = "earth.constant"() <{rms_var = 0.15241337917045406 : f64, value = 98 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1048 = "earth.mul"(%976, %1047) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1049 = "earth.add"(%1046, %1048) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1050 = "earth.constant"() <{rms_var = 0.14871851405931008 : f64, value = 99 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1051 = "earth.mul"(%977, %1050) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1052 = "earth.add"(%1049, %1051) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1053 = "earth.rotate"(%1052) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1054 = "earth.add"(%1052, %1053) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1055 = "earth.rotate"(%1054) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1056 = "earth.add"(%1054, %1055) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1057 = "earth.rotate"(%1056) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1058 = "earth.add"(%1056, %1057) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1059 = "earth.rotate"(%1058) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1060 = "earth.add"(%1058, %1059) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1061 = "earth.rotate"(%1060) <{offset = array<i64: -4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1062 = "earth.constant"() <{rms_var = 0.077111966010965025 : f64, value = 100 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1063 = "earth.mul"(%1061, %1062) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1064 = "earth.add"(%1026, %1063) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1065 = "earth.rotate"(%1060) <{offset = array<i64: 11264>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1066 = "earth.constant"() <{rms_var = 0.083031619072431326 : f64, value = 101 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1067 = "earth.mul"(%1065, %1066) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1068 = "earth.add"(%1064, %1067) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1069 = "earth.rotate"(%1060) <{offset = array<i64: 26624>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1070 = "earth.constant"() <{rms_var = 0.10179747430070245 : f64, value = 102 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1071 = "earth.mul"(%1069, %1070) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1072 = "earth.add"(%1068, %1071) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1073 = "earth.rotate"(%1060) <{offset = array<i64: 41984>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1074 = "earth.constant"() <{rms_var = 0.094465346892113236 : f64, value = 103 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1075 = "earth.mul"(%1073, %1074) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1076 = "earth.add"(%1072, %1075) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1077 = "earth.constant"() <{rms_var = 0.15024445145871693 : f64, value = 104 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1078 = "earth.mul"(%969, %1077) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1079 = "earth.constant"() <{rms_var = 0.14453231007460829 : f64, value = 105 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1080 = "earth.mul"(%970, %1079) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1081 = "earth.add"(%1078, %1080) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1082 = "earth.constant"() <{rms_var = 0.12849137584132345 : f64, value = 106 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1083 = "earth.mul"(%971, %1082) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1084 = "earth.add"(%1081, %1083) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1085 = "earth.constant"() <{rms_var = 0.14749882672058187 : f64, value = 107 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1086 = "earth.mul"(%972, %1085) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1087 = "earth.add"(%1084, %1086) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1088 = "earth.constant"() <{rms_var = 0.19246860855744236 : f64, value = 108 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1089 = "earth.mul"(%973, %1088) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1090 = "earth.add"(%1087, %1089) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1091 = "earth.constant"() <{rms_var = 0.13722066607129441 : f64, value = 109 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1092 = "earth.mul"(%974, %1091) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1093 = "earth.add"(%1090, %1092) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1094 = "earth.constant"() <{rms_var = 0.12879954968209698 : f64, value = 110 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1095 = "earth.mul"(%975, %1094) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1096 = "earth.add"(%1093, %1095) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1097 = "earth.constant"() <{rms_var = 0.17954346371926341 : f64, value = 111 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1098 = "earth.mul"(%976, %1097) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1099 = "earth.add"(%1096, %1098) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1100 = "earth.constant"() <{rms_var = 0.15146363844901589 : f64, value = 112 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1101 = "earth.mul"(%977, %1100) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1102 = "earth.add"(%1099, %1101) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1103 = "earth.rotate"(%1102) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1104 = "earth.add"(%1102, %1103) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1105 = "earth.rotate"(%1104) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1106 = "earth.add"(%1104, %1105) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1107 = "earth.rotate"(%1106) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1108 = "earth.add"(%1106, %1107) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1109 = "earth.rotate"(%1108) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1110 = "earth.add"(%1108, %1109) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1111 = "earth.rotate"(%1110) <{offset = array<i64: -8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1112 = "earth.constant"() <{rms_var = 0.075950720919777073 : f64, value = 113 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1113 = "earth.mul"(%1111, %1112) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1114 = "earth.add"(%1076, %1113) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1115 = "earth.rotate"(%1110) <{offset = array<i64: 7168>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1116 = "earth.constant"() <{rms_var = 0.056143433178170839 : f64, value = 114 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1117 = "earth.mul"(%1115, %1116) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1118 = "earth.add"(%1114, %1117) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1119 = "earth.rotate"(%1110) <{offset = array<i64: 22528>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1120 = "earth.constant"() <{rms_var = 0.080830505320278037 : f64, value = 115 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1121 = "earth.mul"(%1119, %1120) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1122 = "earth.add"(%1118, %1121) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1123 = "earth.rotate"(%1110) <{offset = array<i64: 37888>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1124 = "earth.constant"() <{rms_var = 0.075817629919315718 : f64, value = 116 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1125 = "earth.mul"(%1123, %1124) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1126 = "earth.add"(%1122, %1125) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1127 = "earth.constant"() <{rms_var = 0.16712501424218545 : f64, value = 117 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1128 = "earth.mul"(%969, %1127) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1129 = "earth.constant"() <{rms_var = 0.18306558946978971 : f64, value = 118 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1130 = "earth.mul"(%970, %1129) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1131 = "earth.add"(%1128, %1130) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1132 = "earth.constant"() <{rms_var = 0.15007764827437847 : f64, value = 119 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1133 = "earth.mul"(%971, %1132) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1134 = "earth.add"(%1131, %1133) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1135 = "earth.constant"() <{rms_var = 0.20876102955069348 : f64, value = 120 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1136 = "earth.mul"(%972, %1135) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1137 = "earth.add"(%1134, %1136) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1138 = "earth.constant"() <{rms_var = 0.21195263510998669 : f64, value = 121 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1139 = "earth.mul"(%973, %1138) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1140 = "earth.add"(%1137, %1139) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1141 = "earth.constant"() <{rms_var = 0.1668770265076803 : f64, value = 122 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1142 = "earth.mul"(%974, %1141) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1143 = "earth.add"(%1140, %1142) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1144 = "earth.constant"() <{rms_var = 0.14513823006078308 : f64, value = 123 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1145 = "earth.mul"(%975, %1144) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1146 = "earth.add"(%1143, %1145) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1147 = "earth.constant"() <{rms_var = 0.13183801037256285 : f64, value = 124 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1148 = "earth.mul"(%976, %1147) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1149 = "earth.add"(%1146, %1148) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1150 = "earth.constant"() <{rms_var = 0.15085163994978376 : f64, value = 125 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1151 = "earth.mul"(%977, %1150) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1152 = "earth.add"(%1149, %1151) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1153 = "earth.rotate"(%1152) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1154 = "earth.add"(%1152, %1153) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1155 = "earth.rotate"(%1154) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1156 = "earth.add"(%1154, %1155) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1157 = "earth.rotate"(%1156) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1158 = "earth.add"(%1156, %1157) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1159 = "earth.rotate"(%1158) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1160 = "earth.add"(%1158, %1159) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1161 = "earth.rotate"(%1160) <{offset = array<i64: -12288>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1162 = "earth.constant"() <{rms_var = 0.08598052716615745 : f64, value = 126 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1163 = "earth.mul"(%1161, %1162) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1164 = "earth.add"(%1126, %1163) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1165 = "earth.rotate"(%1160) <{offset = array<i64: 3072>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1166 = "earth.constant"() <{rms_var = 0.082683080257550639 : f64, value = 127 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1167 = "earth.mul"(%1165, %1166) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1168 = "earth.add"(%1164, %1167) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1169 = "earth.rotate"(%1160) <{offset = array<i64: 18432>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1170 = "earth.constant"() <{rms_var = 0.092891974580739542 : f64, value = 128 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1171 = "earth.mul"(%1169, %1170) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1172 = "earth.add"(%1168, %1171) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1173 = "earth.rotate"(%1160) <{offset = array<i64: 33792>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1174 = "earth.constant"() <{rms_var = 0.0677699642845844 : f64, value = 129 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1175 = "earth.mul"(%1173, %1174) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1176 = "earth.add"(%1172, %1175) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1177 = "earth.rotate"(%1176) <{offset = array<i64: -16384>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1178 = "earth.add"(%1176, %1177) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1179 = "earth.rotate"(%1178) <{offset = array<i64: -32768>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1180 = "earth.add"(%1178, %1179) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1181 = "earth.constant"() <{rms_var = 0.035226288337298316 : f64, value = 130 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1182 = "earth.add"(%1180, %1181) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1183 = "earth.add"(%964, %1182) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1184 = "earth.constant"() <{rms_var = 4.000000e+00 : f64, value = 132 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1185 = "earth.add"(%312, %1184) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1186 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 77 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1187 = "earth.add"(%1185, %1186) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-3
    %1188 = "earth.rotate"(%1187) <{offset = array<i64: -33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1189 = "earth.rotate"(%1187) <{offset = array<i64: -32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1190 = "earth.rotate"(%1187) <{offset = array<i64: -31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1191 = "earth.rotate"(%1187) <{offset = array<i64: -1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1192 = "earth.rotate"(%1187) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1193 = "earth.rotate"(%1187) <{offset = array<i64: 1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1194 = "earth.rotate"(%1187) <{offset = array<i64: 31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1195 = "earth.rotate"(%1187) <{offset = array<i64: 32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1196 = "earth.rotate"(%1187) <{offset = array<i64: 33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1197 = "earth.constant"() <{rms_var = 0.096025948135613062 : f64, value = 78 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1198 = "earth.mul"(%1188, %1197) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1199 = "earth.constant"() <{rms_var = 0.11164571811099519 : f64, value = 79 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1200 = "earth.mul"(%1189, %1199) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1201 = "earth.add"(%1198, %1200) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1202 = "earth.constant"() <{rms_var = 0.13017961240207196 : f64, value = 80 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1203 = "earth.mul"(%1190, %1202) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1204 = "earth.add"(%1201, %1203) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1205 = "earth.constant"() <{rms_var = 0.16329299179200105 : f64, value = 81 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1206 = "earth.mul"(%1191, %1205) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1207 = "earth.add"(%1204, %1206) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1208 = "earth.constant"() <{rms_var = 0.17161395650663383 : f64, value = 82 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1209 = "earth.mul"(%1192, %1208) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1210 = "earth.add"(%1207, %1209) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1211 = "earth.constant"() <{rms_var = 0.14327661978876549 : f64, value = 83 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1212 = "earth.mul"(%1193, %1211) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1213 = "earth.add"(%1210, %1212) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1214 = "earth.constant"() <{rms_var = 0.12473295548036838 : f64, value = 84 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1215 = "earth.mul"(%1194, %1214) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1216 = "earth.add"(%1213, %1215) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1217 = "earth.constant"() <{rms_var = 0.12216228874268155 : f64, value = 85 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1218 = "earth.mul"(%1195, %1217) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1219 = "earth.add"(%1216, %1218) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1220 = "earth.constant"() <{rms_var = 0.11762778832214148 : f64, value = 86 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1221 = "earth.mul"(%1196, %1220) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1222 = "earth.add"(%1219, %1221) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1223 = "earth.rotate"(%1222) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1224 = "earth.add"(%1222, %1223) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1225 = "earth.rotate"(%1224) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1226 = "earth.add"(%1224, %1225) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1227 = "earth.rotate"(%1226) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1228 = "earth.add"(%1226, %1227) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1229 = "earth.rotate"(%1228) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1230 = "earth.add"(%1228, %1229) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1231 = "earth.rotate"(%1230) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1232 = "earth.constant"() <{rms_var = 0.087056217584534953 : f64, value = 87 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1233 = "earth.mul"(%1231, %1232) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1234 = "earth.rotate"(%1230) <{offset = array<i64: 15360>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1235 = "earth.constant"() <{rms_var = 0.12502985663775679 : f64, value = 88 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1236 = "earth.mul"(%1234, %1235) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1237 = "earth.add"(%1233, %1236) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1238 = "earth.rotate"(%1230) <{offset = array<i64: 30720>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1239 = "earth.constant"() <{rms_var = 0.096192100162823937 : f64, value = 89 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1240 = "earth.mul"(%1238, %1239) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1241 = "earth.add"(%1237, %1240) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1242 = "earth.rotate"(%1230) <{offset = array<i64: 46080>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1243 = "earth.constant"() <{rms_var = 0.10804870438693016 : f64, value = 90 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1244 = "earth.mul"(%1242, %1243) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1245 = "earth.add"(%1241, %1244) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1246 = "earth.constant"() <{rms_var = 0.11069324581445816 : f64, value = 91 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1247 = "earth.mul"(%1188, %1246) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1248 = "earth.constant"() <{rms_var = 0.10212811710108596 : f64, value = 92 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1249 = "earth.mul"(%1189, %1248) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1250 = "earth.add"(%1247, %1249) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1251 = "earth.constant"() <{rms_var = 0.12515967247344598 : f64, value = 93 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1252 = "earth.mul"(%1190, %1251) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1253 = "earth.add"(%1250, %1252) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1254 = "earth.constant"() <{rms_var = 0.14403503206128876 : f64, value = 94 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1255 = "earth.mul"(%1191, %1254) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1256 = "earth.add"(%1253, %1255) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1257 = "earth.constant"() <{rms_var = 0.1667235282544858 : f64, value = 95 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1258 = "earth.mul"(%1192, %1257) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1259 = "earth.add"(%1256, %1258) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1260 = "earth.constant"() <{rms_var = 0.17100829475650517 : f64, value = 96 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1261 = "earth.mul"(%1193, %1260) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1262 = "earth.add"(%1259, %1261) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1263 = "earth.constant"() <{rms_var = 0.13467979907864649 : f64, value = 97 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1264 = "earth.mul"(%1194, %1263) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1265 = "earth.add"(%1262, %1264) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1266 = "earth.constant"() <{rms_var = 0.15241337917045406 : f64, value = 98 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1267 = "earth.mul"(%1195, %1266) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1268 = "earth.add"(%1265, %1267) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1269 = "earth.constant"() <{rms_var = 0.14871851405931008 : f64, value = 99 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1270 = "earth.mul"(%1196, %1269) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1271 = "earth.add"(%1268, %1270) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1272 = "earth.rotate"(%1271) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1273 = "earth.add"(%1271, %1272) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1274 = "earth.rotate"(%1273) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1275 = "earth.add"(%1273, %1274) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1276 = "earth.rotate"(%1275) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1277 = "earth.add"(%1275, %1276) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1278 = "earth.rotate"(%1277) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1279 = "earth.add"(%1277, %1278) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1280 = "earth.rotate"(%1279) <{offset = array<i64: -4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1281 = "earth.constant"() <{rms_var = 0.077111966010965025 : f64, value = 100 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1282 = "earth.mul"(%1280, %1281) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1283 = "earth.add"(%1245, %1282) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1284 = "earth.rotate"(%1279) <{offset = array<i64: 11264>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1285 = "earth.constant"() <{rms_var = 0.083031619072431326 : f64, value = 101 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1286 = "earth.mul"(%1284, %1285) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1287 = "earth.add"(%1283, %1286) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1288 = "earth.rotate"(%1279) <{offset = array<i64: 26624>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1289 = "earth.constant"() <{rms_var = 0.10179747430070245 : f64, value = 102 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1290 = "earth.mul"(%1288, %1289) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1291 = "earth.add"(%1287, %1290) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1292 = "earth.rotate"(%1279) <{offset = array<i64: 41984>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1293 = "earth.constant"() <{rms_var = 0.094465346892113236 : f64, value = 103 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1294 = "earth.mul"(%1292, %1293) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1295 = "earth.add"(%1291, %1294) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1296 = "earth.constant"() <{rms_var = 0.15024445145871693 : f64, value = 104 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1297 = "earth.mul"(%1188, %1296) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1298 = "earth.constant"() <{rms_var = 0.14453231007460829 : f64, value = 105 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1299 = "earth.mul"(%1189, %1298) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1300 = "earth.add"(%1297, %1299) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1301 = "earth.constant"() <{rms_var = 0.12849137584132345 : f64, value = 106 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1302 = "earth.mul"(%1190, %1301) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1303 = "earth.add"(%1300, %1302) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1304 = "earth.constant"() <{rms_var = 0.14749882672058187 : f64, value = 107 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1305 = "earth.mul"(%1191, %1304) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1306 = "earth.add"(%1303, %1305) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1307 = "earth.constant"() <{rms_var = 0.19246860855744236 : f64, value = 108 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1308 = "earth.mul"(%1192, %1307) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1309 = "earth.add"(%1306, %1308) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1310 = "earth.constant"() <{rms_var = 0.13722066607129441 : f64, value = 109 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1311 = "earth.mul"(%1193, %1310) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1312 = "earth.add"(%1309, %1311) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1313 = "earth.constant"() <{rms_var = 0.12879954968209698 : f64, value = 110 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1314 = "earth.mul"(%1194, %1313) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1315 = "earth.add"(%1312, %1314) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1316 = "earth.constant"() <{rms_var = 0.17954346371926341 : f64, value = 111 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1317 = "earth.mul"(%1195, %1316) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1318 = "earth.add"(%1315, %1317) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1319 = "earth.constant"() <{rms_var = 0.15146363844901589 : f64, value = 112 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1320 = "earth.mul"(%1196, %1319) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1321 = "earth.add"(%1318, %1320) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1322 = "earth.rotate"(%1321) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1323 = "earth.add"(%1321, %1322) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1324 = "earth.rotate"(%1323) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1325 = "earth.add"(%1323, %1324) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1326 = "earth.rotate"(%1325) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1327 = "earth.add"(%1325, %1326) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1328 = "earth.rotate"(%1327) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1329 = "earth.add"(%1327, %1328) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1330 = "earth.rotate"(%1329) <{offset = array<i64: -8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1331 = "earth.constant"() <{rms_var = 0.075950720919777073 : f64, value = 113 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1332 = "earth.mul"(%1330, %1331) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1333 = "earth.add"(%1295, %1332) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1334 = "earth.rotate"(%1329) <{offset = array<i64: 7168>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1335 = "earth.constant"() <{rms_var = 0.056143433178170839 : f64, value = 114 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1336 = "earth.mul"(%1334, %1335) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1337 = "earth.add"(%1333, %1336) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1338 = "earth.rotate"(%1329) <{offset = array<i64: 22528>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1339 = "earth.constant"() <{rms_var = 0.080830505320278037 : f64, value = 115 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1340 = "earth.mul"(%1338, %1339) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1341 = "earth.add"(%1337, %1340) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1342 = "earth.rotate"(%1329) <{offset = array<i64: 37888>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1343 = "earth.constant"() <{rms_var = 0.075817629919315718 : f64, value = 116 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1344 = "earth.mul"(%1342, %1343) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1345 = "earth.add"(%1341, %1344) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1346 = "earth.constant"() <{rms_var = 0.16712501424218545 : f64, value = 117 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1347 = "earth.mul"(%1188, %1346) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1348 = "earth.constant"() <{rms_var = 0.18306558946978971 : f64, value = 118 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1349 = "earth.mul"(%1189, %1348) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1350 = "earth.add"(%1347, %1349) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1351 = "earth.constant"() <{rms_var = 0.15007764827437847 : f64, value = 119 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1352 = "earth.mul"(%1190, %1351) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1353 = "earth.add"(%1350, %1352) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1354 = "earth.constant"() <{rms_var = 0.20876102955069348 : f64, value = 120 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1355 = "earth.mul"(%1191, %1354) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1356 = "earth.add"(%1353, %1355) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1357 = "earth.constant"() <{rms_var = 0.21195263510998669 : f64, value = 121 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1358 = "earth.mul"(%1192, %1357) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1359 = "earth.add"(%1356, %1358) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1360 = "earth.constant"() <{rms_var = 0.1668770265076803 : f64, value = 122 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1361 = "earth.mul"(%1193, %1360) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1362 = "earth.add"(%1359, %1361) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1363 = "earth.constant"() <{rms_var = 0.14513823006078308 : f64, value = 123 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1364 = "earth.mul"(%1194, %1363) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1365 = "earth.add"(%1362, %1364) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1366 = "earth.constant"() <{rms_var = 0.13183801037256285 : f64, value = 124 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1367 = "earth.mul"(%1195, %1366) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1368 = "earth.add"(%1365, %1367) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1369 = "earth.constant"() <{rms_var = 0.15085163994978376 : f64, value = 125 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1370 = "earth.mul"(%1196, %1369) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1371 = "earth.add"(%1368, %1370) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1372 = "earth.rotate"(%1371) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1373 = "earth.add"(%1371, %1372) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1374 = "earth.rotate"(%1373) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1375 = "earth.add"(%1373, %1374) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1376 = "earth.rotate"(%1375) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1377 = "earth.add"(%1375, %1376) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1378 = "earth.rotate"(%1377) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1379 = "earth.add"(%1377, %1378) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1380 = "earth.rotate"(%1379) <{offset = array<i64: -12288>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1381 = "earth.constant"() <{rms_var = 0.08598052716615745 : f64, value = 126 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1382 = "earth.mul"(%1380, %1381) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1383 = "earth.add"(%1345, %1382) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1384 = "earth.rotate"(%1379) <{offset = array<i64: 3072>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1385 = "earth.constant"() <{rms_var = 0.082683080257550639 : f64, value = 127 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1386 = "earth.mul"(%1384, %1385) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1387 = "earth.add"(%1383, %1386) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1388 = "earth.rotate"(%1379) <{offset = array<i64: 18432>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1389 = "earth.constant"() <{rms_var = 0.092891974580739542 : f64, value = 128 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1390 = "earth.mul"(%1388, %1389) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1391 = "earth.add"(%1387, %1390) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1392 = "earth.rotate"(%1379) <{offset = array<i64: 33792>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1393 = "earth.constant"() <{rms_var = 0.0677699642845844 : f64, value = 129 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1394 = "earth.mul"(%1392, %1393) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1395 = "earth.add"(%1391, %1394) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1396 = "earth.rotate"(%1395) <{offset = array<i64: -16384>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1397 = "earth.add"(%1395, %1396) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1398 = "earth.rotate"(%1397) <{offset = array<i64: -32768>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1399 = "earth.add"(%1397, %1398) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1400 = "earth.constant"() <{rms_var = 0.035226288337298316 : f64, value = 130 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1401 = "earth.add"(%1399, %1400) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1402 = "earth.add"(%1183, %1401) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn1-4
    %1403 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1404 = "earth.mul"(%1403, %1402) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1405 = "earth.mul"(%1404, %1402) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1406 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1407 = "earth.add"(%1405, %1406) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1408 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1409 = "earth.mul"(%1408, %1407) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1410 = "earth.mul"(%1409, %1407) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1411 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1412 = "earth.add"(%1410, %1411) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1413 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1414 = "earth.mul"(%1413, %1412) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1415 = "earth.mul"(%1414, %1412) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1416 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1417 = "earth.add"(%1415, %1416) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1418 = "earth.mul"(%1404, %1407) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1419 = "earth.negate"(%1402) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1420 = "earth.add"(%1418, %1419) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1421 = "earth.mul"(%1404, %1412) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1422 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1423 = "earth.mul"(%1422, %1420) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1424 = "earth.mul"(%1423, %1412) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1425 = "earth.negate"(%1420) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1426 = "earth.add"(%1421, %1425) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1427 = "earth.add"(%1424, %1419) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1428 = "earth.mul"(%1404, %1417) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1429 = "earth.mul"(%1423, %1417) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1430 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1431 = "earth.mul"(%1430, %1426) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1432 = "earth.mul"(%1431, %1417) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1433 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1434 = "earth.mul"(%1433, %1427) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1435 = "earth.mul"(%1434, %1417) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1436 = "earth.negate"(%1427) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1437 = "earth.add"(%1428, %1436) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1438 = "earth.negate"(%1426) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1439 = "earth.add"(%1429, %1438) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1440 = "earth.add"(%1432, %1425) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1441 = "earth.add"(%1435, %1419) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1442 = "earth.constant"() <{rms_var = 0.051379788302527769 : f64, value = 28 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1443 = "earth.mul"(%1442, %1402) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1444 = "earth.constant"() <{rms_var = 0.04272842452341858 : f64, value = 29 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1445 = "earth.mul"(%1444, %1420) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1446 = "earth.add"(%1443, %1445) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1447 = "earth.constant"() <{rms_var = 0.036049430000943392 : f64, value = 30 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1448 = "earth.mul"(%1447, %1426) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1449 = "earth.add"(%1446, %1448) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1450 = "earth.constant"() <{rms_var = 0.030937245302699062 : f64, value = 31 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1451 = "earth.mul"(%1450, %1427) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1452 = "earth.add"(%1449, %1451) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1453 = "earth.constant"() <{rms_var = 0.027115258485962086 : f64, value = 32 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1454 = "earth.mul"(%1453, %1437) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1455 = "earth.add"(%1452, %1454) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1456 = "earth.constant"() <{rms_var = 0.024393077542268389 : f64, value = 33 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1457 = "earth.mul"(%1456, %1439) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1458 = "earth.add"(%1455, %1457) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1459 = "earth.constant"() <{rms_var = 0.022642601074970584 : f64, value = 34 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1460 = "earth.mul"(%1459, %1440) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1461 = "earth.add"(%1458, %1460) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1462 = "earth.constant"() <{rms_var = 0.021831260875010087 : f64, value = 35 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1463 = "earth.mul"(%1462, %1441) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1464 = "earth.add"(%1461, %1463) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1465 = "earth.constant"() <{rms_var = 0.021652089836507502 : f64, value = 36 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1466 = "earth.mul"(%1465, %1402) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1467 = "earth.constant"() <{rms_var = 0.018138222134916806 : f64, value = 37 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1468 = "earth.mul"(%1467, %1420) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1469 = "earth.add"(%1466, %1468) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1470 = "earth.constant"() <{rms_var = 0.01542303411540253 : f64, value = 38 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1471 = "earth.mul"(%1470, %1426) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1472 = "earth.add"(%1469, %1471) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1473 = "earth.constant"() <{rms_var = 0.013305654693862069 : f64, value = 39 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1474 = "earth.mul"(%1473, %1427) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1475 = "earth.add"(%1472, %1474) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1476 = "earth.constant"() <{rms_var = 0.011703046220094507 : f64, value = 40 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1477 = "earth.mul"(%1476, %1437) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1478 = "earth.add"(%1475, %1477) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1479 = "earth.constant"() <{rms_var = 0.010552642814751455 : f64, value = 41 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1480 = "earth.mul"(%1479, %1439) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1481 = "earth.add"(%1478, %1480) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1482 = "earth.constant"() <{rms_var = 0.0098096659804816355 : f64, value = 42 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1483 = "earth.mul"(%1482, %1440) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1484 = "earth.add"(%1481, %1483) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1485 = "earth.constant"() <{rms_var = 0.0094452495559895089 : f64, value = 43 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1486 = "earth.mul"(%1485, %1441) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1487 = "earth.add"(%1484, %1486) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1488 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1489 = "earth.mul"(%1488, %1417) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1490 = "earth.mul"(%1489, %1417) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1491 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1492 = "earth.add"(%1490, %1491) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1493 = "earth.mul"(%1487, %1492) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1494 = "earth.add"(%1493, %1464) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1495 = "earth.constant"() <{rms_var = 0.0042995278292336028 : f64, value = 44 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1496 = "earth.mul"(%1495, %1402) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1497 = "earth.constant"() <{rms_var = 0.0036191919476548503 : f64, value = 45 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1498 = "earth.mul"(%1497, %1420) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1499 = "earth.add"(%1496, %1498) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1500 = "earth.constant"() <{rms_var = 0.0030784419617177587 : f64, value = 46 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1501 = "earth.mul"(%1500, %1426) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1502 = "earth.add"(%1499, %1501) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1503 = "earth.constant"() <{rms_var = 0.0026564062051904268 : f64, value = 47 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1504 = "earth.mul"(%1503, %1427) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1505 = "earth.add"(%1502, %1504) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1506 = "earth.constant"() <{rms_var = 0.0023368004457586414 : f64, value = 48 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1507 = "earth.mul"(%1506, %1437) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1508 = "earth.add"(%1505, %1507) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1509 = "earth.constant"() <{rms_var = 0.002107295415524122 : f64, value = 49 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1510 = "earth.mul"(%1509, %1439) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1511 = "earth.add"(%1508, %1510) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1512 = "earth.constant"() <{rms_var = 0.0019590388891824136 : f64, value = 50 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1513 = "earth.mul"(%1512, %1440) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1514 = "earth.add"(%1511, %1513) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1515 = "earth.constant"() <{rms_var = 0.0018863129851277056 : f64, value = 51 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1516 = "earth.mul"(%1515, %1441) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1517 = "earth.add"(%1514, %1516) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1518 = "earth.constant"() <{rms_var = 0.63615477792235875 : f64, value = 52 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1519 = "earth.mul"(%1518, %1402) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1520 = "earth.constant"() <{rms_var = 0.21189795368271022 : f64, value = 53 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1521 = "earth.mul"(%1520, %1420) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1522 = "earth.add"(%1519, %1521) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1523 = "earth.constant"() <{rms_var = 0.12734215758896472 : f64, value = 54 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1524 = "earth.mul"(%1523, %1426) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1525 = "earth.add"(%1522, %1524) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1526 = "earth.constant"() <{rms_var = 0.091626347972833505 : f64, value = 55 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1527 = "earth.mul"(%1526, %1427) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1528 = "earth.add"(%1525, %1527) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1529 = "earth.constant"() <{rms_var = 0.07255941147658998 : f64, value = 56 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1530 = "earth.mul"(%1529, %1437) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1531 = "earth.add"(%1528, %1530) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1532 = "earth.constant"() <{rms_var = 0.061477909207391525 : f64, value = 57 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1533 = "earth.mul"(%1532, %1439) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1534 = "earth.add"(%1531, %1533) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1535 = "earth.constant"() <{rms_var = 0.055169030070493508 : f64, value = 58 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1536 = "earth.mul"(%1535, %1440) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1537 = "earth.add"(%1534, %1536) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1538 = "earth.constant"() <{rms_var = 0.052275954916496656 : f64, value = 59 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1539 = "earth.mul"(%1538, %1441) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1540 = "earth.add"(%1537, %1539) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1541 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1542 = "earth.mul"(%1541, %1492) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1543 = "earth.mul"(%1542, %1492) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1544 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1545 = "earth.add"(%1543, %1544) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1546 = "earth.mul"(%1517, %1545) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1547 = "earth.add"(%1546, %1494) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1548 = "earth.constant"() <{rms_var = 4.9481895575581374E-4 : f64, value = 60 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1549 = "earth.mul"(%1548, %1402) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1550 = "earth.constant"() <{rms_var = 3.771067298313324E-4 : f64, value = 61 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1551 = "earth.mul"(%1550, %1420) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1552 = "earth.add"(%1549, %1551) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1553 = "earth.constant"() <{rms_var = 3.2076765303085136E-4 : f64, value = 62 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1554 = "earth.mul"(%1553, %1426) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1555 = "earth.add"(%1552, %1554) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1556 = "earth.constant"() <{rms_var = 2.7679532400224407E-4 : f64, value = 63 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1557 = "earth.mul"(%1556, %1427) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1558 = "earth.add"(%1555, %1557) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1559 = "earth.constant"() <{rms_var = 2.4349440690240933E-4 : f64, value = 64 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1560 = "earth.mul"(%1559, %1437) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1561 = "earth.add"(%1558, %1560) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1562 = "earth.constant"() <{rms_var = 2.1958101095650637E-4 : f64, value = 65 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1563 = "earth.mul"(%1562, %1439) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1564 = "earth.add"(%1561, %1563) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1565 = "earth.constant"() <{rms_var = 2.0413318001830463E-4 : f64, value = 66 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1566 = "earth.mul"(%1565, %1440) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1567 = "earth.add"(%1564, %1566) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1568 = "earth.constant"() <{rms_var = 1.9655534146569248E-4 : f64, value = 67 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1569 = "earth.mul"(%1568, %1441) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1570 = "earth.add"(%1567, %1569) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1571 = "earth.constant"() <{rms_var = 1.7735088548126505E-4 : f64, value = 68 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1572 = "earth.mul"(%1571, %1402) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1573 = "earth.constant"() <{rms_var = 1.457794164045128E-4 : f64, value = 69 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1574 = "earth.mul"(%1573, %1420) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1575 = "earth.add"(%1572, %1574) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1576 = "earth.constant"() <{rms_var = 1.1982820692495149E-4 : f64, value = 70 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1577 = "earth.mul"(%1576, %1426) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1578 = "earth.add"(%1575, %1577) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1579 = "earth.constant"() <{rms_var = 9.8496754159654756E-5 : f64, value = 71 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1580 = "earth.mul"(%1579, %1427) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1581 = "earth.add"(%1578, %1580) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1582 = "earth.constant"() <{rms_var = 8.0962662339626343E-5 : f64, value = 72 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1583 = "earth.mul"(%1582, %1437) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1584 = "earth.add"(%1581, %1583) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1585 = "earth.constant"() <{rms_var = 6.6549936439435966E-5 : f64, value = 73 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1586 = "earth.mul"(%1585, %1439) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1587 = "earth.add"(%1584, %1586) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1588 = "earth.constant"() <{rms_var = 5.4702920205367267E-5 : f64, value = 74 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1589 = "earth.mul"(%1588, %1440) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1590 = "earth.add"(%1587, %1589) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1591 = "earth.constant"() <{rms_var = 1.3863334887734771E-4 : f64, value = 75 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1592 = "earth.mul"(%1591, %1441) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1593 = "earth.add"(%1590, %1592) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1594 = "earth.mul"(%1547, %1492) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1595 = "earth.add"(%1594, %1540) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1596 = "earth.mul"(%1593, %1492) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1597 = "earth.add"(%1596, %1570) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1598 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1599 = "earth.mul"(%1598, %1545) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1600 = "earth.mul"(%1599, %1545) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1601 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1602 = "earth.add"(%1600, %1601) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1603 = "earth.mul"(%1597, %1602) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1604 = "earth.add"(%1603, %1595) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act1_SiLU_poly
    %1605 = "earth.constant"() <{rms_var = 5.000000e-01 : f64, value = 76 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // add[]layer1_0_act1_SiLU_add
    %1606 = "earth.add"(%1604, %1605) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // add[]layer1_0_act1_SiLU_add
    %1607 = "earth.mul"(%1402, %1606) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // mul[]layer1_0_act1_SiLU_mul
    %1608 = "earth.rotate"(%1607) <{offset = array<i64: -33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1609 = "earth.rotate"(%1607) <{offset = array<i64: -32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1610 = "earth.rotate"(%1607) <{offset = array<i64: -31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1611 = "earth.rotate"(%1607) <{offset = array<i64: -1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1612 = "earth.rotate"(%1607) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1613 = "earth.rotate"(%1607) <{offset = array<i64: 1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1614 = "earth.rotate"(%1607) <{offset = array<i64: 31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1615 = "earth.rotate"(%1607) <{offset = array<i64: 32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1616 = "earth.rotate"(%1607) <{offset = array<i64: 33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1617 = "earth.constant"() <{rms_var = 0.10400615650895387 : f64, value = 133 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1618 = "earth.mul"(%1608, %1617) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1619 = "earth.constant"() <{rms_var = 0.12200783578493199 : f64, value = 134 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1620 = "earth.mul"(%1609, %1619) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1621 = "earth.add"(%1618, %1620) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1622 = "earth.constant"() <{rms_var = 0.10864069078255634 : f64, value = 135 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1623 = "earth.mul"(%1610, %1622) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1624 = "earth.add"(%1621, %1623) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1625 = "earth.constant"() <{rms_var = 0.1813826358721331 : f64, value = 136 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1626 = "earth.mul"(%1611, %1625) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1627 = "earth.add"(%1624, %1626) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1628 = "earth.constant"() <{rms_var = 0.23782715602002519 : f64, value = 137 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1629 = "earth.mul"(%1612, %1628) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1630 = "earth.add"(%1627, %1629) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1631 = "earth.constant"() <{rms_var = 0.1790932130828157 : f64, value = 138 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1632 = "earth.mul"(%1613, %1631) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1633 = "earth.add"(%1630, %1632) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1634 = "earth.constant"() <{rms_var = 0.12293002379897668 : f64, value = 139 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1635 = "earth.mul"(%1614, %1634) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1636 = "earth.add"(%1633, %1635) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1637 = "earth.constant"() <{rms_var = 0.13881195116371711 : f64, value = 140 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1638 = "earth.mul"(%1615, %1637) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1639 = "earth.add"(%1636, %1638) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1640 = "earth.constant"() <{rms_var = 0.12208057183590357 : f64, value = 141 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1641 = "earth.mul"(%1616, %1640) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1642 = "earth.add"(%1639, %1641) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1643 = "earth.rotate"(%1642) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1644 = "earth.add"(%1642, %1643) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1645 = "earth.rotate"(%1644) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1646 = "earth.add"(%1644, %1645) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1647 = "earth.rotate"(%1646) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1648 = "earth.add"(%1646, %1647) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1649 = "earth.rotate"(%1648) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1650 = "earth.add"(%1648, %1649) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1651 = "earth.rotate"(%1650) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1652 = "earth.constant"() <{rms_var = 0.1040950107964377 : f64, value = 142 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1653 = "earth.mul"(%1651, %1652) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1654 = "earth.rotate"(%1650) <{offset = array<i64: 15360>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1655 = "earth.constant"() <{rms_var = 0.054219798709707959 : f64, value = 143 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1656 = "earth.mul"(%1654, %1655) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1657 = "earth.add"(%1653, %1656) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1658 = "earth.rotate"(%1650) <{offset = array<i64: 30720>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1659 = "earth.constant"() <{rms_var = 0.097445054018278298 : f64, value = 144 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1660 = "earth.mul"(%1658, %1659) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1661 = "earth.add"(%1657, %1660) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1662 = "earth.rotate"(%1650) <{offset = array<i64: 46080>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1663 = "earth.constant"() <{rms_var = 0.068681779701445803 : f64, value = 145 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1664 = "earth.mul"(%1662, %1663) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1665 = "earth.add"(%1661, %1664) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1666 = "earth.constant"() <{rms_var = 0.10673230849977647 : f64, value = 146 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1667 = "earth.mul"(%1608, %1666) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1668 = "earth.constant"() <{rms_var = 0.11379122203861868 : f64, value = 147 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1669 = "earth.mul"(%1609, %1668) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1670 = "earth.add"(%1667, %1669) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1671 = "earth.constant"() <{rms_var = 0.1163497903488265 : f64, value = 148 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1672 = "earth.mul"(%1610, %1671) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1673 = "earth.add"(%1670, %1672) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1674 = "earth.constant"() <{rms_var = 0.10890930503439286 : f64, value = 149 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1675 = "earth.mul"(%1611, %1674) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1676 = "earth.add"(%1673, %1675) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1677 = "earth.constant"() <{rms_var = 0.18869231093830438 : f64, value = 150 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1678 = "earth.mul"(%1612, %1677) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1679 = "earth.add"(%1676, %1678) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1680 = "earth.constant"() <{rms_var = 0.17692941546313307 : f64, value = 151 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1681 = "earth.mul"(%1613, %1680) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1682 = "earth.add"(%1679, %1681) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1683 = "earth.constant"() <{rms_var = 0.1182309821634905 : f64, value = 152 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1684 = "earth.mul"(%1614, %1683) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1685 = "earth.add"(%1682, %1684) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1686 = "earth.constant"() <{rms_var = 0.13212182246293011 : f64, value = 153 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1687 = "earth.mul"(%1615, %1686) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1688 = "earth.add"(%1685, %1687) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1689 = "earth.constant"() <{rms_var = 0.14911147671887034 : f64, value = 154 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1690 = "earth.mul"(%1616, %1689) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1691 = "earth.add"(%1688, %1690) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1692 = "earth.rotate"(%1691) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1693 = "earth.add"(%1691, %1692) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1694 = "earth.rotate"(%1693) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1695 = "earth.add"(%1693, %1694) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1696 = "earth.rotate"(%1695) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1697 = "earth.add"(%1695, %1696) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1698 = "earth.rotate"(%1697) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1699 = "earth.add"(%1697, %1698) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1700 = "earth.rotate"(%1699) <{offset = array<i64: -4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1701 = "earth.constant"() <{rms_var = 0.10538071715990344 : f64, value = 155 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1702 = "earth.mul"(%1700, %1701) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1703 = "earth.add"(%1665, %1702) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1704 = "earth.rotate"(%1699) <{offset = array<i64: 11264>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1705 = "earth.constant"() <{rms_var = 0.060568670583673635 : f64, value = 156 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1706 = "earth.mul"(%1704, %1705) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1707 = "earth.add"(%1703, %1706) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1708 = "earth.rotate"(%1699) <{offset = array<i64: 26624>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1709 = "earth.constant"() <{rms_var = 0.036450552647908481 : f64, value = 157 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1710 = "earth.mul"(%1708, %1709) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1711 = "earth.add"(%1707, %1710) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1712 = "earth.rotate"(%1699) <{offset = array<i64: 41984>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1713 = "earth.constant"() <{rms_var = 0.078879837716772497 : f64, value = 158 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1714 = "earth.mul"(%1712, %1713) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1715 = "earth.add"(%1711, %1714) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1716 = "earth.constant"() <{rms_var = 0.1469224672166918 : f64, value = 159 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1717 = "earth.mul"(%1608, %1716) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1718 = "earth.constant"() <{rms_var = 0.14911190039967734 : f64, value = 160 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1719 = "earth.mul"(%1609, %1718) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1720 = "earth.add"(%1717, %1719) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1721 = "earth.constant"() <{rms_var = 0.13124615233523493 : f64, value = 161 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1722 = "earth.mul"(%1610, %1721) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1723 = "earth.add"(%1720, %1722) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1724 = "earth.constant"() <{rms_var = 0.18225946911845031 : f64, value = 162 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1725 = "earth.mul"(%1611, %1724) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1726 = "earth.add"(%1723, %1725) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1727 = "earth.constant"() <{rms_var = 0.1812770404302394 : f64, value = 163 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1728 = "earth.mul"(%1612, %1727) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1729 = "earth.add"(%1726, %1728) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1730 = "earth.constant"() <{rms_var = 0.16168084351116435 : f64, value = 164 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1731 = "earth.mul"(%1613, %1730) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1732 = "earth.add"(%1729, %1731) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1733 = "earth.constant"() <{rms_var = 0.13637109952951873 : f64, value = 165 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1734 = "earth.mul"(%1614, %1733) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1735 = "earth.add"(%1732, %1734) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1736 = "earth.constant"() <{rms_var = 0.21006414360479692 : f64, value = 166 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1737 = "earth.mul"(%1615, %1736) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1738 = "earth.add"(%1735, %1737) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1739 = "earth.constant"() <{rms_var = 0.14658033543052656 : f64, value = 167 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1740 = "earth.mul"(%1616, %1739) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1741 = "earth.add"(%1738, %1740) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1742 = "earth.rotate"(%1741) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1743 = "earth.add"(%1741, %1742) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1744 = "earth.rotate"(%1743) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1745 = "earth.add"(%1743, %1744) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1746 = "earth.rotate"(%1745) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1747 = "earth.add"(%1745, %1746) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1748 = "earth.rotate"(%1747) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1749 = "earth.add"(%1747, %1748) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1750 = "earth.rotate"(%1749) <{offset = array<i64: -8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1751 = "earth.constant"() <{rms_var = 0.069283197622941217 : f64, value = 168 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1752 = "earth.mul"(%1750, %1751) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1753 = "earth.add"(%1715, %1752) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1754 = "earth.rotate"(%1749) <{offset = array<i64: 7168>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1755 = "earth.constant"() <{rms_var = 0.074467567775713839 : f64, value = 169 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1756 = "earth.mul"(%1754, %1755) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1757 = "earth.add"(%1753, %1756) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1758 = "earth.rotate"(%1749) <{offset = array<i64: 22528>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1759 = "earth.constant"() <{rms_var = 0.084142703123225818 : f64, value = 170 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1760 = "earth.mul"(%1758, %1759) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1761 = "earth.add"(%1757, %1760) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1762 = "earth.rotate"(%1749) <{offset = array<i64: 37888>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1763 = "earth.constant"() <{rms_var = 0.07412846710992968 : f64, value = 171 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1764 = "earth.mul"(%1762, %1763) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1765 = "earth.add"(%1761, %1764) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1766 = "earth.constant"() <{rms_var = 0.15533875553942783 : f64, value = 172 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1767 = "earth.mul"(%1608, %1766) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1768 = "earth.constant"() <{rms_var = 0.14637769408175144 : f64, value = 173 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1769 = "earth.mul"(%1609, %1768) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1770 = "earth.add"(%1767, %1769) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1771 = "earth.constant"() <{rms_var = 0.15792116833611017 : f64, value = 174 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1772 = "earth.mul"(%1610, %1771) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1773 = "earth.add"(%1770, %1772) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1774 = "earth.constant"() <{rms_var = 0.13739927438768484 : f64, value = 175 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1775 = "earth.mul"(%1611, %1774) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1776 = "earth.add"(%1773, %1775) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1777 = "earth.constant"() <{rms_var = 0.1957616398218727 : f64, value = 176 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1778 = "earth.mul"(%1612, %1777) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1779 = "earth.add"(%1776, %1778) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1780 = "earth.constant"() <{rms_var = 0.14764849972100441 : f64, value = 177 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1781 = "earth.mul"(%1613, %1780) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1782 = "earth.add"(%1779, %1781) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1783 = "earth.constant"() <{rms_var = 0.11119133880761291 : f64, value = 178 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1784 = "earth.mul"(%1614, %1783) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1785 = "earth.add"(%1782, %1784) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1786 = "earth.constant"() <{rms_var = 0.15286625358137407 : f64, value = 179 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1787 = "earth.mul"(%1615, %1786) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1788 = "earth.add"(%1785, %1787) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1789 = "earth.constant"() <{rms_var = 0.15289762208814006 : f64, value = 180 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1790 = "earth.mul"(%1616, %1789) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1791 = "earth.add"(%1788, %1790) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1792 = "earth.rotate"(%1791) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1793 = "earth.add"(%1791, %1792) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1794 = "earth.rotate"(%1793) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1795 = "earth.add"(%1793, %1794) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1796 = "earth.rotate"(%1795) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1797 = "earth.add"(%1795, %1796) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1798 = "earth.rotate"(%1797) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1799 = "earth.add"(%1797, %1798) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1800 = "earth.rotate"(%1799) <{offset = array<i64: -12288>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1801 = "earth.constant"() <{rms_var = 0.090316089101903224 : f64, value = 181 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1802 = "earth.mul"(%1800, %1801) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1803 = "earth.add"(%1765, %1802) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1804 = "earth.rotate"(%1799) <{offset = array<i64: 3072>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1805 = "earth.constant"() <{rms_var = 0.095745708440118468 : f64, value = 182 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1806 = "earth.mul"(%1804, %1805) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1807 = "earth.add"(%1803, %1806) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1808 = "earth.rotate"(%1799) <{offset = array<i64: 18432>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1809 = "earth.constant"() <{rms_var = 0.081491560873915347 : f64, value = 183 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1810 = "earth.mul"(%1808, %1809) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1811 = "earth.add"(%1807, %1810) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1812 = "earth.rotate"(%1799) <{offset = array<i64: 33792>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1813 = "earth.constant"() <{rms_var = 0.11972632129963265 : f64, value = 184 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1814 = "earth.mul"(%1812, %1813) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1815 = "earth.add"(%1811, %1814) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1816 = "earth.rotate"(%1815) <{offset = array<i64: -16384>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1817 = "earth.add"(%1815, %1816) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1818 = "earth.rotate"(%1817) <{offset = array<i64: -32768>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1819 = "earth.add"(%1817, %1818) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1820 = "earth.constant"() <{rms_var = 0.013394939891858844 : f64, value = 185 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1821 = "earth.add"(%1819, %1820) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_0_convbn2
    %1822 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1823 = "earth.mul"(%1822, %1821) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1824 = "earth.mul"(%1823, %1821) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1825 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1826 = "earth.add"(%1824, %1825) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1827 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1828 = "earth.mul"(%1827, %1826) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1829 = "earth.mul"(%1828, %1826) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1830 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1831 = "earth.add"(%1829, %1830) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1832 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1833 = "earth.mul"(%1832, %1831) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1834 = "earth.mul"(%1833, %1831) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1835 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1836 = "earth.add"(%1834, %1835) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1837 = "earth.mul"(%1823, %1826) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1838 = "earth.negate"(%1821) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1839 = "earth.add"(%1837, %1838) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1840 = "earth.mul"(%1823, %1831) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1841 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1842 = "earth.mul"(%1841, %1839) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1843 = "earth.mul"(%1842, %1831) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1844 = "earth.negate"(%1839) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1845 = "earth.add"(%1840, %1844) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1846 = "earth.add"(%1843, %1838) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1847 = "earth.mul"(%1823, %1836) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1848 = "earth.mul"(%1842, %1836) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1849 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1850 = "earth.mul"(%1849, %1845) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1851 = "earth.mul"(%1850, %1836) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1852 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1853 = "earth.mul"(%1852, %1846) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1854 = "earth.mul"(%1853, %1836) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1855 = "earth.negate"(%1846) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1856 = "earth.add"(%1847, %1855) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1857 = "earth.negate"(%1845) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1858 = "earth.add"(%1848, %1857) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1859 = "earth.add"(%1851, %1844) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1860 = "earth.add"(%1854, %1838) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1861 = "earth.constant"() <{rms_var = 0.051379788302527769 : f64, value = 28 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1862 = "earth.mul"(%1861, %1821) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1863 = "earth.constant"() <{rms_var = 0.04272842452341858 : f64, value = 29 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1864 = "earth.mul"(%1863, %1839) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1865 = "earth.add"(%1862, %1864) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1866 = "earth.constant"() <{rms_var = 0.036049430000943392 : f64, value = 30 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1867 = "earth.mul"(%1866, %1845) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1868 = "earth.add"(%1865, %1867) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1869 = "earth.constant"() <{rms_var = 0.030937245302699062 : f64, value = 31 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1870 = "earth.mul"(%1869, %1846) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1871 = "earth.add"(%1868, %1870) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1872 = "earth.constant"() <{rms_var = 0.027115258485962086 : f64, value = 32 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1873 = "earth.mul"(%1872, %1856) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1874 = "earth.add"(%1871, %1873) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1875 = "earth.constant"() <{rms_var = 0.024393077542268389 : f64, value = 33 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1876 = "earth.mul"(%1875, %1858) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1877 = "earth.add"(%1874, %1876) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1878 = "earth.constant"() <{rms_var = 0.022642601074970584 : f64, value = 34 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1879 = "earth.mul"(%1878, %1859) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1880 = "earth.add"(%1877, %1879) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1881 = "earth.constant"() <{rms_var = 0.021831260875010087 : f64, value = 35 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1882 = "earth.mul"(%1881, %1860) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1883 = "earth.add"(%1880, %1882) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1884 = "earth.constant"() <{rms_var = 0.021652089836507502 : f64, value = 36 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1885 = "earth.mul"(%1884, %1821) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1886 = "earth.constant"() <{rms_var = 0.018138222134916806 : f64, value = 37 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1887 = "earth.mul"(%1886, %1839) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1888 = "earth.add"(%1885, %1887) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1889 = "earth.constant"() <{rms_var = 0.01542303411540253 : f64, value = 38 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1890 = "earth.mul"(%1889, %1845) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1891 = "earth.add"(%1888, %1890) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1892 = "earth.constant"() <{rms_var = 0.013305654693862069 : f64, value = 39 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1893 = "earth.mul"(%1892, %1846) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1894 = "earth.add"(%1891, %1893) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1895 = "earth.constant"() <{rms_var = 0.011703046220094507 : f64, value = 40 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1896 = "earth.mul"(%1895, %1856) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1897 = "earth.add"(%1894, %1896) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1898 = "earth.constant"() <{rms_var = 0.010552642814751455 : f64, value = 41 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1899 = "earth.mul"(%1898, %1858) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1900 = "earth.add"(%1897, %1899) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1901 = "earth.constant"() <{rms_var = 0.0098096659804816355 : f64, value = 42 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1902 = "earth.mul"(%1901, %1859) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1903 = "earth.add"(%1900, %1902) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1904 = "earth.constant"() <{rms_var = 0.0094452495559895089 : f64, value = 43 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1905 = "earth.mul"(%1904, %1860) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1906 = "earth.add"(%1903, %1905) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1907 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1908 = "earth.mul"(%1907, %1836) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1909 = "earth.mul"(%1908, %1836) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1910 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1911 = "earth.add"(%1909, %1910) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1912 = "earth.mul"(%1906, %1911) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1913 = "earth.add"(%1912, %1883) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1914 = "earth.constant"() <{rms_var = 0.0042995278292336028 : f64, value = 44 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1915 = "earth.mul"(%1914, %1821) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1916 = "earth.constant"() <{rms_var = 0.0036191919476548503 : f64, value = 45 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1917 = "earth.mul"(%1916, %1839) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1918 = "earth.add"(%1915, %1917) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1919 = "earth.constant"() <{rms_var = 0.0030784419617177587 : f64, value = 46 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1920 = "earth.mul"(%1919, %1845) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1921 = "earth.add"(%1918, %1920) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1922 = "earth.constant"() <{rms_var = 0.0026564062051904268 : f64, value = 47 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1923 = "earth.mul"(%1922, %1846) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1924 = "earth.add"(%1921, %1923) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1925 = "earth.constant"() <{rms_var = 0.0023368004457586414 : f64, value = 48 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1926 = "earth.mul"(%1925, %1856) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1927 = "earth.add"(%1924, %1926) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1928 = "earth.constant"() <{rms_var = 0.002107295415524122 : f64, value = 49 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1929 = "earth.mul"(%1928, %1858) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1930 = "earth.add"(%1927, %1929) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1931 = "earth.constant"() <{rms_var = 0.0019590388891824136 : f64, value = 50 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1932 = "earth.mul"(%1931, %1859) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1933 = "earth.add"(%1930, %1932) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1934 = "earth.constant"() <{rms_var = 0.0018863129851277056 : f64, value = 51 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1935 = "earth.mul"(%1934, %1860) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1936 = "earth.add"(%1933, %1935) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1937 = "earth.constant"() <{rms_var = 0.63615477792235875 : f64, value = 52 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1938 = "earth.mul"(%1937, %1821) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1939 = "earth.constant"() <{rms_var = 0.21189795368271022 : f64, value = 53 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1940 = "earth.mul"(%1939, %1839) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1941 = "earth.add"(%1938, %1940) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1942 = "earth.constant"() <{rms_var = 0.12734215758896472 : f64, value = 54 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1943 = "earth.mul"(%1942, %1845) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1944 = "earth.add"(%1941, %1943) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1945 = "earth.constant"() <{rms_var = 0.091626347972833505 : f64, value = 55 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1946 = "earth.mul"(%1945, %1846) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1947 = "earth.add"(%1944, %1946) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1948 = "earth.constant"() <{rms_var = 0.07255941147658998 : f64, value = 56 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1949 = "earth.mul"(%1948, %1856) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1950 = "earth.add"(%1947, %1949) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1951 = "earth.constant"() <{rms_var = 0.061477909207391525 : f64, value = 57 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1952 = "earth.mul"(%1951, %1858) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1953 = "earth.add"(%1950, %1952) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1954 = "earth.constant"() <{rms_var = 0.055169030070493508 : f64, value = 58 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1955 = "earth.mul"(%1954, %1859) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1956 = "earth.add"(%1953, %1955) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1957 = "earth.constant"() <{rms_var = 0.052275954916496656 : f64, value = 59 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1958 = "earth.mul"(%1957, %1860) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1959 = "earth.add"(%1956, %1958) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1960 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1961 = "earth.mul"(%1960, %1911) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1962 = "earth.mul"(%1961, %1911) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1963 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1964 = "earth.add"(%1962, %1963) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1965 = "earth.mul"(%1936, %1964) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1966 = "earth.add"(%1965, %1913) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1967 = "earth.constant"() <{rms_var = 4.9481895575581374E-4 : f64, value = 60 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1968 = "earth.mul"(%1967, %1821) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1969 = "earth.constant"() <{rms_var = 3.771067298313324E-4 : f64, value = 61 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1970 = "earth.mul"(%1969, %1839) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1971 = "earth.add"(%1968, %1970) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1972 = "earth.constant"() <{rms_var = 3.2076765303085136E-4 : f64, value = 62 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1973 = "earth.mul"(%1972, %1845) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1974 = "earth.add"(%1971, %1973) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1975 = "earth.constant"() <{rms_var = 2.7679532400224407E-4 : f64, value = 63 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1976 = "earth.mul"(%1975, %1846) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1977 = "earth.add"(%1974, %1976) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1978 = "earth.constant"() <{rms_var = 2.4349440690240933E-4 : f64, value = 64 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1979 = "earth.mul"(%1978, %1856) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1980 = "earth.add"(%1977, %1979) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1981 = "earth.constant"() <{rms_var = 2.1958101095650637E-4 : f64, value = 65 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1982 = "earth.mul"(%1981, %1858) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1983 = "earth.add"(%1980, %1982) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1984 = "earth.constant"() <{rms_var = 2.0413318001830463E-4 : f64, value = 66 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1985 = "earth.mul"(%1984, %1859) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1986 = "earth.add"(%1983, %1985) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1987 = "earth.constant"() <{rms_var = 1.9655534146569248E-4 : f64, value = 67 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1988 = "earth.mul"(%1987, %1860) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1989 = "earth.add"(%1986, %1988) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1990 = "earth.constant"() <{rms_var = 1.7735088548126505E-4 : f64, value = 68 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1991 = "earth.mul"(%1990, %1821) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1992 = "earth.constant"() <{rms_var = 1.457794164045128E-4 : f64, value = 69 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1993 = "earth.mul"(%1992, %1839) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1994 = "earth.add"(%1991, %1993) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1995 = "earth.constant"() <{rms_var = 1.1982820692495149E-4 : f64, value = 70 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1996 = "earth.mul"(%1995, %1845) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1997 = "earth.add"(%1994, %1996) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1998 = "earth.constant"() <{rms_var = 9.8496754159654756E-5 : f64, value = 71 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %1999 = "earth.mul"(%1998, %1846) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2000 = "earth.add"(%1997, %1999) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2001 = "earth.constant"() <{rms_var = 8.0962662339626343E-5 : f64, value = 72 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2002 = "earth.mul"(%2001, %1856) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2003 = "earth.add"(%2000, %2002) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2004 = "earth.constant"() <{rms_var = 6.6549936439435966E-5 : f64, value = 73 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2005 = "earth.mul"(%2004, %1858) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2006 = "earth.add"(%2003, %2005) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2007 = "earth.constant"() <{rms_var = 5.4702920205367267E-5 : f64, value = 74 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2008 = "earth.mul"(%2007, %1859) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2009 = "earth.add"(%2006, %2008) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2010 = "earth.constant"() <{rms_var = 1.3863334887734771E-4 : f64, value = 75 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2011 = "earth.mul"(%2010, %1860) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2012 = "earth.add"(%2009, %2011) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2013 = "earth.mul"(%1966, %1911) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2014 = "earth.add"(%2013, %1959) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2015 = "earth.mul"(%2012, %1911) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2016 = "earth.add"(%2015, %1989) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2017 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2018 = "earth.mul"(%2017, %1964) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2019 = "earth.mul"(%2018, %1964) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2020 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2021 = "earth.add"(%2019, %2020) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2022 = "earth.mul"(%2016, %2021) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2023 = "earth.add"(%2022, %2014) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_0_act2_SiLU_poly
    %2024 = "earth.constant"() <{rms_var = 5.000000e-01 : f64, value = 76 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // add[]layer1_0_act2_SiLU_add
    %2025 = "earth.add"(%2023, %2024) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // add[]layer1_0_act2_SiLU_add
    %2026 = "earth.mul"(%1821, %2025) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // mul[]layer1_0_act2_SiLU_mul
    %2027 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 77 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // mul[]layer1_0_act2_SiLU_mul
    %2028 = "earth.add"(%2026, %2027) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // mul[]layer1_0_act2_SiLU_mul
    %2029 = "earth.rotate"(%2028) <{offset = array<i64: -33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2030 = "earth.rotate"(%2028) <{offset = array<i64: -32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2031 = "earth.rotate"(%2028) <{offset = array<i64: -31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2032 = "earth.rotate"(%2028) <{offset = array<i64: -1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2033 = "earth.rotate"(%2028) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2034 = "earth.rotate"(%2028) <{offset = array<i64: 1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2035 = "earth.rotate"(%2028) <{offset = array<i64: 31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2036 = "earth.rotate"(%2028) <{offset = array<i64: 32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2037 = "earth.rotate"(%2028) <{offset = array<i64: 33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2038 = "earth.constant"() <{rms_var = 0.11074391213488095 : f64, value = 186 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2039 = "earth.mul"(%2029, %2038) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2040 = "earth.constant"() <{rms_var = 0.12531980611525312 : f64, value = 187 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2041 = "earth.mul"(%2030, %2040) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2042 = "earth.add"(%2039, %2041) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2043 = "earth.constant"() <{rms_var = 0.12727800869047212 : f64, value = 188 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2044 = "earth.mul"(%2031, %2043) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2045 = "earth.add"(%2042, %2044) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2046 = "earth.constant"() <{rms_var = 0.16246725078110358 : f64, value = 189 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2047 = "earth.mul"(%2032, %2046) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2048 = "earth.add"(%2045, %2047) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2049 = "earth.constant"() <{rms_var = 0.19720234679310286 : f64, value = 190 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2050 = "earth.mul"(%2033, %2049) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2051 = "earth.add"(%2048, %2050) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2052 = "earth.constant"() <{rms_var = 0.21358974026816768 : f64, value = 191 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2053 = "earth.mul"(%2034, %2052) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2054 = "earth.add"(%2051, %2053) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2055 = "earth.constant"() <{rms_var = 0.14161869601714899 : f64, value = 192 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2056 = "earth.mul"(%2035, %2055) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2057 = "earth.add"(%2054, %2056) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2058 = "earth.constant"() <{rms_var = 0.1688633891734998 : f64, value = 193 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2059 = "earth.mul"(%2036, %2058) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2060 = "earth.add"(%2057, %2059) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2061 = "earth.constant"() <{rms_var = 0.14450697967429132 : f64, value = 194 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2062 = "earth.mul"(%2037, %2061) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2063 = "earth.add"(%2060, %2062) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2064 = "earth.rotate"(%2063) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2065 = "earth.add"(%2063, %2064) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2066 = "earth.rotate"(%2065) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2067 = "earth.add"(%2065, %2066) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2068 = "earth.rotate"(%2067) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2069 = "earth.add"(%2067, %2068) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2070 = "earth.rotate"(%2069) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2071 = "earth.add"(%2069, %2070) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2072 = "earth.rotate"(%2071) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2073 = "earth.constant"() <{rms_var = 0.05503514358579506 : f64, value = 195 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2074 = "earth.mul"(%2072, %2073) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2075 = "earth.rotate"(%2071) <{offset = array<i64: 15360>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2076 = "earth.constant"() <{rms_var = 0.059672732560282299 : f64, value = 196 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2077 = "earth.mul"(%2075, %2076) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2078 = "earth.add"(%2074, %2077) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2079 = "earth.rotate"(%2071) <{offset = array<i64: 30720>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2080 = "earth.constant"() <{rms_var = 0.049451371749458524 : f64, value = 197 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2081 = "earth.mul"(%2079, %2080) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2082 = "earth.add"(%2078, %2081) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2083 = "earth.rotate"(%2071) <{offset = array<i64: 46080>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2084 = "earth.constant"() <{rms_var = 0.052206850266010865 : f64, value = 198 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2085 = "earth.mul"(%2083, %2084) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2086 = "earth.add"(%2082, %2085) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2087 = "earth.constant"() <{rms_var = 0.11806969081226031 : f64, value = 199 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2088 = "earth.mul"(%2029, %2087) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2089 = "earth.constant"() <{rms_var = 0.14225027612330113 : f64, value = 200 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2090 = "earth.mul"(%2030, %2089) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2091 = "earth.add"(%2088, %2090) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2092 = "earth.constant"() <{rms_var = 0.13499556024273035 : f64, value = 201 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2093 = "earth.mul"(%2031, %2092) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2094 = "earth.add"(%2091, %2093) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2095 = "earth.constant"() <{rms_var = 0.14777567315747006 : f64, value = 202 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2096 = "earth.mul"(%2032, %2095) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2097 = "earth.add"(%2094, %2096) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2098 = "earth.constant"() <{rms_var = 0.14844842810606607 : f64, value = 203 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2099 = "earth.mul"(%2033, %2098) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2100 = "earth.add"(%2097, %2099) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2101 = "earth.constant"() <{rms_var = 0.13046996345438708 : f64, value = 204 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2102 = "earth.mul"(%2034, %2101) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2103 = "earth.add"(%2100, %2102) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2104 = "earth.constant"() <{rms_var = 0.15362082284638462 : f64, value = 205 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2105 = "earth.mul"(%2035, %2104) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2106 = "earth.add"(%2103, %2105) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2107 = "earth.constant"() <{rms_var = 0.17432211952277288 : f64, value = 206 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2108 = "earth.mul"(%2036, %2107) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2109 = "earth.add"(%2106, %2108) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2110 = "earth.constant"() <{rms_var = 0.13521859277818976 : f64, value = 207 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2111 = "earth.mul"(%2037, %2110) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2112 = "earth.add"(%2109, %2111) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2113 = "earth.rotate"(%2112) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2114 = "earth.add"(%2112, %2113) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2115 = "earth.rotate"(%2114) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2116 = "earth.add"(%2114, %2115) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2117 = "earth.rotate"(%2116) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2118 = "earth.add"(%2116, %2117) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2119 = "earth.rotate"(%2118) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2120 = "earth.add"(%2118, %2119) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2121 = "earth.rotate"(%2120) <{offset = array<i64: -4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2122 = "earth.constant"() <{rms_var = 0.061510149972610811 : f64, value = 208 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2123 = "earth.mul"(%2121, %2122) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2124 = "earth.add"(%2086, %2123) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2125 = "earth.rotate"(%2120) <{offset = array<i64: 11264>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2126 = "earth.constant"() <{rms_var = 0.060640584583549324 : f64, value = 209 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2127 = "earth.mul"(%2125, %2126) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2128 = "earth.add"(%2124, %2127) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2129 = "earth.rotate"(%2120) <{offset = array<i64: 26624>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2130 = "earth.constant"() <{rms_var = 0.067584785511283094 : f64, value = 210 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2131 = "earth.mul"(%2129, %2130) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2132 = "earth.add"(%2128, %2131) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2133 = "earth.rotate"(%2120) <{offset = array<i64: 41984>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2134 = "earth.constant"() <{rms_var = 0.051041149433808129 : f64, value = 211 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2135 = "earth.mul"(%2133, %2134) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2136 = "earth.add"(%2132, %2135) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2137 = "earth.constant"() <{rms_var = 0.12258397002197094 : f64, value = 212 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2138 = "earth.mul"(%2029, %2137) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2139 = "earth.constant"() <{rms_var = 0.13285130212207255 : f64, value = 213 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2140 = "earth.mul"(%2030, %2139) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2141 = "earth.add"(%2138, %2140) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2142 = "earth.constant"() <{rms_var = 0.13440814372168944 : f64, value = 214 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2143 = "earth.mul"(%2031, %2142) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2144 = "earth.add"(%2141, %2143) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2145 = "earth.constant"() <{rms_var = 0.18090916930843137 : f64, value = 215 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2146 = "earth.mul"(%2032, %2145) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2147 = "earth.add"(%2144, %2146) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2148 = "earth.constant"() <{rms_var = 0.14804298810451103 : f64, value = 216 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2149 = "earth.mul"(%2033, %2148) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2150 = "earth.add"(%2147, %2149) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2151 = "earth.constant"() <{rms_var = 0.17896824436378286 : f64, value = 217 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2152 = "earth.mul"(%2034, %2151) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2153 = "earth.add"(%2150, %2152) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2154 = "earth.constant"() <{rms_var = 0.14943921921292369 : f64, value = 218 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2155 = "earth.mul"(%2035, %2154) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2156 = "earth.add"(%2153, %2155) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2157 = "earth.constant"() <{rms_var = 0.16859153168950297 : f64, value = 219 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2158 = "earth.mul"(%2036, %2157) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2159 = "earth.add"(%2156, %2158) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2160 = "earth.constant"() <{rms_var = 0.1444406386666601 : f64, value = 220 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2161 = "earth.mul"(%2037, %2160) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2162 = "earth.add"(%2159, %2161) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2163 = "earth.rotate"(%2162) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2164 = "earth.add"(%2162, %2163) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2165 = "earth.rotate"(%2164) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2166 = "earth.add"(%2164, %2165) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2167 = "earth.rotate"(%2166) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2168 = "earth.add"(%2166, %2167) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2169 = "earth.rotate"(%2168) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2170 = "earth.add"(%2168, %2169) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2171 = "earth.rotate"(%2170) <{offset = array<i64: -8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2172 = "earth.constant"() <{rms_var = 0.050270665428738155 : f64, value = 221 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2173 = "earth.mul"(%2171, %2172) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2174 = "earth.add"(%2136, %2173) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2175 = "earth.rotate"(%2170) <{offset = array<i64: 7168>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2176 = "earth.constant"() <{rms_var = 0.061382353977469682 : f64, value = 222 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2177 = "earth.mul"(%2175, %2176) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2178 = "earth.add"(%2174, %2177) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2179 = "earth.rotate"(%2170) <{offset = array<i64: 22528>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2180 = "earth.constant"() <{rms_var = 0.060214894887922844 : f64, value = 223 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2181 = "earth.mul"(%2179, %2180) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2182 = "earth.add"(%2178, %2181) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2183 = "earth.rotate"(%2170) <{offset = array<i64: 37888>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2184 = "earth.constant"() <{rms_var = 0.063675039968661404 : f64, value = 224 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2185 = "earth.mul"(%2183, %2184) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2186 = "earth.add"(%2182, %2185) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2187 = "earth.constant"() <{rms_var = 0.11690858105919752 : f64, value = 225 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2188 = "earth.mul"(%2029, %2187) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2189 = "earth.constant"() <{rms_var = 0.12009794490499376 : f64, value = 226 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2190 = "earth.mul"(%2030, %2189) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2191 = "earth.add"(%2188, %2190) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2192 = "earth.constant"() <{rms_var = 0.12035741017124957 : f64, value = 227 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2193 = "earth.mul"(%2031, %2192) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2194 = "earth.add"(%2191, %2193) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2195 = "earth.constant"() <{rms_var = 0.14488434538327719 : f64, value = 228 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2196 = "earth.mul"(%2032, %2195) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2197 = "earth.add"(%2194, %2196) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2198 = "earth.constant"() <{rms_var = 0.1661791906877863 : f64, value = 229 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2199 = "earth.mul"(%2033, %2198) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2200 = "earth.add"(%2197, %2199) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2201 = "earth.constant"() <{rms_var = 0.13438316933182604 : f64, value = 230 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2202 = "earth.mul"(%2034, %2201) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2203 = "earth.add"(%2200, %2202) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2204 = "earth.constant"() <{rms_var = 0.10657858881999215 : f64, value = 231 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2205 = "earth.mul"(%2035, %2204) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2206 = "earth.add"(%2203, %2205) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2207 = "earth.constant"() <{rms_var = 0.10079869020840349 : f64, value = 232 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2208 = "earth.mul"(%2036, %2207) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2209 = "earth.add"(%2206, %2208) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2210 = "earth.constant"() <{rms_var = 0.13000328940413028 : f64, value = 233 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2211 = "earth.mul"(%2037, %2210) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2212 = "earth.add"(%2209, %2211) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2213 = "earth.rotate"(%2212) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2214 = "earth.add"(%2212, %2213) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2215 = "earth.rotate"(%2214) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2216 = "earth.add"(%2214, %2215) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2217 = "earth.rotate"(%2216) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2218 = "earth.add"(%2216, %2217) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2219 = "earth.rotate"(%2218) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2220 = "earth.add"(%2218, %2219) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2221 = "earth.rotate"(%2220) <{offset = array<i64: -12288>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2222 = "earth.constant"() <{rms_var = 0.065656412118626814 : f64, value = 234 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2223 = "earth.mul"(%2221, %2222) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2224 = "earth.add"(%2186, %2223) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2225 = "earth.rotate"(%2220) <{offset = array<i64: 3072>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2226 = "earth.constant"() <{rms_var = 0.054284843174942032 : f64, value = 235 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2227 = "earth.mul"(%2225, %2226) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2228 = "earth.add"(%2224, %2227) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2229 = "earth.rotate"(%2220) <{offset = array<i64: 18432>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2230 = "earth.constant"() <{rms_var = 0.070516076141516545 : f64, value = 236 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2231 = "earth.mul"(%2229, %2230) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2232 = "earth.add"(%2228, %2231) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2233 = "earth.rotate"(%2220) <{offset = array<i64: 33792>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2234 = "earth.constant"() <{rms_var = 0.059368293628538418 : f64, value = 237 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2235 = "earth.mul"(%2233, %2234) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2236 = "earth.add"(%2232, %2235) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2237 = "earth.rotate"(%2236) <{offset = array<i64: -16384>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2238 = "earth.add"(%2236, %2237) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2239 = "earth.rotate"(%2238) <{offset = array<i64: -32768>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2240 = "earth.add"(%2238, %2239) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2241 = "earth.constant"() <{rms_var = 0.037479135326294329 : f64, value = 238 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2242 = "earth.add"(%2240, %2241) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn1-0
    %2243 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2244 = "earth.mul"(%2243, %2242) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2245 = "earth.mul"(%2244, %2242) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2246 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2247 = "earth.add"(%2245, %2246) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2248 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2249 = "earth.mul"(%2248, %2247) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2250 = "earth.mul"(%2249, %2247) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2251 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2252 = "earth.add"(%2250, %2251) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2253 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2254 = "earth.mul"(%2253, %2252) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2255 = "earth.mul"(%2254, %2252) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2256 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2257 = "earth.add"(%2255, %2256) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2258 = "earth.mul"(%2244, %2247) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2259 = "earth.negate"(%2242) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2260 = "earth.add"(%2258, %2259) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2261 = "earth.mul"(%2244, %2252) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2262 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2263 = "earth.mul"(%2262, %2260) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2264 = "earth.mul"(%2263, %2252) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2265 = "earth.negate"(%2260) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2266 = "earth.add"(%2261, %2265) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2267 = "earth.add"(%2264, %2259) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2268 = "earth.mul"(%2244, %2257) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2269 = "earth.mul"(%2263, %2257) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2270 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2271 = "earth.mul"(%2270, %2266) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2272 = "earth.mul"(%2271, %2257) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2273 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2274 = "earth.mul"(%2273, %2267) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2275 = "earth.mul"(%2274, %2257) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2276 = "earth.negate"(%2267) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2277 = "earth.add"(%2268, %2276) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2278 = "earth.negate"(%2266) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2279 = "earth.add"(%2269, %2278) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2280 = "earth.add"(%2272, %2265) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2281 = "earth.add"(%2275, %2259) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2282 = "earth.constant"() <{rms_var = 0.051379788302527769 : f64, value = 28 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2283 = "earth.mul"(%2282, %2242) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2284 = "earth.constant"() <{rms_var = 0.04272842452341858 : f64, value = 29 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2285 = "earth.mul"(%2284, %2260) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2286 = "earth.add"(%2283, %2285) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2287 = "earth.constant"() <{rms_var = 0.036049430000943392 : f64, value = 30 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2288 = "earth.mul"(%2287, %2266) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2289 = "earth.add"(%2286, %2288) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2290 = "earth.constant"() <{rms_var = 0.030937245302699062 : f64, value = 31 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2291 = "earth.mul"(%2290, %2267) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2292 = "earth.add"(%2289, %2291) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2293 = "earth.constant"() <{rms_var = 0.027115258485962086 : f64, value = 32 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2294 = "earth.mul"(%2293, %2277) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2295 = "earth.add"(%2292, %2294) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2296 = "earth.constant"() <{rms_var = 0.024393077542268389 : f64, value = 33 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2297 = "earth.mul"(%2296, %2279) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2298 = "earth.add"(%2295, %2297) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2299 = "earth.constant"() <{rms_var = 0.022642601074970584 : f64, value = 34 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2300 = "earth.mul"(%2299, %2280) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2301 = "earth.add"(%2298, %2300) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2302 = "earth.constant"() <{rms_var = 0.021831260875010087 : f64, value = 35 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2303 = "earth.mul"(%2302, %2281) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2304 = "earth.add"(%2301, %2303) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2305 = "earth.constant"() <{rms_var = 0.021652089836507502 : f64, value = 36 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2306 = "earth.mul"(%2305, %2242) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2307 = "earth.constant"() <{rms_var = 0.018138222134916806 : f64, value = 37 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2308 = "earth.mul"(%2307, %2260) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2309 = "earth.add"(%2306, %2308) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2310 = "earth.constant"() <{rms_var = 0.01542303411540253 : f64, value = 38 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2311 = "earth.mul"(%2310, %2266) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2312 = "earth.add"(%2309, %2311) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2313 = "earth.constant"() <{rms_var = 0.013305654693862069 : f64, value = 39 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2314 = "earth.mul"(%2313, %2267) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2315 = "earth.add"(%2312, %2314) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2316 = "earth.constant"() <{rms_var = 0.011703046220094507 : f64, value = 40 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2317 = "earth.mul"(%2316, %2277) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2318 = "earth.add"(%2315, %2317) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2319 = "earth.constant"() <{rms_var = 0.010552642814751455 : f64, value = 41 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2320 = "earth.mul"(%2319, %2279) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2321 = "earth.add"(%2318, %2320) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2322 = "earth.constant"() <{rms_var = 0.0098096659804816355 : f64, value = 42 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2323 = "earth.mul"(%2322, %2280) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2324 = "earth.add"(%2321, %2323) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2325 = "earth.constant"() <{rms_var = 0.0094452495559895089 : f64, value = 43 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2326 = "earth.mul"(%2325, %2281) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2327 = "earth.add"(%2324, %2326) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2328 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2329 = "earth.mul"(%2328, %2257) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2330 = "earth.mul"(%2329, %2257) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2331 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2332 = "earth.add"(%2330, %2331) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2333 = "earth.mul"(%2327, %2332) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2334 = "earth.add"(%2333, %2304) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2335 = "earth.constant"() <{rms_var = 0.0042995278292336028 : f64, value = 44 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2336 = "earth.mul"(%2335, %2242) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2337 = "earth.constant"() <{rms_var = 0.0036191919476548503 : f64, value = 45 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2338 = "earth.mul"(%2337, %2260) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2339 = "earth.add"(%2336, %2338) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2340 = "earth.constant"() <{rms_var = 0.0030784419617177587 : f64, value = 46 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2341 = "earth.mul"(%2340, %2266) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2342 = "earth.add"(%2339, %2341) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2343 = "earth.constant"() <{rms_var = 0.0026564062051904268 : f64, value = 47 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2344 = "earth.mul"(%2343, %2267) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2345 = "earth.add"(%2342, %2344) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2346 = "earth.constant"() <{rms_var = 0.0023368004457586414 : f64, value = 48 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2347 = "earth.mul"(%2346, %2277) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2348 = "earth.add"(%2345, %2347) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2349 = "earth.constant"() <{rms_var = 0.002107295415524122 : f64, value = 49 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2350 = "earth.mul"(%2349, %2279) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2351 = "earth.add"(%2348, %2350) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2352 = "earth.constant"() <{rms_var = 0.0019590388891824136 : f64, value = 50 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2353 = "earth.mul"(%2352, %2280) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2354 = "earth.add"(%2351, %2353) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2355 = "earth.constant"() <{rms_var = 0.0018863129851277056 : f64, value = 51 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2356 = "earth.mul"(%2355, %2281) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2357 = "earth.add"(%2354, %2356) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2358 = "earth.constant"() <{rms_var = 0.63615477792235875 : f64, value = 52 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2359 = "earth.mul"(%2358, %2242) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2360 = "earth.constant"() <{rms_var = 0.21189795368271022 : f64, value = 53 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2361 = "earth.mul"(%2360, %2260) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2362 = "earth.add"(%2359, %2361) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2363 = "earth.constant"() <{rms_var = 0.12734215758896472 : f64, value = 54 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2364 = "earth.mul"(%2363, %2266) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2365 = "earth.add"(%2362, %2364) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2366 = "earth.constant"() <{rms_var = 0.091626347972833505 : f64, value = 55 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2367 = "earth.mul"(%2366, %2267) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2368 = "earth.add"(%2365, %2367) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2369 = "earth.constant"() <{rms_var = 0.07255941147658998 : f64, value = 56 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2370 = "earth.mul"(%2369, %2277) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2371 = "earth.add"(%2368, %2370) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2372 = "earth.constant"() <{rms_var = 0.061477909207391525 : f64, value = 57 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2373 = "earth.mul"(%2372, %2279) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2374 = "earth.add"(%2371, %2373) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2375 = "earth.constant"() <{rms_var = 0.055169030070493508 : f64, value = 58 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2376 = "earth.mul"(%2375, %2280) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2377 = "earth.add"(%2374, %2376) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2378 = "earth.constant"() <{rms_var = 0.052275954916496656 : f64, value = 59 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2379 = "earth.mul"(%2378, %2281) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2380 = "earth.add"(%2377, %2379) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2381 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2382 = "earth.mul"(%2381, %2332) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2383 = "earth.mul"(%2382, %2332) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2384 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2385 = "earth.add"(%2383, %2384) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2386 = "earth.mul"(%2357, %2385) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2387 = "earth.add"(%2386, %2334) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2388 = "earth.constant"() <{rms_var = 4.9481895575581374E-4 : f64, value = 60 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2389 = "earth.mul"(%2388, %2242) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2390 = "earth.constant"() <{rms_var = 3.771067298313324E-4 : f64, value = 61 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2391 = "earth.mul"(%2390, %2260) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2392 = "earth.add"(%2389, %2391) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2393 = "earth.constant"() <{rms_var = 3.2076765303085136E-4 : f64, value = 62 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2394 = "earth.mul"(%2393, %2266) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2395 = "earth.add"(%2392, %2394) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2396 = "earth.constant"() <{rms_var = 2.7679532400224407E-4 : f64, value = 63 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2397 = "earth.mul"(%2396, %2267) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2398 = "earth.add"(%2395, %2397) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2399 = "earth.constant"() <{rms_var = 2.4349440690240933E-4 : f64, value = 64 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2400 = "earth.mul"(%2399, %2277) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2401 = "earth.add"(%2398, %2400) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2402 = "earth.constant"() <{rms_var = 2.1958101095650637E-4 : f64, value = 65 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2403 = "earth.mul"(%2402, %2279) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2404 = "earth.add"(%2401, %2403) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2405 = "earth.constant"() <{rms_var = 2.0413318001830463E-4 : f64, value = 66 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2406 = "earth.mul"(%2405, %2280) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2407 = "earth.add"(%2404, %2406) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2408 = "earth.constant"() <{rms_var = 1.9655534146569248E-4 : f64, value = 67 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2409 = "earth.mul"(%2408, %2281) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2410 = "earth.add"(%2407, %2409) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2411 = "earth.constant"() <{rms_var = 1.7735088548126505E-4 : f64, value = 68 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2412 = "earth.mul"(%2411, %2242) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2413 = "earth.constant"() <{rms_var = 1.457794164045128E-4 : f64, value = 69 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2414 = "earth.mul"(%2413, %2260) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2415 = "earth.add"(%2412, %2414) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2416 = "earth.constant"() <{rms_var = 1.1982820692495149E-4 : f64, value = 70 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2417 = "earth.mul"(%2416, %2266) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2418 = "earth.add"(%2415, %2417) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2419 = "earth.constant"() <{rms_var = 9.8496754159654756E-5 : f64, value = 71 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2420 = "earth.mul"(%2419, %2267) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2421 = "earth.add"(%2418, %2420) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2422 = "earth.constant"() <{rms_var = 8.0962662339626343E-5 : f64, value = 72 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2423 = "earth.mul"(%2422, %2277) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2424 = "earth.add"(%2421, %2423) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2425 = "earth.constant"() <{rms_var = 6.6549936439435966E-5 : f64, value = 73 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2426 = "earth.mul"(%2425, %2279) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2427 = "earth.add"(%2424, %2426) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2428 = "earth.constant"() <{rms_var = 5.4702920205367267E-5 : f64, value = 74 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2429 = "earth.mul"(%2428, %2280) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2430 = "earth.add"(%2427, %2429) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2431 = "earth.constant"() <{rms_var = 1.3863334887734771E-4 : f64, value = 75 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2432 = "earth.mul"(%2431, %2281) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2433 = "earth.add"(%2430, %2432) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2434 = "earth.mul"(%2387, %2332) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2435 = "earth.add"(%2434, %2380) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2436 = "earth.mul"(%2433, %2332) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2437 = "earth.add"(%2436, %2410) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2438 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2439 = "earth.mul"(%2438, %2385) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2440 = "earth.mul"(%2439, %2385) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2441 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2442 = "earth.add"(%2440, %2441) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2443 = "earth.mul"(%2437, %2442) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2444 = "earth.add"(%2443, %2435) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act1_SiLU_poly
    %2445 = "earth.constant"() <{rms_var = 5.000000e-01 : f64, value = 76 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // add[]layer1_1_act1_SiLU_add
    %2446 = "earth.add"(%2444, %2445) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // add[]layer1_1_act1_SiLU_add
    %2447 = "earth.mul"(%2242, %2446) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // mul[]layer1_1_act1_SiLU_mul
    %2448 = "earth.rotate"(%2447) <{offset = array<i64: -33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2449 = "earth.rotate"(%2447) <{offset = array<i64: -32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2450 = "earth.rotate"(%2447) <{offset = array<i64: -31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2451 = "earth.rotate"(%2447) <{offset = array<i64: -1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2452 = "earth.rotate"(%2447) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2453 = "earth.rotate"(%2447) <{offset = array<i64: 1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2454 = "earth.rotate"(%2447) <{offset = array<i64: 31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2455 = "earth.rotate"(%2447) <{offset = array<i64: 32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2456 = "earth.rotate"(%2447) <{offset = array<i64: 33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2457 = "earth.constant"() <{rms_var = 0.11165627691863619 : f64, value = 239 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2458 = "earth.mul"(%2448, %2457) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2459 = "earth.constant"() <{rms_var = 0.13517601435902757 : f64, value = 240 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2460 = "earth.mul"(%2449, %2459) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2461 = "earth.add"(%2458, %2460) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2462 = "earth.constant"() <{rms_var = 0.08784659578456501 : f64, value = 241 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2463 = "earth.mul"(%2450, %2462) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2464 = "earth.add"(%2461, %2463) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2465 = "earth.constant"() <{rms_var = 0.1542799232137515 : f64, value = 242 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2466 = "earth.mul"(%2451, %2465) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2467 = "earth.add"(%2464, %2466) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2468 = "earth.constant"() <{rms_var = 0.19772936555363327 : f64, value = 243 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2469 = "earth.mul"(%2452, %2468) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2470 = "earth.add"(%2467, %2469) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2471 = "earth.constant"() <{rms_var = 0.14770151156816236 : f64, value = 244 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2472 = "earth.mul"(%2453, %2471) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2473 = "earth.add"(%2470, %2472) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2474 = "earth.constant"() <{rms_var = 0.13386063116336655 : f64, value = 245 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2475 = "earth.mul"(%2454, %2474) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2476 = "earth.add"(%2473, %2475) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2477 = "earth.constant"() <{rms_var = 0.14687374760261146 : f64, value = 246 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2478 = "earth.mul"(%2455, %2477) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2479 = "earth.add"(%2476, %2478) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2480 = "earth.constant"() <{rms_var = 0.11631272117565154 : f64, value = 247 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2481 = "earth.mul"(%2456, %2480) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2482 = "earth.add"(%2479, %2481) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2483 = "earth.rotate"(%2482) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2484 = "earth.add"(%2482, %2483) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2485 = "earth.rotate"(%2484) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2486 = "earth.add"(%2484, %2485) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2487 = "earth.rotate"(%2486) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2488 = "earth.add"(%2486, %2487) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2489 = "earth.rotate"(%2488) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2490 = "earth.add"(%2488, %2489) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2491 = "earth.rotate"(%2490) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2492 = "earth.constant"() <{rms_var = 0.12682078782903164 : f64, value = 248 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2493 = "earth.mul"(%2491, %2492) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2494 = "earth.rotate"(%2490) <{offset = array<i64: 15360>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2495 = "earth.constant"() <{rms_var = 0.065500666600071575 : f64, value = 249 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2496 = "earth.mul"(%2494, %2495) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2497 = "earth.add"(%2493, %2496) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2498 = "earth.rotate"(%2490) <{offset = array<i64: 30720>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2499 = "earth.constant"() <{rms_var = 0.10689998610976181 : f64, value = 250 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2500 = "earth.mul"(%2498, %2499) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2501 = "earth.add"(%2497, %2500) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2502 = "earth.rotate"(%2490) <{offset = array<i64: 46080>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2503 = "earth.constant"() <{rms_var = 0.075088038168589527 : f64, value = 251 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2504 = "earth.mul"(%2502, %2503) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2505 = "earth.add"(%2501, %2504) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2506 = "earth.constant"() <{rms_var = 0.10325167757908202 : f64, value = 252 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2507 = "earth.mul"(%2448, %2506) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2508 = "earth.constant"() <{rms_var = 0.098530254489669886 : f64, value = 253 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2509 = "earth.mul"(%2449, %2508) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2510 = "earth.add"(%2507, %2509) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2511 = "earth.constant"() <{rms_var = 0.12298830891566782 : f64, value = 254 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2512 = "earth.mul"(%2450, %2511) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2513 = "earth.add"(%2510, %2512) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2514 = "earth.constant"() <{rms_var = 0.12845797142845838 : f64, value = 255 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2515 = "earth.mul"(%2451, %2514) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2516 = "earth.add"(%2513, %2515) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2517 = "earth.constant"() <{rms_var = 0.14024566580046546 : f64, value = 256 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2518 = "earth.mul"(%2452, %2517) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2519 = "earth.add"(%2516, %2518) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2520 = "earth.constant"() <{rms_var = 0.18036207393241649 : f64, value = 257 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2521 = "earth.mul"(%2453, %2520) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2522 = "earth.add"(%2519, %2521) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2523 = "earth.constant"() <{rms_var = 0.10841442061565071 : f64, value = 258 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2524 = "earth.mul"(%2454, %2523) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2525 = "earth.add"(%2522, %2524) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2526 = "earth.constant"() <{rms_var = 0.11451646290416842 : f64, value = 259 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2527 = "earth.mul"(%2455, %2526) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2528 = "earth.add"(%2525, %2527) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2529 = "earth.constant"() <{rms_var = 0.13504335885341112 : f64, value = 260 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2530 = "earth.mul"(%2456, %2529) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2531 = "earth.add"(%2528, %2530) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2532 = "earth.rotate"(%2531) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2533 = "earth.add"(%2531, %2532) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2534 = "earth.rotate"(%2533) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2535 = "earth.add"(%2533, %2534) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2536 = "earth.rotate"(%2535) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2537 = "earth.add"(%2535, %2536) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2538 = "earth.rotate"(%2537) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2539 = "earth.add"(%2537, %2538) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2540 = "earth.rotate"(%2539) <{offset = array<i64: -4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2541 = "earth.constant"() <{rms_var = 0.12955883655503089 : f64, value = 261 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2542 = "earth.mul"(%2540, %2541) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2543 = "earth.add"(%2505, %2542) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2544 = "earth.rotate"(%2539) <{offset = array<i64: 11264>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2545 = "earth.constant"() <{rms_var = 0.057932548615650045 : f64, value = 262 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2546 = "earth.mul"(%2544, %2545) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2547 = "earth.add"(%2543, %2546) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2548 = "earth.rotate"(%2539) <{offset = array<i64: 26624>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2549 = "earth.constant"() <{rms_var = 0.051521288896764363 : f64, value = 263 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2550 = "earth.mul"(%2548, %2549) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2551 = "earth.add"(%2547, %2550) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2552 = "earth.rotate"(%2539) <{offset = array<i64: 41984>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2553 = "earth.constant"() <{rms_var = 0.095239249812067189 : f64, value = 264 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2554 = "earth.mul"(%2552, %2553) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2555 = "earth.add"(%2551, %2554) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2556 = "earth.constant"() <{rms_var = 0.1310655465952579 : f64, value = 265 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2557 = "earth.mul"(%2448, %2556) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2558 = "earth.constant"() <{rms_var = 0.16166267991645517 : f64, value = 266 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2559 = "earth.mul"(%2449, %2558) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2560 = "earth.add"(%2557, %2559) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2561 = "earth.constant"() <{rms_var = 0.099655687199722912 : f64, value = 267 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2562 = "earth.mul"(%2450, %2561) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2563 = "earth.add"(%2560, %2562) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2564 = "earth.constant"() <{rms_var = 0.15397700539555453 : f64, value = 268 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2565 = "earth.mul"(%2451, %2564) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2566 = "earth.add"(%2563, %2565) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2567 = "earth.constant"() <{rms_var = 0.21042602824192885 : f64, value = 269 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2568 = "earth.mul"(%2452, %2567) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2569 = "earth.add"(%2566, %2568) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2570 = "earth.constant"() <{rms_var = 0.14834518138129604 : f64, value = 270 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2571 = "earth.mul"(%2453, %2570) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2572 = "earth.add"(%2569, %2571) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2573 = "earth.constant"() <{rms_var = 0.10793938026179398 : f64, value = 271 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2574 = "earth.mul"(%2454, %2573) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2575 = "earth.add"(%2572, %2574) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2576 = "earth.constant"() <{rms_var = 0.14399856079232731 : f64, value = 272 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2577 = "earth.mul"(%2455, %2576) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2578 = "earth.add"(%2575, %2577) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2579 = "earth.constant"() <{rms_var = 0.10439116293866986 : f64, value = 273 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2580 = "earth.mul"(%2456, %2579) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2581 = "earth.add"(%2578, %2580) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2582 = "earth.rotate"(%2581) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2583 = "earth.add"(%2581, %2582) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2584 = "earth.rotate"(%2583) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2585 = "earth.add"(%2583, %2584) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2586 = "earth.rotate"(%2585) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2587 = "earth.add"(%2585, %2586) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2588 = "earth.rotate"(%2587) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2589 = "earth.add"(%2587, %2588) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2590 = "earth.rotate"(%2589) <{offset = array<i64: -8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2591 = "earth.constant"() <{rms_var = 0.060083786653052862 : f64, value = 274 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2592 = "earth.mul"(%2590, %2591) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2593 = "earth.add"(%2555, %2592) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2594 = "earth.rotate"(%2589) <{offset = array<i64: 7168>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2595 = "earth.constant"() <{rms_var = 0.10861164518030124 : f64, value = 275 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2596 = "earth.mul"(%2594, %2595) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2597 = "earth.add"(%2593, %2596) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2598 = "earth.rotate"(%2589) <{offset = array<i64: 22528>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2599 = "earth.constant"() <{rms_var = 0.08100422885802691 : f64, value = 276 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2600 = "earth.mul"(%2598, %2599) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2601 = "earth.add"(%2597, %2600) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2602 = "earth.rotate"(%2589) <{offset = array<i64: 37888>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2603 = "earth.constant"() <{rms_var = 0.090070441250369615 : f64, value = 277 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2604 = "earth.mul"(%2602, %2603) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2605 = "earth.add"(%2601, %2604) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2606 = "earth.constant"() <{rms_var = 0.096536313802035478 : f64, value = 278 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2607 = "earth.mul"(%2448, %2606) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2608 = "earth.constant"() <{rms_var = 0.16084806770243959 : f64, value = 279 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2609 = "earth.mul"(%2449, %2608) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2610 = "earth.add"(%2607, %2609) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2611 = "earth.constant"() <{rms_var = 0.12057511574037012 : f64, value = 280 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2612 = "earth.mul"(%2450, %2611) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2613 = "earth.add"(%2610, %2612) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2614 = "earth.constant"() <{rms_var = 0.13540781861338488 : f64, value = 281 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2615 = "earth.mul"(%2451, %2614) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2616 = "earth.add"(%2613, %2615) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2617 = "earth.constant"() <{rms_var = 0.17316350925324017 : f64, value = 282 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2618 = "earth.mul"(%2452, %2617) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2619 = "earth.add"(%2616, %2618) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2620 = "earth.constant"() <{rms_var = 0.13409522209285124 : f64, value = 283 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2621 = "earth.mul"(%2453, %2620) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2622 = "earth.add"(%2619, %2621) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2623 = "earth.constant"() <{rms_var = 0.11947104655348903 : f64, value = 284 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2624 = "earth.mul"(%2454, %2623) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2625 = "earth.add"(%2622, %2624) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2626 = "earth.constant"() <{rms_var = 0.12584344888021384 : f64, value = 285 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2627 = "earth.mul"(%2455, %2626) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2628 = "earth.add"(%2625, %2627) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2629 = "earth.constant"() <{rms_var = 0.10810235432578488 : f64, value = 286 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2630 = "earth.mul"(%2456, %2629) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2631 = "earth.add"(%2628, %2630) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2632 = "earth.rotate"(%2631) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2633 = "earth.add"(%2631, %2632) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2634 = "earth.rotate"(%2633) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2635 = "earth.add"(%2633, %2634) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2636 = "earth.rotate"(%2635) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2637 = "earth.add"(%2635, %2636) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2638 = "earth.rotate"(%2637) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2639 = "earth.add"(%2637, %2638) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2640 = "earth.rotate"(%2639) <{offset = array<i64: -12288>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2641 = "earth.constant"() <{rms_var = 0.1003152811724233 : f64, value = 287 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2642 = "earth.mul"(%2640, %2641) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2643 = "earth.add"(%2605, %2642) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2644 = "earth.rotate"(%2639) <{offset = array<i64: 3072>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2645 = "earth.constant"() <{rms_var = 0.12447880504705401 : f64, value = 288 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2646 = "earth.mul"(%2644, %2645) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2647 = "earth.add"(%2643, %2646) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2648 = "earth.rotate"(%2639) <{offset = array<i64: 18432>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2649 = "earth.constant"() <{rms_var = 0.075559916937942354 : f64, value = 289 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2650 = "earth.mul"(%2648, %2649) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2651 = "earth.add"(%2647, %2650) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2652 = "earth.rotate"(%2639) <{offset = array<i64: 33792>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2653 = "earth.constant"() <{rms_var = 0.082042427726752118 : f64, value = 290 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2654 = "earth.mul"(%2652, %2653) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2655 = "earth.add"(%2651, %2654) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2656 = "earth.rotate"(%2655) <{offset = array<i64: -16384>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2657 = "earth.add"(%2655, %2656) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2658 = "earth.rotate"(%2657) <{offset = array<i64: -32768>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2659 = "earth.add"(%2657, %2658) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2660 = "earth.constant"() <{rms_var = 0.013153931678559818 : f64, value = 291 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2661 = "earth.add"(%2659, %2660) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_1_convbn2
    %2662 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2663 = "earth.mul"(%2662, %2661) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2664 = "earth.mul"(%2663, %2661) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2665 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2666 = "earth.add"(%2664, %2665) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2667 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2668 = "earth.mul"(%2667, %2666) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2669 = "earth.mul"(%2668, %2666) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2670 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2671 = "earth.add"(%2669, %2670) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2672 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2673 = "earth.mul"(%2672, %2671) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2674 = "earth.mul"(%2673, %2671) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2675 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2676 = "earth.add"(%2674, %2675) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2677 = "earth.mul"(%2663, %2666) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2678 = "earth.negate"(%2661) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2679 = "earth.add"(%2677, %2678) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2680 = "earth.mul"(%2663, %2671) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2681 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2682 = "earth.mul"(%2681, %2679) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2683 = "earth.mul"(%2682, %2671) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2684 = "earth.negate"(%2679) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2685 = "earth.add"(%2680, %2684) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2686 = "earth.add"(%2683, %2678) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2687 = "earth.mul"(%2663, %2676) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2688 = "earth.mul"(%2682, %2676) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2689 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2690 = "earth.mul"(%2689, %2685) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2691 = "earth.mul"(%2690, %2676) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2692 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2693 = "earth.mul"(%2692, %2686) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2694 = "earth.mul"(%2693, %2676) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2695 = "earth.negate"(%2686) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2696 = "earth.add"(%2687, %2695) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2697 = "earth.negate"(%2685) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2698 = "earth.add"(%2688, %2697) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2699 = "earth.add"(%2691, %2684) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2700 = "earth.add"(%2694, %2678) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2701 = "earth.constant"() <{rms_var = 0.051379788302527769 : f64, value = 28 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2702 = "earth.mul"(%2701, %2661) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2703 = "earth.constant"() <{rms_var = 0.04272842452341858 : f64, value = 29 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2704 = "earth.mul"(%2703, %2679) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2705 = "earth.add"(%2702, %2704) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2706 = "earth.constant"() <{rms_var = 0.036049430000943392 : f64, value = 30 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2707 = "earth.mul"(%2706, %2685) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2708 = "earth.add"(%2705, %2707) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2709 = "earth.constant"() <{rms_var = 0.030937245302699062 : f64, value = 31 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2710 = "earth.mul"(%2709, %2686) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2711 = "earth.add"(%2708, %2710) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2712 = "earth.constant"() <{rms_var = 0.027115258485962086 : f64, value = 32 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2713 = "earth.mul"(%2712, %2696) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2714 = "earth.add"(%2711, %2713) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2715 = "earth.constant"() <{rms_var = 0.024393077542268389 : f64, value = 33 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2716 = "earth.mul"(%2715, %2698) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2717 = "earth.add"(%2714, %2716) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2718 = "earth.constant"() <{rms_var = 0.022642601074970584 : f64, value = 34 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2719 = "earth.mul"(%2718, %2699) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2720 = "earth.add"(%2717, %2719) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2721 = "earth.constant"() <{rms_var = 0.021831260875010087 : f64, value = 35 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2722 = "earth.mul"(%2721, %2700) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2723 = "earth.add"(%2720, %2722) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2724 = "earth.constant"() <{rms_var = 0.021652089836507502 : f64, value = 36 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2725 = "earth.mul"(%2724, %2661) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2726 = "earth.constant"() <{rms_var = 0.018138222134916806 : f64, value = 37 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2727 = "earth.mul"(%2726, %2679) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2728 = "earth.add"(%2725, %2727) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2729 = "earth.constant"() <{rms_var = 0.01542303411540253 : f64, value = 38 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2730 = "earth.mul"(%2729, %2685) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2731 = "earth.add"(%2728, %2730) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2732 = "earth.constant"() <{rms_var = 0.013305654693862069 : f64, value = 39 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2733 = "earth.mul"(%2732, %2686) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2734 = "earth.add"(%2731, %2733) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2735 = "earth.constant"() <{rms_var = 0.011703046220094507 : f64, value = 40 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2736 = "earth.mul"(%2735, %2696) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2737 = "earth.add"(%2734, %2736) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2738 = "earth.constant"() <{rms_var = 0.010552642814751455 : f64, value = 41 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2739 = "earth.mul"(%2738, %2698) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2740 = "earth.add"(%2737, %2739) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2741 = "earth.constant"() <{rms_var = 0.0098096659804816355 : f64, value = 42 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2742 = "earth.mul"(%2741, %2699) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2743 = "earth.add"(%2740, %2742) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2744 = "earth.constant"() <{rms_var = 0.0094452495559895089 : f64, value = 43 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2745 = "earth.mul"(%2744, %2700) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2746 = "earth.add"(%2743, %2745) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2747 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2748 = "earth.mul"(%2747, %2676) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2749 = "earth.mul"(%2748, %2676) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2750 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2751 = "earth.add"(%2749, %2750) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2752 = "earth.mul"(%2746, %2751) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2753 = "earth.add"(%2752, %2723) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2754 = "earth.constant"() <{rms_var = 0.0042995278292336028 : f64, value = 44 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2755 = "earth.mul"(%2754, %2661) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2756 = "earth.constant"() <{rms_var = 0.0036191919476548503 : f64, value = 45 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2757 = "earth.mul"(%2756, %2679) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2758 = "earth.add"(%2755, %2757) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2759 = "earth.constant"() <{rms_var = 0.0030784419617177587 : f64, value = 46 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2760 = "earth.mul"(%2759, %2685) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2761 = "earth.add"(%2758, %2760) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2762 = "earth.constant"() <{rms_var = 0.0026564062051904268 : f64, value = 47 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2763 = "earth.mul"(%2762, %2686) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2764 = "earth.add"(%2761, %2763) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2765 = "earth.constant"() <{rms_var = 0.0023368004457586414 : f64, value = 48 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2766 = "earth.mul"(%2765, %2696) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2767 = "earth.add"(%2764, %2766) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2768 = "earth.constant"() <{rms_var = 0.002107295415524122 : f64, value = 49 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2769 = "earth.mul"(%2768, %2698) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2770 = "earth.add"(%2767, %2769) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2771 = "earth.constant"() <{rms_var = 0.0019590388891824136 : f64, value = 50 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2772 = "earth.mul"(%2771, %2699) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2773 = "earth.add"(%2770, %2772) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2774 = "earth.constant"() <{rms_var = 0.0018863129851277056 : f64, value = 51 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2775 = "earth.mul"(%2774, %2700) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2776 = "earth.add"(%2773, %2775) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2777 = "earth.constant"() <{rms_var = 0.63615477792235875 : f64, value = 52 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2778 = "earth.mul"(%2777, %2661) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2779 = "earth.constant"() <{rms_var = 0.21189795368271022 : f64, value = 53 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2780 = "earth.mul"(%2779, %2679) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2781 = "earth.add"(%2778, %2780) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2782 = "earth.constant"() <{rms_var = 0.12734215758896472 : f64, value = 54 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2783 = "earth.mul"(%2782, %2685) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2784 = "earth.add"(%2781, %2783) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2785 = "earth.constant"() <{rms_var = 0.091626347972833505 : f64, value = 55 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2786 = "earth.mul"(%2785, %2686) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2787 = "earth.add"(%2784, %2786) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2788 = "earth.constant"() <{rms_var = 0.07255941147658998 : f64, value = 56 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2789 = "earth.mul"(%2788, %2696) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2790 = "earth.add"(%2787, %2789) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2791 = "earth.constant"() <{rms_var = 0.061477909207391525 : f64, value = 57 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2792 = "earth.mul"(%2791, %2698) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2793 = "earth.add"(%2790, %2792) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2794 = "earth.constant"() <{rms_var = 0.055169030070493508 : f64, value = 58 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2795 = "earth.mul"(%2794, %2699) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2796 = "earth.add"(%2793, %2795) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2797 = "earth.constant"() <{rms_var = 0.052275954916496656 : f64, value = 59 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2798 = "earth.mul"(%2797, %2700) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2799 = "earth.add"(%2796, %2798) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2800 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2801 = "earth.mul"(%2800, %2751) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2802 = "earth.mul"(%2801, %2751) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2803 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2804 = "earth.add"(%2802, %2803) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2805 = "earth.mul"(%2776, %2804) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2806 = "earth.add"(%2805, %2753) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2807 = "earth.constant"() <{rms_var = 4.9481895575581374E-4 : f64, value = 60 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2808 = "earth.mul"(%2807, %2661) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2809 = "earth.constant"() <{rms_var = 3.771067298313324E-4 : f64, value = 61 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2810 = "earth.mul"(%2809, %2679) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2811 = "earth.add"(%2808, %2810) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2812 = "earth.constant"() <{rms_var = 3.2076765303085136E-4 : f64, value = 62 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2813 = "earth.mul"(%2812, %2685) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2814 = "earth.add"(%2811, %2813) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2815 = "earth.constant"() <{rms_var = 2.7679532400224407E-4 : f64, value = 63 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2816 = "earth.mul"(%2815, %2686) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2817 = "earth.add"(%2814, %2816) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2818 = "earth.constant"() <{rms_var = 2.4349440690240933E-4 : f64, value = 64 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2819 = "earth.mul"(%2818, %2696) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2820 = "earth.add"(%2817, %2819) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2821 = "earth.constant"() <{rms_var = 2.1958101095650637E-4 : f64, value = 65 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2822 = "earth.mul"(%2821, %2698) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2823 = "earth.add"(%2820, %2822) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2824 = "earth.constant"() <{rms_var = 2.0413318001830463E-4 : f64, value = 66 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2825 = "earth.mul"(%2824, %2699) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2826 = "earth.add"(%2823, %2825) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2827 = "earth.constant"() <{rms_var = 1.9655534146569248E-4 : f64, value = 67 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2828 = "earth.mul"(%2827, %2700) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2829 = "earth.add"(%2826, %2828) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2830 = "earth.constant"() <{rms_var = 1.7735088548126505E-4 : f64, value = 68 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2831 = "earth.mul"(%2830, %2661) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2832 = "earth.constant"() <{rms_var = 1.457794164045128E-4 : f64, value = 69 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2833 = "earth.mul"(%2832, %2679) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2834 = "earth.add"(%2831, %2833) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2835 = "earth.constant"() <{rms_var = 1.1982820692495149E-4 : f64, value = 70 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2836 = "earth.mul"(%2835, %2685) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2837 = "earth.add"(%2834, %2836) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2838 = "earth.constant"() <{rms_var = 9.8496754159654756E-5 : f64, value = 71 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2839 = "earth.mul"(%2838, %2686) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2840 = "earth.add"(%2837, %2839) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2841 = "earth.constant"() <{rms_var = 8.0962662339626343E-5 : f64, value = 72 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2842 = "earth.mul"(%2841, %2696) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2843 = "earth.add"(%2840, %2842) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2844 = "earth.constant"() <{rms_var = 6.6549936439435966E-5 : f64, value = 73 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2845 = "earth.mul"(%2844, %2698) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2846 = "earth.add"(%2843, %2845) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2847 = "earth.constant"() <{rms_var = 5.4702920205367267E-5 : f64, value = 74 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2848 = "earth.mul"(%2847, %2699) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2849 = "earth.add"(%2846, %2848) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2850 = "earth.constant"() <{rms_var = 1.3863334887734771E-4 : f64, value = 75 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2851 = "earth.mul"(%2850, %2700) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2852 = "earth.add"(%2849, %2851) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2853 = "earth.mul"(%2806, %2751) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2854 = "earth.add"(%2853, %2799) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2855 = "earth.mul"(%2852, %2751) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2856 = "earth.add"(%2855, %2829) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2857 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2858 = "earth.mul"(%2857, %2804) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2859 = "earth.mul"(%2858, %2804) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2860 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2861 = "earth.add"(%2859, %2860) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2862 = "earth.mul"(%2856, %2861) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2863 = "earth.add"(%2862, %2854) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_1_act2_SiLU_poly
    %2864 = "earth.constant"() <{rms_var = 5.000000e-01 : f64, value = 76 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // add[]layer1_1_act2_SiLU_add
    %2865 = "earth.add"(%2863, %2864) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // add[]layer1_1_act2_SiLU_add
    %2866 = "earth.mul"(%2661, %2865) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // mul[]layer1_1_act2_SiLU_mul
    %2867 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 77 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // mul[]layer1_1_act2_SiLU_mul
    %2868 = "earth.add"(%2866, %2867) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // mul[]layer1_1_act2_SiLU_mul
    %2869 = "earth.rotate"(%2868) <{offset = array<i64: -33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2870 = "earth.rotate"(%2868) <{offset = array<i64: -32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2871 = "earth.rotate"(%2868) <{offset = array<i64: -31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2872 = "earth.rotate"(%2868) <{offset = array<i64: -1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2873 = "earth.rotate"(%2868) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2874 = "earth.rotate"(%2868) <{offset = array<i64: 1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2875 = "earth.rotate"(%2868) <{offset = array<i64: 31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2876 = "earth.rotate"(%2868) <{offset = array<i64: 32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2877 = "earth.rotate"(%2868) <{offset = array<i64: 33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2878 = "earth.constant"() <{rms_var = 0.1448343515281629 : f64, value = 292 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2879 = "earth.mul"(%2869, %2878) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2880 = "earth.constant"() <{rms_var = 0.17253677620124233 : f64, value = 293 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2881 = "earth.mul"(%2870, %2880) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2882 = "earth.add"(%2879, %2881) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2883 = "earth.constant"() <{rms_var = 0.13737079800859539 : f64, value = 294 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2884 = "earth.mul"(%2871, %2883) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2885 = "earth.add"(%2882, %2884) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2886 = "earth.constant"() <{rms_var = 0.1408051820947358 : f64, value = 295 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2887 = "earth.mul"(%2872, %2886) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2888 = "earth.add"(%2885, %2887) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2889 = "earth.constant"() <{rms_var = 0.16851771191857826 : f64, value = 296 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2890 = "earth.mul"(%2873, %2889) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2891 = "earth.add"(%2888, %2890) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2892 = "earth.constant"() <{rms_var = 0.13582164971323474 : f64, value = 297 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2893 = "earth.mul"(%2874, %2892) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2894 = "earth.add"(%2891, %2893) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2895 = "earth.constant"() <{rms_var = 0.15403085838199027 : f64, value = 298 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2896 = "earth.mul"(%2875, %2895) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2897 = "earth.add"(%2894, %2896) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2898 = "earth.constant"() <{rms_var = 0.15349051088145582 : f64, value = 299 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2899 = "earth.mul"(%2876, %2898) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2900 = "earth.add"(%2897, %2899) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2901 = "earth.constant"() <{rms_var = 0.14939016898063393 : f64, value = 300 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2902 = "earth.mul"(%2877, %2901) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2903 = "earth.add"(%2900, %2902) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2904 = "earth.rotate"(%2903) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2905 = "earth.add"(%2903, %2904) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2906 = "earth.rotate"(%2905) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2907 = "earth.add"(%2905, %2906) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2908 = "earth.rotate"(%2907) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2909 = "earth.add"(%2907, %2908) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2910 = "earth.rotate"(%2909) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2911 = "earth.add"(%2909, %2910) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2912 = "earth.rotate"(%2911) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2913 = "earth.constant"() <{rms_var = 0.040771141287606179 : f64, value = 301 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2914 = "earth.mul"(%2912, %2913) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2915 = "earth.rotate"(%2911) <{offset = array<i64: 15360>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2916 = "earth.constant"() <{rms_var = 0.050182338992427536 : f64, value = 302 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2917 = "earth.mul"(%2915, %2916) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2918 = "earth.add"(%2914, %2917) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2919 = "earth.rotate"(%2911) <{offset = array<i64: 30720>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2920 = "earth.constant"() <{rms_var = 0.065751189369755644 : f64, value = 303 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2921 = "earth.mul"(%2919, %2920) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2922 = "earth.add"(%2918, %2921) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2923 = "earth.rotate"(%2911) <{offset = array<i64: 46080>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2924 = "earth.constant"() <{rms_var = 0.05702194035371428 : f64, value = 304 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2925 = "earth.mul"(%2923, %2924) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2926 = "earth.add"(%2922, %2925) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2927 = "earth.constant"() <{rms_var = 0.13348529038072263 : f64, value = 305 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2928 = "earth.mul"(%2869, %2927) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2929 = "earth.constant"() <{rms_var = 0.15090111770668124 : f64, value = 306 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2930 = "earth.mul"(%2870, %2929) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2931 = "earth.add"(%2928, %2930) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2932 = "earth.constant"() <{rms_var = 0.10226288141473613 : f64, value = 307 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2933 = "earth.mul"(%2871, %2932) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2934 = "earth.add"(%2931, %2933) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2935 = "earth.constant"() <{rms_var = 0.14463188739639954 : f64, value = 308 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2936 = "earth.mul"(%2872, %2935) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2937 = "earth.add"(%2934, %2936) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2938 = "earth.constant"() <{rms_var = 0.21824472855902366 : f64, value = 309 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2939 = "earth.mul"(%2873, %2938) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2940 = "earth.add"(%2937, %2939) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2941 = "earth.constant"() <{rms_var = 0.12134875967049039 : f64, value = 310 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2942 = "earth.mul"(%2874, %2941) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2943 = "earth.add"(%2940, %2942) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2944 = "earth.constant"() <{rms_var = 0.14908256901628208 : f64, value = 311 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2945 = "earth.mul"(%2875, %2944) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2946 = "earth.add"(%2943, %2945) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2947 = "earth.constant"() <{rms_var = 0.18262686313031912 : f64, value = 312 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2948 = "earth.mul"(%2876, %2947) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2949 = "earth.add"(%2946, %2948) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2950 = "earth.constant"() <{rms_var = 0.14452143928308106 : f64, value = 313 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2951 = "earth.mul"(%2877, %2950) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2952 = "earth.add"(%2949, %2951) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2953 = "earth.rotate"(%2952) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2954 = "earth.add"(%2952, %2953) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2955 = "earth.rotate"(%2954) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2956 = "earth.add"(%2954, %2955) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2957 = "earth.rotate"(%2956) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2958 = "earth.add"(%2956, %2957) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2959 = "earth.rotate"(%2958) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2960 = "earth.add"(%2958, %2959) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2961 = "earth.rotate"(%2960) <{offset = array<i64: -4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2962 = "earth.constant"() <{rms_var = 0.063194567115260031 : f64, value = 314 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2963 = "earth.mul"(%2961, %2962) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2964 = "earth.add"(%2926, %2963) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2965 = "earth.rotate"(%2960) <{offset = array<i64: 11264>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2966 = "earth.constant"() <{rms_var = 0.03568904701269085 : f64, value = 315 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2967 = "earth.mul"(%2965, %2966) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2968 = "earth.add"(%2964, %2967) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2969 = "earth.rotate"(%2960) <{offset = array<i64: 26624>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2970 = "earth.constant"() <{rms_var = 0.04417952890972332 : f64, value = 316 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2971 = "earth.mul"(%2969, %2970) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2972 = "earth.add"(%2968, %2971) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2973 = "earth.rotate"(%2960) <{offset = array<i64: 41984>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2974 = "earth.constant"() <{rms_var = 0.042239580825932245 : f64, value = 317 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2975 = "earth.mul"(%2973, %2974) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2976 = "earth.add"(%2972, %2975) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2977 = "earth.constant"() <{rms_var = 0.11482308768305451 : f64, value = 318 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2978 = "earth.mul"(%2869, %2977) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2979 = "earth.constant"() <{rms_var = 0.13406258191418552 : f64, value = 319 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2980 = "earth.mul"(%2870, %2979) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2981 = "earth.add"(%2978, %2980) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2982 = "earth.constant"() <{rms_var = 0.12687861276525964 : f64, value = 320 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2983 = "earth.mul"(%2871, %2982) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2984 = "earth.add"(%2981, %2983) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2985 = "earth.constant"() <{rms_var = 0.14901419923651785 : f64, value = 321 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2986 = "earth.mul"(%2872, %2985) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2987 = "earth.add"(%2984, %2986) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2988 = "earth.constant"() <{rms_var = 0.167835106426766 : f64, value = 322 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2989 = "earth.mul"(%2873, %2988) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2990 = "earth.add"(%2987, %2989) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2991 = "earth.constant"() <{rms_var = 0.15158525006309373 : f64, value = 323 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2992 = "earth.mul"(%2874, %2991) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2993 = "earth.add"(%2990, %2992) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2994 = "earth.constant"() <{rms_var = 0.12795716924306696 : f64, value = 324 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2995 = "earth.mul"(%2875, %2994) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2996 = "earth.add"(%2993, %2995) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2997 = "earth.constant"() <{rms_var = 0.17094863397273519 : f64, value = 325 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2998 = "earth.mul"(%2876, %2997) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %2999 = "earth.add"(%2996, %2998) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3000 = "earth.constant"() <{rms_var = 0.14580974808287905 : f64, value = 326 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3001 = "earth.mul"(%2877, %3000) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3002 = "earth.add"(%2999, %3001) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3003 = "earth.rotate"(%3002) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3004 = "earth.add"(%3002, %3003) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3005 = "earth.rotate"(%3004) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3006 = "earth.add"(%3004, %3005) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3007 = "earth.rotate"(%3006) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3008 = "earth.add"(%3006, %3007) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3009 = "earth.rotate"(%3008) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3010 = "earth.add"(%3008, %3009) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3011 = "earth.rotate"(%3010) <{offset = array<i64: -8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3012 = "earth.constant"() <{rms_var = 0.055681207744076928 : f64, value = 327 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3013 = "earth.mul"(%3011, %3012) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3014 = "earth.add"(%2976, %3013) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3015 = "earth.rotate"(%3010) <{offset = array<i64: 7168>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3016 = "earth.constant"() <{rms_var = 0.051863867076369453 : f64, value = 328 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3017 = "earth.mul"(%3015, %3016) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3018 = "earth.add"(%3014, %3017) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3019 = "earth.rotate"(%3010) <{offset = array<i64: 22528>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3020 = "earth.constant"() <{rms_var = 0.047475737505073411 : f64, value = 329 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3021 = "earth.mul"(%3019, %3020) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3022 = "earth.add"(%3018, %3021) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3023 = "earth.rotate"(%3010) <{offset = array<i64: 37888>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3024 = "earth.constant"() <{rms_var = 0.058437892740478295 : f64, value = 330 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3025 = "earth.mul"(%3023, %3024) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3026 = "earth.add"(%3022, %3025) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3027 = "earth.constant"() <{rms_var = 0.11355525812287834 : f64, value = 331 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3028 = "earth.mul"(%2869, %3027) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3029 = "earth.constant"() <{rms_var = 0.13779595859352878 : f64, value = 332 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3030 = "earth.mul"(%2870, %3029) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3031 = "earth.add"(%3028, %3030) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3032 = "earth.constant"() <{rms_var = 0.10956438865409543 : f64, value = 333 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3033 = "earth.mul"(%2871, %3032) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3034 = "earth.add"(%3031, %3033) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3035 = "earth.constant"() <{rms_var = 0.15243826871878347 : f64, value = 334 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3036 = "earth.mul"(%2872, %3035) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3037 = "earth.add"(%3034, %3036) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3038 = "earth.constant"() <{rms_var = 0.14552273825884648 : f64, value = 335 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3039 = "earth.mul"(%2873, %3038) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3040 = "earth.add"(%3037, %3039) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3041 = "earth.constant"() <{rms_var = 0.14589860652079478 : f64, value = 336 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3042 = "earth.mul"(%2874, %3041) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3043 = "earth.add"(%3040, %3042) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3044 = "earth.constant"() <{rms_var = 0.11287152282561738 : f64, value = 337 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3045 = "earth.mul"(%2875, %3044) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3046 = "earth.add"(%3043, %3045) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3047 = "earth.constant"() <{rms_var = 0.11504485956952334 : f64, value = 338 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3048 = "earth.mul"(%2876, %3047) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3049 = "earth.add"(%3046, %3048) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3050 = "earth.constant"() <{rms_var = 0.1513209902747322 : f64, value = 339 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3051 = "earth.mul"(%2877, %3050) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3052 = "earth.add"(%3049, %3051) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3053 = "earth.rotate"(%3052) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3054 = "earth.add"(%3052, %3053) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3055 = "earth.rotate"(%3054) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3056 = "earth.add"(%3054, %3055) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3057 = "earth.rotate"(%3056) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3058 = "earth.add"(%3056, %3057) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3059 = "earth.rotate"(%3058) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3060 = "earth.add"(%3058, %3059) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3061 = "earth.rotate"(%3060) <{offset = array<i64: -12288>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3062 = "earth.constant"() <{rms_var = 0.064482181428996851 : f64, value = 340 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3063 = "earth.mul"(%3061, %3062) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3064 = "earth.add"(%3026, %3063) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3065 = "earth.rotate"(%3060) <{offset = array<i64: 3072>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3066 = "earth.constant"() <{rms_var = 0.06114689690470939 : f64, value = 341 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3067 = "earth.mul"(%3065, %3066) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3068 = "earth.add"(%3064, %3067) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3069 = "earth.rotate"(%3060) <{offset = array<i64: 18432>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3070 = "earth.constant"() <{rms_var = 0.043301168841381417 : f64, value = 342 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3071 = "earth.mul"(%3069, %3070) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3072 = "earth.add"(%3068, %3071) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3073 = "earth.rotate"(%3060) <{offset = array<i64: 33792>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3074 = "earth.constant"() <{rms_var = 0.04581317219763241 : f64, value = 343 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3075 = "earth.mul"(%3073, %3074) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3076 = "earth.add"(%3072, %3075) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3077 = "earth.rotate"(%3076) <{offset = array<i64: -16384>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3078 = "earth.add"(%3076, %3077) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3079 = "earth.rotate"(%3078) <{offset = array<i64: -32768>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3080 = "earth.add"(%3078, %3079) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3081 = "earth.constant"() <{rms_var = 0.021355933216739151 : f64, value = 344 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3082 = "earth.add"(%3080, %3081) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn1-0
    %3083 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3084 = "earth.mul"(%3083, %3082) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3085 = "earth.mul"(%3084, %3082) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3086 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3087 = "earth.add"(%3085, %3086) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3088 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3089 = "earth.mul"(%3088, %3087) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3090 = "earth.mul"(%3089, %3087) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3091 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3092 = "earth.add"(%3090, %3091) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3093 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3094 = "earth.mul"(%3093, %3092) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3095 = "earth.mul"(%3094, %3092) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3096 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3097 = "earth.add"(%3095, %3096) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3098 = "earth.mul"(%3084, %3087) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3099 = "earth.negate"(%3082) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3100 = "earth.add"(%3098, %3099) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3101 = "earth.mul"(%3084, %3092) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3102 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3103 = "earth.mul"(%3102, %3100) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3104 = "earth.mul"(%3103, %3092) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3105 = "earth.negate"(%3100) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3106 = "earth.add"(%3101, %3105) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3107 = "earth.add"(%3104, %3099) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3108 = "earth.mul"(%3084, %3097) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3109 = "earth.mul"(%3103, %3097) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3110 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3111 = "earth.mul"(%3110, %3106) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3112 = "earth.mul"(%3111, %3097) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3113 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3114 = "earth.mul"(%3113, %3107) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3115 = "earth.mul"(%3114, %3097) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3116 = "earth.negate"(%3107) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3117 = "earth.add"(%3108, %3116) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3118 = "earth.negate"(%3106) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3119 = "earth.add"(%3109, %3118) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3120 = "earth.add"(%3112, %3105) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3121 = "earth.add"(%3115, %3099) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3122 = "earth.constant"() <{rms_var = 0.051379788302527769 : f64, value = 28 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3123 = "earth.mul"(%3122, %3082) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3124 = "earth.constant"() <{rms_var = 0.04272842452341858 : f64, value = 29 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3125 = "earth.mul"(%3124, %3100) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3126 = "earth.add"(%3123, %3125) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3127 = "earth.constant"() <{rms_var = 0.036049430000943392 : f64, value = 30 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3128 = "earth.mul"(%3127, %3106) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3129 = "earth.add"(%3126, %3128) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3130 = "earth.constant"() <{rms_var = 0.030937245302699062 : f64, value = 31 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3131 = "earth.mul"(%3130, %3107) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3132 = "earth.add"(%3129, %3131) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3133 = "earth.constant"() <{rms_var = 0.027115258485962086 : f64, value = 32 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3134 = "earth.mul"(%3133, %3117) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3135 = "earth.add"(%3132, %3134) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3136 = "earth.constant"() <{rms_var = 0.024393077542268389 : f64, value = 33 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3137 = "earth.mul"(%3136, %3119) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3138 = "earth.add"(%3135, %3137) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3139 = "earth.constant"() <{rms_var = 0.022642601074970584 : f64, value = 34 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3140 = "earth.mul"(%3139, %3120) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3141 = "earth.add"(%3138, %3140) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3142 = "earth.constant"() <{rms_var = 0.021831260875010087 : f64, value = 35 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3143 = "earth.mul"(%3142, %3121) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3144 = "earth.add"(%3141, %3143) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3145 = "earth.constant"() <{rms_var = 0.021652089836507502 : f64, value = 36 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3146 = "earth.mul"(%3145, %3082) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3147 = "earth.constant"() <{rms_var = 0.018138222134916806 : f64, value = 37 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3148 = "earth.mul"(%3147, %3100) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3149 = "earth.add"(%3146, %3148) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3150 = "earth.constant"() <{rms_var = 0.01542303411540253 : f64, value = 38 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3151 = "earth.mul"(%3150, %3106) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3152 = "earth.add"(%3149, %3151) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3153 = "earth.constant"() <{rms_var = 0.013305654693862069 : f64, value = 39 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3154 = "earth.mul"(%3153, %3107) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3155 = "earth.add"(%3152, %3154) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3156 = "earth.constant"() <{rms_var = 0.011703046220094507 : f64, value = 40 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3157 = "earth.mul"(%3156, %3117) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3158 = "earth.add"(%3155, %3157) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3159 = "earth.constant"() <{rms_var = 0.010552642814751455 : f64, value = 41 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3160 = "earth.mul"(%3159, %3119) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3161 = "earth.add"(%3158, %3160) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3162 = "earth.constant"() <{rms_var = 0.0098096659804816355 : f64, value = 42 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3163 = "earth.mul"(%3162, %3120) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3164 = "earth.add"(%3161, %3163) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3165 = "earth.constant"() <{rms_var = 0.0094452495559895089 : f64, value = 43 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3166 = "earth.mul"(%3165, %3121) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3167 = "earth.add"(%3164, %3166) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3168 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3169 = "earth.mul"(%3168, %3097) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3170 = "earth.mul"(%3169, %3097) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3171 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3172 = "earth.add"(%3170, %3171) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3173 = "earth.mul"(%3167, %3172) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3174 = "earth.add"(%3173, %3144) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3175 = "earth.constant"() <{rms_var = 0.0042995278292336028 : f64, value = 44 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3176 = "earth.mul"(%3175, %3082) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3177 = "earth.constant"() <{rms_var = 0.0036191919476548503 : f64, value = 45 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3178 = "earth.mul"(%3177, %3100) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3179 = "earth.add"(%3176, %3178) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3180 = "earth.constant"() <{rms_var = 0.0030784419617177587 : f64, value = 46 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3181 = "earth.mul"(%3180, %3106) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3182 = "earth.add"(%3179, %3181) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3183 = "earth.constant"() <{rms_var = 0.0026564062051904268 : f64, value = 47 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3184 = "earth.mul"(%3183, %3107) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3185 = "earth.add"(%3182, %3184) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3186 = "earth.constant"() <{rms_var = 0.0023368004457586414 : f64, value = 48 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3187 = "earth.mul"(%3186, %3117) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3188 = "earth.add"(%3185, %3187) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3189 = "earth.constant"() <{rms_var = 0.002107295415524122 : f64, value = 49 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3190 = "earth.mul"(%3189, %3119) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3191 = "earth.add"(%3188, %3190) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3192 = "earth.constant"() <{rms_var = 0.0019590388891824136 : f64, value = 50 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3193 = "earth.mul"(%3192, %3120) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3194 = "earth.add"(%3191, %3193) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3195 = "earth.constant"() <{rms_var = 0.0018863129851277056 : f64, value = 51 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3196 = "earth.mul"(%3195, %3121) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3197 = "earth.add"(%3194, %3196) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3198 = "earth.constant"() <{rms_var = 0.63615477792235875 : f64, value = 52 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3199 = "earth.mul"(%3198, %3082) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3200 = "earth.constant"() <{rms_var = 0.21189795368271022 : f64, value = 53 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3201 = "earth.mul"(%3200, %3100) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3202 = "earth.add"(%3199, %3201) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3203 = "earth.constant"() <{rms_var = 0.12734215758896472 : f64, value = 54 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3204 = "earth.mul"(%3203, %3106) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3205 = "earth.add"(%3202, %3204) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3206 = "earth.constant"() <{rms_var = 0.091626347972833505 : f64, value = 55 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3207 = "earth.mul"(%3206, %3107) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3208 = "earth.add"(%3205, %3207) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3209 = "earth.constant"() <{rms_var = 0.07255941147658998 : f64, value = 56 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3210 = "earth.mul"(%3209, %3117) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3211 = "earth.add"(%3208, %3210) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3212 = "earth.constant"() <{rms_var = 0.061477909207391525 : f64, value = 57 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3213 = "earth.mul"(%3212, %3119) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3214 = "earth.add"(%3211, %3213) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3215 = "earth.constant"() <{rms_var = 0.055169030070493508 : f64, value = 58 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3216 = "earth.mul"(%3215, %3120) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3217 = "earth.add"(%3214, %3216) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3218 = "earth.constant"() <{rms_var = 0.052275954916496656 : f64, value = 59 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3219 = "earth.mul"(%3218, %3121) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3220 = "earth.add"(%3217, %3219) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3221 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3222 = "earth.mul"(%3221, %3172) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3223 = "earth.mul"(%3222, %3172) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3224 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3225 = "earth.add"(%3223, %3224) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3226 = "earth.mul"(%3197, %3225) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3227 = "earth.add"(%3226, %3174) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3228 = "earth.constant"() <{rms_var = 4.9481895575581374E-4 : f64, value = 60 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3229 = "earth.mul"(%3228, %3082) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3230 = "earth.constant"() <{rms_var = 3.771067298313324E-4 : f64, value = 61 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3231 = "earth.mul"(%3230, %3100) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3232 = "earth.add"(%3229, %3231) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3233 = "earth.constant"() <{rms_var = 3.2076765303085136E-4 : f64, value = 62 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3234 = "earth.mul"(%3233, %3106) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3235 = "earth.add"(%3232, %3234) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3236 = "earth.constant"() <{rms_var = 2.7679532400224407E-4 : f64, value = 63 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3237 = "earth.mul"(%3236, %3107) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3238 = "earth.add"(%3235, %3237) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3239 = "earth.constant"() <{rms_var = 2.4349440690240933E-4 : f64, value = 64 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3240 = "earth.mul"(%3239, %3117) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3241 = "earth.add"(%3238, %3240) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3242 = "earth.constant"() <{rms_var = 2.1958101095650637E-4 : f64, value = 65 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3243 = "earth.mul"(%3242, %3119) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3244 = "earth.add"(%3241, %3243) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3245 = "earth.constant"() <{rms_var = 2.0413318001830463E-4 : f64, value = 66 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3246 = "earth.mul"(%3245, %3120) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3247 = "earth.add"(%3244, %3246) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3248 = "earth.constant"() <{rms_var = 1.9655534146569248E-4 : f64, value = 67 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3249 = "earth.mul"(%3248, %3121) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3250 = "earth.add"(%3247, %3249) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3251 = "earth.constant"() <{rms_var = 1.7735088548126505E-4 : f64, value = 68 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3252 = "earth.mul"(%3251, %3082) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3253 = "earth.constant"() <{rms_var = 1.457794164045128E-4 : f64, value = 69 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3254 = "earth.mul"(%3253, %3100) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3255 = "earth.add"(%3252, %3254) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3256 = "earth.constant"() <{rms_var = 1.1982820692495149E-4 : f64, value = 70 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3257 = "earth.mul"(%3256, %3106) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3258 = "earth.add"(%3255, %3257) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3259 = "earth.constant"() <{rms_var = 9.8496754159654756E-5 : f64, value = 71 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3260 = "earth.mul"(%3259, %3107) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3261 = "earth.add"(%3258, %3260) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3262 = "earth.constant"() <{rms_var = 8.0962662339626343E-5 : f64, value = 72 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3263 = "earth.mul"(%3262, %3117) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3264 = "earth.add"(%3261, %3263) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3265 = "earth.constant"() <{rms_var = 6.6549936439435966E-5 : f64, value = 73 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3266 = "earth.mul"(%3265, %3119) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3267 = "earth.add"(%3264, %3266) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3268 = "earth.constant"() <{rms_var = 5.4702920205367267E-5 : f64, value = 74 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3269 = "earth.mul"(%3268, %3120) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3270 = "earth.add"(%3267, %3269) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3271 = "earth.constant"() <{rms_var = 1.3863334887734771E-4 : f64, value = 75 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3272 = "earth.mul"(%3271, %3121) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3273 = "earth.add"(%3270, %3272) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3274 = "earth.mul"(%3227, %3172) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3275 = "earth.add"(%3274, %3220) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3276 = "earth.mul"(%3273, %3172) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3277 = "earth.add"(%3276, %3250) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3278 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3279 = "earth.mul"(%3278, %3225) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3280 = "earth.mul"(%3279, %3225) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3281 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3282 = "earth.add"(%3280, %3281) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3283 = "earth.mul"(%3277, %3282) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3284 = "earth.add"(%3283, %3275) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act1_SiLU_poly
    %3285 = "earth.constant"() <{rms_var = 5.000000e-01 : f64, value = 76 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // add[]layer1_2_act1_SiLU_add
    %3286 = "earth.add"(%3284, %3285) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // add[]layer1_2_act1_SiLU_add
    %3287 = "earth.mul"(%3082, %3286) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // mul[]layer1_2_act1_SiLU_mul
    %3288 = "earth.rotate"(%3287) <{offset = array<i64: -33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3289 = "earth.rotate"(%3287) <{offset = array<i64: -32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3290 = "earth.rotate"(%3287) <{offset = array<i64: -31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3291 = "earth.rotate"(%3287) <{offset = array<i64: -1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3292 = "earth.rotate"(%3287) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3293 = "earth.rotate"(%3287) <{offset = array<i64: 1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3294 = "earth.rotate"(%3287) <{offset = array<i64: 31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3295 = "earth.rotate"(%3287) <{offset = array<i64: 32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3296 = "earth.rotate"(%3287) <{offset = array<i64: 33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3297 = "earth.constant"() <{rms_var = 0.09417896124252037 : f64, value = 345 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3298 = "earth.mul"(%3288, %3297) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3299 = "earth.constant"() <{rms_var = 0.12166343827124206 : f64, value = 346 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3300 = "earth.mul"(%3289, %3299) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3301 = "earth.add"(%3298, %3300) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3302 = "earth.constant"() <{rms_var = 0.094962630158233252 : f64, value = 347 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3303 = "earth.mul"(%3290, %3302) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3304 = "earth.add"(%3301, %3303) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3305 = "earth.constant"() <{rms_var = 0.13156104054598688 : f64, value = 348 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3306 = "earth.mul"(%3291, %3305) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3307 = "earth.add"(%3304, %3306) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3308 = "earth.constant"() <{rms_var = 0.17216214320421064 : f64, value = 349 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3309 = "earth.mul"(%3292, %3308) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3310 = "earth.add"(%3307, %3309) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3311 = "earth.constant"() <{rms_var = 0.12731507496335237 : f64, value = 350 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3312 = "earth.mul"(%3293, %3311) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3313 = "earth.add"(%3310, %3312) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3314 = "earth.constant"() <{rms_var = 0.12963108687629882 : f64, value = 351 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3315 = "earth.mul"(%3294, %3314) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3316 = "earth.add"(%3313, %3315) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3317 = "earth.constant"() <{rms_var = 0.14811407851175215 : f64, value = 352 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3318 = "earth.mul"(%3295, %3317) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3319 = "earth.add"(%3316, %3318) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3320 = "earth.constant"() <{rms_var = 0.10911621584820613 : f64, value = 353 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3321 = "earth.mul"(%3296, %3320) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3322 = "earth.add"(%3319, %3321) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3323 = "earth.rotate"(%3322) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3324 = "earth.add"(%3322, %3323) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3325 = "earth.rotate"(%3324) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3326 = "earth.add"(%3324, %3325) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3327 = "earth.rotate"(%3326) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3328 = "earth.add"(%3326, %3327) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3329 = "earth.rotate"(%3328) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3330 = "earth.add"(%3328, %3329) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3331 = "earth.rotate"(%3330) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3332 = "earth.constant"() <{rms_var = 0.1005830777002315 : f64, value = 354 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3333 = "earth.mul"(%3331, %3332) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3334 = "earth.rotate"(%3330) <{offset = array<i64: 15360>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3335 = "earth.constant"() <{rms_var = 0.076331626355853776 : f64, value = 355 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3336 = "earth.mul"(%3334, %3335) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3337 = "earth.add"(%3333, %3336) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3338 = "earth.rotate"(%3330) <{offset = array<i64: 30720>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3339 = "earth.constant"() <{rms_var = 0.074214567990358277 : f64, value = 356 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3340 = "earth.mul"(%3338, %3339) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3341 = "earth.add"(%3337, %3340) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3342 = "earth.rotate"(%3330) <{offset = array<i64: 46080>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3343 = "earth.constant"() <{rms_var = 0.066301341884227954 : f64, value = 357 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3344 = "earth.mul"(%3342, %3343) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3345 = "earth.add"(%3341, %3344) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3346 = "earth.constant"() <{rms_var = 0.091007997056750042 : f64, value = 358 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3347 = "earth.mul"(%3288, %3346) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3348 = "earth.constant"() <{rms_var = 0.11430343738098596 : f64, value = 359 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3349 = "earth.mul"(%3289, %3348) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3350 = "earth.add"(%3347, %3349) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3351 = "earth.constant"() <{rms_var = 0.10381757071098809 : f64, value = 360 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3352 = "earth.mul"(%3290, %3351) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3353 = "earth.add"(%3350, %3352) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3354 = "earth.constant"() <{rms_var = 0.12224145673118199 : f64, value = 361 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3355 = "earth.mul"(%3291, %3354) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3356 = "earth.add"(%3353, %3355) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3357 = "earth.constant"() <{rms_var = 0.13744423720458498 : f64, value = 362 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3358 = "earth.mul"(%3292, %3357) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3359 = "earth.add"(%3356, %3358) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3360 = "earth.constant"() <{rms_var = 0.11158996756314321 : f64, value = 363 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3361 = "earth.mul"(%3293, %3360) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3362 = "earth.add"(%3359, %3361) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3363 = "earth.constant"() <{rms_var = 0.11463103273788414 : f64, value = 364 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3364 = "earth.mul"(%3294, %3363) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3365 = "earth.add"(%3362, %3364) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3366 = "earth.constant"() <{rms_var = 0.12748098993555912 : f64, value = 365 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3367 = "earth.mul"(%3295, %3366) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3368 = "earth.add"(%3365, %3367) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3369 = "earth.constant"() <{rms_var = 0.11258418446021062 : f64, value = 366 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3370 = "earth.mul"(%3296, %3369) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3371 = "earth.add"(%3368, %3370) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3372 = "earth.rotate"(%3371) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3373 = "earth.add"(%3371, %3372) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3374 = "earth.rotate"(%3373) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3375 = "earth.add"(%3373, %3374) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3376 = "earth.rotate"(%3375) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3377 = "earth.add"(%3375, %3376) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3378 = "earth.rotate"(%3377) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3379 = "earth.add"(%3377, %3378) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3380 = "earth.rotate"(%3379) <{offset = array<i64: -4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3381 = "earth.constant"() <{rms_var = 0.076940708508931796 : f64, value = 367 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3382 = "earth.mul"(%3380, %3381) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3383 = "earth.add"(%3345, %3382) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3384 = "earth.rotate"(%3379) <{offset = array<i64: 11264>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3385 = "earth.constant"() <{rms_var = 0.044429638664955237 : f64, value = 368 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3386 = "earth.mul"(%3384, %3385) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3387 = "earth.add"(%3383, %3386) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3388 = "earth.rotate"(%3379) <{offset = array<i64: 26624>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3389 = "earth.constant"() <{rms_var = 0.039225102966515497 : f64, value = 369 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3390 = "earth.mul"(%3388, %3389) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3391 = "earth.add"(%3387, %3390) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3392 = "earth.rotate"(%3379) <{offset = array<i64: 41984>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3393 = "earth.constant"() <{rms_var = 0.082484672230606446 : f64, value = 370 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3394 = "earth.mul"(%3392, %3393) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3395 = "earth.add"(%3391, %3394) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3396 = "earth.constant"() <{rms_var = 0.11235989780425451 : f64, value = 371 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3397 = "earth.mul"(%3288, %3396) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3398 = "earth.constant"() <{rms_var = 0.13276930941802947 : f64, value = 372 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3399 = "earth.mul"(%3289, %3398) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3400 = "earth.add"(%3397, %3399) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3401 = "earth.constant"() <{rms_var = 0.11055192391677204 : f64, value = 373 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3402 = "earth.mul"(%3290, %3401) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3403 = "earth.add"(%3400, %3402) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3404 = "earth.constant"() <{rms_var = 0.11394777291397178 : f64, value = 374 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3405 = "earth.mul"(%3291, %3404) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3406 = "earth.add"(%3403, %3405) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3407 = "earth.constant"() <{rms_var = 0.1932543595384176 : f64, value = 375 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3408 = "earth.mul"(%3292, %3407) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3409 = "earth.add"(%3406, %3408) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3410 = "earth.constant"() <{rms_var = 0.15407186602849954 : f64, value = 376 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3411 = "earth.mul"(%3293, %3410) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3412 = "earth.add"(%3409, %3411) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3413 = "earth.constant"() <{rms_var = 0.10793877039835924 : f64, value = 377 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3414 = "earth.mul"(%3294, %3413) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3415 = "earth.add"(%3412, %3414) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3416 = "earth.constant"() <{rms_var = 0.15274420277115439 : f64, value = 378 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3417 = "earth.mul"(%3295, %3416) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3418 = "earth.add"(%3415, %3417) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3419 = "earth.constant"() <{rms_var = 0.13589298546741671 : f64, value = 379 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3420 = "earth.mul"(%3296, %3419) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3421 = "earth.add"(%3418, %3420) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3422 = "earth.rotate"(%3421) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3423 = "earth.add"(%3421, %3422) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3424 = "earth.rotate"(%3423) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3425 = "earth.add"(%3423, %3424) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3426 = "earth.rotate"(%3425) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3427 = "earth.add"(%3425, %3426) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3428 = "earth.rotate"(%3427) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3429 = "earth.add"(%3427, %3428) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3430 = "earth.rotate"(%3429) <{offset = array<i64: -8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3431 = "earth.constant"() <{rms_var = 0.060650837181097199 : f64, value = 380 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3432 = "earth.mul"(%3430, %3431) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3433 = "earth.add"(%3395, %3432) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3434 = "earth.rotate"(%3429) <{offset = array<i64: 7168>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3435 = "earth.constant"() <{rms_var = 0.10800146377973827 : f64, value = 381 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3436 = "earth.mul"(%3434, %3435) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3437 = "earth.add"(%3433, %3436) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3438 = "earth.rotate"(%3429) <{offset = array<i64: 22528>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3439 = "earth.constant"() <{rms_var = 0.072948450676760646 : f64, value = 382 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3440 = "earth.mul"(%3438, %3439) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3441 = "earth.add"(%3437, %3440) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3442 = "earth.rotate"(%3429) <{offset = array<i64: 37888>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3443 = "earth.constant"() <{rms_var = 0.058311421561305664 : f64, value = 383 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3444 = "earth.mul"(%3442, %3443) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3445 = "earth.add"(%3441, %3444) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3446 = "earth.constant"() <{rms_var = 0.11029006129004447 : f64, value = 384 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3447 = "earth.mul"(%3288, %3446) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3448 = "earth.constant"() <{rms_var = 0.15297351203582063 : f64, value = 385 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3449 = "earth.mul"(%3289, %3448) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3450 = "earth.add"(%3447, %3449) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3451 = "earth.constant"() <{rms_var = 0.09527692395333491 : f64, value = 386 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3452 = "earth.mul"(%3290, %3451) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3453 = "earth.add"(%3450, %3452) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3454 = "earth.constant"() <{rms_var = 0.13117647971809407 : f64, value = 387 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3455 = "earth.mul"(%3291, %3454) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3456 = "earth.add"(%3453, %3455) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3457 = "earth.constant"() <{rms_var = 0.19385132686815418 : f64, value = 388 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3458 = "earth.mul"(%3292, %3457) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3459 = "earth.add"(%3456, %3458) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3460 = "earth.constant"() <{rms_var = 0.15733730947250063 : f64, value = 389 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3461 = "earth.mul"(%3293, %3460) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3462 = "earth.add"(%3459, %3461) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3463 = "earth.constant"() <{rms_var = 0.11550841827163928 : f64, value = 390 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3464 = "earth.mul"(%3294, %3463) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3465 = "earth.add"(%3462, %3464) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3466 = "earth.constant"() <{rms_var = 0.14727295530896603 : f64, value = 391 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3467 = "earth.mul"(%3295, %3466) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3468 = "earth.add"(%3465, %3467) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3469 = "earth.constant"() <{rms_var = 0.12577219172987547 : f64, value = 392 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3470 = "earth.mul"(%3296, %3469) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3471 = "earth.add"(%3468, %3470) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3472 = "earth.rotate"(%3471) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3473 = "earth.add"(%3471, %3472) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3474 = "earth.rotate"(%3473) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3475 = "earth.add"(%3473, %3474) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3476 = "earth.rotate"(%3475) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3477 = "earth.add"(%3475, %3476) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3478 = "earth.rotate"(%3477) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3479 = "earth.add"(%3477, %3478) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3480 = "earth.rotate"(%3479) <{offset = array<i64: -12288>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3481 = "earth.constant"() <{rms_var = 0.076489619787229651 : f64, value = 393 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3482 = "earth.mul"(%3480, %3481) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3483 = "earth.add"(%3445, %3482) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3484 = "earth.rotate"(%3479) <{offset = array<i64: 3072>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3485 = "earth.constant"() <{rms_var = 0.1215376613658682 : f64, value = 394 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3486 = "earth.mul"(%3484, %3485) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3487 = "earth.add"(%3483, %3486) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3488 = "earth.rotate"(%3479) <{offset = array<i64: 18432>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3489 = "earth.constant"() <{rms_var = 0.065863004675865311 : f64, value = 395 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3490 = "earth.mul"(%3488, %3489) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3491 = "earth.add"(%3487, %3490) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3492 = "earth.rotate"(%3479) <{offset = array<i64: 33792>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3493 = "earth.constant"() <{rms_var = 0.092435173039375082 : f64, value = 396 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3494 = "earth.mul"(%3492, %3493) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3495 = "earth.add"(%3491, %3494) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3496 = "earth.rotate"(%3495) <{offset = array<i64: -16384>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3497 = "earth.add"(%3495, %3496) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3498 = "earth.rotate"(%3497) <{offset = array<i64: -32768>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3499 = "earth.add"(%3497, %3498) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3500 = "earth.constant"() <{rms_var = 0.021758331668353285 : f64, value = 397 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3501 = "earth.add"(%3499, %3500) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer1_2_convbn2
    %3502 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3503 = "earth.mul"(%3502, %3501) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3504 = "earth.mul"(%3503, %3501) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3505 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3506 = "earth.add"(%3504, %3505) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3507 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3508 = "earth.mul"(%3507, %3506) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3509 = "earth.mul"(%3508, %3506) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3510 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3511 = "earth.add"(%3509, %3510) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3512 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3513 = "earth.mul"(%3512, %3511) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3514 = "earth.mul"(%3513, %3511) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3515 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3516 = "earth.add"(%3514, %3515) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3517 = "earth.mul"(%3503, %3506) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3518 = "earth.negate"(%3501) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3519 = "earth.add"(%3517, %3518) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3520 = "earth.mul"(%3503, %3511) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3521 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3522 = "earth.mul"(%3521, %3519) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3523 = "earth.mul"(%3522, %3511) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3524 = "earth.negate"(%3519) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3525 = "earth.add"(%3520, %3524) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3526 = "earth.add"(%3523, %3518) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3527 = "earth.mul"(%3503, %3516) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3528 = "earth.mul"(%3522, %3516) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3529 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3530 = "earth.mul"(%3529, %3525) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3531 = "earth.mul"(%3530, %3516) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3532 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3533 = "earth.mul"(%3532, %3526) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3534 = "earth.mul"(%3533, %3516) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3535 = "earth.negate"(%3526) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3536 = "earth.add"(%3527, %3535) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3537 = "earth.negate"(%3525) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3538 = "earth.add"(%3528, %3537) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3539 = "earth.add"(%3531, %3524) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3540 = "earth.add"(%3534, %3518) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3541 = "earth.constant"() <{rms_var = 0.051379788302527769 : f64, value = 28 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3542 = "earth.mul"(%3541, %3501) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3543 = "earth.constant"() <{rms_var = 0.04272842452341858 : f64, value = 29 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3544 = "earth.mul"(%3543, %3519) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3545 = "earth.add"(%3542, %3544) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3546 = "earth.constant"() <{rms_var = 0.036049430000943392 : f64, value = 30 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3547 = "earth.mul"(%3546, %3525) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3548 = "earth.add"(%3545, %3547) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3549 = "earth.constant"() <{rms_var = 0.030937245302699062 : f64, value = 31 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3550 = "earth.mul"(%3549, %3526) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3551 = "earth.add"(%3548, %3550) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3552 = "earth.constant"() <{rms_var = 0.027115258485962086 : f64, value = 32 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3553 = "earth.mul"(%3552, %3536) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3554 = "earth.add"(%3551, %3553) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3555 = "earth.constant"() <{rms_var = 0.024393077542268389 : f64, value = 33 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3556 = "earth.mul"(%3555, %3538) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3557 = "earth.add"(%3554, %3556) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3558 = "earth.constant"() <{rms_var = 0.022642601074970584 : f64, value = 34 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3559 = "earth.mul"(%3558, %3539) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3560 = "earth.add"(%3557, %3559) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3561 = "earth.constant"() <{rms_var = 0.021831260875010087 : f64, value = 35 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3562 = "earth.mul"(%3561, %3540) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3563 = "earth.add"(%3560, %3562) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3564 = "earth.constant"() <{rms_var = 0.021652089836507502 : f64, value = 36 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3565 = "earth.mul"(%3564, %3501) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3566 = "earth.constant"() <{rms_var = 0.018138222134916806 : f64, value = 37 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3567 = "earth.mul"(%3566, %3519) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3568 = "earth.add"(%3565, %3567) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3569 = "earth.constant"() <{rms_var = 0.01542303411540253 : f64, value = 38 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3570 = "earth.mul"(%3569, %3525) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3571 = "earth.add"(%3568, %3570) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3572 = "earth.constant"() <{rms_var = 0.013305654693862069 : f64, value = 39 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3573 = "earth.mul"(%3572, %3526) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3574 = "earth.add"(%3571, %3573) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3575 = "earth.constant"() <{rms_var = 0.011703046220094507 : f64, value = 40 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3576 = "earth.mul"(%3575, %3536) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3577 = "earth.add"(%3574, %3576) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3578 = "earth.constant"() <{rms_var = 0.010552642814751455 : f64, value = 41 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3579 = "earth.mul"(%3578, %3538) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3580 = "earth.add"(%3577, %3579) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3581 = "earth.constant"() <{rms_var = 0.0098096659804816355 : f64, value = 42 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3582 = "earth.mul"(%3581, %3539) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3583 = "earth.add"(%3580, %3582) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3584 = "earth.constant"() <{rms_var = 0.0094452495559895089 : f64, value = 43 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3585 = "earth.mul"(%3584, %3540) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3586 = "earth.add"(%3583, %3585) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3587 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3588 = "earth.mul"(%3587, %3516) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3589 = "earth.mul"(%3588, %3516) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3590 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3591 = "earth.add"(%3589, %3590) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3592 = "earth.mul"(%3586, %3591) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3593 = "earth.add"(%3592, %3563) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3594 = "earth.constant"() <{rms_var = 0.0042995278292336028 : f64, value = 44 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3595 = "earth.mul"(%3594, %3501) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3596 = "earth.constant"() <{rms_var = 0.0036191919476548503 : f64, value = 45 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3597 = "earth.mul"(%3596, %3519) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3598 = "earth.add"(%3595, %3597) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3599 = "earth.constant"() <{rms_var = 0.0030784419617177587 : f64, value = 46 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3600 = "earth.mul"(%3599, %3525) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3601 = "earth.add"(%3598, %3600) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3602 = "earth.constant"() <{rms_var = 0.0026564062051904268 : f64, value = 47 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3603 = "earth.mul"(%3602, %3526) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3604 = "earth.add"(%3601, %3603) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3605 = "earth.constant"() <{rms_var = 0.0023368004457586414 : f64, value = 48 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3606 = "earth.mul"(%3605, %3536) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3607 = "earth.add"(%3604, %3606) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3608 = "earth.constant"() <{rms_var = 0.002107295415524122 : f64, value = 49 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3609 = "earth.mul"(%3608, %3538) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3610 = "earth.add"(%3607, %3609) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3611 = "earth.constant"() <{rms_var = 0.0019590388891824136 : f64, value = 50 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3612 = "earth.mul"(%3611, %3539) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3613 = "earth.add"(%3610, %3612) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3614 = "earth.constant"() <{rms_var = 0.0018863129851277056 : f64, value = 51 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3615 = "earth.mul"(%3614, %3540) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3616 = "earth.add"(%3613, %3615) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3617 = "earth.constant"() <{rms_var = 0.63615477792235875 : f64, value = 52 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3618 = "earth.mul"(%3617, %3501) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3619 = "earth.constant"() <{rms_var = 0.21189795368271022 : f64, value = 53 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3620 = "earth.mul"(%3619, %3519) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3621 = "earth.add"(%3618, %3620) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3622 = "earth.constant"() <{rms_var = 0.12734215758896472 : f64, value = 54 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3623 = "earth.mul"(%3622, %3525) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3624 = "earth.add"(%3621, %3623) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3625 = "earth.constant"() <{rms_var = 0.091626347972833505 : f64, value = 55 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3626 = "earth.mul"(%3625, %3526) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3627 = "earth.add"(%3624, %3626) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3628 = "earth.constant"() <{rms_var = 0.07255941147658998 : f64, value = 56 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3629 = "earth.mul"(%3628, %3536) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3630 = "earth.add"(%3627, %3629) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3631 = "earth.constant"() <{rms_var = 0.061477909207391525 : f64, value = 57 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3632 = "earth.mul"(%3631, %3538) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3633 = "earth.add"(%3630, %3632) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3634 = "earth.constant"() <{rms_var = 0.055169030070493508 : f64, value = 58 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3635 = "earth.mul"(%3634, %3539) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3636 = "earth.add"(%3633, %3635) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3637 = "earth.constant"() <{rms_var = 0.052275954916496656 : f64, value = 59 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3638 = "earth.mul"(%3637, %3540) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3639 = "earth.add"(%3636, %3638) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3640 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3641 = "earth.mul"(%3640, %3591) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3642 = "earth.mul"(%3641, %3591) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3643 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3644 = "earth.add"(%3642, %3643) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3645 = "earth.mul"(%3616, %3644) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3646 = "earth.add"(%3645, %3593) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3647 = "earth.constant"() <{rms_var = 4.9481895575581374E-4 : f64, value = 60 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3648 = "earth.mul"(%3647, %3501) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3649 = "earth.constant"() <{rms_var = 3.771067298313324E-4 : f64, value = 61 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3650 = "earth.mul"(%3649, %3519) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3651 = "earth.add"(%3648, %3650) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3652 = "earth.constant"() <{rms_var = 3.2076765303085136E-4 : f64, value = 62 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3653 = "earth.mul"(%3652, %3525) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3654 = "earth.add"(%3651, %3653) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3655 = "earth.constant"() <{rms_var = 2.7679532400224407E-4 : f64, value = 63 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3656 = "earth.mul"(%3655, %3526) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3657 = "earth.add"(%3654, %3656) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3658 = "earth.constant"() <{rms_var = 2.4349440690240933E-4 : f64, value = 64 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3659 = "earth.mul"(%3658, %3536) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3660 = "earth.add"(%3657, %3659) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3661 = "earth.constant"() <{rms_var = 2.1958101095650637E-4 : f64, value = 65 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3662 = "earth.mul"(%3661, %3538) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3663 = "earth.add"(%3660, %3662) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3664 = "earth.constant"() <{rms_var = 2.0413318001830463E-4 : f64, value = 66 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3665 = "earth.mul"(%3664, %3539) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3666 = "earth.add"(%3663, %3665) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3667 = "earth.constant"() <{rms_var = 1.9655534146569248E-4 : f64, value = 67 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3668 = "earth.mul"(%3667, %3540) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3669 = "earth.add"(%3666, %3668) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3670 = "earth.constant"() <{rms_var = 1.7735088548126505E-4 : f64, value = 68 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3671 = "earth.mul"(%3670, %3501) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3672 = "earth.constant"() <{rms_var = 1.457794164045128E-4 : f64, value = 69 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3673 = "earth.mul"(%3672, %3519) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3674 = "earth.add"(%3671, %3673) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3675 = "earth.constant"() <{rms_var = 1.1982820692495149E-4 : f64, value = 70 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3676 = "earth.mul"(%3675, %3525) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3677 = "earth.add"(%3674, %3676) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3678 = "earth.constant"() <{rms_var = 9.8496754159654756E-5 : f64, value = 71 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3679 = "earth.mul"(%3678, %3526) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3680 = "earth.add"(%3677, %3679) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3681 = "earth.constant"() <{rms_var = 8.0962662339626343E-5 : f64, value = 72 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3682 = "earth.mul"(%3681, %3536) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3683 = "earth.add"(%3680, %3682) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3684 = "earth.constant"() <{rms_var = 6.6549936439435966E-5 : f64, value = 73 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3685 = "earth.mul"(%3684, %3538) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3686 = "earth.add"(%3683, %3685) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3687 = "earth.constant"() <{rms_var = 5.4702920205367267E-5 : f64, value = 74 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3688 = "earth.mul"(%3687, %3539) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3689 = "earth.add"(%3686, %3688) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3690 = "earth.constant"() <{rms_var = 1.3863334887734771E-4 : f64, value = 75 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3691 = "earth.mul"(%3690, %3540) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3692 = "earth.add"(%3689, %3691) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3693 = "earth.mul"(%3646, %3591) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3694 = "earth.add"(%3693, %3639) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3695 = "earth.mul"(%3692, %3591) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3696 = "earth.add"(%3695, %3669) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3697 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3698 = "earth.mul"(%3697, %3644) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3699 = "earth.mul"(%3698, %3644) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3700 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3701 = "earth.add"(%3699, %3700) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3702 = "earth.mul"(%3696, %3701) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3703 = "earth.add"(%3702, %3694) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer1_2_act2_SiLU_poly
    %3704 = "earth.constant"() <{rms_var = 5.000000e-01 : f64, value = 76 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // add[]layer1_2_act2_SiLU_add
    %3705 = "earth.add"(%3703, %3704) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // add[]layer1_2_act2_SiLU_add
    %3706 = "earth.mul"(%3501, %3705) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // mul[]layer1_2_act2_SiLU_mul
    %3707 = "earth.rotate"(%3706) <{offset = array<i64: -33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3708 = "earth.rotate"(%3706) <{offset = array<i64: -32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3709 = "earth.rotate"(%3706) <{offset = array<i64: -31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3710 = "earth.rotate"(%3706) <{offset = array<i64: -1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3711 = "earth.rotate"(%3706) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3712 = "earth.rotate"(%3706) <{offset = array<i64: 1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3713 = "earth.rotate"(%3706) <{offset = array<i64: 31>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3714 = "earth.rotate"(%3706) <{offset = array<i64: 32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3715 = "earth.rotate"(%3706) <{offset = array<i64: 33>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3716 = "earth.constant"() <{rms_var = 0.11587405225331468 : f64, value = 398 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3717 = "earth.mul"(%3707, %3716) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3718 = "earth.constant"() <{rms_var = 0.1244462237246183 : f64, value = 399 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3719 = "earth.mul"(%3708, %3718) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3720 = "earth.add"(%3717, %3719) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3721 = "earth.constant"() <{rms_var = 0.11699880362236059 : f64, value = 400 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3722 = "earth.mul"(%3709, %3721) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3723 = "earth.add"(%3720, %3722) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3724 = "earth.constant"() <{rms_var = 0.12836125882956834 : f64, value = 401 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3725 = "earth.mul"(%3710, %3724) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3726 = "earth.add"(%3723, %3725) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3727 = "earth.constant"() <{rms_var = 0.13856450478826343 : f64, value = 402 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3728 = "earth.mul"(%3711, %3727) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3729 = "earth.add"(%3726, %3728) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3730 = "earth.constant"() <{rms_var = 0.14133028867787037 : f64, value = 403 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3731 = "earth.mul"(%3712, %3730) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3732 = "earth.add"(%3729, %3731) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3733 = "earth.constant"() <{rms_var = 0.11208895027448523 : f64, value = 404 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3734 = "earth.mul"(%3713, %3733) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3735 = "earth.add"(%3732, %3734) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3736 = "earth.constant"() <{rms_var = 0.14641334546202595 : f64, value = 405 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3737 = "earth.mul"(%3714, %3736) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3738 = "earth.add"(%3735, %3737) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3739 = "earth.constant"() <{rms_var = 0.1350443009373975 : f64, value = 406 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3740 = "earth.mul"(%3715, %3739) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3741 = "earth.add"(%3738, %3740) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3742 = "earth.rotate"(%3741) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3743 = "earth.add"(%3741, %3742) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3744 = "earth.rotate"(%3743) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3745 = "earth.add"(%3743, %3744) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3746 = "earth.rotate"(%3745) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3747 = "earth.add"(%3745, %3746) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3748 = "earth.rotate"(%3747) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3749 = "earth.add"(%3747, %3748) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3750 = "earth.rotate"(%3749) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3751 = "earth.constant"() <{rms_var = 0.025530310408942661 : f64, value = 407 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3752 = "earth.mul"(%3750, %3751) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3753 = "earth.rotate"(%3749) <{offset = array<i64: 16383>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3754 = "earth.constant"() <{rms_var = 0.023864629598244945 : f64, value = 408 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3755 = "earth.mul"(%3753, %3754) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3756 = "earth.add"(%3752, %3755) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3757 = "earth.rotate"(%3749) <{offset = array<i64: 32736>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3758 = "earth.constant"() <{rms_var = 0.026423367619428328 : f64, value = 409 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3759 = "earth.mul"(%3757, %3758) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3760 = "earth.add"(%3756, %3759) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3761 = "earth.rotate"(%3749) <{offset = array<i64: 49119>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3762 = "earth.constant"() <{rms_var = 0.029940495505713621 : f64, value = 410 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3763 = "earth.mul"(%3761, %3762) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3764 = "earth.add"(%3760, %3763) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3765 = "earth.constant"() <{rms_var = 0.11403315564709555 : f64, value = 411 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3766 = "earth.mul"(%3707, %3765) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3767 = "earth.constant"() <{rms_var = 0.10400034518785665 : f64, value = 412 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3768 = "earth.mul"(%3708, %3767) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3769 = "earth.add"(%3766, %3768) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3770 = "earth.constant"() <{rms_var = 0.10069998367161159 : f64, value = 413 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3771 = "earth.mul"(%3709, %3770) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3772 = "earth.add"(%3769, %3771) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3773 = "earth.constant"() <{rms_var = 0.13042044063138716 : f64, value = 414 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3774 = "earth.mul"(%3710, %3773) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3775 = "earth.add"(%3772, %3774) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3776 = "earth.constant"() <{rms_var = 0.20105923627037478 : f64, value = 415 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3777 = "earth.mul"(%3711, %3776) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3778 = "earth.add"(%3775, %3777) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3779 = "earth.constant"() <{rms_var = 0.11295430825386571 : f64, value = 416 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3780 = "earth.mul"(%3712, %3779) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3781 = "earth.add"(%3778, %3780) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3782 = "earth.constant"() <{rms_var = 0.10868863456008128 : f64, value = 417 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3783 = "earth.mul"(%3713, %3782) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3784 = "earth.add"(%3781, %3783) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3785 = "earth.constant"() <{rms_var = 0.11052290226956266 : f64, value = 418 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3786 = "earth.mul"(%3714, %3785) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3787 = "earth.add"(%3784, %3786) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3788 = "earth.constant"() <{rms_var = 0.10241954008698477 : f64, value = 419 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3789 = "earth.mul"(%3715, %3788) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3790 = "earth.add"(%3787, %3789) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3791 = "earth.rotate"(%3790) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3792 = "earth.add"(%3790, %3791) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3793 = "earth.rotate"(%3792) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3794 = "earth.add"(%3792, %3793) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3795 = "earth.rotate"(%3794) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3796 = "earth.add"(%3794, %3795) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3797 = "earth.rotate"(%3796) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3798 = "earth.add"(%3796, %3797) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3799 = "earth.rotate"(%3798) <{offset = array<i64: -1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3800 = "earth.constant"() <{rms_var = 0.022538789307280156 : f64, value = 420 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3801 = "earth.mul"(%3799, %3800) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3802 = "earth.add"(%3764, %3801) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3803 = "earth.rotate"(%3798) <{offset = array<i64: 15359>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3804 = "earth.constant"() <{rms_var = 0.025434442875072561 : f64, value = 421 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3805 = "earth.mul"(%3803, %3804) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3806 = "earth.add"(%3802, %3805) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3807 = "earth.rotate"(%3798) <{offset = array<i64: 31712>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3808 = "earth.constant"() <{rms_var = 0.019139399437026798 : f64, value = 422 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3809 = "earth.mul"(%3807, %3808) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3810 = "earth.add"(%3806, %3809) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3811 = "earth.rotate"(%3798) <{offset = array<i64: 48095>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3812 = "earth.constant"() <{rms_var = 0.028823896616542066 : f64, value = 423 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3813 = "earth.mul"(%3811, %3812) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3814 = "earth.add"(%3810, %3813) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3815 = "earth.constant"() <{rms_var = 0.09613269694135633 : f64, value = 424 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3816 = "earth.mul"(%3707, %3815) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3817 = "earth.constant"() <{rms_var = 0.10200512098214745 : f64, value = 425 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3818 = "earth.mul"(%3708, %3817) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3819 = "earth.add"(%3816, %3818) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3820 = "earth.constant"() <{rms_var = 0.10574848654836355 : f64, value = 426 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3821 = "earth.mul"(%3709, %3820) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3822 = "earth.add"(%3819, %3821) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3823 = "earth.constant"() <{rms_var = 0.12801661596187025 : f64, value = 427 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3824 = "earth.mul"(%3710, %3823) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3825 = "earth.add"(%3822, %3824) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3826 = "earth.constant"() <{rms_var = 0.12481016976353559 : f64, value = 428 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3827 = "earth.mul"(%3711, %3826) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3828 = "earth.add"(%3825, %3827) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3829 = "earth.constant"() <{rms_var = 0.14781414910091423 : f64, value = 429 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3830 = "earth.mul"(%3712, %3829) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3831 = "earth.add"(%3828, %3830) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3832 = "earth.constant"() <{rms_var = 0.12046158482378673 : f64, value = 430 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3833 = "earth.mul"(%3713, %3832) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3834 = "earth.add"(%3831, %3833) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3835 = "earth.constant"() <{rms_var = 0.15133315151945953 : f64, value = 431 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3836 = "earth.mul"(%3714, %3835) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3837 = "earth.add"(%3834, %3836) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3838 = "earth.constant"() <{rms_var = 0.13729027023148013 : f64, value = 432 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3839 = "earth.mul"(%3715, %3838) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3840 = "earth.add"(%3837, %3839) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3841 = "earth.rotate"(%3840) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3842 = "earth.add"(%3840, %3841) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3843 = "earth.rotate"(%3842) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3844 = "earth.add"(%3842, %3843) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3845 = "earth.rotate"(%3844) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3846 = "earth.add"(%3844, %3845) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3847 = "earth.rotate"(%3846) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3848 = "earth.add"(%3846, %3847) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3849 = "earth.rotate"(%3848) <{offset = array<i64: -2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3850 = "earth.constant"() <{rms_var = 0.025020045446593052 : f64, value = 433 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3851 = "earth.mul"(%3849, %3850) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3852 = "earth.add"(%3814, %3851) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3853 = "earth.rotate"(%3848) <{offset = array<i64: 14335>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3854 = "earth.constant"() <{rms_var = 0.031830329803904329 : f64, value = 434 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3855 = "earth.mul"(%3853, %3854) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3856 = "earth.add"(%3852, %3855) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3857 = "earth.rotate"(%3848) <{offset = array<i64: 30688>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3858 = "earth.constant"() <{rms_var = 0.027791770387208339 : f64, value = 435 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3859 = "earth.mul"(%3857, %3858) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3860 = "earth.add"(%3856, %3859) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3861 = "earth.rotate"(%3848) <{offset = array<i64: 47071>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3862 = "earth.constant"() <{rms_var = 0.022929950915575412 : f64, value = 436 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3863 = "earth.mul"(%3861, %3862) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3864 = "earth.add"(%3860, %3863) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3865 = "earth.constant"() <{rms_var = 0.11525721326238633 : f64, value = 437 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3866 = "earth.mul"(%3707, %3865) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3867 = "earth.constant"() <{rms_var = 0.12743922447435982 : f64, value = 438 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3868 = "earth.mul"(%3708, %3867) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3869 = "earth.add"(%3866, %3868) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3870 = "earth.constant"() <{rms_var = 0.10287124341311839 : f64, value = 439 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3871 = "earth.mul"(%3709, %3870) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3872 = "earth.add"(%3869, %3871) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3873 = "earth.constant"() <{rms_var = 0.13783067462397067 : f64, value = 440 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3874 = "earth.mul"(%3710, %3873) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3875 = "earth.add"(%3872, %3874) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3876 = "earth.constant"() <{rms_var = 0.16982660379961304 : f64, value = 441 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3877 = "earth.mul"(%3711, %3876) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3878 = "earth.add"(%3875, %3877) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3879 = "earth.constant"() <{rms_var = 0.1595136696936571 : f64, value = 442 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3880 = "earth.mul"(%3712, %3879) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3881 = "earth.add"(%3878, %3880) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3882 = "earth.constant"() <{rms_var = 0.11162599458984748 : f64, value = 443 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3883 = "earth.mul"(%3713, %3882) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3884 = "earth.add"(%3881, %3883) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3885 = "earth.constant"() <{rms_var = 0.1531891426899385 : f64, value = 444 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3886 = "earth.mul"(%3714, %3885) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3887 = "earth.add"(%3884, %3886) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3888 = "earth.constant"() <{rms_var = 0.1409562479730995 : f64, value = 445 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3889 = "earth.mul"(%3715, %3888) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3890 = "earth.add"(%3887, %3889) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3891 = "earth.rotate"(%3890) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3892 = "earth.add"(%3890, %3891) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3893 = "earth.rotate"(%3892) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3894 = "earth.add"(%3892, %3893) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3895 = "earth.rotate"(%3894) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3896 = "earth.add"(%3894, %3895) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3897 = "earth.rotate"(%3896) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3898 = "earth.add"(%3896, %3897) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3899 = "earth.rotate"(%3898) <{offset = array<i64: -3072>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3900 = "earth.constant"() <{rms_var = 0.024379248984718691 : f64, value = 446 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3901 = "earth.mul"(%3899, %3900) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3902 = "earth.add"(%3864, %3901) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3903 = "earth.rotate"(%3898) <{offset = array<i64: 13311>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3904 = "earth.constant"() <{rms_var = 0.025500867515086515 : f64, value = 447 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3905 = "earth.mul"(%3903, %3904) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3906 = "earth.add"(%3902, %3905) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3907 = "earth.rotate"(%3898) <{offset = array<i64: 29664>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3908 = "earth.constant"() <{rms_var = 0.026916422810615161 : f64, value = 448 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3909 = "earth.mul"(%3907, %3908) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3910 = "earth.add"(%3906, %3909) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3911 = "earth.rotate"(%3898) <{offset = array<i64: 46047>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3912 = "earth.constant"() <{rms_var = 0.020925164212159644 : f64, value = 449 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3913 = "earth.mul"(%3911, %3912) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3914 = "earth.add"(%3910, %3913) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3915 = "earth.constant"() <{rms_var = 0.10447704639042563 : f64, value = 450 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3916 = "earth.mul"(%3707, %3915) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3917 = "earth.constant"() <{rms_var = 0.11810248607076904 : f64, value = 451 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3918 = "earth.mul"(%3708, %3917) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3919 = "earth.add"(%3916, %3918) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3920 = "earth.constant"() <{rms_var = 0.11066211046271161 : f64, value = 452 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3921 = "earth.mul"(%3709, %3920) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3922 = "earth.add"(%3919, %3921) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3923 = "earth.constant"() <{rms_var = 0.13406673437491196 : f64, value = 453 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3924 = "earth.mul"(%3710, %3923) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3925 = "earth.add"(%3922, %3924) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3926 = "earth.constant"() <{rms_var = 0.15828575069089226 : f64, value = 454 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3927 = "earth.mul"(%3711, %3926) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3928 = "earth.add"(%3925, %3927) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3929 = "earth.constant"() <{rms_var = 0.15018141638519422 : f64, value = 455 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3930 = "earth.mul"(%3712, %3929) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3931 = "earth.add"(%3928, %3930) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3932 = "earth.constant"() <{rms_var = 0.12471157092240237 : f64, value = 456 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3933 = "earth.mul"(%3713, %3932) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3934 = "earth.add"(%3931, %3933) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3935 = "earth.constant"() <{rms_var = 0.1627828836423563 : f64, value = 457 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3936 = "earth.mul"(%3714, %3935) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3937 = "earth.add"(%3934, %3936) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3938 = "earth.constant"() <{rms_var = 0.13518298539552356 : f64, value = 458 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3939 = "earth.mul"(%3715, %3938) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3940 = "earth.add"(%3937, %3939) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3941 = "earth.rotate"(%3940) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3942 = "earth.add"(%3940, %3941) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3943 = "earth.rotate"(%3942) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3944 = "earth.add"(%3942, %3943) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3945 = "earth.rotate"(%3944) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3946 = "earth.add"(%3944, %3945) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3947 = "earth.rotate"(%3946) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3948 = "earth.add"(%3946, %3947) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3949 = "earth.rotate"(%3948) <{offset = array<i64: -4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3950 = "earth.constant"() <{rms_var = 0.025254008436597385 : f64, value = 459 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3951 = "earth.mul"(%3949, %3950) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3952 = "earth.add"(%3914, %3951) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3953 = "earth.rotate"(%3948) <{offset = array<i64: 12287>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3954 = "earth.constant"() <{rms_var = 0.027591003613688353 : f64, value = 460 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3955 = "earth.mul"(%3953, %3954) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3956 = "earth.add"(%3952, %3955) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3957 = "earth.rotate"(%3948) <{offset = array<i64: 28640>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3958 = "earth.constant"() <{rms_var = 0.021075123322115676 : f64, value = 461 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3959 = "earth.mul"(%3957, %3958) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3960 = "earth.add"(%3956, %3959) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3961 = "earth.rotate"(%3948) <{offset = array<i64: 45023>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3962 = "earth.constant"() <{rms_var = 0.022036375358980693 : f64, value = 462 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3963 = "earth.mul"(%3961, %3962) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3964 = "earth.add"(%3960, %3963) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3965 = "earth.constant"() <{rms_var = 0.085756413763658876 : f64, value = 463 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3966 = "earth.mul"(%3707, %3965) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3967 = "earth.constant"() <{rms_var = 0.11114693894710674 : f64, value = 464 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3968 = "earth.mul"(%3708, %3967) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3969 = "earth.add"(%3966, %3968) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3970 = "earth.constant"() <{rms_var = 0.093669698287079828 : f64, value = 465 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3971 = "earth.mul"(%3709, %3970) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3972 = "earth.add"(%3969, %3971) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3973 = "earth.constant"() <{rms_var = 0.12351002092867701 : f64, value = 466 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3974 = "earth.mul"(%3710, %3973) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3975 = "earth.add"(%3972, %3974) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3976 = "earth.constant"() <{rms_var = 0.16832369669841207 : f64, value = 467 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3977 = "earth.mul"(%3711, %3976) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3978 = "earth.add"(%3975, %3977) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3979 = "earth.constant"() <{rms_var = 0.13419712969186218 : f64, value = 468 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3980 = "earth.mul"(%3712, %3979) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3981 = "earth.add"(%3978, %3980) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3982 = "earth.constant"() <{rms_var = 0.10597401291267186 : f64, value = 469 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3983 = "earth.mul"(%3713, %3982) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3984 = "earth.add"(%3981, %3983) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3985 = "earth.constant"() <{rms_var = 0.14862762453119049 : f64, value = 470 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3986 = "earth.mul"(%3714, %3985) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3987 = "earth.add"(%3984, %3986) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3988 = "earth.constant"() <{rms_var = 0.12436040825006645 : f64, value = 471 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3989 = "earth.mul"(%3715, %3988) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3990 = "earth.add"(%3987, %3989) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3991 = "earth.rotate"(%3990) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3992 = "earth.add"(%3990, %3991) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3993 = "earth.rotate"(%3992) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3994 = "earth.add"(%3992, %3993) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3995 = "earth.rotate"(%3994) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3996 = "earth.add"(%3994, %3995) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3997 = "earth.rotate"(%3996) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3998 = "earth.add"(%3996, %3997) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %3999 = "earth.rotate"(%3998) <{offset = array<i64: -5120>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4000 = "earth.constant"() <{rms_var = 0.022311173801897947 : f64, value = 472 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4001 = "earth.mul"(%3999, %4000) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4002 = "earth.add"(%3964, %4001) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4003 = "earth.rotate"(%3998) <{offset = array<i64: 11263>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4004 = "earth.constant"() <{rms_var = 0.024323670268022402 : f64, value = 473 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4005 = "earth.mul"(%4003, %4004) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4006 = "earth.add"(%4002, %4005) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4007 = "earth.rotate"(%3998) <{offset = array<i64: 27616>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4008 = "earth.constant"() <{rms_var = 0.02365009288227049 : f64, value = 474 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4009 = "earth.mul"(%4007, %4008) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4010 = "earth.add"(%4006, %4009) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4011 = "earth.rotate"(%3998) <{offset = array<i64: 43999>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4012 = "earth.constant"() <{rms_var = 0.026038546792699343 : f64, value = 475 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4013 = "earth.mul"(%4011, %4012) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4014 = "earth.add"(%4010, %4013) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4015 = "earth.constant"() <{rms_var = 0.11026901254803198 : f64, value = 476 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4016 = "earth.mul"(%3707, %4015) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4017 = "earth.constant"() <{rms_var = 0.12403733152040955 : f64, value = 477 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4018 = "earth.mul"(%3708, %4017) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4019 = "earth.add"(%4016, %4018) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4020 = "earth.constant"() <{rms_var = 0.1147375724888464 : f64, value = 478 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4021 = "earth.mul"(%3709, %4020) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4022 = "earth.add"(%4019, %4021) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4023 = "earth.constant"() <{rms_var = 0.12809961670390638 : f64, value = 479 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4024 = "earth.mul"(%3710, %4023) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4025 = "earth.add"(%4022, %4024) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4026 = "earth.constant"() <{rms_var = 0.14229166792064343 : f64, value = 480 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4027 = "earth.mul"(%3711, %4026) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4028 = "earth.add"(%4025, %4027) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4029 = "earth.constant"() <{rms_var = 0.12747610243717289 : f64, value = 481 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4030 = "earth.mul"(%3712, %4029) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4031 = "earth.add"(%4028, %4030) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4032 = "earth.constant"() <{rms_var = 0.11653334570058418 : f64, value = 482 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4033 = "earth.mul"(%3713, %4032) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4034 = "earth.add"(%4031, %4033) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4035 = "earth.constant"() <{rms_var = 0.14889633125004018 : f64, value = 483 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4036 = "earth.mul"(%3714, %4035) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4037 = "earth.add"(%4034, %4036) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4038 = "earth.constant"() <{rms_var = 0.12208790096846343 : f64, value = 484 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4039 = "earth.mul"(%3715, %4038) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4040 = "earth.add"(%4037, %4039) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4041 = "earth.rotate"(%4040) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4042 = "earth.add"(%4040, %4041) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4043 = "earth.rotate"(%4042) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4044 = "earth.add"(%4042, %4043) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4045 = "earth.rotate"(%4044) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4046 = "earth.add"(%4044, %4045) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4047 = "earth.rotate"(%4046) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4048 = "earth.add"(%4046, %4047) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4049 = "earth.rotate"(%4048) <{offset = array<i64: -6144>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4050 = "earth.constant"() <{rms_var = 0.028319899886396483 : f64, value = 485 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4051 = "earth.mul"(%4049, %4050) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4052 = "earth.add"(%4014, %4051) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4053 = "earth.rotate"(%4048) <{offset = array<i64: 10239>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4054 = "earth.constant"() <{rms_var = 0.022928538596148702 : f64, value = 486 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4055 = "earth.mul"(%4053, %4054) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4056 = "earth.add"(%4052, %4055) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4057 = "earth.rotate"(%4048) <{offset = array<i64: 26592>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4058 = "earth.constant"() <{rms_var = 0.023321635027901872 : f64, value = 487 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4059 = "earth.mul"(%4057, %4058) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4060 = "earth.add"(%4056, %4059) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4061 = "earth.rotate"(%4048) <{offset = array<i64: 42975>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4062 = "earth.constant"() <{rms_var = 0.02937330118292783 : f64, value = 488 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4063 = "earth.mul"(%4061, %4062) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4064 = "earth.add"(%4060, %4063) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4065 = "earth.constant"() <{rms_var = 0.1264093204426491 : f64, value = 489 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4066 = "earth.mul"(%3707, %4065) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4067 = "earth.constant"() <{rms_var = 0.14714148359907614 : f64, value = 490 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4068 = "earth.mul"(%3708, %4067) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4069 = "earth.add"(%4066, %4068) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4070 = "earth.constant"() <{rms_var = 0.1324349382377138 : f64, value = 491 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4071 = "earth.mul"(%3709, %4070) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4072 = "earth.add"(%4069, %4071) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4073 = "earth.constant"() <{rms_var = 0.13024815414059138 : f64, value = 492 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4074 = "earth.mul"(%3710, %4073) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4075 = "earth.add"(%4072, %4074) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4076 = "earth.constant"() <{rms_var = 0.15516032710363725 : f64, value = 493 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4077 = "earth.mul"(%3711, %4076) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4078 = "earth.add"(%4075, %4077) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4079 = "earth.constant"() <{rms_var = 0.15743453899992854 : f64, value = 494 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4080 = "earth.mul"(%3712, %4079) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4081 = "earth.add"(%4078, %4080) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4082 = "earth.constant"() <{rms_var = 0.10773796266832768 : f64, value = 495 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4083 = "earth.mul"(%3713, %4082) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4084 = "earth.add"(%4081, %4083) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4085 = "earth.constant"() <{rms_var = 0.12317609712568746 : f64, value = 496 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4086 = "earth.mul"(%3714, %4085) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4087 = "earth.add"(%4084, %4086) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4088 = "earth.constant"() <{rms_var = 0.12619758519994936 : f64, value = 497 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4089 = "earth.mul"(%3715, %4088) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4090 = "earth.add"(%4087, %4089) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4091 = "earth.rotate"(%4090) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4092 = "earth.add"(%4090, %4091) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4093 = "earth.rotate"(%4092) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4094 = "earth.add"(%4092, %4093) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4095 = "earth.rotate"(%4094) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4096 = "earth.add"(%4094, %4095) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4097 = "earth.rotate"(%4096) <{offset = array<i64: 8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4098 = "earth.add"(%4096, %4097) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4099 = "earth.rotate"(%4098) <{offset = array<i64: -7168>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4100 = "earth.constant"() <{rms_var = 0.028198099818622663 : f64, value = 498 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4101 = "earth.mul"(%4099, %4100) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4102 = "earth.add"(%4064, %4101) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4103 = "earth.rotate"(%4098) <{offset = array<i64: 9215>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4104 = "earth.constant"() <{rms_var = 0.030523579098820385 : f64, value = 499 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4105 = "earth.mul"(%4103, %4104) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4106 = "earth.add"(%4102, %4105) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4107 = "earth.rotate"(%4098) <{offset = array<i64: 25568>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4108 = "earth.constant"() <{rms_var = 0.024372853466473297 : f64, value = 500 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4109 = "earth.mul"(%4107, %4108) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4110 = "earth.add"(%4106, %4109) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4111 = "earth.rotate"(%4098) <{offset = array<i64: 41951>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4112 = "earth.constant"() <{rms_var = 0.028609305641054329 : f64, value = 501 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4113 = "earth.mul"(%4111, %4112) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4114 = "earth.add"(%4110, %4113) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4115 = "earth.rotate"(%4114) <{offset = array<i64: -8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4116 = "earth.add"(%4114, %4115) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4117 = "earth.rotate"(%4116) <{offset = array<i64: -16384>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4118 = "earth.add"(%4116, %4117) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4119 = "earth.rotate"(%4118) <{offset = array<i64: -32768>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4120 = "earth.add"(%4118, %4119) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4121 = "earth.constant"() <{rms_var = 0.029288722340477111 : f64, value = 502 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4122 = "earth.add"(%4120, %4121) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn1
    %4123 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4124 = "earth.mul"(%4123, %4122) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4125 = "earth.mul"(%4124, %4122) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4126 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4127 = "earth.add"(%4125, %4126) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4128 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4129 = "earth.mul"(%4128, %4127) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4130 = "earth.mul"(%4129, %4127) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4131 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4132 = "earth.add"(%4130, %4131) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4133 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4134 = "earth.mul"(%4133, %4132) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4135 = "earth.mul"(%4134, %4132) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4136 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4137 = "earth.add"(%4135, %4136) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4138 = "earth.mul"(%4124, %4127) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4139 = "earth.negate"(%4122) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4140 = "earth.add"(%4138, %4139) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4141 = "earth.mul"(%4124, %4132) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4142 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4143 = "earth.mul"(%4142, %4140) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4144 = "earth.mul"(%4143, %4132) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4145 = "earth.negate"(%4140) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4146 = "earth.add"(%4141, %4145) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4147 = "earth.add"(%4144, %4139) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4148 = "earth.mul"(%4124, %4137) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4149 = "earth.mul"(%4143, %4137) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4150 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4151 = "earth.mul"(%4150, %4146) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4152 = "earth.mul"(%4151, %4137) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4153 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4154 = "earth.mul"(%4153, %4147) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4155 = "earth.mul"(%4154, %4137) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4156 = "earth.negate"(%4147) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4157 = "earth.add"(%4148, %4156) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4158 = "earth.negate"(%4146) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4159 = "earth.add"(%4149, %4158) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4160 = "earth.add"(%4152, %4145) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4161 = "earth.add"(%4155, %4139) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4162 = "earth.constant"() <{rms_var = 0.051379788302527769 : f64, value = 28 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4163 = "earth.mul"(%4162, %4122) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4164 = "earth.constant"() <{rms_var = 0.04272842452341858 : f64, value = 29 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4165 = "earth.mul"(%4164, %4140) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4166 = "earth.add"(%4163, %4165) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4167 = "earth.constant"() <{rms_var = 0.036049430000943392 : f64, value = 30 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4168 = "earth.mul"(%4167, %4146) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4169 = "earth.add"(%4166, %4168) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4170 = "earth.constant"() <{rms_var = 0.030937245302699062 : f64, value = 31 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4171 = "earth.mul"(%4170, %4147) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4172 = "earth.add"(%4169, %4171) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4173 = "earth.constant"() <{rms_var = 0.027115258485962086 : f64, value = 32 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4174 = "earth.mul"(%4173, %4157) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4175 = "earth.add"(%4172, %4174) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4176 = "earth.constant"() <{rms_var = 0.024393077542268389 : f64, value = 33 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4177 = "earth.mul"(%4176, %4159) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4178 = "earth.add"(%4175, %4177) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4179 = "earth.constant"() <{rms_var = 0.022642601074970584 : f64, value = 34 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4180 = "earth.mul"(%4179, %4160) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4181 = "earth.add"(%4178, %4180) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4182 = "earth.constant"() <{rms_var = 0.021831260875010087 : f64, value = 35 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4183 = "earth.mul"(%4182, %4161) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4184 = "earth.add"(%4181, %4183) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4185 = "earth.constant"() <{rms_var = 0.021652089836507502 : f64, value = 36 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4186 = "earth.mul"(%4185, %4122) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4187 = "earth.constant"() <{rms_var = 0.018138222134916806 : f64, value = 37 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4188 = "earth.mul"(%4187, %4140) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4189 = "earth.add"(%4186, %4188) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4190 = "earth.constant"() <{rms_var = 0.01542303411540253 : f64, value = 38 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4191 = "earth.mul"(%4190, %4146) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4192 = "earth.add"(%4189, %4191) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4193 = "earth.constant"() <{rms_var = 0.013305654693862069 : f64, value = 39 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4194 = "earth.mul"(%4193, %4147) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4195 = "earth.add"(%4192, %4194) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4196 = "earth.constant"() <{rms_var = 0.011703046220094507 : f64, value = 40 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4197 = "earth.mul"(%4196, %4157) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4198 = "earth.add"(%4195, %4197) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4199 = "earth.constant"() <{rms_var = 0.010552642814751455 : f64, value = 41 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4200 = "earth.mul"(%4199, %4159) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4201 = "earth.add"(%4198, %4200) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4202 = "earth.constant"() <{rms_var = 0.0098096659804816355 : f64, value = 42 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4203 = "earth.mul"(%4202, %4160) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4204 = "earth.add"(%4201, %4203) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4205 = "earth.constant"() <{rms_var = 0.0094452495559895089 : f64, value = 43 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4206 = "earth.mul"(%4205, %4161) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4207 = "earth.add"(%4204, %4206) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4208 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4209 = "earth.mul"(%4208, %4137) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4210 = "earth.mul"(%4209, %4137) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4211 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4212 = "earth.add"(%4210, %4211) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4213 = "earth.mul"(%4207, %4212) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4214 = "earth.add"(%4213, %4184) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4215 = "earth.constant"() <{rms_var = 0.0042995278292336028 : f64, value = 44 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4216 = "earth.mul"(%4215, %4122) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4217 = "earth.constant"() <{rms_var = 0.0036191919476548503 : f64, value = 45 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4218 = "earth.mul"(%4217, %4140) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4219 = "earth.add"(%4216, %4218) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4220 = "earth.constant"() <{rms_var = 0.0030784419617177587 : f64, value = 46 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4221 = "earth.mul"(%4220, %4146) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4222 = "earth.add"(%4219, %4221) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4223 = "earth.constant"() <{rms_var = 0.0026564062051904268 : f64, value = 47 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4224 = "earth.mul"(%4223, %4147) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4225 = "earth.add"(%4222, %4224) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4226 = "earth.constant"() <{rms_var = 0.0023368004457586414 : f64, value = 48 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4227 = "earth.mul"(%4226, %4157) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4228 = "earth.add"(%4225, %4227) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4229 = "earth.constant"() <{rms_var = 0.002107295415524122 : f64, value = 49 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4230 = "earth.mul"(%4229, %4159) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4231 = "earth.add"(%4228, %4230) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4232 = "earth.constant"() <{rms_var = 0.0019590388891824136 : f64, value = 50 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4233 = "earth.mul"(%4232, %4160) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4234 = "earth.add"(%4231, %4233) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4235 = "earth.constant"() <{rms_var = 0.0018863129851277056 : f64, value = 51 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4236 = "earth.mul"(%4235, %4161) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4237 = "earth.add"(%4234, %4236) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4238 = "earth.constant"() <{rms_var = 0.63615477792235875 : f64, value = 52 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4239 = "earth.mul"(%4238, %4122) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4240 = "earth.constant"() <{rms_var = 0.21189795368271022 : f64, value = 53 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4241 = "earth.mul"(%4240, %4140) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4242 = "earth.add"(%4239, %4241) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4243 = "earth.constant"() <{rms_var = 0.12734215758896472 : f64, value = 54 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4244 = "earth.mul"(%4243, %4146) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4245 = "earth.add"(%4242, %4244) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4246 = "earth.constant"() <{rms_var = 0.091626347972833505 : f64, value = 55 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4247 = "earth.mul"(%4246, %4147) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4248 = "earth.add"(%4245, %4247) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4249 = "earth.constant"() <{rms_var = 0.07255941147658998 : f64, value = 56 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4250 = "earth.mul"(%4249, %4157) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4251 = "earth.add"(%4248, %4250) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4252 = "earth.constant"() <{rms_var = 0.061477909207391525 : f64, value = 57 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4253 = "earth.mul"(%4252, %4159) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4254 = "earth.add"(%4251, %4253) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4255 = "earth.constant"() <{rms_var = 0.055169030070493508 : f64, value = 58 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4256 = "earth.mul"(%4255, %4160) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4257 = "earth.add"(%4254, %4256) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4258 = "earth.constant"() <{rms_var = 0.052275954916496656 : f64, value = 59 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4259 = "earth.mul"(%4258, %4161) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4260 = "earth.add"(%4257, %4259) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4261 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4262 = "earth.mul"(%4261, %4212) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4263 = "earth.mul"(%4262, %4212) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4264 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4265 = "earth.add"(%4263, %4264) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4266 = "earth.mul"(%4237, %4265) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4267 = "earth.add"(%4266, %4214) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4268 = "earth.constant"() <{rms_var = 4.9481895575581374E-4 : f64, value = 60 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4269 = "earth.mul"(%4268, %4122) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4270 = "earth.constant"() <{rms_var = 3.771067298313324E-4 : f64, value = 61 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4271 = "earth.mul"(%4270, %4140) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4272 = "earth.add"(%4269, %4271) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4273 = "earth.constant"() <{rms_var = 3.2076765303085136E-4 : f64, value = 62 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4274 = "earth.mul"(%4273, %4146) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4275 = "earth.add"(%4272, %4274) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4276 = "earth.constant"() <{rms_var = 2.7679532400224407E-4 : f64, value = 63 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4277 = "earth.mul"(%4276, %4147) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4278 = "earth.add"(%4275, %4277) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4279 = "earth.constant"() <{rms_var = 2.4349440690240933E-4 : f64, value = 64 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4280 = "earth.mul"(%4279, %4157) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4281 = "earth.add"(%4278, %4280) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4282 = "earth.constant"() <{rms_var = 2.1958101095650637E-4 : f64, value = 65 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4283 = "earth.mul"(%4282, %4159) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4284 = "earth.add"(%4281, %4283) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4285 = "earth.constant"() <{rms_var = 2.0413318001830463E-4 : f64, value = 66 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4286 = "earth.mul"(%4285, %4160) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4287 = "earth.add"(%4284, %4286) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4288 = "earth.constant"() <{rms_var = 1.9655534146569248E-4 : f64, value = 67 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4289 = "earth.mul"(%4288, %4161) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4290 = "earth.add"(%4287, %4289) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4291 = "earth.constant"() <{rms_var = 1.7735088548126505E-4 : f64, value = 68 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4292 = "earth.mul"(%4291, %4122) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4293 = "earth.constant"() <{rms_var = 1.457794164045128E-4 : f64, value = 69 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4294 = "earth.mul"(%4293, %4140) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4295 = "earth.add"(%4292, %4294) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4296 = "earth.constant"() <{rms_var = 1.1982820692495149E-4 : f64, value = 70 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4297 = "earth.mul"(%4296, %4146) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4298 = "earth.add"(%4295, %4297) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4299 = "earth.constant"() <{rms_var = 9.8496754159654756E-5 : f64, value = 71 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4300 = "earth.mul"(%4299, %4147) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4301 = "earth.add"(%4298, %4300) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4302 = "earth.constant"() <{rms_var = 8.0962662339626343E-5 : f64, value = 72 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4303 = "earth.mul"(%4302, %4157) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4304 = "earth.add"(%4301, %4303) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4305 = "earth.constant"() <{rms_var = 6.6549936439435966E-5 : f64, value = 73 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4306 = "earth.mul"(%4305, %4159) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4307 = "earth.add"(%4304, %4306) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4308 = "earth.constant"() <{rms_var = 5.4702920205367267E-5 : f64, value = 74 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4309 = "earth.mul"(%4308, %4160) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4310 = "earth.add"(%4307, %4309) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4311 = "earth.constant"() <{rms_var = 1.3863334887734771E-4 : f64, value = 75 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4312 = "earth.mul"(%4311, %4161) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4313 = "earth.add"(%4310, %4312) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4314 = "earth.mul"(%4267, %4212) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4315 = "earth.add"(%4314, %4260) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4316 = "earth.mul"(%4313, %4212) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4317 = "earth.add"(%4316, %4290) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4318 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4319 = "earth.mul"(%4318, %4265) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4320 = "earth.mul"(%4319, %4265) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4321 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4322 = "earth.add"(%4320, %4321) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4323 = "earth.mul"(%4317, %4322) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4324 = "earth.add"(%4323, %4315) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act1_SiLU_poly
    %4325 = "earth.constant"() <{rms_var = 5.000000e-01 : f64, value = 76 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // add[]layer2_0_act1_SiLU_add
    %4326 = "earth.add"(%4324, %4325) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // add[]layer2_0_act1_SiLU_add
    %4327 = "earth.mul"(%4122, %4326) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // mul[]layer2_0_act1_SiLU_mul
    %4328 = "earth.rotate"(%4327) <{offset = array<i64: -66>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4329 = "earth.rotate"(%4327) <{offset = array<i64: -64>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4330 = "earth.rotate"(%4327) <{offset = array<i64: -62>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4331 = "earth.rotate"(%4327) <{offset = array<i64: -2>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4332 = "earth.rotate"(%4327) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4333 = "earth.rotate"(%4327) <{offset = array<i64: 2>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4334 = "earth.rotate"(%4327) <{offset = array<i64: 62>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4335 = "earth.rotate"(%4327) <{offset = array<i64: 64>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4336 = "earth.rotate"(%4327) <{offset = array<i64: 66>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4337 = "earth.constant"() <{rms_var = 0.082134801628763104 : f64, value = 503 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4338 = "earth.mul"(%4328, %4337) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4339 = "earth.constant"() <{rms_var = 0.1024358100160576 : f64, value = 504 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4340 = "earth.mul"(%4329, %4339) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4341 = "earth.add"(%4338, %4340) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4342 = "earth.constant"() <{rms_var = 0.099749702439570403 : f64, value = 505 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4343 = "earth.mul"(%4330, %4342) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4344 = "earth.add"(%4341, %4343) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4345 = "earth.constant"() <{rms_var = 0.094515518044384599 : f64, value = 506 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4346 = "earth.mul"(%4331, %4345) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4347 = "earth.add"(%4344, %4346) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4348 = "earth.constant"() <{rms_var = 0.14084702636544305 : f64, value = 507 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4349 = "earth.mul"(%4332, %4348) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4350 = "earth.add"(%4347, %4349) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4351 = "earth.constant"() <{rms_var = 0.12503570281018311 : f64, value = 508 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4352 = "earth.mul"(%4333, %4351) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4353 = "earth.add"(%4350, %4352) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4354 = "earth.constant"() <{rms_var = 0.1009744772738372 : f64, value = 509 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4355 = "earth.mul"(%4334, %4354) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4356 = "earth.add"(%4353, %4355) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4357 = "earth.constant"() <{rms_var = 0.1196030688970406 : f64, value = 510 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4358 = "earth.mul"(%4335, %4357) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4359 = "earth.add"(%4356, %4358) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4360 = "earth.constant"() <{rms_var = 0.095938051803294719 : f64, value = 511 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4361 = "earth.mul"(%4336, %4360) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4362 = "earth.add"(%4359, %4361) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4363 = "earth.rotate"(%4362) <{offset = array<i64: 1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4364 = "earth.add"(%4362, %4363) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4365 = "earth.rotate"(%4364) <{offset = array<i64: 32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4366 = "earth.add"(%4364, %4365) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4367 = "earth.rotate"(%4366) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4368 = "earth.add"(%4366, %4367) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4369 = "earth.rotate"(%4368) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4370 = "earth.add"(%4368, %4369) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4371 = "earth.rotate"(%4370) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4372 = "earth.add"(%4370, %4371) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4373 = "earth.rotate"(%4372) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4374 = "earth.constant"() <{rms_var = 0.043959297226222681 : f64, value = 512 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4375 = "earth.mul"(%4373, %4374) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4376 = "earth.rotate"(%4372) <{offset = array<i64: 8191>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4377 = "earth.constant"() <{rms_var = 0.040942583366902499 : f64, value = 513 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4378 = "earth.mul"(%4376, %4377) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4379 = "earth.add"(%4375, %4378) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4380 = "earth.rotate"(%4372) <{offset = array<i64: 16352>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4381 = "earth.constant"() <{rms_var = 0.038934823281105098 : f64, value = 514 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4382 = "earth.mul"(%4380, %4381) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4383 = "earth.add"(%4379, %4382) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4384 = "earth.rotate"(%4372) <{offset = array<i64: 24543>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4385 = "earth.constant"() <{rms_var = 0.050370631307341741 : f64, value = 515 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4386 = "earth.mul"(%4384, %4385) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4387 = "earth.add"(%4383, %4386) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4388 = "earth.rotate"(%4372) <{offset = array<i64: 31744>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4389 = "earth.constant"() <{rms_var = 0.04873185577228753 : f64, value = 516 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4390 = "earth.mul"(%4388, %4389) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4391 = "earth.add"(%4387, %4390) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4392 = "earth.rotate"(%4372) <{offset = array<i64: 39935>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4393 = "earth.constant"() <{rms_var = 0.055291647986972565 : f64, value = 517 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4394 = "earth.mul"(%4392, %4393) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4395 = "earth.add"(%4391, %4394) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4396 = "earth.rotate"(%4372) <{offset = array<i64: 48096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4397 = "earth.constant"() <{rms_var = 0.04194385995690874 : f64, value = 518 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4398 = "earth.mul"(%4396, %4397) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4399 = "earth.add"(%4395, %4398) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4400 = "earth.rotate"(%4372) <{offset = array<i64: 56287>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4401 = "earth.constant"() <{rms_var = 0.043220802327916662 : f64, value = 519 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4402 = "earth.mul"(%4400, %4401) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4403 = "earth.add"(%4399, %4402) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4404 = "earth.constant"() <{rms_var = 0.073799948312766608 : f64, value = 520 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4405 = "earth.mul"(%4328, %4404) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4406 = "earth.constant"() <{rms_var = 0.080450534124323 : f64, value = 521 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4407 = "earth.mul"(%4329, %4406) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4408 = "earth.add"(%4405, %4407) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4409 = "earth.constant"() <{rms_var = 0.087011420583298152 : f64, value = 522 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4410 = "earth.mul"(%4330, %4409) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4411 = "earth.add"(%4408, %4410) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4412 = "earth.constant"() <{rms_var = 0.10589916837321711 : f64, value = 523 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4413 = "earth.mul"(%4331, %4412) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4414 = "earth.add"(%4411, %4413) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4415 = "earth.constant"() <{rms_var = 0.14704911576387503 : f64, value = 524 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4416 = "earth.mul"(%4332, %4415) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4417 = "earth.add"(%4414, %4416) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4418 = "earth.constant"() <{rms_var = 0.11399703523997401 : f64, value = 525 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4419 = "earth.mul"(%4333, %4418) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4420 = "earth.add"(%4417, %4419) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4421 = "earth.constant"() <{rms_var = 0.083360347439407789 : f64, value = 526 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4422 = "earth.mul"(%4334, %4421) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4423 = "earth.add"(%4420, %4422) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4424 = "earth.constant"() <{rms_var = 0.10455395940019889 : f64, value = 527 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4425 = "earth.mul"(%4335, %4424) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4426 = "earth.add"(%4423, %4425) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4427 = "earth.constant"() <{rms_var = 0.091058978893993323 : f64, value = 528 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4428 = "earth.mul"(%4336, %4427) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4429 = "earth.add"(%4426, %4428) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4430 = "earth.rotate"(%4429) <{offset = array<i64: 1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4431 = "earth.add"(%4429, %4430) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4432 = "earth.rotate"(%4431) <{offset = array<i64: 32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4433 = "earth.add"(%4431, %4432) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4434 = "earth.rotate"(%4433) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4435 = "earth.add"(%4433, %4434) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4436 = "earth.rotate"(%4435) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4437 = "earth.add"(%4435, %4436) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4438 = "earth.rotate"(%4437) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4439 = "earth.add"(%4437, %4438) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4440 = "earth.rotate"(%4439) <{offset = array<i64: -2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4441 = "earth.constant"() <{rms_var = 0.08727345415137111 : f64, value = 529 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4442 = "earth.mul"(%4440, %4441) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4443 = "earth.add"(%4403, %4442) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4444 = "earth.rotate"(%4439) <{offset = array<i64: 6143>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4445 = "earth.constant"() <{rms_var = 0.030299359114017724 : f64, value = 530 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4446 = "earth.mul"(%4444, %4445) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4447 = "earth.add"(%4443, %4446) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4448 = "earth.rotate"(%4439) <{offset = array<i64: 14304>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4449 = "earth.constant"() <{rms_var = 0.044054552909278483 : f64, value = 531 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4450 = "earth.mul"(%4448, %4449) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4451 = "earth.add"(%4447, %4450) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4452 = "earth.rotate"(%4439) <{offset = array<i64: 22495>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4453 = "earth.constant"() <{rms_var = 0.037950096909936906 : f64, value = 532 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4454 = "earth.mul"(%4452, %4453) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4455 = "earth.add"(%4451, %4454) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4456 = "earth.rotate"(%4439) <{offset = array<i64: 29696>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4457 = "earth.constant"() <{rms_var = 0.046452021570060009 : f64, value = 533 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4458 = "earth.mul"(%4456, %4457) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4459 = "earth.add"(%4455, %4458) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4460 = "earth.rotate"(%4439) <{offset = array<i64: 37887>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4461 = "earth.constant"() <{rms_var = 0.019524457847883869 : f64, value = 534 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4462 = "earth.mul"(%4460, %4461) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4463 = "earth.add"(%4459, %4462) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4464 = "earth.rotate"(%4439) <{offset = array<i64: 46048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4465 = "earth.constant"() <{rms_var = 0.022099601365286874 : f64, value = 535 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4466 = "earth.mul"(%4464, %4465) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4467 = "earth.add"(%4463, %4466) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4468 = "earth.rotate"(%4439) <{offset = array<i64: 54239>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4469 = "earth.constant"() <{rms_var = 0.054148714578300311 : f64, value = 536 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4470 = "earth.mul"(%4468, %4469) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4471 = "earth.add"(%4467, %4470) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4472 = "earth.constant"() <{rms_var = 0.086831915682563431 : f64, value = 537 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4473 = "earth.mul"(%4328, %4472) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4474 = "earth.constant"() <{rms_var = 0.10785422016947509 : f64, value = 538 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4475 = "earth.mul"(%4329, %4474) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4476 = "earth.add"(%4473, %4475) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4477 = "earth.constant"() <{rms_var = 0.075672417365272457 : f64, value = 539 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4478 = "earth.mul"(%4330, %4477) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4479 = "earth.add"(%4476, %4478) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4480 = "earth.constant"() <{rms_var = 0.10238454538377437 : f64, value = 540 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4481 = "earth.mul"(%4331, %4480) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4482 = "earth.add"(%4479, %4481) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4483 = "earth.constant"() <{rms_var = 0.20089054823980926 : f64, value = 541 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4484 = "earth.mul"(%4332, %4483) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4485 = "earth.add"(%4482, %4484) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4486 = "earth.constant"() <{rms_var = 0.096051997733828053 : f64, value = 542 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4487 = "earth.mul"(%4333, %4486) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4488 = "earth.add"(%4485, %4487) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4489 = "earth.constant"() <{rms_var = 0.085780515889839495 : f64, value = 543 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4490 = "earth.mul"(%4334, %4489) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4491 = "earth.add"(%4488, %4490) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4492 = "earth.constant"() <{rms_var = 0.10864781706984845 : f64, value = 544 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4493 = "earth.mul"(%4335, %4492) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4494 = "earth.add"(%4491, %4493) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4495 = "earth.constant"() <{rms_var = 0.09614655324700258 : f64, value = 545 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4496 = "earth.mul"(%4336, %4495) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4497 = "earth.add"(%4494, %4496) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4498 = "earth.rotate"(%4497) <{offset = array<i64: 1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4499 = "earth.add"(%4497, %4498) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4500 = "earth.rotate"(%4499) <{offset = array<i64: 32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4501 = "earth.add"(%4499, %4500) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4502 = "earth.rotate"(%4501) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4503 = "earth.add"(%4501, %4502) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4504 = "earth.rotate"(%4503) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4505 = "earth.add"(%4503, %4504) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4506 = "earth.rotate"(%4505) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4507 = "earth.add"(%4505, %4506) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4508 = "earth.rotate"(%4507) <{offset = array<i64: -4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4509 = "earth.constant"() <{rms_var = 0.025629459002203352 : f64, value = 546 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4510 = "earth.mul"(%4508, %4509) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4511 = "earth.add"(%4471, %4510) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4512 = "earth.rotate"(%4507) <{offset = array<i64: 4095>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4513 = "earth.constant"() <{rms_var = 0.073308780556518405 : f64, value = 547 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4514 = "earth.mul"(%4512, %4513) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4515 = "earth.add"(%4511, %4514) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4516 = "earth.rotate"(%4507) <{offset = array<i64: 12256>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4517 = "earth.constant"() <{rms_var = 0.03788504431663188 : f64, value = 548 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4518 = "earth.mul"(%4516, %4517) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4519 = "earth.add"(%4515, %4518) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4520 = "earth.rotate"(%4507) <{offset = array<i64: 20447>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4521 = "earth.constant"() <{rms_var = 0.047850311446919022 : f64, value = 549 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4522 = "earth.mul"(%4520, %4521) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4523 = "earth.add"(%4519, %4522) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4524 = "earth.rotate"(%4507) <{offset = array<i64: 27648>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4525 = "earth.constant"() <{rms_var = 0.028712213782065999 : f64, value = 550 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4526 = "earth.mul"(%4524, %4525) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4527 = "earth.add"(%4523, %4526) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4528 = "earth.rotate"(%4507) <{offset = array<i64: 35839>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4529 = "earth.constant"() <{rms_var = 0.083555627245750844 : f64, value = 551 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4530 = "earth.mul"(%4528, %4529) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4531 = "earth.add"(%4527, %4530) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4532 = "earth.rotate"(%4507) <{offset = array<i64: 44000>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4533 = "earth.constant"() <{rms_var = 0.037616148157379188 : f64, value = 552 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4534 = "earth.mul"(%4532, %4533) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4535 = "earth.add"(%4531, %4534) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4536 = "earth.rotate"(%4507) <{offset = array<i64: 52191>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4537 = "earth.constant"() <{rms_var = 0.039737293025768429 : f64, value = 553 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4538 = "earth.mul"(%4536, %4537) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4539 = "earth.add"(%4535, %4538) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4540 = "earth.constant"() <{rms_var = 0.085796326020878111 : f64, value = 554 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4541 = "earth.mul"(%4328, %4540) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4542 = "earth.constant"() <{rms_var = 0.097797166672682039 : f64, value = 555 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4543 = "earth.mul"(%4329, %4542) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4544 = "earth.add"(%4541, %4543) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4545 = "earth.constant"() <{rms_var = 0.09581760448223392 : f64, value = 556 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4546 = "earth.mul"(%4330, %4545) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4547 = "earth.add"(%4544, %4546) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4548 = "earth.constant"() <{rms_var = 0.09700492704137019 : f64, value = 557 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4549 = "earth.mul"(%4331, %4548) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4550 = "earth.add"(%4547, %4549) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4551 = "earth.constant"() <{rms_var = 0.14443943992392547 : f64, value = 558 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4552 = "earth.mul"(%4332, %4551) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4553 = "earth.add"(%4550, %4552) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4554 = "earth.constant"() <{rms_var = 0.12884631844905856 : f64, value = 559 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4555 = "earth.mul"(%4333, %4554) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4556 = "earth.add"(%4553, %4555) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4557 = "earth.constant"() <{rms_var = 0.091747326266504711 : f64, value = 560 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4558 = "earth.mul"(%4334, %4557) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4559 = "earth.add"(%4556, %4558) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4560 = "earth.constant"() <{rms_var = 0.12151295421274166 : f64, value = 561 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4561 = "earth.mul"(%4335, %4560) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4562 = "earth.add"(%4559, %4561) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4563 = "earth.constant"() <{rms_var = 0.11810894501728257 : f64, value = 562 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4564 = "earth.mul"(%4336, %4563) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4565 = "earth.add"(%4562, %4564) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4566 = "earth.rotate"(%4565) <{offset = array<i64: 1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4567 = "earth.add"(%4565, %4566) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4568 = "earth.rotate"(%4567) <{offset = array<i64: 32>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4569 = "earth.add"(%4567, %4568) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4570 = "earth.rotate"(%4569) <{offset = array<i64: 1024>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4571 = "earth.add"(%4569, %4570) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4572 = "earth.rotate"(%4571) <{offset = array<i64: 2048>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4573 = "earth.add"(%4571, %4572) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4574 = "earth.rotate"(%4573) <{offset = array<i64: 4096>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4575 = "earth.add"(%4573, %4574) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4576 = "earth.rotate"(%4575) <{offset = array<i64: -6144>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4577 = "earth.constant"() <{rms_var = 0.039479121291820095 : f64, value = 563 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4578 = "earth.mul"(%4576, %4577) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4579 = "earth.add"(%4539, %4578) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4580 = "earth.rotate"(%4575) <{offset = array<i64: 2047>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4581 = "earth.constant"() <{rms_var = 0.038379119866049999 : f64, value = 564 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4582 = "earth.mul"(%4580, %4581) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4583 = "earth.add"(%4579, %4582) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4584 = "earth.rotate"(%4575) <{offset = array<i64: 10208>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4585 = "earth.constant"() <{rms_var = 0.036996646727129927 : f64, value = 565 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4586 = "earth.mul"(%4584, %4585) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4587 = "earth.add"(%4583, %4586) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4588 = "earth.rotate"(%4575) <{offset = array<i64: 18399>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4589 = "earth.constant"() <{rms_var = 0.048818458489916471 : f64, value = 566 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4590 = "earth.mul"(%4588, %4589) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4591 = "earth.add"(%4587, %4590) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4592 = "earth.rotate"(%4575) <{offset = array<i64: 25600>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4593 = "earth.constant"() <{rms_var = 0.051752149550394727 : f64, value = 567 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4594 = "earth.mul"(%4592, %4593) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4595 = "earth.add"(%4591, %4594) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4596 = "earth.rotate"(%4575) <{offset = array<i64: 33791>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4597 = "earth.constant"() <{rms_var = 0.0450174117012129 : f64, value = 568 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4598 = "earth.mul"(%4596, %4597) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4599 = "earth.add"(%4595, %4598) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4600 = "earth.rotate"(%4575) <{offset = array<i64: 41952>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4601 = "earth.constant"() <{rms_var = 0.04187542108972251 : f64, value = 569 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4602 = "earth.mul"(%4600, %4601) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4603 = "earth.add"(%4599, %4602) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4604 = "earth.rotate"(%4575) <{offset = array<i64: 50143>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4605 = "earth.constant"() <{rms_var = 0.036606564455593973 : f64, value = 570 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4606 = "earth.mul"(%4604, %4605) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4607 = "earth.add"(%4603, %4606) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4608 = "earth.rotate"(%4607) <{offset = array<i64: -8192>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4609 = "earth.add"(%4607, %4608) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4610 = "earth.rotate"(%4609) <{offset = array<i64: -16384>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4611 = "earth.add"(%4609, %4610) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4612 = "earth.rotate"(%4611) <{offset = array<i64: -32768>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4613 = "earth.add"(%4611, %4612) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4614 = "earth.constant"() <{rms_var = 0.013502675213112148 : f64, value = 571 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4615 = "earth.add"(%4613, %4614) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // convbn[]layer2_0_convbn2
    %4616 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4617 = "earth.mul"(%4616, %4615) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4618 = "earth.mul"(%4617, %4615) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4619 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4620 = "earth.add"(%4618, %4619) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4621 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4622 = "earth.mul"(%4621, %4620) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4623 = "earth.mul"(%4622, %4620) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4624 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4625 = "earth.add"(%4623, %4624) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4626 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4627 = "earth.mul"(%4626, %4625) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4628 = "earth.mul"(%4627, %4625) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4629 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4630 = "earth.add"(%4628, %4629) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4631 = "earth.mul"(%4617, %4620) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4632 = "earth.negate"(%4615) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4633 = "earth.add"(%4631, %4632) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4634 = "earth.mul"(%4617, %4625) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4635 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4636 = "earth.mul"(%4635, %4633) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4637 = "earth.mul"(%4636, %4625) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4638 = "earth.negate"(%4633) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4639 = "earth.add"(%4634, %4638) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4640 = "earth.add"(%4637, %4632) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4641 = "earth.mul"(%4617, %4630) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4642 = "earth.mul"(%4636, %4630) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4643 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4644 = "earth.mul"(%4643, %4639) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4645 = "earth.mul"(%4644, %4630) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4646 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4647 = "earth.mul"(%4646, %4640) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4648 = "earth.mul"(%4647, %4630) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4649 = "earth.negate"(%4640) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4650 = "earth.add"(%4641, %4649) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4651 = "earth.negate"(%4639) : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4652 = "earth.add"(%4642, %4651) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4653 = "earth.add"(%4645, %4638) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4654 = "earth.add"(%4648, %4632) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4655 = "earth.constant"() <{rms_var = 0.051379788302527769 : f64, value = 28 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4656 = "earth.mul"(%4655, %4615) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4657 = "earth.constant"() <{rms_var = 0.04272842452341858 : f64, value = 29 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4658 = "earth.mul"(%4657, %4633) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4659 = "earth.add"(%4656, %4658) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4660 = "earth.constant"() <{rms_var = 0.036049430000943392 : f64, value = 30 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4661 = "earth.mul"(%4660, %4639) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4662 = "earth.add"(%4659, %4661) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4663 = "earth.constant"() <{rms_var = 0.030937245302699062 : f64, value = 31 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4664 = "earth.mul"(%4663, %4640) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4665 = "earth.add"(%4662, %4664) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4666 = "earth.constant"() <{rms_var = 0.027115258485962086 : f64, value = 32 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4667 = "earth.mul"(%4666, %4650) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4668 = "earth.add"(%4665, %4667) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4669 = "earth.constant"() <{rms_var = 0.024393077542268389 : f64, value = 33 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4670 = "earth.mul"(%4669, %4652) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4671 = "earth.add"(%4668, %4670) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4672 = "earth.constant"() <{rms_var = 0.022642601074970584 : f64, value = 34 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4673 = "earth.mul"(%4672, %4653) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4674 = "earth.add"(%4671, %4673) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4675 = "earth.constant"() <{rms_var = 0.021831260875010087 : f64, value = 35 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4676 = "earth.mul"(%4675, %4654) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4677 = "earth.add"(%4674, %4676) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4678 = "earth.constant"() <{rms_var = 0.021652089836507502 : f64, value = 36 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4679 = "earth.mul"(%4678, %4615) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4680 = "earth.constant"() <{rms_var = 0.018138222134916806 : f64, value = 37 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4681 = "earth.mul"(%4680, %4633) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4682 = "earth.add"(%4679, %4681) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4683 = "earth.constant"() <{rms_var = 0.01542303411540253 : f64, value = 38 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4684 = "earth.mul"(%4683, %4639) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4685 = "earth.add"(%4682, %4684) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4686 = "earth.constant"() <{rms_var = 0.013305654693862069 : f64, value = 39 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4687 = "earth.mul"(%4686, %4640) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4688 = "earth.add"(%4685, %4687) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4689 = "earth.constant"() <{rms_var = 0.011703046220094507 : f64, value = 40 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4690 = "earth.mul"(%4689, %4650) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4691 = "earth.add"(%4688, %4690) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4692 = "earth.constant"() <{rms_var = 0.010552642814751455 : f64, value = 41 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4693 = "earth.mul"(%4692, %4652) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4694 = "earth.add"(%4691, %4693) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4695 = "earth.constant"() <{rms_var = 0.0098096659804816355 : f64, value = 42 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4696 = "earth.mul"(%4695, %4653) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4697 = "earth.add"(%4694, %4696) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4698 = "earth.constant"() <{rms_var = 0.0094452495559895089 : f64, value = 43 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4699 = "earth.mul"(%4698, %4654) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4700 = "earth.add"(%4697, %4699) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4701 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4702 = "earth.mul"(%4701, %4630) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4703 = "earth.mul"(%4702, %4630) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4704 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4705 = "earth.add"(%4703, %4704) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4706 = "earth.mul"(%4700, %4705) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4707 = "earth.add"(%4706, %4677) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4708 = "earth.constant"() <{rms_var = 0.0042995278292336028 : f64, value = 44 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4709 = "earth.mul"(%4708, %4615) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4710 = "earth.constant"() <{rms_var = 0.0036191919476548503 : f64, value = 45 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4711 = "earth.mul"(%4710, %4633) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4712 = "earth.add"(%4709, %4711) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4713 = "earth.constant"() <{rms_var = 0.0030784419617177587 : f64, value = 46 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4714 = "earth.mul"(%4713, %4639) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4715 = "earth.add"(%4712, %4714) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4716 = "earth.constant"() <{rms_var = 0.0026564062051904268 : f64, value = 47 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4717 = "earth.mul"(%4716, %4640) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4718 = "earth.add"(%4715, %4717) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4719 = "earth.constant"() <{rms_var = 0.0023368004457586414 : f64, value = 48 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4720 = "earth.mul"(%4719, %4650) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4721 = "earth.add"(%4718, %4720) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4722 = "earth.constant"() <{rms_var = 0.002107295415524122 : f64, value = 49 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4723 = "earth.mul"(%4722, %4652) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4724 = "earth.add"(%4721, %4723) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4725 = "earth.constant"() <{rms_var = 0.0019590388891824136 : f64, value = 50 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4726 = "earth.mul"(%4725, %4653) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4727 = "earth.add"(%4724, %4726) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4728 = "earth.constant"() <{rms_var = 0.0018863129851277056 : f64, value = 51 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4729 = "earth.mul"(%4728, %4654) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4730 = "earth.add"(%4727, %4729) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4731 = "earth.constant"() <{rms_var = 0.63615477792235875 : f64, value = 52 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4732 = "earth.mul"(%4731, %4615) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4733 = "earth.constant"() <{rms_var = 0.21189795368271022 : f64, value = 53 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4734 = "earth.mul"(%4733, %4633) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4735 = "earth.add"(%4732, %4734) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4736 = "earth.constant"() <{rms_var = 0.12734215758896472 : f64, value = 54 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4737 = "earth.mul"(%4736, %4639) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4738 = "earth.add"(%4735, %4737) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4739 = "earth.constant"() <{rms_var = 0.091626347972833505 : f64, value = 55 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4740 = "earth.mul"(%4739, %4640) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4741 = "earth.add"(%4738, %4740) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4742 = "earth.constant"() <{rms_var = 0.07255941147658998 : f64, value = 56 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4743 = "earth.mul"(%4742, %4650) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4744 = "earth.add"(%4741, %4743) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4745 = "earth.constant"() <{rms_var = 0.061477909207391525 : f64, value = 57 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4746 = "earth.mul"(%4745, %4652) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4747 = "earth.add"(%4744, %4746) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4748 = "earth.constant"() <{rms_var = 0.055169030070493508 : f64, value = 58 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4749 = "earth.mul"(%4748, %4653) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4750 = "earth.add"(%4747, %4749) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4751 = "earth.constant"() <{rms_var = 0.052275954916496656 : f64, value = 59 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4752 = "earth.mul"(%4751, %4654) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4753 = "earth.add"(%4750, %4752) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4754 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4755 = "earth.mul"(%4754, %4705) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4756 = "earth.mul"(%4755, %4705) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4757 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4758 = "earth.add"(%4756, %4757) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4759 = "earth.mul"(%4730, %4758) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4760 = "earth.add"(%4759, %4707) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4761 = "earth.constant"() <{rms_var = 4.9481895575581374E-4 : f64, value = 60 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4762 = "earth.mul"(%4761, %4615) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4763 = "earth.constant"() <{rms_var = 3.771067298313324E-4 : f64, value = 61 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4764 = "earth.mul"(%4763, %4633) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4765 = "earth.add"(%4762, %4764) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4766 = "earth.constant"() <{rms_var = 3.2076765303085136E-4 : f64, value = 62 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4767 = "earth.mul"(%4766, %4639) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4768 = "earth.add"(%4765, %4767) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4769 = "earth.constant"() <{rms_var = 2.7679532400224407E-4 : f64, value = 63 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4770 = "earth.mul"(%4769, %4640) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4771 = "earth.add"(%4768, %4770) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4772 = "earth.constant"() <{rms_var = 2.4349440690240933E-4 : f64, value = 64 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4773 = "earth.mul"(%4772, %4650) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4774 = "earth.add"(%4771, %4773) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4775 = "earth.constant"() <{rms_var = 2.1958101095650637E-4 : f64, value = 65 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4776 = "earth.mul"(%4775, %4652) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4777 = "earth.add"(%4774, %4776) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4778 = "earth.constant"() <{rms_var = 2.0413318001830463E-4 : f64, value = 66 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4779 = "earth.mul"(%4778, %4653) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4780 = "earth.add"(%4777, %4779) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4781 = "earth.constant"() <{rms_var = 1.9655534146569248E-4 : f64, value = 67 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4782 = "earth.mul"(%4781, %4654) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4783 = "earth.add"(%4780, %4782) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4784 = "earth.constant"() <{rms_var = 1.7735088548126505E-4 : f64, value = 68 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4785 = "earth.mul"(%4784, %4615) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4786 = "earth.constant"() <{rms_var = 1.457794164045128E-4 : f64, value = 69 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4787 = "earth.mul"(%4786, %4633) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4788 = "earth.add"(%4785, %4787) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4789 = "earth.constant"() <{rms_var = 1.1982820692495149E-4 : f64, value = 70 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4790 = "earth.mul"(%4789, %4639) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4791 = "earth.add"(%4788, %4790) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4792 = "earth.constant"() <{rms_var = 9.8496754159654756E-5 : f64, value = 71 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4793 = "earth.mul"(%4792, %4640) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4794 = "earth.add"(%4791, %4793) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4795 = "earth.constant"() <{rms_var = 8.0962662339626343E-5 : f64, value = 72 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4796 = "earth.mul"(%4795, %4650) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4797 = "earth.add"(%4794, %4796) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4798 = "earth.constant"() <{rms_var = 6.6549936439435966E-5 : f64, value = 73 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4799 = "earth.mul"(%4798, %4652) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4800 = "earth.add"(%4797, %4799) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4801 = "earth.constant"() <{rms_var = 5.4702920205367267E-5 : f64, value = 74 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4802 = "earth.mul"(%4801, %4653) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4803 = "earth.add"(%4800, %4802) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4804 = "earth.constant"() <{rms_var = 1.3863334887734771E-4 : f64, value = 75 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4805 = "earth.mul"(%4804, %4654) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4806 = "earth.add"(%4803, %4805) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4807 = "earth.mul"(%4760, %4705) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4808 = "earth.add"(%4807, %4753) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4809 = "earth.mul"(%4806, %4705) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4810 = "earth.add"(%4809, %4783) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4811 = "earth.constant"() <{rms_var = 2.000000e+00 : f64, value = 26 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4812 = "earth.mul"(%4811, %4758) : (tensor<1x!earth.pl<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4813 = "earth.mul"(%4812, %4758) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4814 = "earth.constant"() <{rms_var = 1.000000e+00 : f64, value = 27 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4815 = "earth.add"(%4813, %4814) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4816 = "earth.mul"(%4810, %4815) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4817 = "earth.add"(%4816, %4808) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // poly[]layer2_0_act2_SiLU_poly
    %4818 = "earth.constant"() <{rms_var = 5.000000e-01 : f64, value = 76 : i64}> : () -> (tensor<1x!earth.pl<0 * 0>>)  loc(unknown) // add[]layer2_0_act2_SiLU_add
    %4819 = "earth.add"(%4817, %4818) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // add[]layer2_0_act2_SiLU_add
    %4820 = "earth.mul"(%4615, %4819) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)  loc(unknown) // mul[]layer2_0_act2_SiLU_mul
    "func.return"(%4820) : (tensor<1x!earth.ci<0 * 0>>) -> () loc("/home/ubuntu/dacaporion/dacapo/examples/benchmarks/CompPartSiLU64k.py":39:0)  }) : () -> () loc("/home/ubuntu/dacaporion/dacapo/examples/benchmarks/CompPartSiLU64k.py":39:0)  // mul[]layer2_0_act2_SiLU_mul
}) : () -> () loc(unknown)  // mul[]layer2_0_act2_SiLU_mul
