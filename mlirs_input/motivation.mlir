"builtin.module"() <{sym_name = "traced/motivation.mlir"}> ({
  "func.func"() <{function_type = (tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>, sym_name = "_hecate_Motivation"}> ({
  ^bb0(%arg0: tensor<1x!earth.ci<0 * 0>> loc("/home/ubuntu/dacaporion/dacapo/examples/benchmarks/motivation.py":6:0)):
    %0 = "earth.mul"(%arg0, %arg0) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)
    %1 = "earth.mul"(%0, %arg0) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)
    %2 = "earth.mul"(%1, %1) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)
    %3 = "earth.rotate"(%arg0) <{offset = array<i64: 1>}> : (tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)
    %4 = "earth.mul"(%3, %3) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)
    %5 = "earth.mul"(%1, %4) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)
    %6 = "earth.add"(%2, %5) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)
    %7 = "earth.mul"(%6, %6) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>)
    "func.return"(%7) : (tensor<1x!earth.ci<0 * 0>>) -> () loc("/home/ubuntu/dacaporion/dacapo/examples/benchmarks/motivation.py":6:0)  }) : () -> () loc("/home/ubuntu/dacaporion/dacapo/examples/benchmarks/motivation.py":6:0)
}) : () -> () loc(unknown)
