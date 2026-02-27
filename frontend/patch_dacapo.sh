#!/bin/bash

cp dacapo_patch/benchmarks/* dacapo/examples/benchmarks/
cp dacapo_patch/gen_input_data.py dacapo/examples/tests/
cp dacapo_patch/python/hecate/expr.py dacapo/python/hecate/hecate/
cp dacapo_patch/python/poly/models/* dacapo/python/poly/poly/models/
cp dacapo_patch/python/poly/Func.py dacapo/python/poly/poly/
cp dacapo_patch/python/poly/MPCB.py dacapo/python/poly/poly/
cp dacapo_patch/python/poly/Poly.py dacapo/python/poly/poly/
cp dacapo_patch/python/propagate_comments.py dacapo/python/
cp dacapo_patch/tools/frontend.cpp dacapo/tools/