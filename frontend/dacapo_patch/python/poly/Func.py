import poly.MPCB as MPCB
import poly.Poly as Poly
import numpy as np
import hecate as hc
import pathlib
import os

def HE_BN (name, close, mpp, bn, scale=1.0) :
    G, H = MPCB.abstractBN(bn)
    mpcb = close["BN"](name, mpp, G, H)
    return mpcb

def HE_MPBN (name, mpp, bn, scale=1.0) :
    G, H = MPCB.abstractBN(bn)
    mpcb = MPCB.BN(name, mpp, G, H/scale, 2**16)
    return mpcb

def HE_Conv (name, close, mpp, conv) :
    mpcb =  close["MPC"] (name, mpp, conv.weight, conv.bias)
    return mpcb

def HE_ConvBN (name, close, mpp, conv, bn) :
    mpcb =  close["MPCB"] (name, mpp, conv.weight, *MPCB.abstractBN(bn))
    return mpcb

def HE_MaxPad (name, close, mpp) :
    def maximum (mname, a, b):
        out = Poly.maxx(mname, a, b)
        # out = hc.bootstrap(out)
        return out
    MPCB.maximum = maximum
    mpcb =  close["MPD"] (name, mpp)
    return mpcb

def HE_Max (name, close, mpp) :
    def maximum (mname, a, b):
        out = Poly.maxx(mname, a, b)
        # out = hc.bootstrap(out)
        return out
    MPCB.maximum = maximum
    mpcb =  close["MP"] (name, mpp)
    return mpcb

def HE_Avg (name, close, mpp) :
    mpcb =  close["MA"] (name, mpp)
    return mpcb

def HE_DS (name, close, mpp) :
    mpcb = close["DS"](name, mpp)
    return mpcb

def HE_Pool (name, close, mpp) :
    return close["AP"](name, mpp)

def HE_Linear(name, close, mpp, linear, p = 1.0, scale = 1.0) :
    mpcb = MPCB.Linear(name, mpp, linear.weight * p , linear.bias.cpu() / scale, 2**16)
    return mpcb

def HE_ReshapeLinear(name, close, mpp, linear, p = 1.0, scale = 1.0, reshape = {}) :
    weight = MPCB.Reshape (linear.weight, reshape)
    mpcb = MPCB.Linear(name, mpp, weight * p , linear.bias.cpu() / scale, 2**16)
    return mpcb

def HE_DwConv (name, close, mpp, conv, bn) :
    G, H = MPCB.abstractBN(bn)
    mpcb =  close["DW"] (name, mpp, conv.weight, G, H+conv.bias)
    return mpcb


def HE_Concat (name, close, mpp_1, mpp_2) :
    mpcb =  close["CC"] (name, mpp_1, mpp_2)
    return mpcb

def HE_ReLU (name, x) : 
    hc.add_comment(f"// poly@{name}_ReLU_poly1")
    y = Poly.poly1(x)
    hc.add_comment(f"// poly@{name}_ReLU_poly2")
    y = Poly.poly2(y)
    # y = hc.bootstrap(y)
    hc.add_comment(f"// poly@{name}_ReLU_poly3")
    y = Poly.poly3(y)
    hc.add_comment(f"// add@{name}_ReLU_add")
    y = y + 0.5
    hc.add_comment(f"// mul@{name}_ReLU_mul")
    y = y * x
    return y

def HE_SiLU (name, x) :
    calculation = Poly.GenPoly()
    hc.add_comment(f"// poly@{name}_SiLU_poly")
    y = calculation(x)
    hc.add_comment(f"// add@{name}_SiLU_add")
    y = y + 0.5
    hc.add_comment(f"// mul@{name}_SiLU_mul")
    y = x * y
    return y



