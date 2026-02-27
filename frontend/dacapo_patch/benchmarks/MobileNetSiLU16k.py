#!/usr/bin/env python


import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import resnet

import numpy as np

import poly
from poly.models.MobileNetSiLU import *
from poly.MPCB import *
from poly.Func import *


import hecate as hc
import sys

def getModel():
    from pathlib import Path
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent
 
    model = torch.nn.DataParallel(mobilenetsilu())

    model_dict = torch.load(str(source_dir)+"/../data/mobileNet_silu_model", map_location=torch.device('cpu'))
    # model_dict = torch.load(str(source_dir)+"/../data/mobileNet_relu_model", map_location=torch.device('cpu'))
    model.module.load_state_dict(model_dict)
    model = model.eval()
    return model

@hc.func("c")
def MobileNetSiLU16k (ctxt) :
    model = getModel()
    model = model.type(torch.double)
    model = model.cpu()
    # input_var = input_var.type(torch.double)
    input_var = np.empty((1), dtype=object)
    input_var[0] = ctxt

    def act(name, x) :
        return HE_SiLU(name, x)
        # return HE_ReLU(name, x)
    initial_shapes = {
        # Constant
        # "nt" : 2**16,
        "nt" : 2**14,
        "bb" : 32,
        # Input Characteristics (Cascaded)
        "ko" : 1,
        "ho" : 32,
        "wo" : 32
    }
    print("pre_layer")
    conv1_shapes = CascadeConv(initial_shapes, model.module.pre_layer.Conv2d)
    close = shapeClosure(**conv1_shapes)
    out = HE_ConvBN("convbn1", close, input_var, model.module.pre_layer.Conv2d, model.module.pre_layer.bn)
    # out = hc.bootstrap(out)
    out = act("act1", out)
    block_in = conv1_shapes
    for i in range(0, len(model.module.Depthwise)):
        print(i,"layer")
        inconv_0_shapes = CascadeConv(block_in, model.module.Depthwise[i].dwConv2d)
        close = shapeClosure(**inconv_0_shapes)
        out = HE_DwConv(f"layer_{i}_dw", close, out, model.module.Depthwise[i].dwConv2d, model.module.Depthwise[i].bn)
        # out = hc.bootstrap(out)
        out = act(f"layer_{i}_act1", out)

        inconv_0_pointwise_shapes = CascadeConv(inconv_0_shapes, model.module.Depthwise[i].pointwiseConv2d.Conv2d)
        close = shapeClosure(**inconv_0_pointwise_shapes)
        out = HE_ConvBN(f"layer_{i}_convbn", close, out, model.module.Depthwise[i].pointwiseConv2d.Conv2d, model.module.Depthwise[i].pointwiseConv2d.bn)
        # out = hc.bootstrap(out)
        out = act(f"layer_{i}_act2", out)
        block_in = inconv_0_pointwise_shapes

    # avgpool_1_shapes = CascadeMax (block_in, model.module.avgpool)
    avgpool_1_shapes = CascadePool (block_in)
    close = shapeClosure(**avgpool_1_shapes)
    # out = HE_Avg(close, out)
    out = HE_Pool("final_pool", close, out)
    block_in = avgpool_1_shapes

    out = HE_Linear("final_linear", close["OP"], out, model.module.linear, scale = 32.0)

    print("end")
    return out

modName = hc.save("traced", "traced")
print (modName)

