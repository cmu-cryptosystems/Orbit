#!/usr/bin/env python

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import hecate as hc
import sys

import poly
from poly.models.ResNetSiLU import *
from poly.MPCB import *
from poly.Func import *


def getModel():
    # model_dict = torch.load("../data/resnet20.silu.model", map_location=torch.device('cpu'))
    from pathlib import Path
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent
    model = torch.nn.DataParallel(resnet20silu())
    model_dict = torch.load(str(source_dir)+"/../data/resnet20.silu.model", map_location=torch.device('cpu'))
    # model_dict = torch.load(str(source_dir)+"/../data/resnet20.relu.model", map_location=torch.device('cpu'))
    model.load_state_dict(model_dict['state_dict'])
    model = model.eval()
    return model


@hc.func("c")
def CompPartSiLU64k (ctxt) :

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
        "nt" : 2**16,
        # "nt" : 2**14,
        "bb" : 32,
        # Input Characteristics (Cascaded)
        "ko" : 1,
        "ho" : 32,
        "wo" : 32
    }
    conv1_shapes = CascadeConv(initial_shapes, model.module.conv1)
    close = shapeClosure(**conv1_shapes)
    out = HE_ConvBN("convbn1-0", close, input_var,model.module.conv1, model.module.bn1)
    # for i in range(1, 20):
    #     this_input = input_var + i
    #     outi = HE_ConvBN(f"convbn1-{i}", close, this_input ,model.module.conv1, model.module.bn1)
    #     out = out + outi
    # out = hc.bootstrap(out)
    
    out = act("act1", out)
    block_in = conv1_shapes
    print ("layer1")
    for i in range(0, 3) :
        print (i)
        # dsout = out
        inconv1_shapes = CascadeConv (block_in, model.module.layer1[i].conv1)
        close = shapeClosure(**inconv1_shapes)
        out0 = HE_ConvBN(f"layer1_{i}_convbn1-0", close, out + 1, model.module.layer1[i].conv1, model.module.layer1[i].bn1)
        if i < 1:
            for j in range(1, 5):
                outj = HE_ConvBN(f"layer1_{i}_convbn1-{j}", close, out + j + 1, model.module.layer1[i].conv1, model.module.layer1[i].bn1)
                out0 = out0 + outj
        # out = hc.bootstrap(out)
        out = act(f"layer1_{i}_act1", out0)
        inconv2_shapes = CascadeConv (inconv1_shapes, model.module.layer1[i].conv2)
        close = shapeClosure(**inconv2_shapes)
        out = HE_ConvBN(f"layer1_{i}_convbn2", close, out,model.module.layer1[i].conv2, model.module.layer1[i].bn2)
        # hc.add_comment(f"// add@layer1_{i}_add")
        # out = out + dsout
        # out = hc.bootstrap(out)
        out = act(f"layer1_{i}_act2", out)
        block_in = inconv2_shapes

    print ("layer2")
    ds1_shapes = CascadeDS (block_in)
    close = shapeClosure(**ds1_shapes)
    dsout = HE_DS("layer2_ds", close, out)
    for i in range(0, 1) :
        print (i)
        # if not (i == 0) :
        #     dsout = out
        inconv1_shapes = CascadeConv (block_in, model.module.layer2[i].conv1)
        close = shapeClosure(**inconv1_shapes)
        out = HE_ConvBN(f"layer2_{i}_convbn1", close, out, model.module.layer2[i].conv1, model.module.layer2[i].bn1)
        # out = hc.bootstrap(out)
        out = act (f"layer2_{i}_act1", out)
        inconv2_shapes = CascadeConv (inconv1_shapes, model.module.layer2[i].conv2)
        close = shapeClosure(**inconv2_shapes)
        out = HE_ConvBN(f"layer2_{i}_convbn2", close, out, model.module.layer2[i].conv2, model.module.layer2[i].bn2)
        # hc.add_comment(f"// add@layer2_{i}_add")
        # out = out +dsout
        # out = hc.bootstrap(out)
        out = act (f"layer2_{i}_act2", out)
        block_in = inconv2_shapes
    
    return out

modName = hc.save("traced", "traced")
print (modName)

