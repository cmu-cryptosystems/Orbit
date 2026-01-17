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
from poly.models.SqueezeNetReLU import *
from poly.MPCB import *
from poly.Func import *

import sys

def getModel():
    from pathlib import Path
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent
    model = torch.nn.DataParallel(squeezenetrelu())

    # model_dict = torch.load(str(source_dir)+"/../data/squeezeNet_silu_maxpool_model", map_location=torch.device('cpu'))
    model_dict = torch.load(str(source_dir)+"/../data/squeezeNet_relu_maxpool_model", map_location=torch.device('cpu'))
    # There is no state_dict with checkpoint
    #model.load_state_dict(model_dict['state_dict'])
    model.module.load_state_dict(model_dict)
    model = model.eval()
    return model

@hc.func("c")
def SqueezeNetReLU16k (ctxt) :
    model = getModel()
    model = model.type(torch.double)
    model = model.cpu()
    for p in model.parameters():
        p.requires_grad = False
    input_var = np.empty((1), dtype= object)
    input_var[0] = ctxt

    def act (name, x) : 
        # return HE_SiLU(name, x)
        return HE_ReLU(name, x)
    def pooling (name, close, x):
        # return HE_Avg(name, close, x)
        return HE_MaxPad(name, close, x)
    
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
    print("Conv_1")        
    conv_1_shapes = CascadeConv(initial_shapes, model.module.conv_1.Conv2d)
    close = shapeClosure(**conv_1_shapes)
    out = HE_ConvBN("conv1_convbn", close, input_var,model.module.conv_1.Conv2d, model.module.conv_1.bn)
    out = act ("conv1_act", out)
    # out = hc.bootstrap(out)
    block_in = conv_1_shapes
 
    print("maxpool_1")
    maxpool_1_shapes = CascadeMax (block_in, model.module.maxpool_1)
    close = shapeClosure(**maxpool_1_shapes)
    out = pooling("maxpool_1", close, out)
    block_in = maxpool_1_shapes

    print("fire_2")
    fire_2_squeeze_shapes = CascadeConv(block_in, model.module.fire_2.squeeze.Conv2d)
    close = shapeClosure(**fire_2_squeeze_shapes)
    # out = hc.bootstrap(out)
    out = HE_ConvBN("fire2_convbn", close, out, model.module.fire_2.squeeze.Conv2d, model.module.fire_2.squeeze.bn)
    out = act ("fire2_act", out)
    # out = hc.bootstrap(out)
    block_in = fire_2_squeeze_shapes
 
    fire_2_expand1x1_shapes = CascadeConv(block_in, model.module.fire_2.expand1x1)
    close = shapeClosure(**fire_2_expand1x1_shapes)
    out1 = HE_Conv("fire2_conv", close, out, model.module.fire_2.expand1x1)

    fire_2_expand3x3_shapes = CascadeConv(block_in, model.module.fire_2.expand3x3)
    close = shapeClosure(**fire_2_expand3x3_shapes)
    out2 = HE_Conv("fire2_conv", close, out, model.module.fire_2.expand3x3)

    ##############concat################
#     ori = torch.cat([], dim=1)
    block_in = CascadeConcat(fire_2_expand1x1_shapes, fire_2_expand3x3_shapes)
    close = shapeClosure(**block_in)
    out = HE_Concat("fire2_concat", close, out1, out2)
    ###################################

    print("fire_3")
    fire_3_squeeze_shapes = CascadeConv(block_in, model.module.fire_3.squeeze.Conv2d)
    close = shapeClosure(**fire_3_squeeze_shapes)
    out = HE_ConvBN("fire_3_convbn", close, out, model.module.fire_3.squeeze.Conv2d, model.module.fire_3.squeeze.bn)
    # out = hc.bootstrap(out)
    out = act ("fire_3_act", out)
    # out = hc.bootstrap(out)
    block_in = fire_3_squeeze_shapes

    fire_3_expand1x1_shapes = CascadeConv(block_in, model.module.fire_3.expand1x1)
    close = shapeClosure(**fire_3_expand1x1_shapes)
    out1 = HE_Conv("fire_3_conv", close, out, model.module.fire_3.expand1x1)

    fire_3_expand3x3_shapes = CascadeConv(block_in, model.module.fire_3.expand3x3)
    close = shapeClosure(**fire_3_expand3x3_shapes)
    out2 = HE_Conv("fire_3_conv", close, out, model.module.fire_3.expand3x3)
    ##############concat################
    block_in = CascadeConcat(fire_3_expand1x1_shapes, fire_3_expand3x3_shapes)
    close = shapeClosure(**block_in)
    out = HE_Concat("fire_3_concat", close, out1, out2)
    ###################################

    print("fire_4")
    fire_4_squeeze_shapes = CascadeConv(block_in, model.module.fire_4.squeeze.Conv2d)
    close = shapeClosure(**fire_4_squeeze_shapes)
    out = HE_ConvBN("fire_4_convbn", close, out, model.module.fire_4.squeeze.Conv2d, model.module.fire_4.squeeze.bn)
    # out = hc.bootstrap(out)
    out = act("fire_4_act", out)
    # out = hc.bootstrap(out)
    print ("additional")
    block_in = fire_4_squeeze_shapes
    

    fire_4_expand1x1_shapes = CascadeConv(block_in, model.module.fire_4.expand1x1)
    close = shapeClosure(**fire_4_expand1x1_shapes)
    out1 = HE_Conv("fire_4_conv", close, out, model.module.fire_4.expand1x1)

    fire_4_expand3x3_shapes = CascadeConv(block_in, model.module.fire_4.expand3x3)
    close = shapeClosure(**fire_4_expand3x3_shapes)
    out2 = HE_Conv("fire_4_conv", close, out, model.module.fire_4.expand3x3)
    ##############concat################
    block_in = CascadeConcat(fire_4_expand1x1_shapes, fire_4_expand3x3_shapes)
    close = shapeClosure(**block_in)
    out = HE_Concat("fire_4_concat", close, out1, out2)
    ##################################    
    
 
    print("maxpool_4")
    maxpool_4_shapes = CascadeMax (block_in, model.module.maxpool_4)
    close = shapeClosure(**maxpool_4_shapes)
    out = pooling("maxpool_4", close, out)
    # out = hc.bootstrap(out)
    block_in = maxpool_4_shapes

    print("fire_5")
    fire_5_squeeze_shapes = CascadeConv(block_in, model.module.fire_5.squeeze.Conv2d)
    close = shapeClosure(**fire_5_squeeze_shapes)
    out = HE_ConvBN("fire_5_convbn", close, out, model.module.fire_5.squeeze.Conv2d, model.module.fire_5.squeeze.bn)
    out = act("fire_5_act", out)
    # out = hc.bootstrap(out)
    block_in = fire_5_squeeze_shapes

    fire_5_expand1x1_shapes = CascadeConv(block_in, model.module.fire_5.expand1x1)
    close = shapeClosure(**fire_5_expand1x1_shapes)
    out1 = HE_Conv("fire_5_conv", close, out, model.module.fire_5.expand1x1)

    fire_5_expand3x3_shapes = CascadeConv(block_in, model.module.fire_5.expand3x3)
    close = shapeClosure(**fire_5_expand3x3_shapes)
    out2 = HE_Conv("fire_5_conv", close, out, model.module.fire_5.expand3x3)
    ##############concat################
    block_in = CascadeConcat(fire_5_expand1x1_shapes, fire_5_expand3x3_shapes)
    close = shapeClosure(**block_in)
    out = HE_Concat("fire_5_concat", close, out1, out2)
    ###################################
    print("fire_6")
    fire_6_squeeze_shapes = CascadeConv(block_in, model.module.fire_6.squeeze.Conv2d)
    close = shapeClosure(**fire_6_squeeze_shapes)
    out = HE_ConvBN("fire_6_convbn", close, out, model.module.fire_6.squeeze.Conv2d, model.module.fire_6.squeeze.bn)
    # out = hc.bootstrap(out)
    out = act("fire_6_act", out)
    # out = hc.bootstrap(out)
    block_in = fire_6_squeeze_shapes

    fire_6_expand1x1_shapes = CascadeConv(block_in, model.module.fire_6.expand1x1)
    close = shapeClosure(**fire_6_expand1x1_shapes)
    out1 = HE_Conv("fire_6_conv", close, out, model.module.fire_6.expand1x1)

    fire_6_expand3x3_shapes = CascadeConv(block_in, model.module.fire_6.expand3x3)
    close = shapeClosure(**fire_6_expand3x3_shapes)
    out2 = HE_Conv("fire_6_conv", close, out, model.module.fire_6.expand3x3)
    ##############concat################

    block_in = CascadeConcat(fire_6_expand1x1_shapes, fire_6_expand3x3_shapes)
    close = shapeClosure(**block_in)
    out = HE_Concat("fire_6_concat", close, out1, out2)
    ###################################
    
    
    print("fire_7")
    fire_7_squeeze_shapes = CascadeConv(block_in, model.module.fire_7.squeeze.Conv2d)
    close = shapeClosure(**fire_7_squeeze_shapes)
    out = HE_ConvBN("fire_7_convbn", close, out, model.module.fire_7.squeeze.Conv2d, model.module.fire_7.squeeze.bn)
    # out = hc.bootstrap(out)
    out = act("fire_7_act", out)
    # out = hc.bootstrap(out)
    block_in = fire_7_squeeze_shapes

    fire_7_expand1x1_shapes = CascadeConv(block_in, model.module.fire_7.expand1x1)
    close = shapeClosure(**fire_7_expand1x1_shapes)
    out1 = HE_Conv("fire_7_conv", close, out, model.module.fire_7.expand1x1)

    fire_7_expand3x3_shapes = CascadeConv(block_in, model.module.fire_7.expand3x3)
    close = shapeClosure(**fire_7_expand3x3_shapes)
    out2 = HE_Conv("fire_7_conv", close, out, model.module.fire_7.expand3x3)
    ##############concat################
    block_in = CascadeConcat(fire_7_expand1x1_shapes, fire_7_expand3x3_shapes)
    close = shapeClosure(**block_in)
    out = HE_Concat("fire_7_concat", close, out1, out2)
    ###################################
    
    print("fire_8")
    fire_8_squeeze_shapes = CascadeConv(block_in, model.module.fire_8.squeeze.Conv2d)
    close = shapeClosure(**fire_8_squeeze_shapes)
    out = HE_ConvBN("fire_8_convbn", close, out, model.module.fire_8.squeeze.Conv2d, model.module.fire_8.squeeze.bn)
    # out = hc.bootstrap(out)
    out = act("fire_8_act", out)
    # out = hc.bootstrap(out)
    block_in = fire_8_squeeze_shapes

    fire_8_expand1x1_shapes = CascadeConv(block_in, model.module.fire_8.expand1x1)
    close = shapeClosure(**fire_8_expand1x1_shapes)
    out1 = HE_Conv("fire_8_conv", close, out, model.module.fire_8.expand1x1)

    fire_8_expand3x3_shapes = CascadeConv(block_in, model.module.fire_8.expand3x3)
    close = shapeClosure(**fire_8_expand3x3_shapes)
    out2 = HE_Conv("fire_8_conv", close, out, model.module.fire_8.expand3x3)
    ##############concat################
    block_in = CascadeConcat(fire_8_expand1x1_shapes, fire_8_expand3x3_shapes)
    close = shapeClosure(**block_in)
    out = HE_Concat("fire_8_concat", close, out1, out2)
    ###################################
    
    print("maxpool_8")
    maxpool_8_shapes = CascadeMax (block_in, model.module.maxpool_8)
    close = shapeClosure(**maxpool_8_shapes)
    out = pooling("maxpool_8", close, out)
    # out = hc.bootstrap(out)
    block_in = maxpool_8_shapes

    print("fire_9")
    fire_9_squeeze_shapes = CascadeConv(block_in, model.module.fire_9.squeeze.Conv2d)
    close = shapeClosure(**fire_9_squeeze_shapes)
    out = HE_ConvBN("fire_9_convbn", close, out, model.module.fire_9.squeeze.Conv2d, model.module.fire_9.squeeze.bn)
    out = act("fire_9_act", out)
    # out = hc.bootstrap(out)
    block_in = fire_9_squeeze_shapes

    fire_9_expand1x1_shapes = CascadeConv(block_in, model.module.fire_9.expand1x1)
    close = shapeClosure(**fire_9_expand1x1_shapes)
    out1 = HE_Conv("fire_9_conv", close, out, model.module.fire_9.expand1x1)

    fire_9_expand3x3_shapes = CascadeConv(block_in, model.module.fire_9.expand3x3)
    close = shapeClosure(**fire_9_expand3x3_shapes)
    out2 = HE_Conv("fire_9_conv", close, out, model.module.fire_9.expand3x3)
    ##############concat################
    block_in = CascadeConcat(fire_9_expand1x1_shapes, fire_9_expand3x3_shapes)
    close = shapeClosure(**block_in)
    out = HE_Concat("fire_9_concat", close, out1, out2)
    ###################################
 
   
    print("Conv_10")
    conv_10_shapes = CascadeConv(block_in, model.module.conv_10.Conv2d)
    close = shapeClosure(**conv_10_shapes)
    out = HE_ConvBN("conv_10_convbn", close, out, model.module.conv_10.Conv2d, model.module.conv_10.bn)
    # out = hc.bootstrap(out)
    out = act("conv_10_act", out)
    block_in = conv_10_shapes
    
    print("avgpool_10")
    avgpool_10_shapes = CascadePool (block_in)
    close = shapeClosure(**avgpool_10_shapes)
    out = HE_Pool("avgpool_10", close, out)
    block_in = avgpool_10_shapes
 
    print("end")
    return out
    

modName = hc.save("traced", "traced")
print (modName)

