import hecate as hc
import sys
import os
import argparse
import poly
from poly.models.AlexNetReLU import *
from poly.models.AlexNetSiLU import *
from poly.models.ResNetReLU import *
from poly.models.ResNetSiLU import *
from poly.models.VGG16ReLU import *
from poly.models.VGG16SiLU import *
from poly.models.MobileNetReLU import *
from poly.models.MobileNetSiLU import *
from poly.models.SqueezeNetReLU import *
from poly.models.SqueezeNetSiLU import *
from poly.MPCB import *

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from PIL import Image
import numpy as np
from random import *
import pprint


from pathlib import Path

seed(100)
source_path = Path(__file__).resolve()
source_dir = source_path.parent
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=str(source_dir)+"/../data/CIFAR10", train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
])),
batch_size=128, shuffle=False,
num_workers=4, pin_memory=True)

def getModel(modelname, modelact):
    assert modelname in ["alexnet", "resnet", "vgg16", "squeezenet", "mobilenet"]
    assert modelact in ["silu", "relu"]
    
    if modelname == "resnet":
        if modelact == "relu":
            model = torch.nn.DataParallel(resnet20relu())
        else:
            model = torch.nn.DataParallel(resnet20silu())
        
        if modelact == "silu":
            model_dict = torch.load(str(source_dir)+"/../data/resnet20."+modelact+".model", map_location=torch.device('cpu'))
            new_model_dict = {}
            for k, v in model_dict['state_dict'].items():
                assert k.startswith("module.")
                new_model_dict[k[7:]] = v
            model_dict = new_model_dict
        else:
            model_dict = torch.load(str(source_dir)+"/../data/resnet20_"+modelact+"_model", map_location=torch.device('cpu'))
    elif modelname == "mobilenet":
        if modelact == "relu":
            model = torch.nn.DataParallel(mobilenetrelu())
        else:
             model = torch.nn.DataParallel(mobilenetsilu())
        model_dict = torch.load(str(source_dir)+"/../data/mobileNet_"+modelact+"_model", map_location=torch.device('cpu'))
    elif modelname == "alexnet":
        act_name = "relu_maxpool" if modelact == "relu" else "silu_avgpool"
        if modelact == "relu":
            model = torch.nn.DataParallel(alexnetrelu())
        else:
            model = torch.nn.DataParallel(alexnetsilu())
        model_dict = torch.load(str(source_dir)+f"/../data/alexNet_{act_name}_model", map_location=torch.device('cpu'))
    elif modelname == "vgg16":
        act_name = "relu_maxpool" if modelact == "relu" else "silu_avgpool"
        if modelact == "relu":
            model = torch.nn.DataParallel(vgg16relu())
        else:
            model = torch.nn.DataParallel(vgg16silu())
        model_dict = torch.load(str(source_dir)+f"/../data/vgg16_{act_name}_model", map_location=torch.device('cpu'))
    elif modelname == "squeezenet":
        act_name = "relu_maxpool" if modelact == "relu" else "silu_avgpool"
        if modelact == "relu":
            model = torch.nn.DataParallel(squeezenetrelu())
        else:
            model = torch.nn.DataParallel(squeezenetsilu())
        model_dict = torch.load(str(source_dir)+f"/../data/squeezeNet_{act_name}_model", map_location=torch.device('cpu'))
    
    model.module.load_state_dict(model_dict)
    model = model.eval()
    return model

def preprocess(x, model, modelname, n):
    initial_shapes = {
    # Constant
    "nt" : n, # 2**14, 2**15, 2**16
    "bb" : 32,
    # Input Characteristics (Cascaded)
    "ko" : 1,
    "ho" : 32,
    "wo" : 32
    }
    if modelname == "resnet":
        conv1_shapes = CascadeConv(initial_shapes, model.module.conv1)
    elif modelname == "mobilenet":
        conv1_shapes = CascadeConv(initial_shapes, model.module.pre_layer.Conv2d)
    elif modelname == "alexnet":
        conv1_shapes = CascadeConv(initial_shapes, model.module.Conv2d_1)
    elif modelname == "vgg16":
        conv1_shapes = CascadeConv(initial_shapes, model.module.conv_1_1.Conv2d)
    elif modelname == "squeezenet":
        conv1_shapes = CascadeConv(initial_shapes, model.module.conv_1.Conv2d)
    close = shapeClosure(**conv1_shapes)
    return close["MPP"](modelname, x)[0]

def process(model, x):
    torch_res = model(x) 
    torch_res = torch_res.cpu().detach().numpy()[0]
    return torch_res

def postprocess(res, torch_res_shape):
    torch_res_size = 1
    for i in range(len(torch_res_shape)):
        torch_res_size *= torch_res_shape[i]
    return res[0,:torch_res_size].reshape(torch_res_shape) *32
    

def parse_args():
    parser = argparse.ArgumentParser(description="Generate DaCapo benchmark input/reference samples.")
    parser.add_argument("num_inputs", type=int)
    parser.add_argument("--model", choices=["alexnet", "resnet", "vgg16", "squeezenet", "mobilenet"], default=None)
    parser.add_argument("--activation", choices=["silu", "relu"], default=None)
    parser.add_argument("--n", choices=["16", "16k", "64", "64k"], default=None)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    num_inputs = args.num_inputs
    
    io_pairs = []
    for i in range(num_inputs):
        input, target = val_loader.dataset[i]
        input_var = input.unsqueeze(0)
        io_pairs.append( (i, input_var, target) )
    
    model_names = [args.model] if args.model else ["resnet", "alexnet", "vgg16", "squeezenet", "mobilenet"]
    model_acts = [args.activation] if args.activation else ["silu", "relu"]
    slot_sizes = [2**16 if args.n in {"64", "64k"} else 2**14] if args.n else [2**14, 2**16]

    for modelname in model_names:
        for modelact in model_acts:
            model = getModel(modelname, modelact)
            model = model.eval()
            for n in slot_sizes:
                if modelname == "mobilenet" and n != 2**16:
                    continue
                n_name = "16k" if n == 2**14 else "64k"
                dir_path = str(source_dir)+f"/../../../../input_data/{n_name}/{modelname}/{modelact}/"
                input_path = dir_path + f"inputs/"
                ref_path = dir_path + f"plrefs/"
                
                os.makedirs(input_path, exist_ok=True)
                os.makedirs(ref_path, exist_ok=True)
                
                num_correct = 0
                num_total = 0
                print(f"Generating input and reference for {modelname} with {modelact} activation, n={n_name}")
                for (idx, input_var, target) in io_pairs:
                    reference = process(model, input_var)
                    
                    ref_length = reference.shape[0]
                    assert ref_length == 10
                    ref_vals = []
                    for val in range(ref_length):
                        ref_vals.append(reference[val])
                    ref_max = None
                    ref_ans = None
                    for class_idx in range(10):
                        if ref_max is None or ref_vals[class_idx] > ref_max:
                            ref_max = ref_vals[class_idx]
                            ref_ans = class_idx
                    if ref_ans == target:
                        num_correct += 1
                    num_total += 1
                    
                    preprocessed_input = preprocess(input_var, model, modelname, n)
                    
                    if idx % 100 == 0:
                        print(f"    idx={idx}")
                    with open(f"{input_path}input{idx}.txt", "w") as f:
                        length = preprocessed_input.shape[0]
                        f.write(f"{length}\n")
                        for val in range(length):
                            f.write(f"{preprocessed_input[val].item()}\n")
                    
                    with open(f"{ref_path}plref{idx}.txt", "w") as f:
                        length = reference.shape[0]
                        f.write(f"{length}\n")
                        for val in range(length):
                            f.write(f"{reference[val].item()}\n")

                accuracy = num_correct / num_total
                print(f"Accuracy for {modelname} with {modelact} activation, n={n_name}: {accuracy*100:.2f}%")

                with open(f"{dir_path}true_labels.txt", "w") as f:
                    for (idx, _, target) in io_pairs:
                        f.write(f"input{idx}.txt: {target}\n")
