import os
import json
import torch
import torch.nn as nn 
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import yolov6

model = torch.load('yolov6l.pt')
model = model['model']

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def find(module, name=""):
    if "ConvBNSiLU" in name:
        print(name)
        return {name: module}
    res = {}
    for name1, module in model.named_modules():
        res.update(find_layers(
            module, name=name + '.' + name1 if name != '' else name1
        ))
        
res = find(model, "")

conv_weigths = {}
for name, module in model.named_modules():
    if 'block.conv' in name:
        conv_weigths.update({name : module})
        
key_list = list(conv_weigths.keys())

def prune(weight, rate):
    W = weight.data
    o, c, r, d = W.shape
    num = int(o * rate)


    l2_norms = torch.sqrt(torch.sum(W ** 2, dim=(1, 2, 3)))
    _, top_idx = torch.topk(l2_norms, o - num, dim=0, largest=True)  

    pruned_W = W[top_idx, :, :, :]
    
    return pruned_W

import copy

def prune_model_and_update(model, pruning_rate):

    pruned_model = copy.deepcopy(model)
    
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
 
            weight = module.weight.data
            pruned_weight = prune(weight, pruning_rate)

            new_out_channels = pruned_weight.shape[0]
            module.out_channels = new_out_channels
            module.weight = nn.Parameter(pruned_weight)
            
            for next_name, next_module in pruned_model.named_modules():
                if next_name.startswith(name) and isinstance(next_module, nn.Conv2d):
                    next_module.in_channels = new_out_channels
                    break  

    return pruned_model

pruned_model = prune_model_and_update(model, pruning_rate=0.5)

state_dict = model.state_dict()

new_state_dict = {}
for key in state_dict.keys():
    if 'block.conv' in key: 
        pruned_weight = prune(state_dict[key], 0.5)
        new_state_dict[key] = pruned_weight
    else:
        new_state_dict[key] = state_dict[key]

        new_state_dict[key] = state_dict[key]

    cnt += 1
torch.save(new_state_dict, 'pruned.pt')
