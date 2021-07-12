import os
import glob
import numpy as np
import random
#import matplotlib
#import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image

from args import make_args
from utils import *
from renderer import PhongRenderer

class SketchDataset(Dataset):
    def __init__(self, args, transform=None):
        '''
        ## Goal of sampling data ##

        Anchor & Positive sketches (1 class, K sketches)
        Negative sketches (C-1 class, K sketches each)

        ## Variable Description ##

        self.C : sampled class number
        self.K : sampled sketch number for each class

        self.cls_name : ["airplane", "ant", ...]
        self.cls : [0, 1, ...]
        self.cls2name : {0 : "airplane", 1 : "ant", ... }
        self.cls2sketches = {0 : ["1.png", "2.png", "5.png", ...], 1: ["3.png", "4.png"], ... }
        self.cls_list = [0, 0, 0, 1, 1, ...]
        self.sketches_list = ["1.png", "2.png", "5.png", "3.png", "4.png", ...]
        '''
        self.C_sketch = args.C_sketch
        self.K_sketch = args.K_sketch
        self.sketch_train_dir = args.sketch_train_dir
        self.total_cls_num = 0
        self.total_sketches_num = 0

        # Load whole data (data protocol: class & sketches (index: file_name))
        self.cls_dir = sorted(glob.glob(self.sketch_train_dir + "/*"))
        self.cls_name = [x.split("/")[-1] for x in self.cls_dir]
        self.total_cls_num = len(self.cls_name)
        
        self.cls = list(range(self.total_cls_num)) # Express Class as Index (0, 1, 2, ... -> 0 means airplane)
        self.cls2name = {} # match class & class_name (0:"airplane")
        for i in range(self.total_cls_num):
            self.cls2name[i] = self.cls_name[i]
        
        self.cls2sketches = {} # match class & according sketches (0:["1.png path", "2.png path"])
        for i in range(self.total_cls_num):
            self.cls2sketches[i] = sorted(glob.glob(self.cls_dir[i] + "/train/*"))
            self.total_sketches_num += len(glob.glob(self.cls_dir[i] + "/train/*"))
        
        self.cls_list = list()
        self.sketches_list = list()
        for cl, sk in self.cls2sketches.items():
            self.cls_list += [cl] * len(sk)
            self.sketches_list += sk
        
        self.transform = transform

        ###### transform for train #####
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        ################################
    def __len__(self):
        return self.total_sketches_num

    def __getitem__(self, idx):
        
        cls_list = list()
        sketches_list = list()
        make_tensor = transforms.ToTensor()
        
        # Selected pair with index
        selected_cls = self.cls_list[idx]
        selected_sketch = self.sketches_list[idx]

        cls_sub = self.cls.copy()
        cls_sub.remove(selected_cls)
        sampled_cls = sorted(random.sample(cls_sub, self.C_sketch-1))
        sampled_cls.append(selected_cls)
        for c in sampled_cls:
            if c == selected_cls:
                cls2sketches_sub = self.cls2sketches[c].copy()
                cls2sketches_sub.remove(selected_sketch)
                sampled_sketches = sorted(random.sample(cls2sketches_sub, self.K_sketch-1))
                sampled_sketches.append(selected_sketch)
                cls_list += [c] * self.K_sketch
                sketches_list += sampled_sketches
            else:
                sampled_sketches = sorted(random.sample(self.cls2sketches[c], self.K_sketch))
                cls_list += [c] * self.K_sketch
                sketches_list += sampled_sketches

        # Make sampled class to torch.tensor
        cls_tensor = torch.Tensor(cls_list)

        # Read image & convert 3 channels & make to torch.tensor
        decide_expand_dim = True
        for path in sketches_list:
            sketch_pil = Image.open(path).convert(mode="RGB")
            sketch_tensor = make_tensor(sketch_pil)
            sketch_tensor = torch.unsqueeze(sketch_tensor,0)
            if decide_expand_dim:
                sketches_tensor = sketch_tensor
                decide_expand_dim = False
            else:
                sketches_tensor = torch.cat((sketches_tensor, sketch_tensor),0)

        # transform input
        if self.transform is not None:
            sketches_tensor = self.transform(sketches_tensor)

        return sketches_tensor, cls_tensor

class ModelDataset(Dataset):
    def __init__(self, args, renderer):
        '''
        ## Goal of sampling data ##

        Anchor & Positive sketches (1 class, K sketches)
        Negative sketches (C-1 class, K sketches each)

        ## Variable Description ##

        self.C : sampled class number
        self.K : sampled model number for each class

        self.class2model : {"airplane" : ["1.obj", "2.obj", "5.obj"], "ant" : ["3.obj", "4.obj"], ... }
        self.cls_name : ["airplane", "ant", ... ]
        self.cls : [0, 1, ... ]
        self.cls2name : {0 : "airplane", 1 : "ant", ... }
        self.cls2model = {0 : ["1.obj", "2.obj", "5.obj"], 1: ["3.obj", "4.obj"], ... }
        self.cls_list = [0, 0, 0, 1, 1, ... ]
        self.model_list = ["1.obj", "2.obj", "5.obj", "3.obj", "4.obj", ... ]
        '''

        self.C_model = args.C_model
        self.K_model = args.K_model
        self.renderer = renderer
        self.cla_file = args.model_cla_file

        class2model = read_cla(self.cla_file)
        self.cls_name = list(class2model.keys())
        self.total_cls_num = len(self.cls_name)

        self.cls = list(range(self.total_cls_num))
        self.cls2name = {} # match class & class_name (0:"airplane")
        for i in range(self.total_cls_num):
            self.cls2name[i] = self.cls_name[i]

        self.cls2model = {} # match class & according model (0:["1.obj path", "2.obj path"])
        self.total_model_num = 0
        for i in range(self.total_cls_num):
            model_idx_list = class2model[self.cls2name[i]]
            self.cls2model[i] = [args.model_dir + "/" + idx + ".obj" for idx in model_idx_list]
            self.total_model_num += len(model_idx_list)

        self.cls_list = list()
        self.model_list = list()
        for cl, m in self.cls2model.items():
            self.cls_list += [cl] * len(m)
            self.model_list += m

    def __len__(self):
        return self.total_model_num

    def __getitem__(self, idx):
        cls_list = list()
        model_list = list()
        make_tensor = transforms.ToTensor()
        
        # Selected pair with index
        selected_cls = self.cls_list[idx]
        selected_model = self.model_list[idx]

        cls_sub = self.cls.copy()
        cls_sub.remove(selected_cls)
        sampled_cls = sorted(random.sample(cls_sub, self.C_model-1))
        sampled_cls.append(selected_cls)
        for c in sampled_cls:
            if c == selected_cls:
                cls2model_sub = self.cls2model[c].copy()
                cls2model_sub.remove(selected_model)
                sampled_models = sorted(random.sample(cls2model_sub, self.K_model-1))
                sampled_models.append(selected_model)
                cls_list += [c] * self.K_model
                model_list += sampled_models
            else:
                sampled_models = sorted(random.sample(self.cls2model[c], self.K_model))
                cls_list += [c] * self.K_model
                model_list += sampled_models

        # Make sampled class to torch.tensor
        cls_tensor = torch.Tensor(cls_list)

        # View rendering for each model
        decide_expand_dim = True
        for model in model_list:
            rendered_model = self.renderer(model)
            rendered_model = torch.unsqueeze(rendered_model,0)
            if decide_expand_dim:
                rendered_models_tensor = rendered_model
                decide_expand_dim = False
            else:
                rendered_models_tensor = torch.cat((rendered_models_tensor, rendered_model),0)

        return rendered_models_tensor, cls_tensor
        
if __name__ == "__main__":
    args = make_args()
    # sketch_dataset = SketchDataset(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    params = {
        "image_size": 256,
        "camera_dist": 1.8,   
        "elevation": [-45,-45,-45,-45,0,0,0,0,45,45,45,45],
        "azim_angle": [0,90,180,270]*3,
        "obj_filename": "./input/m0.obj"
    }
    phong_renderer = PhongRenderer(args, params["image_size"], params["camera_dist"], params["elevation"], params["azim_angle"])
    model_dataset = ModelDataset(args, phong_renderer)
    model_dataset.__getitem__(1)
