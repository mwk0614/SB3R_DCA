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

class ModelDataset():
    def __init__(self, args):
        '''
        This data loader makes rendered-model batch offline (from already renderd images)

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

        # self.C_model = args.C_model
        self.args = args
        self.model_dir = args.model_dir
        self.K_model = args.K_model
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

    def get_item(self, sampled_cls):
        view_num = 12
        cls_list = list()
        model_list = list()
        rendered_models_path_list = list()
        make_tensor = transforms.ToTensor()
        
        # Selected pair with index
        for c in sampled_cls:
            sampled_models = sorted(random.sample(self.cls2model[c], self.K_model))
            cls_list += [c] * self.K_model
            model_list += sampled_models
        
        for model in model_list:
            assert model[-4:] == ".obj"
            for v in range(view_num):
                rendered_models_path_list.append(model[:-4]+"_"+str(v)+".png")

        # Make sampled class to torch.tensor
        cls_model = torch.Tensor(cls_list)

        # Load Rendered View Images
        save_single_container = True
        save_multi_container = True
        view_count = 0
        for path in rendered_models_path_list:
            view_count += 1
            rendered_img_pil = Image.open(path).convert(mode="RGB")
            rendered_img = make_tensor(rendered_img_pil)
            rendered_img = torch.unsqueeze(torch.unsqueeze(rendered_img,0),0)
            if save_single_container:
                rendered_imgs = rendered_img
                save_single_container = False
            else:
                rendered_imgs = torch.cat((rendered_imgs, rendered_img),1)
                if view_count == view_num:
                    if save_multi_container:
                        total_rendered_imgs = rendered_imgs
                        save_multi_container = False
                    else:
                        total_rendered_imgs = torch.cat((total_rendered_imgs, rendered_imgs),0)
                    view_count = 0
                    save_single_container = True
            
        return total_rendered_imgs, cls_model


class SketchModelDataset(Dataset):
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
        self.C = args.C
        self.K_sketch = args.K_sketch
        self.sketch_train_dir = args.sketch_train_dir
        self.total_cls_num = 0
        self.total_sketches_num = 0

        # Import ModelDataset
        self.model_dataset = ModelDataset(args)

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

    def __len__(self):
        return self.total_sketches_num

    def __getitem__(self, idx):

        ############ Sketch Part ############
        cls_list = list()
        sketches_list = list()
        make_tensor = transforms.ToTensor()
        
        # Sample class-sketch pair
        sampled_cls = sorted(random.sample(self.cls, self.C))

        for c in sampled_cls:
            sampled_sketches = sorted(random.sample(self.cls2sketches[c], self.K_sketch))
            cls_list += [c] * self.K_sketch
            sketches_list += sampled_sketches

        # Make sampled class to torch.tensor
        cls_sketch = torch.Tensor(cls_list)

        # Read image & convert 3 channels & make to torch.tensor
        decide_expand_dim = True
        for path in sketches_list:
            sketch_pil = Image.open(path).convert(mode="RGB")
            sketch = make_tensor(sketch_pil)
            sketch = torch.unsqueeze(sketch,0)
            if decide_expand_dim:
                sketches = sketch
                decide_expand_dim = False
            else:
                sketches = torch.cat((sketches, sketch),0)

        # transform input
        if self.transform is not None:
            sketches = self.transform(sketches)

        ############ Model Part ############
        rendered_models, cls_model = self.model_dataset.get_item(sampled_cls) # [batch, view_num, C, H, W]
        return sketches, cls_sketch, rendered_models, cls_model, self.cls2name

        
if __name__ == "__main__":
    args = make_args()
    # sketch_dataset = SketchDataset(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model_dataset = SketchModelDataset(args)
    model_dataset.__getitem__(1)
