import os
import cv2
import meshio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.utils as vutils

def read_image(filepath):
    raise NotImplementedError

def read_cla(cla_file):
    with open(cla_file, 'r') as f:
        contents = list()
        for line in f:
            contents.append(line.strip())
    split_idx = [idx for idx, el in enumerate(contents) if el == ""]

    class_list = list()
    for i in range(len(split_idx)-1):
        start = split_idx[i]
        end = split_idx[i+1]
        class_list.append(contents[start+1:end])
    
    class_model = {}
    class_number_check = list()
    model_number_check = list()
    for pair in class_list:
        class_name = pair[0].split(" ")[0]
        if "\t" in class_name:
            class_name = class_name.replace("\t","")
        model_id = ["m"+idx for idx in pair[1:]]
        class_model[class_name] = model_id
        class_number_check.append(class_name)
        model_number_check += model_id
        
    assert len(class_number_check) == int(contents[1].split(" ")[0]) 
    assert len(model_number_check) == int(contents[1].split(" ")[1])

    return class_model

def save_checkpoint(model, optimizer, save_path):
    ckpt_format = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(ckpt_format, save_path)

def input_check(input_tensor):
    img_tensor = input_tensor[0].cpu()
    img_npy = img_tensor.numpy()
    img_npy = np.transpose(img_tensor, (1,2,0))
    plt.imshow(img_npy)
    plt.show()

def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
    # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
    # m.bias.data should be 0
        m.bias.data.fill_(0)

def resize_image(img_path, resize_shape=(224,224), filter_name="Lanzcos"):
    assert filter_name in ["Bilinear", "Bicubic", "Lanzcos"] 
    img = Image.open(img_path)
    if filter_name == "Bilinear":
        img_resized = img.resize(resize_shape, Image.BILINEAR)
    if filter_name == "Bicubic":
        img_resized = img.resize(resize_shape, Image.BICUBIC)
    if filter_name == "Lanzcos":
        img_resized = img.resize(resize_shape, Image.LANCZOS)
    return img_resized

def check_input(data_loader, out_dir="check_input"):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    data_zip = next(iter(data_loader))
    sketches = torch.squeeze(data_zip[0]).detach().cpu()
    sketches = sketches.numpy()
    sketches_cls = (torch.squeeze(data_zip[1]).detach().cpu()).numpy()
    cls2name = data_zip[2]
    print("Sketch tensor shape: {}".format(sketches.shape))
    # print("Model tensor shape: {}".format(sketches.shape))
    for i in range(sketches.shape[0]):
        sketch = np.transpose(sketches[i],(1,2,0))
        cv2.imwrite(out_dir+"/sketch_"+str(i)+"_"+cls2name[int(sketches_cls[i])][0]+".png", (sketch*255).astype(np.uint8))
    import ipdb; ipdb.set_trace(context=21)



if __name__ == "__main__":
    cla_file = "./cla_files/SHREC13_SBR_Model.cla"
    class_model = read_cla(cla_file)
    
