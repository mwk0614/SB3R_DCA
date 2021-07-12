import os
import glob
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, models, transforms

from args import make_args
from utils import *
from data import SketchModelDataset
from renderer import PhongRenderer
from network import MVCNN, MetricCNN, TransformNetwork, Discriminator, average_view_pooling
from loss import IAML_loss, CMD_loss

def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Just for Developing
    render_params = {
        "image_size": 224,
        "camera_dist": 1.8,   
        "elevation": [-45,-45,-45,-45,0,0,0,0,45,45,45,45],
        "azim_angle": [0,90,180,270]*3,
        "obj_filename": "./input/m0.obj"
    }

    # Import Phong Renderer
    renderer = PhongRenderer(args, render_params["image_size"], render_params["camera_dist"], render_params["elevation"], render_params["azim_angle"])

    # Dataset(Sketch-Model) & DataLoader
    sketch_model_dataset = SketchModelDataset(args, renderer)
    train_loader = DataLoader(sketch_model_dataset, batch_size=1, shuffle=True)
    sketches, cls_sketch, rendered_models, cls_model = next(iter(train_loader))

    sketches = torch.squeeze(sketches)
    cls_sketch = torch.squeeze(cls_sketch)
    rendered_models = torch.squeeze(rendered_models)
    cls_model = torch.squeeze(cls_model)
    # print(sketches.shape, cls_sketch.shape, rendered_models.shape, cls_model.shape)

    # Network
    sketch_cnn = MVCNN()
    sketch_metric = MetricCNN()

    model_cnn = MVCNN()
    model_metric = MetricCNN()

    transform_net = TransformNetwork()

    discriminator = Discriminator()

    ### Network Input Output & Loss Test ###
    sketches = sketches.to(device)
    rendered_models = rendered_models.to(device)

    sketch_cnn = sketch_cnn.to(device)
    sketch_metric = sketch_metric.to(device)
    transform_net = transform_net.to(device)

    model_cnn = model_cnn.to(device)
    model_metric = model_metric.to(device)

    discriminator = discriminator.to(device)

    s_cnn_features = sketch_cnn(sketches)
    s_metric_features = sketch_metric(s_cnn_features)
    trans_features = transform_net(s_metric_features)

    decide_expand_dim = True
    view_num = rendered_models.shape[1]
    for i in range(view_num):
        m_cnn_feature = model_cnn(rendered_models[ : , i, ... ])
        if decide_expand_dim:
            m_cnn_features_sub = torch.unsqueeze(m_cnn_feature, 1)
            decide_expand_dim = False
        else:
            m_cnn_feature = torch.unsqueeze(m_cnn_feature, 1)
            m_cnn_features_sub = torch.cat((m_cnn_features_sub, m_cnn_feature), 1)

    m_cnn_features = average_view_pooling(m_cnn_features_sub)
    m_metric_features = model_metric(m_cnn_features)

    iaml_loss_sketch = IAML_loss(s_metric_features, s_metric_features, cls_sketch)
    iaml_loss_trans = IAML_loss(trans_features, trans_features, cls_sketch)
    cmd_loss = CMD_loss(trans_features, m_metric_features, cls_sketch, cls_model)
    # import ipdb; ipdb.set_trace(context=21)

if __name__ == "__main__":
    args = make_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    train(args)