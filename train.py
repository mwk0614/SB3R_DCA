import os
import glob
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets, models, transforms

from args import make_args
from utils import *
from data import SketchModelDataset
from renderer import PhongRenderer
from network import MVCNN, MetricCNN, TransformNetwork, Discriminator, average_view_pooling
from loss import IAML_loss, CMD_loss, G_loss, D_loss

'''
- make "render_params" hyper-parameter || random sampled
- training sketch transformation (normalization etc...)
- network weight "initialization" for "metric, transformation, discriminator network"
- GAN Loss [o]
- Training pipeline
- Testing pipeline
'''

def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    ### Just for Developing ###
    render_params = {
        "image_size": 224,
        "camera_dist": 1.8,   
        "elevation": [-45,-45,-45,-45,0,0,0,0,45,45,45,45],
        "azim_angle": [0,90,180,270]*3,
        "obj_filename": "./input/m0.obj"
    }

    ### Import Phong Renderer ###
    renderer = PhongRenderer(args, render_params["image_size"], render_params["camera_dist"], render_params["elevation"], render_params["azim_angle"])

    ### Dataset(Sketch-Model) & DataLoader ###
    sketch_model_train = SketchModelDataset(args, renderer)
    train_loader = DataLoader(sketch_model_train, batch_size=1, shuffle=True)

    ### Network ###
    sketch_cnn = MVCNN()
    sketch_metric = MetricCNN()

    model_cnn = MVCNN()
    model_metric = MetricCNN()

    transform_net = TransformNetwork()
    discriminator = Discriminator()

    if torch.cuda.is_available():
        sketch_cnn = sketch_cnn.to(device)
        sketch_metric = sketch_metric.to(device)

        model_cnn = model_cnn.to(device)
        model_metric = model_metric.to(device)

        transform_net = transform_net.to(device)
        discriminator = discriminator.to(device)

    ### Optimizer ###
    sketch_optim_vars = [{"params": sketch_cnn.parameters()}, {"params": sketch_metric.parameters()}]
    model_optim_vars = [{"params": model_cnn.parameters()}, {"params": model_metric.parameters()}]
    trans_optim_vars = [{"params": transform_net.parameters()}]
    disc_optim_vars = [{"params": discriminator.parameters()}]

    sketch_adam = optim.Adam(sketch_optim_vars, lr=args.lr)
    model_adam = optim.Adam(model_optim_vars, lr=args.lr)
    trans_adam = optim.Adam(trans_optim_vars, lr=args.lr)
    disc_adam = optim.Adam(disc_optim_vars, lr=args.lr)

    ### Loss (Objective Function) ###
    # IAML, CMD loss, G_loss, D_loss imported in iteration

    ### Pre-training step ###
    epoch_count = 0
    iter_count = 0
    while epoch_count < args.max_epoch:
        epoch_count += 1
        for i, data in enumerate(sketch_model_train, 0):
            iter_count += 1
            # Data Load
            sketches = data[0].to(device) if torch.cuda.is_available() else data[0]
            cls_sketch = data[1].to(device) if torch.cuda.is_available() else data[1]
            rendered_models = data[2].to(device) if torch.cuda.is_available() else data[2]
            cls_model = data[3].to(device) if torch.cuda.is_available() else data[3]

            ### Update Sketch CNN & Metric network ###
            s_cnn_features = sketch_cnn(sketches)
            s_metric_features = sketch_metric(s_cnn_features)

            sketch_cnn.zero_grad()
            sketch_metric.zero_grad()

            iaml_loss_sketch = IAML_loss(s_metric_features, s_metric_features, cls_sketch)
            try:
                iaml_loss_sketch.backward()
            except AttributeError as e:
                print(e)
                import ipdb; ipdb.set_trace(context=21)
            sketch_adam.step()

            # ### Update Model CNN & Metric network ###
            # model_cnn.zero_grad()
            # model_metric.zero_grad()

            # decide_expand_dim = True
            # view_num = rendered_models.shape[1]
            # for i in range(view_num):
            #     m_cnn_feature = model_cnn(rendered_models[ : , i, ... ])
            #     if decide_expand_dim:
            #         m_cnn_features_sub = torch.unsqueeze(m_cnn_feature, 1)
            #         decide_expand_dim = False
            #     else:
            #         m_cnn_feature = torch.unsqueeze(m_cnn_feature, 1)
            #         m_cnn_features_sub = torch.cat((m_cnn_features_sub, m_cnn_feature), 1)
            # m_cnn_features = average_view_pooling(m_cnn_features_sub)
            # m_metric_features = model_metric(m_cnn_features)

            # iaml_loss_model = IAML_loss(m_metric_features, m_metric_features, cls_model)
            # # import ipdb; ipdb.set_trace(context=21)
            # iaml_loss_model.backward()
            
            # model_adam.step()
            writer.add_scalar("Loss/IAML_sketch", iaml_loss_sketch, iter_count)
            if iter_count % 5 == 0:
                print("Iteration Check: {}".format(iter_count))

        if epoch_count % 5 == 0:
            # Save Models, 
            sketch_cnn_ckpt_path = args.sketch_ckpt_dir + "/sketch_cnn_ckpt_" + str(epoch_count) + ".pth"
            sketch_metric_ckpt_path = args.sketch_ckpt_dir + "/sketch_metric_ckpt_" + str(epoch_count) + ".pth"
            sketch_optim_ckpt_path = args.sketch_ckpt_dir + "/sketch_optim_ckpt_" + str(epoch_count) + ".pth"
            torch.save(sketch_cnn.state_dict(), sketch_cnn_ckpt_path)
            torch.save(sketch_metric.state_dict(), sketch_metric_ckpt_path)
            torch.save(sketch_optim.state_dict(), sketch_optim_ckpt_path)
            # ### Update Discriminator ###
            # import ipdb; ipdb.set_trace(context=21)

            # ### Update Transformation Network ###
            # trans_features = transform_net(s_metric_features)

            # trans_disc = discriminator(trans_features)
            # model_disc = discriminator(m_metric_features)
            # import ipdb; ipdb.set_trace(context=21)

            # # Define Loss
            # iaml_loss_sketch = IAML_loss(s_metric_features, s_metric_features, cls_sketch)


            # iaml_loss_trans = IAML_loss(trans_features, trans_features, cls_sketch)
            # cmd_loss = CMD_loss(trans_features, m_metric_features, cls_sketch, cls_model)
            # gen_loss = G_loss(trans_disc)
            # disc_loss = D_loss(model_disc, trans_disc)
            # import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    args = make_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    train(args)