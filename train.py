import os
import glob
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

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

# def pretrain_sketch(args, data_loader, model, optim):

def train(args, render_params={}):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # mp.set_start_method("spawn")
    writer = SummaryWriter()

    ### Import Phong Renderer ###
    renderer = PhongRenderer(args, render_params["image_size"], render_params["camera_dist"], render_params["elevation"], render_params["azim_angle"])

    ### Dataset(Sketch-Model) & DataLoader ###
    sketch_model_train = SketchModelDataset(args, renderer)
    train_loader = DataLoader(sketch_model_train, batch_size=1, shuffle=True, num_workers=4)
    train_iterator = iter(train_loader)
    # train_loader = DataLoader(sketch_model_train, batch_size=1, shuffle=True)
    
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

    ### Step 0-1:  Pre-training sketch ###
    if args.pretraining:
        
        if not os.path.exists(args.model_pretrained_ckpt_dir):
            os.mkdir(args.model_pretrained_ckpt_dir)
        if not os.path.exists(args.trans_pretrained_ckpt_dir):
            os.mkdir(args.trans_pretrained_ckpt_dir)

        if '1' in args.pretraining_mode:
            epoch_count = 0
            iter_count = 0
            while iter_count < args.max_iter:
                # for i, data in enumerate(train_loader, 0):
                iter_count += 1
                print(iter_count)
                # Data Load
                data = next(train_iterator)
                sketches = data[0].to(device) if torch.cuda.is_available() else data[0]
                cls_sketch = data[1].to(device) if torch.cuda.is_available() else data[1]
                sketches = torch.squeeze(sketches)
                cls_sketch = torch.squeeze(cls_sketch)

                ### Update Sketch CNN & Metric network ###
                sketch_cnn.zero_grad()
                sketch_metric.zero_grad()

                s_cnn_features = sketch_cnn(sketches)
                s_metric_features = sketch_metric(s_cnn_features)

                iaml_loss_sketch = IAML_loss(s_metric_features, s_metric_features, cls_sketch)
                iaml_loss_sketch.backward()
                sketch_adam.step()

                writer.add_scalar("Loss/Sketch_iaml_pre", iaml_loss_sketch, iter_count)
                if iter_count % 100 == 0:
                    print("Pre-train Sketch network step... Iteration Check: {}".format(iter_count))
                    
                    # Save Models
                    if not os.path.exists(args.sketch_pretrained_ckpt_dir + "/" + str(iter_count)):
                        os.mkdir(args.sketch_pretrained_ckpt_dir + "/" + str(iter_count))
                    sketch_cnn_ckpt_path = args.sketch_pretrained_ckpt_dir + "/" + str(iter_count) + "/sketch_cnn_ckpt.pth"
                    sketch_metric_ckpt_path = args.sketch_pretrained_ckpt_dir + "/" + str(iter_count) + "/sketch_metric_ckpt.pth"
                    sketch_optim_ckpt_path = args.sketch_pretrained_ckpt_dir + "/" + str(iter_count) + "/sketch_optim_ckpt.pth"
                    torch.save(sketch_cnn.state_dict(), sketch_cnn_ckpt_path)
                    torch.save(sketch_metric.state_dict(), sketch_metric_ckpt_path)
                    torch.save(sketch_adam.state_dict(), sketch_optim_ckpt_path)

        if '2' in args.pretraining_mode:
            ### Step 0-2:  Pre-training model ###
            iter_count = 0
            while iter_count < args.max_iter:
                # for i, data in enumerate(train_loader, 0):
                iter_count += 1
                print(iter_count)

                # Data Load
                data = next(train_iterator)
                rendered_models = data[2].to(device) if torch.cuda.is_available() else data[2]
                cls_model = data[3].to(device) if torch.cuda.is_available() else data[3]
                rendered_models = torch.squeeze(rendered_models)
                cls_model = torch.squeeze(cls_model)

                model_cnn.zero_grad()
                model_metric.zero_grad()

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

                iaml_loss_model = IAML_loss(m_metric_features, m_metric_features, cls_model)
                iaml_loss_model.backward()
                
                model_adam.step()

                writer.add_scalar("Loss/Model_iaml_pre", iaml_loss_model, iter_count)
                if iter_count % 100 == 0:
                    print("Pre-train Model network step... Iteration Check: {}".format(iter_count))

            if epoch_count % 5 == 0:
                # Save Models                    
                model_cnn_ckpt_path = args.model_pretrained_ckpt_dir + "/model_cnn_ckpt_" + str(epoch_count) + ".pth"
                model_metric_ckpt_path = args.model_pretrained_ckpt_dir + "/model_metric_ckpt_" + str(epoch_count) + ".pth"
                model_optim_ckpt_path = args.model_pretrained_ckpt_dir + "/model_optim_ckpt_" + str(epoch_count) + ".pth"
                torch.save(model_cnn.state_dict(), model_cnn_ckpt_path)
                torch.save(model_metric.state_dict(), model_metric_ckpt_path)
                torch.save(model_optim.state_dict(), model_optim_ckpt_path)

        if '3' in args.pretraining_mode:
            ### Load Pre-trained Networks ###
            sketch_ckpt_dir_list = sorted(glob.glob(args.sketch_pretrained_ckpt_dir + "/*"))
            sketch_load_ckpt_dir = sketch_ckpt_dir_list[-1]
            sketch_ckpt_list = sorted(glob.glob(sketch_load_ckpt_dir+"/*"))

            sketch_cnn.load_state_dict(torch.load([ckpt for ckpt in sketch_ckpt_list if "cnn" in ckpt][0]))
            sketch_metric.load_state_dict(torch.load([ckpt for ckpt in sketch_ckpt_list if "metric" in ckpt][0]))
            import ipdb; ipdb.set_trace(context=21) 


            # model_ckpt_dir_list = sorted(glob.glob(args.model_pretrained_ckpt_dir + "/*"))
            # model_load_ckpt_dir = ckpt_dir_list[-1]
            # model_ckpt_list = sorted(glob.glob(load_ckpt_dir+"/*"))

            # model_cnn.load_state_dict(torch.load(ckpt_list["model_cnn" in ckpt_list]))
            # model_metric.load_state_dict(torch.load(ckpt_list["model_metric" in ckpt_list]))


            ### Update Transformation Network ###
            epoch_count = 0
            while epoch_count < args.max_epoch:
                epoch_count += 1
                iter_count = 0
                for i, data in enumerate(train_loader, 0):
                    iter_count += 1
                    # Data Load
                    sketches = data[0].to(device) if torch.cuda.is_available() else data[0]
                    cls_sketch = data[1].to(device) if torch.cuda.is_available() else data[1]
                    rendered_models = data[2].to(device) if torch.cuda.is_available() else data[2]
                    cls_model = data[3].to(device) if torch.cuda.is_available() else data[3]

                    sketches = torch.squeeze(sketches)
                    cls_sketch = torch.squeeze(cls_sketch)
                    rendered_models = torch.squeeze(rendered_models)
                    cls_model = torch.squeeze(cls_model)

                    # Gradient Initialization
                    sketch_cnn.zero_grad()
                    sketch_metric.zero_grad()

                    model_cnn.zero_grad()
                    model_metric.zero_grad()

                    trans_features.zero_grad()
                    discriminator.zero_grad()

                    # Sketch network forward
                    s_cnn_features = sketch_cnn(sketches)
                    s_metric_features = sketch_metric(s_cnn_features)

                    # CAD Model network forward
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
                    
                    # Transformation network forward
                    trans_features = transform_net(s_metric_features)

                    # Discriminator network forward
                    trans_disc = discriminator(trans_features)
                    model_disc = discriminator(m_metric_features)

                    # Loss
                    trans_loss = IAML_loss(trans_features, trans_features, cls_sketch) + \
                        (G_loss(trans_disc) + CMD_loss(trans_features, m_metric_features, cls_sketch, cls_model))
                    disc_loss = D_loss(model_disc, trans_disc)

                    trans_loss.backward()
                    disc_loss.backward()

                    trans_adam.step()

                    writer.add_scalar("Loss/Trans_trans_pre", iaml_loss_model, iter_count)
                    writer.add_scalar("Loss/Trans_disc_pre", iaml_loss_model, iter_count)

                    if iter_count % 100 == 0:
                        print("Pre-train Transformation network step... Iteration Check: {}".format(iter_count))

                # if epoch_count % 5 == 0:
                #     # Save Models, 
                #     model_cnn_ckpt_path = args.model_pretrained_ckpt_dir + "/model_cnn_ckpt_" + str(epoch_count) + ".pth"
                #     model_metric_ckpt_path = args.model_pretrained_ckpt_dir + "/model_metric_ckpt_" + str(epoch_count) + ".pth"
                #     model_optim_ckpt_path = args.model_pretrained_ckpt_dir + "/model_optim_ckpt_" + str(epoch_count) + ".pth"
                #     torch.save(model_cnn.state_dict(), model_cnn_ckpt_path)
                #     torch.save(model_metric.state_dict(), model_metric_ckpt_path)
                #     torch.save(model_optim.state_dict(), model_optim_ckpt_path)
    else:
        ### Update Transformation Network ###
        epoch_count = 0
        while epoch_count < args.max_epoch:
            epoch_count += 1
            iter_count = 0
            for i, data in enumerate(train_loader, 0):
                iter_count += 1
                # Data Load
                sketches = data[0].to(device) if torch.cuda.is_available() else data[0]
                cls_sketch = data[1].to(device) if torch.cuda.is_available() else data[1]
                rendered_models = data[2].to(device) if torch.cuda.is_available() else data[2]
                cls_model = data[3].to(device) if torch.cuda.is_available() else data[3]

                sketches = torch.squeeze(sketches)
                cls_sketch = torch.squeeze(cls_sketch)
                rendered_models = torch.squeeze(rendered_models)
                cls_model = torch.squeeze(cls_model)

                # Sketch Network update
                sketch_cnn.zero_grad()
                sketch_metric.zero_grad()

                s_cnn_features = sketch_cnn(sketches)
                s_metric_features = sketch_metric(s_cnn_features)

                iaml_loss_sketch = IAML_loss(s_metric_features,s_metric_features,cls_sketch)
                iaml_loss_sketch.backward()

                sketch_adam.step()

                # CAD Model network forward

                model_cnn.zero_grad()
                model_metric.zero_grad()

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

                iaml_loss_model = IAML_loss(m_metric_features,m_metric_features,cls_model)
                iaml_loss_model.backward()

                model_adam.step()
                
                # Transformation & Discriminator network forward
                trans_features.zero_grad()
                discriminator.zero_grad()

                trans_features = transform_net(s_metric_features)
                trans_disc = discriminator(trans_features)
                model_disc = discriminator(m_metric_features)

                trans_loss = IAML_loss(trans_features, trans_features, cls_sketch) + \
                    (G_loss(trans_disc) + CMD_loss(trans_features, m_metric_features, cls_sketch, cls_model))
                disc_loss = D_loss(model_disc, trans_disc)

                disc_loss.backward()
                disc_adam.step()

                trans_loss.backward()
                trans_adam.step()

                # writer.add_scalar("Loss/Trans_trans_pre", iaml_loss_model, iter_count)
                # writer.add_scalar("Loss/Trans_disc_pre", iaml_loss_model, iter_count)

                if iter_count % 100 == 0:
                    print("Whole Training step... Iteration Check: {}".format(iter_count))

            # if epoch_count % 5 == 0:
            #     # Save Models, 
            #     model_cnn_ckpt_path = args.model_pretrained_ckpt_dir + "/model_cnn_ckpt_" + str(epoch_count) + ".pth"
            #     model_metric_ckpt_path = args.model_pretrained_ckpt_dir + "/model_metric_ckpt_" + str(epoch_count) + ".pth"
            #     model_optim_ckpt_path = args.model_pretrained_ckpt_dir + "/model_optim_ckpt_" + str(epoch_count) + ".pth"
            #     torch.save(model_cnn.state_dict(), model_cnn_ckpt_path)
            #     torch.save(model_metric.state_dict(), model_metric_ckpt_path)
            #     torch.save(model_optim.state_dict(), model_optim_ckpt_path)
    

if __name__ == "__main__":
    args = make_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("Working GPU: {}".format(args.gpu))
    print("Pretraining or not: {}".format(args.pretraining))
    print("Pretraining Mode: {}".format(args.pretraining_mode))

    # Random setting?
    render_params = {
        "image_size": 224,
        "camera_dist": 1.8,   
        "elevation": [-45,-45,-45,-45,0,0,0,0,45,45,45,45],
        "azim_angle": [0,90,180,270]*3,
    }

    train(args, render_params)
    import ipdb; ipdb.set_trace(context=21)