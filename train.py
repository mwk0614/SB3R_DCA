import os
import glob
import matplotlib
import matplotlib.pyplot as plt
from termcolor import colored

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
from train_utils import *
# from renderer import PhongRenderer

from data_offline import SketchModelDataset
from network import MVCNN, MetricCNN, TransformNetwork, Discriminator, average_view_pooling
from trainer import *
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

class Train():
    def __init__(self, args, render_params={}):
        # mp.set_start_method("spawn")
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ### Import Phong Renderer ###
        # renderer = PhongRenderer(args, render_params["image_size"], render_params["camera_dist"], render_params["elevation"], render_params["azim_angle"])

        ### Dataset(Sketch-Model) & DataLoader ###
        sketch_transform = transforms.Compose([
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        sketch_model_train = SketchModelDataset(args, sketch_transform)
        self.train_loader = DataLoader(sketch_model_train, batch_size=1, shuffle=True, num_workers=4)
        
        ### Network ###
        self.sketch_cnn = MVCNN().to(self.device) if torch.cuda.is_available() else MVCNN()
        self.sketch_metric = MetricCNN().to(self.device) if torch.cuda.is_available() else MetricCNN()

        self.model_cnn = MVCNN().to(self.device) if torch.cuda.is_available() else MVCNN()
        self.model_metric = MetricCNN().to(self.device) if torch.cuda.is_available() else MetricCNN()

        self.transform_net = TransformNetwork().to(self.device) if torch.cuda.is_available() else TransformNetwork()
        self.discriminator = Discriminator().to(self.device) if torch.cuda.is_available() else Discriminator()

        ### Optimizer ###
        sketch_optim_vars = [{"params": self.sketch_cnn.parameters()}, {"params": self.sketch_metric.parameters()}]
        model_optim_vars = [{"params": self.model_cnn.parameters()}, {"params": self.model_metric.parameters()}]
        trans_optim_vars = [{"params": self.transform_net.parameters()}]
        disc_optim_vars = [{"params": self.discriminator.parameters()}]

        self.sketch_optim = optim.Adam(sketch_optim_vars, lr=args.lr)
        self.model_optim = optim.Adam(model_optim_vars, lr=args.lr)
        self.trans_optim = optim.Adam(trans_optim_vars, lr=args.lr)
        self.disc_optim = optim.Adam(disc_optim_vars, lr=args.lr)

        # For Tensorboard
        self.writer = SummaryWriter()

    def sketch_pretraining(self):
        assert "1" in self.args.pretraining_mode
        self.epoch_count = 0
        self.total_iter_count = 0
        while self.epoch_count < self.args.max_epoch:
            self.epoch_count += 1
            print("Start {}th Epoch".format(self.epoch_count))
            sketch_pretrainer(self)

    def model_pretraining(self):
        ''' Step 0-2:  Pre-training model '''
        self.epoch_count = 0
        self.total_iter_count = 0
        while self.epoch_count < self.args.max_epoch:
            self.epoch_count += 1
            print("Start {}th Epoch".format(epoch_count))
            model_pretrainer(self)

    def trans_pretraining(self):

        ''' Step 0-3:  Pre-training Transformation Network '''
        if '3' in args.pretraining_mode:
            ### Load Pre-trained Networks (sketch, model) ###
            sketch_ckpt_dir_list = sorted(glob.glob("{}/".format(args.trials) + args.sketch_pretrained_ckpt_dir + "/*"))
            sketch_load_ckpt_dir = sketch_ckpt_dir_list[-1]
            sketch_ckpt_list = sorted(glob.glob(sketch_load_ckpt_dir+"/*"))

            self.sketch_cnn.load_state_dict(torch.load([ckpt for ckpt in sketch_ckpt_list if "cnn" in ckpt][0]))
            self.sketch_metric.load_state_dict(torch.load([ckpt for ckpt in sketch_ckpt_list if "metric" in ckpt][0]))

            model_ckpt_dir_list = sorted(glob.glob("{}/".format(args.trials) + args.model_pretrained_ckpt_dir + "/*"))
            model_load_ckpt_dir = model_ckpt_dir_list[-1]
            model_ckpt_list = sorted(glob.glob(model_load_ckpt_dir+"/*"))

            self.model_cnn.load_state_dict(torch.load([ckpt for ckpt in model_ckpt_list if "cnn" in ckpt][0]))
            self.model_metric.load_state_dict(torch.load([ckpt for ckpt in model_ckpt_list if "metric" in ckpt][0]))

            ### Update Transformation Network ###
            self.total_iter_count = 0
            self.epoch_count = 0
            while self.epoch_count < args.max_epoch:
                self.epoch_count += 1
                print("Start {}th Epoch".format(self.epoch_count))
                trans_pretrainer(self)

    def whole_training(self):
        '''Iterative Update all networks'''
        ### Load Pre-trained Networks (sketch, model, transform network) ###
        # Sketch
        sketch_ckpt_dir_list = sorted(glob.glob("{}/".format(args.trials) + args.sketch_pretrained_ckpt_dir + "/*"))
        sketch_load_ckpt_dir = sketch_ckpt_dir_list[-1]
        sketch_ckpt_list = sorted(glob.glob(sketch_load_ckpt_dir+"/*"))

        sketch_cnn.load_state_dict(torch.load([ckpt for ckpt in sketch_ckpt_list if "cnn" in ckpt][0]))
        sketch_metric.load_state_dict(torch.load([ckpt for ckpt in sketch_ckpt_list if "metric" in ckpt][0]))

        # Model
        model_ckpt_dir_list = sorted(glob.glob("{}/".format(args.trials) + args.model_pretrained_ckpt_dir + "/*"))
        model_load_ckpt_dir = model_ckpt_dir_list[-1]
        model_ckpt_list = sorted(glob.glob(model_load_ckpt_dir+"/*"))

        model_cnn.load_state_dict(torch.load([ckpt for ckpt in model_ckpt_list if "cnn" in ckpt][0]))
        model_metric.load_state_dict(torch.load([ckpt for ckpt in model_ckpt_list if "metric" in ckpt][0]))

        # Transform Network
        trans_ckpt_dir_list = sorted(glob.glob("{}/".format(args.trials) + args.trans_pretrained_ckpt_dir + "/*"))
        trans_load_ckpt_dir = trans_ckpt_dir_list[-1]
        trans_ckpt_list = sorted(glob.glob(trans_load_ckpt_dir+"/*"))

        transform_net.load_state_dict(torch.load([ckpt for ckpt in trans_ckpt_list if "net" in ckpt][0]))

        self.total_iter_count = 0
        self.epoch_count = 0
        while self.total_iter_count < args.max_iter:
            self.epoch_count += 1
            print("Start {}th Epoch".format(self.epoch_count))
            whole_trainer(self)

if __name__ == "__main__":
    args = make_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("Working GPU: {}".format(args.gpu))
    print("Pretraining or not: {}".format(args.pretraining))
    print("Pretraining Mode: {}".format(args.pretraining_mode))

    trainer_container = Train(args)
    if args.pretraining:
        if "1" in args.pretraining_mode:
            trainer_container.sketch_pretraining()
        if "2" in args.pretraining_mode:
            trainer_container.model_pretraining()
        if "3" in args.pretraining_mode:
            trainer_container.trans_pretraining()
    else:
        trainer_container.whole_trainer()
