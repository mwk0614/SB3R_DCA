from args import make_args
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from train import Train

def train(args):
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

if __name__ == "__main__":
    args = make_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("Working GPU: {}".format(args.gpu))
    print("Pretraining or not: {}".format(args.pretraining))
    print("Pretraining Mode: {}".format(args.pretraining_mode))

    train(args)

    