from args import make_args
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim


from network import MVCNN, MetricCNN, TransformNetwork

if __name__ == "__main__":

    # Parameter Setting
    args = make_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    import ipdb; ipdb.set_trace()

    # DataLoader

    # Network
    # sketch_cnn = MVCNN()
    # sketch_metric = MetricCNN()

    # model_cnn = MVCNN()
    # model_metric = MetricCNN()

    # transform_net = TransformNetwork()