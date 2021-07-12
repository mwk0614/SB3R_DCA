import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models
import torchvision.transforms as transforms

class MVCNN(nn.Module):
    def __init__(self, type=None, model_name="resnet50"):
        super(MVCNN, self).__init__()
        self.flatten = nn.Flatten()
        if "resnet50" in model_name:
            # Use Pooling 5 layer of Resnet-50
            model = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(model.children())[:-1])

    def forward(self,x):
        x = self.model(x)
        x = self.flatten(x)
        return x


class MetricCNN(nn.Module):
    def __init__(self, ):
        super(MetricCNN,self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.BatchNorm1d(128),
        )

    def forward(self,x):
        return self.network(x)

class TransformNetwork(nn.Module):
    def __init__(self):
        super(TransformNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.network(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(128,64),
            nn.Linear(64,1)
        )

    def forward(self, x):
        return self.network(x)

def average_view_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)


if __name__ == "__main__":
    ############# Test Network #################
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load rendered images (take one image)
    rendered_imgs = np.load("./input/ant_batch.npy")
    img_original = rendered_imgs[...,:3]
    img = torch.tensor(np.transpose(img_original,(0,3,1,2)))

    expanded = np.expand_dims(rendered_imgs, axis=0)
    stacked = np.concatenate((expanded,expanded), axis=0)

    # pre-processing
    transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    img_transformed = transform(img).to(device)

    # Network
    mvcnn = MVCNN().to(device)
    metric = MetricCNN().to(device)
    transformation = TransformNetwork().to(device)

    flattened_1 = mvcnn(img_transformed)
    view_pooled_1 = average_view_pooling(flattened_1)
    flattened_2 = mvcnn(img_transformed)
    view_pooled_2 = average_view_pooling(flattened_2)
    view_pooled = torch.stack((torch.squeeze(view_pooled_1), torch.squeeze(view_pooled_2)),dim=0)
    metric_passed = metric(view_pooled)


