import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, args, arch='resnet50'):
        super(Encoder, self).__init__()

        self.args = args

        if arch == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
        elif arch == "resnet34":
            self.backbone = models.resnet34(pretrained=True)
        elif arch == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
        else:
            raise ValueError("Unsupported architecture: {}".format(arch))
        
        dim_backbone = self.backbone.fc.out_features
        self.MLP = nn.Sequential(
            nn.Linear(dim_backbone, 128),
            nn.ReLU()
        )

    def forward(self, x_i, x_j):
        h_i = self.backbone(x_i)
        z_i = self.MLP(h_i)

        h_j = self.backbone(x_j)
        z_j = self.MLP(h_j)

        return z_i, z_j