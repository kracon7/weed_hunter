import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CorridorNet(nn.Module):
    def __init__(self):
        super(CorridorNet, self).__init__()

        self.branch_1 = SingleBranch()
        self.branch_2 = SingleBranch()
        self.nn_layers = nn.ModuleList()

    def forward(self, x):
        vpz = self.branch_1(x)
        lines = self.branch_2(x)
        out = torch.cat([vpz, lines], dim=-1)
        return out


class SingleBranch(nn.Module):
    def __init__(self):
        super(SingleBranch, self).__init__()

        resnet = models.resnet18(pretrained=True)
        self.resnet_backbone = nn.Sequential(*(list(resnet.children())[:-2]))

        self.predictor = nn.Sequential(
                nn.Conv2d(512, 512, 5, 2),
                nn.ReLU(),
                nn.Conv2d(512, 512, 5, 2),
                nn.ReLU()
            )
        self.fc = nn.Sequential(
                nn.Linear(2048, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
        )
        
    def forward(self, x):
        out = self.resnet_backbone(x)
        out = self.predictor(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
