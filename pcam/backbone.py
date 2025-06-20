import torch
from torchvision.models import densenet201

def generate_backbone(weights=None):
    backbone = densenet201(weights=weights)
    backbone.classifier = torch.nn.Identity()
    return backbone