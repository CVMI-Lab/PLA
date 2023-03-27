import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input, target, mask):
        selected_input = input[mask]
        cos_similarity = nn.functional.cosine_similarity(selected_input, target).mean()
        return 1 - cos_similarity


class BYOLLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, target):
        loss = 2 - 2 * (input * target).sum(dim=-1)
        return loss.mean()
