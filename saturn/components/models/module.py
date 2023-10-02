import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def dot_product_scores(compr: torch.Tensor, refer: torch.Tensor) -> torch.Tensor:
    r = torch.matmul(compr, torch.transpose(refer, 0, 1))
    return r

def dot_product_scores_numpy(compr, refer):
    r = np.matmul(compr, np.transpose(refer, (1, 0)))
    return r
    
class SimilarityFunction(nn.Module):
    def __init__(self, name_fn="cosine"):
        super().__init__()
        self.fn = dot_product_scores
    def forward(self, x, y):
        return self.fn(x, y)
