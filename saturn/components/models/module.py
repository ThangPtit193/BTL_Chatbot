import torch
import torch.nn as nn
import torch.nn.functional as F

def dot_product_scores(compr, refer):
    r = torch.matmul(compr, torch.transpose(refer, 0, 1))
    return r
    
class SimilarityFunction(nn.Module):
    def __init__(self, name_fn="dot"):
        super().__init__()
        self.fn = dot_product_scores
    def forward(self, x, y):
        return self.fn(x, y)
