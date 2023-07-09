import torch
import torch.nn as nn


class UniformityLoss(nn.Module):
    def __init__(self, t=2):
        
        self.t = t
    
    def forward(self, embedding):
        """
        Computes the uniform loss for a given tensor embedding, which measures the uniformity of pairwise distances between points.

        Args:
            embedding (torch.Tensor): The input tensor representing a set of points.
            t (float, optional): The temperature parameter controlling the spread of the distances. Default is 2.

        Returns:
            torch.Tensor: The uniform loss computed as the logarithm of the mean exponential of pairwise distances.

        Raises:
            TypeError: If embedding is not a torch.Tensor object.

        Examples:
            >>> embedding = torch.tensor([[1, 2], [3, 4], [5, 6]])
            >>> uniform_loss(embedding, t=2)
            tensor(-0.8113)
        """

        return torch.pdist(embedding, p=2).pow(2).mul(-self.t).exp().mean().log()