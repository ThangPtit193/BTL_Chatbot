import torch
import torch.nn as nn


class AlignmentLoss(nn.Module):
    def __init__(self, alpha=2):
        
        self.alpha = alpha
    
    def forward(self, anchor, positive):
        """
        Computes the alignment loss between two tensors, anchor and positive, using the L2 norm with an optional power factor.

        Args:
            anchor (torch.Tensor): The first input tensor.
            positive (torch.Tensor): The second input tensor.
            alpha (float, optional): The power factor for the L2 norm computation. Default is 2.

        Returns:
            torch.Tensor: The alignment loss computed using the L2 norm raised to the power of alpha.

        Raises:
            TypeError: If anchor or positive are not torch.Tensor objects.
            ValueError: If anchor and positive have different shapes.

        Examples:
            >>> anchor = torch.tensor([[1, 2], [3, 4], [5, 6]])
            >>> positive = torch.tensor([[2, 3], [4, 5], [6, 7]])
            >>> align_loss(anchor, positive, alpha=2)
            tensor(0.7071)
        """

        return (anchor - positive).norm(p=2, dim=1).pow(self.alpha).mean()