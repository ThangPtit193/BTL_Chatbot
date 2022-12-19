import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from enum import Enum
from sentence_transformers import SentenceTransformer

class QuadrupletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)

class QuadrupletLoss(nn.Module):
    """
    Quadruplet loss
    """
    def __init__(
            self, 
            model: SentenceTransformer, 
            distance_metric=QuadrupletDistanceMetric.EUCLIDEAN, 
            quadruple_margin_1: float = 2.0,
            quadruple_margin_2: float = 1.0,
        ):
        super(QuadrupletLoss, self).__init__()
        self.model = model
        self.distance_metric = distance_metric
        self.quadruple_margin_1 = quadruple_margin_1
        self.quadruple_margin_2 = quadruple_margin_2


    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(QuadrupletDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "TripletDistanceMetric.{}".format(name)
                break

        return {
            'distance_metric': distance_metric_name, 
            'quadruple_margin_1': self.quadruple_margin_1,
            'quadruple_margin_2': self.quadruple_margin_2
            }

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        rep_anchor, rep_pos, rep_neg_1, rep_neg_2  = reps
        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg_1 = self.distance_metric(rep_anchor, rep_neg_1)
        distance_neg_2 = self.distance_metric(rep_anchor, rep_neg_2)
        
        losses_1 = F.relu(distance_pos - distance_neg_1 + self.quadruple_margin_1)
        losses_2 = F.relu(distance_pos - distance_neg_2 + self.quadruple_margin_2)
        losses = losses_1 + losses_2
        return losses.mean()