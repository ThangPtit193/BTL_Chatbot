from abc import abstractmethod
from pathlib import Path
from typing import Union, List

import torch.utils.data
from loguru import logger


class BaseEvaluation(object):
    """
    Base class for implementing evaluation of given model over a dataset
    """

    def __init__(
            self,
            debug: bool = False,
            open_domain: bool = False,
            top_k: int = 5
    ):
        """

        :param debug: if true, a record of each sample and its evaluation will be stored in EvalDocument.log
        :param open_domain: if true, a document is considered correctly retrieved so long as the answer string can be
                            found within it. Otherwise, correct retrieval is evaluated based document id
        :param top_k: calculate eval metrics for top k results
        """
        super().__init__()
        self.debug = debug
        self.open_domain = open_domain
        self.top_k = top_k
        self.no_answer_warning = True
        self.log: List = []

    def __call__(
            self,
            model: Union[str, Path],
            output_path: str = None,
            epoch: int = -1,
            steps: int = -1
    ) -> float:
        """
        This is called during training
        :param model:
        :param output_path:
        :param epoch:
        :param steps:
        :return:
        """





