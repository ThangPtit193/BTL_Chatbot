from pathlib import Path

from venus.document_store import DocumentStore
from venus.pipelines.pipeline import Pipeline, BaseStandardPipeline
from venus.ranker import BaseRanker
from venus.reader.base import BaseReader
from venus.retriever import Retriever

from pipelines.pipelines.classifier import CustomQueryClassifier


class CustomRetrieverPipeline(BaseStandardPipeline):
    def __init__(
            self,
            # retriever: Retriever,
            # ranker: BaseRanker,
            # query_classifier: CustomQueryClassifier

    ):
        """
        Base class for implementing a custom block-building that is a very simple to ensemble together different
        components into a solid pipeline.

        :param retriever:This is an instance of Retriever Class. The Retriever performs Document
                        Retrieval by sweeping through a document store and returning a set of candidate documents that
                        are relevant to the query.
                         `Query`: Who is the father of Arya Stark?
                         {   'content': '\n'
                                       '===In the Riverlands===\n'
                                       'The Stark army reaches the Twins, a bridge stronghold '
                                       'controlled by Walder Frey, who agrees to allow the army to '
                                       'cross the river and to commit his troops in return for Robb '
                                       'an...',
                            'name': '450_Baelor.txt'}
                        Input: Query
                        Output: Document
        # :param ranker: A Ranker reorders a set of Documents based on their relevance to the Query
        #                 Input: Document
        #                 Output: Document
        # :param reader: The Reader takes a question and a set of Documents as input and returns an Answer by selecting a
        #                 text span within the Documents.
        #                 Input: Document
        #                 Output: Answer
        """
        self.pipeline = Pipeline()
        # self.pipeline.add_node(component=query_classifier, name="CustomQueryClassifier", inputs=["Query"])
        # self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query_Classifier"])

    # def load_config(self):

