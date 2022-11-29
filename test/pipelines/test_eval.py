import pytest

from meteor.pipelines.standard_pipelines import DocumentSearchPipeline, SearchSummarizationPipeline
from meteor.schema import MultiLabel, Label, Answer, Span, Document, EvaluationResult


@pytest.mark.parametrize("retriever_with_docs", ["embedding"], indirect=True)
def test_document_search_calculate_metrics(retriever_with_docs):
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(labels=LABELS, params={"Retriever": {"top_k": 22}},
                                                  dynamic_top_k=False)
    print(eval_result.save("test"))

    eval_result.calculate_metrics(document_scope="document_id")
    pipeline.print_eval_report(eval_result=eval_result)

# @pytest.mark.parametrize("retriever_with_docs", ["embedding"], indirect=True)
# @pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
# def test_run_retrieval_pipeline(retriever_with_docs):
#     query = "Người sáng lập tập đoàn Microsoft qua đời ở tuổi bao nhiêu"
#     pipeline = SearchSummarizationPipeline(retriever=retriever_with_docs)
#     eval_result = pipeline.run(query=query, params={"Retriever": {"top_k": 5}})
#     print(pipeline.draw(f"test/{pipeline.__class__.__name__}.png"))
#     print(eval_result)
