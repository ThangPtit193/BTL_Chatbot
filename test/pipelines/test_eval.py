import pytest

from meteor.pipelines.standard_pipelines import DocumentSearchPipeline
from meteor.schema import MultiLabel, Label, Answer, Span, Document, EvaluationResult

EVAL_LABELS = [
    MultiLabel(
        labels=[
            Label(
                query="Who lives in Berlin?",
                answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]),
                document=Document(
                    id="a0747b83aea0b60c4b114b15476dd32d",
                    content_type="text",
                    content="My name is Carla and I live in Berlin",
                ),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
            )
        ]
    ),
    MultiLabel(
        labels=[
            Label(
                query="Who lives in Munich?",
                answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]),
                document=Document(
                    id="something_else", content_type="text", content="My name is Carla and I live in Munich"
                ),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
            )
        ]
    ),
]


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_document_search_calculate_metrics(retriever_with_docs):
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(labels=EVAL_LABELS, params={"Retriever": {"top_k": 5}})

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    assert "Retriever" in eval_result
    assert len(eval_result) == 1
    retriever_result = eval_result["Retriever"]
    retriever_berlin = retriever_result[retriever_result["query"] == "Who lives in Berlin?"]
    retriever_munich = retriever_result[retriever_result["query"] == "Who lives in Munich?"]
    print(retriever_berlin)
