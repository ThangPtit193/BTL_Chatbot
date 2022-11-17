import pytest

from meteor.document_stores import InMemoryDocumentStore
from meteor.nodes import EmbeddingRetriever
from meteor.pipelines.standard_pipelines import DocumentSearchPipeline, SearchSummarizationPipeline
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
                    id="something_else", content_type="text",
                    content="My name is Carla and I live in Munich"
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


sample_document_1 = [
    {
        "content": "Cảnh báo được gửi đến các tư lệnh Lục quân và Hải quân Hoa Kỳ tại Hawaii nhưng tin tức này không được nhận đúng lúc vì lỗi của bộ máy hành chính."
    },
    {
        "content": "Văn hóa Canada rút ra từ những ảnh hưởng của các dân tộc thành phần"
    },
    {
        "content": "Các tỉnh khác không có ngôn ngữ chính thức như vậy"
    },
    {
        "content": "Paris nằm ở điểm gặp nhau của các hành trình thương mại đường bộ và đường sông, và là trung tâm của một vùng nông nghiệp giàu có"
    },
    {
        "content": "Chultem phân biệt ba kiểu kiến trúc truyền thống Mông Cổ: Mông Cổ, Tây Tạng và Trung Quốc và kiểu kết hợp."
    },
    {
        "content": "'Người đồng sáng lập tập đoàn Microsoft Paul Allen qua đời ở tuổi 65'",
    },
    {
        "content": "Sinh ra tại Seattle, tiểu bang Washington, Paul Allen trở thành người bạn thân thiết từ thuở niên thiếu của Bill Gates. Chính ông là người đã thuyết phục Bill Gates bỏ Đại học Harvard để cùng nhau thành lập Tập đoàn Microsoft vào năm 1975.",
    },
    {
        "content": "Chính ông là người đã thuyết phục Bill Gates bỏ Đại học Harvard để cùng nhau thành lập Tập đoàn Microsoft vào năm 1975.",
    },
    {
        "content": "Paris nằm ở điểm gặp nhau của các hành trình thương mại đường bộ và đường sông, và là trung tâm của một vùng nông nghiệp giàu có"
    },
    {
        "content": "Chultem phân biệt ba kiểu kiến trúc truyền thống Mông Cổ: Mông Cổ, Tây Tạng và Trung Quốc và kiểu kết hợp."
    }

]


def test_run_retrieval_pipeline():
    retriever_model_name_or_path = 'ms-viquad-bi-encoder-phobert-base'
    document_store = InMemoryDocumentStore(index="document")
    document_store.delete_documents(index="document")
    document_store.write_documents(sample_document_1)
    print(document_store.get_all_documents(index="document"))

    retriever = EmbeddingRetriever(embedding_model=retriever_model_name_or_path, document_store=document_store)
    retriever.update_embeddings(index='document')

    pipeline = SearchSummarizationPipeline(retriever=retriever)

    question = "người sáng lập microsoft là ai?"
    o = pipeline.run(query=question, params={"Retriever": {"top_k": 5}})
    print(o)

