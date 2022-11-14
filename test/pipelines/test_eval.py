from meteor.schema import MultiLabel, Label, Answer, Span, Document

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
