import os
import re
from home.models import get_answer
from django.shortcuts import render
from django.views import View
from django.http import JsonResponse
from transformers import AutoModel, AutoTokenizer

from saturn.components.loaders.utils import convert_text_to_features
from saturn.utils.utils import preprocessing
from saturn.utils.normalize import normalize_encode, normalize_word_diacritic


# Create your models here.
class IndexView(View):
    template_name = 'index.html'

    def __init__(self, **kwargs):
        super(IndexView).__init__(**kwargs)
        self.model = AutoModel.from_pretrained("models")
        self.tokenizer = AutoTokenizer.from_pretrained("models")

    def get(self, request):
        context = {
            'image_url': os.path.join('/static', 'ptit.png')
        }
        return render(request, self.template_name, context=context)

    def post(self, request):
        desc = request.POST['desciption']
        
        input_ids, attention_mask = convert_text_to_features(
            text=normalize_encode(normalize_word_diacritic(preprocessing(desc))),
            tokenizer=self.tokenizer,
            max_seq_len=64
        )
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        answer = get_answer(outputs, self.tokenizer)
        
        data = {
            'vi_text': desc,
            'answer': answer
        }
        return JsonResponse(data)
