import os
import re
from home.models import Chromadb
from django.shortcuts import render
from django.views import View
from django.http import JsonResponse

model = Chromadb()
model.add_document()


# Create your models here.
class IndexView(View):
    template_name = 'index.html'

    def __init__(self, **kwargs):
        super(IndexView).__init__(**kwargs)
        self.model = model

    def get(self, request):
        context = {
            'image_url': os.path.join('/static', 'ptit.png')
        }
        return render(request, self.template_name, context=context)

    def post(self, request):
        desc = request.POST['desciption']
        chunk = re.split(r'(?<=[.!?])\s+', desc)
        answer = ""
        if len(chunk) == 1:
            answer = self.model.search(desc)
        else:
            for id, sentence in enumerate(chunk):
                result = self.model.search(sentence)
                result = f"Câu thứ {id + 1}: " + result + '\n'
                answer += result

        data = {
            'vi_text': desc,
            'answer': answer
        }
        return JsonResponse(data)
