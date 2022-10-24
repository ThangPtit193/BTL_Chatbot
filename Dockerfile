FROM quoinedev/python3.7-pandas-alpine
WORKDIR /knowledge-retrieval

RUN python3 -m pip install -U pip
RUN python3 -m pip install -U setuptools
RUN pip install torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install requirements
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY . /knowledge-retrieval
ENV PYTHONPATH "${PYTHONPATH}:/knowledge-retrieval"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8891"]