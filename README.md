<p align="center">
  <a href="https://gitlab.ftech.ai/nlp/va/knowledge-retrieval"><img src="./images/meteor.svg" alt="Meteor Services"></a>
</p>

<p>

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models%20on%20Hub-yellow)](https://huggingface.co/models?filter=keytotext)
[![AxiomHub](https://img.shields.io/badge/Axiom-Axiom%20Hub-blue)](https://axiom.dev.ftech.ai/ui/home)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-ff69b4)(https://gitlab.ftech.ai/nlp/va/knowledge-retrieval)]
</p>



Meteor Services is a service that enables you to upload document to DocumentStore, semantic document search use case and training
model (**coming soon**).
Whether you want to perform Question Answering or semantic document search, you can use the SOTA NLP models
in Venus hub to provide unique search experiences and allow your users to query in natural language.

## Core Features

- [x] **Latest models**: Utilize all latest transformer-based models (e.g., BERT, RoBERTa, PhoBert) for extractive QA,
  generative QA, and document retrieval.
- [x] **Open**: 100% compatible with HuggingFace's model hub and Venus hub *(internal use only)*. Tight interfaces to other
  frameworks (e.g., Transformers, FARM, sentence-transformers)
- [x] **Scalable**: Scale to millions of docs via retrievers, production-ready backends like Elasticsearch / FAISS (coming
  soon)
- [x] **End-to-End**: All tooling in one place: training, eval, inference, etc.
- [ ] **RestAPI**: Coming soon ...
- [ ] **Streamlit**: Coming soon ...

## 💾 Installation


The following command will install the latest version of Meteor from the main branch.

```shell
pip install .
```

## Requirements
### Using pip

```shell
python3 -m venv ./venv 
. ./venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

The minimum required packages to train and run the baseline ML models are listed in [requirements.txt](requirements.txt).

For GPU support there are additional platform-specific instructions:

For PyTorch, [see here](https://pytorch.org/get-started/locally/).

### Conda
If you like you can also install dependencies using anaconda. We suggest using miniforge (and possibly mamba) as distribution. Otherwise you may have to enable the conda-forge channel for the following commands.

Starting from a fresh environment:

```shell
conda create -n meteor python==3.7.9
conda activate meteor
pip install .

# Test version 
meteor version
```

To support for most current servers, we highly recommend install pytorch in stable version if using GPU (**Optional**)
```commandline

pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

## Configuration

To run benchmark for your dataset and model, you config it at [config file](config/config.yaml) or pass custom config path
```yaml
GENERAL:
  run_mode: inference

EMBEDDER:
  class: NaiveEmbedder
  package: meteor.components.embeddings.embedding_models
  TRAINER:
    # A data_config.json file
    data_config_path: data/data_config.json
    # The pretrained model
    #    pretrained_model: sentence-transformers/quora-distilbert-multilingual
    pretrained_model: vinai/phobert-base
    # Total of steps, it depends on the number of question answering pair that you have
    steps: 1000
    # When reach this step, we will be saved checkpoint
    save_steps: 10000
    # The max length of tokenizer
    max_length: 128
    # Use 20 for cossim, and 1 when you work with un-normalized embeddings with dot product
    scale: 20
    batch_size: 32
    model_save_path: models
    warmup_steps: 50
    datasets_per_batch: 2
    num_same_dataset: 1
    evaluation_steps: 200
    weight_decay: 0.01
    max_grad_norm: 1.0
    use_amp: False
    save_best_model: True
    show_progress_bar: True

EVALUATION:
  # Corpus name or path, meteor will load your corpus dataset from local or axiom hub (coming soon)
  corpus_name_or_path: "assets/corpus_company.json"
  # Corpus name or path, meteor will load your query dataset from local or axiom hub ((coming soon))
  query_name_or_path: "assets/query_company.json"
  model_name_or_path: [ "distilbert-multilingual-faq-v3.2" , "khanhpd2/sbert_phobert_large_cosine_sim", "timi-idol-paraphrase-multilingual-MiniLM-L12-v2-v.1.0.1"]


```

## Data Preparation
We provide you with two approaches to prepare dataset for training, evaluating and inferring models. However, you should follow my data format to avoid any errors while executing.

<span style="color:red">**Corpus and Query** Datasets</span> should be prepared before running services.
```yaml
{
  "science.ask_application.VIRUS": [
    "Cho biết  1 số ưng dụng of vi rút"
  ],
  "science.ask_role.DI_TRUYEN_LIEN_KET": [
    "ad cho t hỏi về ý nghĩa của di truyền liên kết",
    "ad cho tôi hỏi về v/trò của di truyen lieen kết"
  ],
  "chemistry.ask_nature_state.SILIC": [
    "Silic có trang thái tn ra sao",
    "Silic có trang thái tu nhien như răng"
  ]
}

```
Noted that query and corpus have the same keys.


## Local Run
You should export python path to run this service in local. To see why we export it [here](https://www.simplilearn.com/tutorials/python-tutorial/python-path)
```commandline
export PYTHONPATH=./
```

If configuration path is none, the default configuration will be loaded. Use `--c` or `-config` if you use your configuration path.

```shell
meteor test 
```
The benchmark results will be saved at [./reports](reports) folder.
## Services

(Coming soon)


## Information Retriever Metrics
There are three metrics for evaluating information retriever models, [see here](https://docs.google.com/document/d/1bTPGMUd4q0591bIRb9g28X1qxcc3B5i0WdQPvegZINE/edit)

## Reference
[Evaluation Metrics For Information Retrieval](https://amitness.com/2020/08/information-retrieval-evaluation/), Amit Chaudhary, 2020


## Code Contributors
[Hao Nguyen Van](https://gitlab.ftech.ai/haonv)

[Phong Nguyen Thanh](https://gitlab.ftech.ai/phongnt)

[Tam Pham The](https://gitlab.ftech.ai/tampt)

[Hieu Le Van](https://gitlab.ftech.ai/leanhhieu231)










