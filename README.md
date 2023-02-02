<p align="center">
  <a href="https://gitlab.ftech.ai/nlp/va/knowledge-retrieval"><img src="./images/saturn_readme.svg" alt="Saturn Services"></a>
</p>

<p>

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models%20on%20Hub-yellow)](https://huggingface.co/models?filter=keytotext)
[![AxiomHub](https://img.shields.io/badge/Axiom-Axiom%20Hub-blue)](https://axiom.dev.ftech.ai/ui/home)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-ff69b4)](http://192.168.1.11:8501/)
</p>


Saturn is a service that enables you to upload document to DocumentStore, semantic document search use case and
training
model (**coming soon**).
Whether you want to perform Question Answering or semantic document search, you can use the SOTA NLP models
in Venus hub to provide unique search experiences and allow your users to query in natural language.

# **Contents**

- [**Contents**](#contents)
  - [Core Features ](#core-features-)
  - [Installation ](#-installation-)
    - [Install from repository](#install-from-repository)
    - [Conda](#conda)
  - [Configuration ](#configuration-)
  - [Data Preparation ](#data-preparation-)
  - [Services ](#services-)
  - [Information Retrieval Metrics ](#information-retrieval-metrics-)
  - [Command Line Interface ](#command-line-interface-)
      - [Training](#training)
      - [Evaluation](#evaluation)
      - [Inference](#inference)
      - [Push and pull model to/from Axiom Hub](#push-and-pull-model-tofrom-axiom-hub)
  - [Experiments ](#experiments-)
  - [Reference ](#reference-)
  - [Code Contributors ](#code-contributors-)


## Core Features <div id="core-features"></div>

- [x] **Latest models**: Utilize all latest transformer-based models (e.g., BERT, RoBERTa, PhoBert) for extractive QA,
  generative QA, and document retrieval.
- [x] **Open**: 100% compatible with HuggingFace's model hub and Venus hub *(internal use only)*. Tight interfaces to
  other
  frameworks (e.g., Transformers, FARM, sentence-transformers)
- [x] **Scalable**: Scale to millions of docs via retrievers, production-ready backends like Elasticsearch / FAISS (
  coming
  soon)
- [x] **End-to-End**: All tooling in one place: training, eval, inference, etc.
- [ ] **RestAPI**: Coming soon ...
- [x] **Streamlit**: [https://saturn.dev.ftech.ai/home](http://192.168.1.11:8501/)

## Installation <div id="core-features"></div>

The following command will install the latest version of Saturn from the main branch.
You can install a basic version of Saturn's latest release by using [pip](https://github.com/pypa/pip).

```shell
pip install <update later>
```

### Install from repository

```shell
python3 -m venv ./venv 
. ./venv/bin/activate
pip install --upgrade pip wheel
pip install .
```

The minimum required packages to train and run the baseline ML models are listed in [requirements.txt](requirements.txt)
.

For GPU support there are additional platform-specific instructions:

For PyTorch, [see here](https://pytorch.org/get-started/locally/).

### Conda

If you like you can also install dependencies using anaconda. We suggest using miniforge (and possibly mamba) as
distribution. Otherwise you may have to enable the conda-forge channel for the following commands.

Starting from a fresh environment:

```shell
conda create -n saturn python==3.7.9
conda activate saturn
pip install .

# Test version 
saturn version
```

To support for most current servers, we highly recommend install pytorch in stable version if using GPU (**Optional**)

```commandline

pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

## Configuration <div id="configuration"></div>

To run benchmark for your dataset and model, you config it at [config file](config/config.yaml) or pass custom config
path

Sample configuration
```yaml
GENERAL:
  device: cuda
  project: dummy
  version: v1.0.0
  is_warning_action: False
  output_data: assets
  output_model: models
  output_report: reports

  # Skipped signal
  skipped_gen_data: True
  skipped_training: True
  skipped_eval: False

DATA_GENERATION:
  data_dir: data/raw/dummy
  search_mode: bm25_embedder
  embedding_batch_size: 100
  neg_sim_batch_size: 10
  max_triple_per_file: 50
  max_sentence_repeated: 2
  EMBEDDER:
    cache_path: ./embedder_caches
    pretrained_name_or_abspath: timi-idol-keepitreal-vn-sbert-faq-9M-v1.0.0
#  COMBINER:


TRAINER:
  class: SBertSemanticSimilarity
  package: saturn.components.embeddings.embedding_models
  pretrained_name_or_abspath: keepitreal/vietnamese-sbert
  n_samples: 10
  batch_size: 256
  epochs: 2
  warmup_steps: 5000
  evaluation_steps: 2000
  weight_decay: 0.01
  max_grad_norm: 1.0
  use_amp: True
  save_best_model: True
  show_progress_bar: True
  checkpoint_save_epoch: 4
  checkpoint_save_total_limit: 10
  save_by_epoch: 4
#    resume_from_checkpoint: checkpoints/epoch-14/checkpoint.pt

EVALUATION:
  type: "evaluation_pipeline"
  corpus_name_or_path: "data/eval-data/dummy/corpus_docs.json"
  query_name_or_path: "data/eval-data/dummy/query_docs.json"
  pretrained_name_or_abspath:
    - timi-keepitreal-H768-faq-2M-v1.1.3
    - timi-idol-keepitreal-vn-sbert-faq-9M-v1.0.0
  top_k: 10
  retriever_threshold: 0.7
  default_faq_label: 'faq/out_of_scope'

RELEASE:
  model_path: models
  pretrained_model: MiniLM-L12-H384-uncased
  data_size: 9M
```

Sample configuration for quadruplet-loss
```yaml
...

TRAINER:
  class: QuadrupletSemanticSimilarity
  package: saturn.components.embeddings.embedding_models
  pretrained_name_or_abspath: keepitreal/vietnamese-sbert
  n_samples: 10
  batch_size: 256
  ...
```

## Data Preparation <div id="data-preparation"></div>

We provide you with two approaches to prepare dataset for training, evaluating and inferring models. However, you should
follow my data format to avoid any errors while executing.

<span style="color:green">Corpus and Query Datasets</span> should be prepared before running services.

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

## Services <div id="services"></div>

Currently, we provide with you some services such as evaluation, inference.

## Information Retrieval Metrics <div id="ir-metrics"></div>

There are three metrics for evaluating information retrieval
models, Learn more: [IR metrics](https://docs.google.com/document/d/1bTPGMUd4q0591bIRb9g28X1qxcc3B5i0WdQPvegZINE/edit)

## Command Line Interface <div id="cli"></div>

#### Run end to end

To run e2e:

```shell
saturn run-e2e -c <config/config.yaml>
```

#### Training only

```shell
saturn train -c <config/config.yaml>
```

#### Evaluation only

To evaluate a model, run:

```shell
saturn test -c <config/config.yaml>
```

Options

```commandline  
-c, --config    TEXT  path to config  [required]
-rt, --rtype    TEXT  supported report type
-k, --top_k     int   k retriveal docs
-md, --save_md  bool  save report with markdown file
```

#### Inference

To run the UI, run the following from the root directory of the Saturn repo

```shell
saturn ui
```

Options

```commandline
-p, --path TEXT path to streamlit file [optional]
```

#### Push and pull model to/from Axiom Hub

To push your model to Axiom Hub, you should have an account and login to Axiom Hub. Then, you can push your model to
Axiom Hub by using the following command.

```console
Options:
  -p, --model_path TEXT  path to folder  [required]
  -n, --name TEXT        The name of the model  [required]
  -rf, --replace         Replace existed data/model
  --help                 Show this message and exit.
```

```shell
saturn push --model_path <model_path> --name <name> --replace <replace>
```

To pull your model from Axiom Hub, you can use the following command.

```console
Usage: saturn pull [OPTIONS]

Options:
  -n, --name TEXT          The name of the model to pull  [required]
  -dp, --output_path TEXT  Where to save the model  [required]
  -rf, --replace           Replace existed data/model
  --help                   Show this message and exit.
```

```shell
saturn pull --name <name> --output_path <output_path> --replace <replace>
```

To list all models in Axiom Hub, you can use the following command.

```shell
saturn ls
```

## Experiments <div id="experiments"></div>

| Model name / Data name                                        | Timi-eval-data-v1.0.0      | Timi-eval-data-v1.2.0   | Timi-eval-data-v1.3.0   |
|:--------------------------------------------------------------|:---------------------------|:------------------------|:------------------------|
| distilbert-multilingual-faq-v3.2                              | 0.252mAP 0.688mRR          | 0.293mAP 0.540mRR       | 0.29mAP 0.54mRR         |
| timi-idol-microsoft-MiniLM-L12-H384-uncased-faq-9M-v1.0.0     | **0.733**mAP **0.793**mRR  | 0.653mAP 0.751mRR       | 0.65mAP 0.75mRR         |
| timi-idol-microsoft-MiniLM-L12-H384-uncased-faq-9M-v1.1.0     | 0.6686mAP 0.783mRR         | 0.749mAP 0.770mRR       | 0.75mAP 0.77mRR         |
| timi-idol-paraphrase-multilingual-MiniLM-L12-v2-faq-8M-v1.0.1 |                            | 0.78mAP 0.79mRR         | 0.78mAP 0.79mRR         |
| timi-idol-keepitreal-vn-sbert-faq-9M-v1.0.0                   |                            | **0.82**mAP **0.83**mRR | **0.82**mAP **0.83**mRR |

**Note**

1. distilbert-multilingual-faq-v3.2

- No information retriever

2. timi-idol-microsoft-MiniLM-L12-H384-uncased-faq-9M-v1.0.0

```text
- Pretrained model: [microsoft/MiniLM-L12-H384-uncased](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased)
- Training data: 9M FAQ data from timi idol (ID 464, version: v1.0.0)
- Method: Naive fine-tuning
- Steps: 600000
- Batch size: 128
```

3. timi-idol-microsoft-MiniLM-L12-H384-uncased-faq-9M-v1.2.0

```text
- Pretrained model: [microsoft/MiniLM-L12-H384-uncased](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased)
- Training data: 9M FAQ data from timi idol (ID 464, version: v1.1.0)
- Method: Naive fine-tuning
- Steps: 500000
- Batch size: 128
```

4. timi-idol-timi-idol-microsoft-MiniLM-L12-H384-uncased-faq-9M-v1.1.0-faq-9M-15-v1.0.0

```text
- Pretrained model: [timi-idol-microsoft-MiniLM-L12-H384-uncased-faq-9M-v1.1.0](http://minio.dev.ftech.ai/venus-model-v0.1-ca24fe0d/timi-idol-microsoft-MiniLM-L12-H384-uncased-faq-9M-v1.1.0.zip)
- Training data: 9M FAQ data from timi idol (ID 464, version: v1.3.0)
- Method: SentenceBert fine-tuning
- Epochs: 15
- Batch size: 256
- use_amp: True
```

5. timi-idol-keepitreal-vn-sbert-faq-9M-v1.0.0

```text
- Pretrained model: [keepitreal/vietnamese-sbert](https://huggingface.co/keepitreal/vietnamese-sbert)
- Training data: 9M FAQ data from timi idol (ID 464, version: v1.3.0)
- Method: SentenceBert fine-tuning
- Epochs: 15
- Batch size: 256
- use_amp: True
```

## Reference <div id="reference"></div>

[Evaluation Metrics For Information Retrieval](https://amitness.com/2020/08/information-retrieval-evaluation/), Amit
Chaudhary, 2020

## Code Contributors <div id="contributors"></div>

[Hao Nguyen Van](https://gitlab.ftech.ai/haonv)

[Phong Nguyen Thanh](https://gitlab.ftech.ai/phongnt)

[Tam Pham The](https://gitlab.ftech.ai/tampt)

[Hieu Le Van](https://gitlab.ftech.ai/leanhhieu231)










