<p align="center">
  <a href="https://gitlab.ftech.ai/nlp/va/knowledge-retrieval"><img src="./images/venus_banner.svg" alt="Venus Services"></a>
</p>

<p>

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

</p>



Venus Services is a service that enables you to upload document to DocumentStore, semantic document search use case and training
model (**coming soon**).
Whether you want to perform Question Answering or semantic document search, you can use the SOTA NLP models
in Venus hub to provide unique search experiences and allow your users to query in natural language.

## Core Features

- **Latest models**: Utilize all latest transformer-based models (e.g., BERT, RoBERTa, PhoBert) for extractive QA,
  generative QA, and document retrieval.
- **Open**: 100% compatible with HuggingFace's model hub and Venus hub *(internal use only)*. Tight interfaces to other
  frameworks (e.g., Transformers, FARM, sentence-transformers)
- **Scalable**: Scale to millions of docs via retrievers, production-ready backends like Elasticsearch / FAISS (coming
  soon), and a fastAPI REST API
- **End-to-End**: All tooling in one place: training, eval, inference, etc.

## 💾 Installation

**1. Installation**

The following command will install the latest version of Venus from the main branch.

```shell
git clone https://gitlab.ftech.ai/nlp/va/knowledge-retrieval
```

**2. Python Dependencies**

```shell
cd knowledge-retrieval
pip install - r requirements.txt
```

**3. Docker**

**4. Online services**

Note that VPN could be required to access this domain in private network.

👇 Please click link below

<a href="./product/download.html" target="_top">Venus Services</a>

## Document Store

You can think of the Document Store as a "database" that:

- stores your texts and meta data
- provides them to the Retriever at query time

To store your texts and their meta, you should follow the document format below:
