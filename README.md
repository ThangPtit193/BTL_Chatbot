<p align="center">
  <a href="https://gitlab.ftech.ai/nlp/va/knowledge-retrieval"><img src="./docs/images/venus_banner.svg" alt="Venus Services"></a>
</p>

<p>



</p>



Venus Services is a service that enables you to upload document to DocumentStore, semantic document search use case and training
model (**coming soon**).
Whether you want to perform Question Answering or semantic document search, you can use the SOTA NLP models
in Venus hub to provide unique search experiences and allow your users to query in natural language.

## Core Features

- **Latest models**: Utilize all latest transformer-based models (e.g., BERT, RoBERTa, MiniLM) for extractive QA,
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

```json
[
  {
    "text": "Tế bào là gì",
    "meta": {
      "answer": "utter_science_define {'trigger_slot': 'TE_BAO'}",
      "adjacency_pair": "ask_define/utter_science_define {'trigger_slot': 'TE_BAO'}",
      "domain": "science",
      "index": "fschool_science_index"
    }
  },
  {
    "text": "Cho mình hỏi định nghĩa tế bào là gì?",
    "meta": {
      "answer": "utter_science_define {'trigger_slot': 'TE_BAO'}",
      "adjacency_pair": "ask_define/utter_science_define {'trigger_slot': 'TE_BAO'}",
      "domain": "science",
      "index": "fschool_science_index"
    }
  },
  {
    "text": "Định nghĩa số phức là gì",
    "meta": {
      "answer": "utter_math_define {'trigger_slot': 'SO_PHUC'}",
      "adjacency_pair": "ask_define/utter_math_define {'trigger_slot': 'SO_PHUC'}",
      "domain": "math",
      "index": "fschool_math_index"
    }
  },
  {
    "text": "Hỏi về chiều cao của Timi",
    "meta": {
      "answer": "utter_greet",
      "adjacency_pair": "ask_general/greet",
      "domain": "general",
      "index": "general_index"
    }
  }
]
```

**<span style="color:red"> Note that</span>** index for each text should be based on standard format **
domain_sub-domain_index** that improves the retrieval performance.
