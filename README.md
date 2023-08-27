## Semantic - Similarity

## How to Reproduce

First, make sure your Python version is 3.7+, and then install the required packages using the command below:

```script
cd saturn

export PYTHONPATH=/absolute/saturn
```

#### PreStage: Training Data Development

```script
# download processed data for pretraining
```

Format data
```
{"query": "abc", "document": "xyz"}
{"query": "abc", "document": "xyz"}
```

and save it as `data.jsonl` filname

Structure data folders:
```
├── data                   # Dataset
    ├── train              # Training folder
        ├── data.jsonl     # Training dataset
    ├── eval               # Eval folder
        ├── data.jsonl     # Evaluate dataset

```
#### Stage 1: Training Unsuppervised Model Using Contrastive Learning


```
bash scripts/train_biencoder.sh
```


#### Stage 2: Future Works
##### 1. Data Sampling

Positive (Self-Supervised/Unsupervised)
- [x] Inverse Cloze Task
- [x] Dropout as Positive Instance
- [ ] Text Augmentation
- [ ] Recurring Span Retrieval
- [ ] Others (TBD)


Negative

- [x] In-Batch Negative
- [ ] Hard Negative
- [ ] Cross Batch Negative


##### 2. Auxilaury Task
- [x] Alignment Task
- [x] Uniformity Vector Distribution
- [ ] Masked Language Modeling
- [ ] SPLADE
