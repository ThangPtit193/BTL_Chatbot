## Semantic - Similarity

## How to Reproduce

First, make sure your Python version is 3.7+, and then install the required packages using the command below:

```script
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
#### Training Unsuppervised Model Using Contrastive Learning


```
bash scripts/train_biencoder.sh
```
