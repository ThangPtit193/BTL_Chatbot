import random

from comet.lib import file_util, logger

_logger = logger.get_logger(__name__)

PRJ_DIR = "transformer_applications/sentences_similarity"


def data_producer(queue, filepaths, dataset_indices, config):
    datasets = []
    for filepath in filepaths:
        data_obj = Dataset(filepath)
        datasets.append(iter(data_obj))

        # Store if dataset is in a 2 col or 3 col format
    num_cols = {idx: len(next(dataset)) for idx, dataset in enumerate(datasets)}
    while True:
        texts_in_batch = set()
        batch_format = None  # 2 vs 3 col format for this batch
        # Add data from several sub datasets
        for _ in range(config['datasets_per_batch']):
            is_valid_dataset = False
            while not is_valid_dataset:
                data_idx = random.choice(dataset_indices)
                if batch_format is None:
                    batch_format = num_cols[data_idx]
                    is_valid_dataset = True
                else:
                    # Check that this dataset has the same format
                    is_valid_dataset = (batch_format == num_cols[data_idx])
            # Get data from this dataset
            dataset = datasets[data_idx]
            for _ in range(config['num_same_dataset']):
                batch_device = []  # A batch for one device
                while len(batch_device) < config['batch_size']:
                    sample = next(dataset)
                    # if not in_batch:
                    for text in sample:
                        texts_in_batch.add(text)
                    batch_device.append(sample)
                queue.put(batch_device)


class Dataset:
    """
    A class that handles one dataset
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        max_dataset_size = 10 * 1000 * 1000  # Cache small datasets in memory
        dataset = []
        while dataset is None or len(dataset) == 0:
            triplets_data = file_util.load_json(self.filepath)['data']
            _logger.info("Loaded %d triplets from %s", len(triplets_data), self.filepath)

            for triplet_data in triplets_data:
                data = [triplet_data['query'], triplet_data['pos'], triplet_data['neg']]
                if dataset is not None:
                    dataset.append(data)
                    if len(dataset) >= max_dataset_size:
                        dataset = None
                yield data

        # Data loaded. Now stream to the queue
        # Shuffle for each epoch
        while True:
            random.shuffle(dataset)
            for data in dataset:
                yield data


if __name__ == "__main__":
    dataset = Dataset("data/triples/triples_10.json")
    for data in dataset:
        print(data)
