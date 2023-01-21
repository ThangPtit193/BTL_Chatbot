from comet.lib import file_util
import os
import random
import tqdm

random.seed(42)


def generate_eval_data_from_raw_data(
    max_sentence_per_intent: int = 100,
    raw_data_path: str = "data/raw/timi/v1.1.3/positives",
    output_path: str = "data/raw/timi/v1.1.3/eval",
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Get query data
    query_data = {}
    files = file_util.get_all_files_in_directory(raw_data_path)
    for file in tqdm.tqdm(files):
        positives = file_util.load_yaml(file)['positives']
        for positive in positives:
            intent, examples = positive['intent'], positive['examples']
            if intent not in query_data:
                query_data[intent] = []
            limited_examples = examples[:max_sentence_per_intent]
            random.shuffle(limited_examples)
            query_data[intent].extend(limited_examples)
    # Write to file
    file_util.write_json_beautifier(
        os.path.join(output_path, "query.json"), query_data
    )

    # Get corpus data
    corpus_data = {}
    for file in tqdm.tqdm(files):
        positives = file_util.load_yaml(file)['positives']
        for positive in positives:
            intent, examples = positive['intent'], positive['examples']
            if intent not in corpus_data:
                corpus_data[intent] = []
            corpus_data[intent].extend(examples)
    # Write to file
    file_util.write_json_beautifier(
        os.path.join(output_path, "corpus.json"), corpus_data
    )


if __name__ == "__main__":
    generate_eval_data_from_raw_data()
