import datasets
from functools import partial


class DiffusionLoader:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _load(self, task_name, split):
        dataset = datasets.load_dataset('lm1b', split=split)
        print(f'Example in {split} set:')
        print(dataset[0])
        dataset = dataset.map(partial(self.convert_to_features, tokenizer=self.tokenizer), batched=True, remove_columns='text')
        return dataset

    def my_load(self, task_name, splits):
        return [self._load(task_name, name) for name in splits]

    @staticmethod
    def convert_to_features(example_batch, tokenizer):
        input_encodings = tokenizer.batch_encode_plus(example_batch['text'], max_length=128, truncation=True, add_special_tokens=False)
        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
        }

        return encodings


if __name__ == "__main__":
    data = DiffusionLoader().my_load(task_name="lm1b", splits=["test"])

    print(data)

    print(data["input_ids"])
    print(data["attention_mask"])

