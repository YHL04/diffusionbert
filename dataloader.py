

import torch
from torch.nn.utils.rnn import pad_sequence

import datasets
from functools import partial


class DiffusionLoader:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _load(self, task_name, split):
        dataset = datasets.load_dataset('lm1b', split=split)
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


def word_freq_preprocess(wf):
    wf = wf + 1
    wf = wf.log()
    wf = wf / wf.max()

    # range: 0 - 1
    return wf


def get_word_freq():
    word_freq = torch.load(f'./word_freq/bert-base-uncased_lm1b.pt')
    word_freq = word_freq_preprocess(word_freq)

    return word_freq


def process_fn_in_collate(wf):
    return wf - wf.mean()


def collate_fn(batch_input):
    word_freq = get_word_freq()

    input_ids = [torch.tensor(d['input_ids']) for d in batch_input]
    attention_mask = [torch.tensor(d['attention_mask']) for d in batch_input]
    word_freq_logits = [process_fn_in_collate(word_freq.gather(0, torch.tensor(d['input_ids']))) for d in batch_input]
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    word_freq_logits = pad_sequence(word_freq_logits, batch_first=True)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'word_freq_logits': word_freq_logits
    }


def get_dataloader(tokenizer, batch_size):
    data = DiffusionLoader(tokenizer=tokenizer).my_load(task_name="lm1b", splits=["test"])[0]
    dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=batch_size,
                                             collate_fn=collate_fn
                                             )
    return dataloader


if __name__ == "__main__":
    # distributed sampler used for dataset in multiple gpus
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    data = DiffusionLoader(tokenizer=tokenizer).my_load(task_name="lm1b", splits=["test"])[0]

    print(data[0])


