

"""
Count all the words in the dataset and store them in a dictionary
and then save it as a .pt file
"""
import os

from transformers import BertTokenizer
import torch
from dataloader import DiffusionLoader
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_data = DiffusionLoader().my_load(task_name='lm1b', splits=['test'])[0]

word_freq = torch.zeros((tokenizer.vocab_size,), dtype=torch.int64)

# count the frequencies
for data in tqdm(train_data):
    for iid in data['input_ids']:
        word_freq[iid] += 1

if not os.path.exists('./word_freq'):
    os.mkdir('word_freq')

torch.save(word_freq, f'./word_freq/bert-base-uncased_lm1b.pt')

