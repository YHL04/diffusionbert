

import torch
import torch.nn.functional as F

from diffusion import DiffusionTrainer
from models import DiffusionBERT
from transformers import BertTokenizer

T = 100
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = DiffusionBERT()
maxlen = 128

test = DiffusionTrainer(T=T,
                        tokenizer=tokenizer,
                        model=model,
                        maxlen=maxlen)

print("mask_token_ids ", tokenizer.mask_token_id)
print("pad_token_ids ", tokenizer.pad_token_id)
x_t = F.one_hot(torch.tensor(tokenizer.mask_token_id), num_classes=30522).repeat(128, 1).unsqueeze(0).cuda()
attention_mask = torch.ones((1, 128)).cuda()

print(x_t.shape)
print(attention_mask.shape)

result = test.predict_text(x_t=x_t, attention_mask=attention_mask)

print(result)
