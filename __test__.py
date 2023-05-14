

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

x_t = F.one_hot(torch.tensor(102), num_classes=30522).repeat(128, 1).unsqueeze(0)
target_mask = torch.ones((1, 128))

print(x_t.shape)
print(target_mask.shape)

result = test.predict_text(x_t=x_t, target_mask=target_mask)

print(result)
