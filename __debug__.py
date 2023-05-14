

import torch.nn.functional as F

from transformers import BertTokenizer
from diffusion import DiffusionTrainer, categorical_sample
from models import DiffusionBERT
from dataloader import get_dataloader


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = DiffusionBERT()

test = DiffusionTrainer(T=300,
                        tokenizer=tokenizer,
                        model=model,
                        maxlen=128)

dataloader = get_dataloader(tokenizer=tokenizer,
                            batch_size=1)

batch = next(iter(dataloader))

input_ids = batch["input_ids"].cuda()
masks = batch["attention_mask"].cuda()

print(test.convert_to_text(input_ids))

x_prob = F.one_hot(input_ids, num_classes=30522)
prob_t = test.get_qt(x_0=x_prob, t=50)

# ans = x_prob * prob_t
# ans = ans / (ans.sum(-1, keepdims=True) + 1e-10)

text = test.convert_to_text(categorical_sample(prob_t))

print(text)

