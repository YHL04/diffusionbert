

import torch
import torch.nn as nn

from transformers import BertForMaskedLM


class DiffusionBERT(nn.Module):

    def __init__(self):
        super(DiffusionBERT, self).__init__()

        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased').cuda()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, t, attention_mask):
        """
        :param x_t:            Tensor[batch_size, maxlen, vocab_size]
        :param t:              Tensor[]
        :param attention_mask: Tensor[batch_size, maxlen, vocab_size]
        :return:
        """
        x = self.bert.forward(x, attention_mask=attention_mask)
        x = self.softmax(x["logits"])

        return x


if __name__ == "__main__":
    model = DiffusionBERT()

    x = torch.randint(low=0, high=30522, size=(3, 4, 5))

    print(model(x, t=0., target_mask=0.))
