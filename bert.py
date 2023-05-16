

import torch
import torch.nn as nn

from models.modeling_bert import BertForMaskedLM


class DiffusionBERT(nn.Module):

    def __init__(self, vocab_size):
        super(DiffusionBERT, self).__init__()
        self.vocab_size = vocab_size

        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased', num_labels=vocab_size).cuda()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, t, attention_mask):
        """
        :param x_t:            Tensor[batch_size, maxlen, vocab_size]
        :param t:              Tensor[]
        :param attention_mask: Tensor[batch_size, maxlen, vocab_size]
        :return:
        """
        assert x.isnan().sum() == 0, x
        # assert x.sum() != 0, t
        # print("input ", x)

        x = self.bert.forward(x)["logits"]
        # x = self.bert.forward(x, attention_mask=attention_mask)["logits"]
        # print("vocab size ", self.vocab_size)
        # print(x.shape)
        # print("output ", x)
        assert x.isnan().sum() == 0, x

        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DiffusionBERT()

    x = torch.randint(low=0, high=30522, size=(3, 4, 5))

    print(model(x, t=0., target_mask=0.))
