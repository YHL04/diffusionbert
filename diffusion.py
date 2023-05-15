

import torch
import torch.nn.functional as F
from torch.optim import Adam

import math


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02, device="cuda"):
    """
    linear schedule, proposed in original ddpm paper
    """
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

def mutual_beta_schedule(timesteps, device="cuda"):
    """
    mutual schedule for betas
    """
    steps = torch.arange(timesteps, device=device)
    return 1. / (timesteps - steps)


def cosine_beta_schedule(timesteps, s=0.008, device="cuda"):
    """
    cosine schedule for betas
    """
    steps = torch.arange(timesteps, device=device)
    return torch.cos((steps / timesteps + s) / (1 + s) * math.pi / 2)


def categorical_sample(probs):
    return torch.distributions.categorical.Categorical(probs=probs).sample()


class DiffusionTrainer:

    def __init__(self,
                 T,
                 tokenizer,
                 model,
                 vocab_size=30522,
                 maxlen=128,
                 word_freq_lambda=0.3,
                 device="cuda"):

        # general parameters
        self.T = T
        self.tokenizer = tokenizer
        self.model = model

        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.device = device

        # precomputed variables
        self.betas = mutual_beta_schedule(timesteps=T, device=device)
        self.alphas = 1. - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, dim=0)
        self.alphas_prod[-1] = 0.

        # diffusion bert variables
        self.mask_token_id = self.tokenizer.mask_token_id
        self.mask = F.one_hot(torch.tensor(self.mask_token_id, device=device), num_classes=vocab_size
                              ).unsqueeze(1).repeat(1, vocab_size).float()

        # word freq
        self.word_freq = torch.load(f'./word_freq/bert-base-uncased_lm1b.pt').to(torch.float32).cuda()
        # word_freq_lambda * sin wave from 0 to pi interval across time-steps, Tensor[T + 1]
        self.word_freq_lambda = word_freq_lambda * torch.sin(torch.arange(T + 1, device=self.device) / T * math.pi)

        # optimizer
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)

    def get_qt(self, x_0, t):
        """
        :param x_0: Tensor[batch_size, maxlen, vocab_size]
        :param t: int
        :param epsilon: float

        :return x_t: Tensor[batch_size, maxlen, vocab_size]

        - get token probabilities at time t
        - mask_prob gets higher with t and non mask prob gets lower with t
        """
        if t == 0: return x_0

        assert x_0.isnan().sum() == 0

        p = self.alphas_prod[t]

        # get frequency of each word as another tensor and normalize between -0.5 to 0.5
        # then multiply by word_freq_lambda of current timestep
        # word_freq_logits = self.word_freq.repeat(x_0.size(0), 1).gather(1, x_0.argmax(-1))
        # word_freq_logits = word_freq_logits - word_freq_logits.mean(-1, keepdims=True)
        # word_freq_probs = word_freq_logits.unsqueeze(-1) * self.word_freq_lambda[t]
        # print(word_freq_probs)

        # adding word_freq_probs to p makes infrequent word more likely and frequent word less likely
        # p has to be bigger than 0 to avoid nans

        # p = torch.clip(p + word_freq_probs, 0.001, 0.999)
        p = torch.clip(p, 0.001, 0.999)

        # Tensor[batch_size, maxlen, vocab_size]
        non_mask_prob = p * x_0
        print(non_mask_prob.shape)

        # mask_prob = 1. - sum(non_mask_prob) + mask_token_id
        mask_prob = 1. - non_mask_prob.sum(-1, keepdims=True) + non_mask_prob[..., self.mask_token_id].unsqueeze(-1)

        # concatenate ([:mask_token_id] + [mask_prob] + [mask_token_id + 1:]
        prob_t = torch.cat((
            non_mask_prob[..., :self.mask_token_id], mask_prob,
            non_mask_prob[..., self.mask_token_id + 1:]
        ), dim=-1)

        assert prob_t.isnan().sum() == 0
        prob_t = prob_t / prob_t.sum(-1, keepdims=True)
        assert prob_t.isnan().sum() == 0

        return prob_t

    def get_transition_probs(self, x_t1, t):
        """
        :param x_t1: one hot tensor of x_t1
        :param t:    int

        :return: Tensor[batch_size, maxlen, vocab_size]
                 q(x_{t+1} | x_t)
        """
        # get beta at time t
        beta = self.betas[t]

        qtpls1_at_mask = x_t1[..., self.mask_token_id: self.mask_token_id + 1]

        non_mask_prob0 = (1 - beta) * x_t1[..., :self.mask_token_id] + beta * qtpls1_at_mask
        non_mask_prob1 = (1 - beta) * x_t1[..., self.mask_token_id + 1:] + beta * qtpls1_at_mask

        prob_t = torch.cat((non_mask_prob0, qtpls1_at_mask, non_mask_prob1), dim=-1)
        return prob_t

    def forward_step(self, x_0, t, transition_probs=None):
        """
        :param x_0             : Tensor[batch_size, maxlen]
        :param t               : Tensor[]
        :param transition_probs: Tensor
        :param make_one_hot    : bool
        :param epsilon         : float

        :return: posterior       : x_{t+1}
                 samples         : q(x_t | x_{t+1})
                 transition_probs: q(x_{t+1} | x_t)
        """
        # prob_t and next_prob_t are probabilities of turning id into mask
        prob_t = self.get_qt(x_0, t)
        next_prob_t = self.get_qt(x_0, t+1)

        # go from probabilities to one hot
        samples = categorical_sample(next_prob_t)
        samples = F.one_hot(samples, self.vocab_size).reshape(samples.shape + (self.vocab_size,))

        # get q(x_{t+1} | x_t) from x_{t+1}
        if transition_probs is None:
            transition_probs = self.get_transition_probs(samples, t)

        posterior = prob_t * transition_probs
        # cannot divide by zero (causes nan)
        posterior = posterior / (posterior.sum(-1, keepdims=True) + 1e-10)

        assert posterior.isnan().sum() == 0
        return posterior, samples, transition_probs

    def backward_step(self, x_t1, t1, attention_mask, transition_probs=None):
        """
        :param x_t1: x_{t+1}
                     Tensor[batch_size, maxlen, vocab_size]
        :param t: int
        :param attention_mask:
        :param transition_probs:

        :return: Tensor[batch_size, maxlen, vocab_size]
                 probabilities for q(x_{t-1} | x_t)
        """

        # testing = categorical_sample(x_t[0].unsqueeze(0))
        # print(self.convert_to_text(testing))

        probs = self.model(categorical_sample(x_t1), t=t1, attention_mask=attention_mask)
        assert probs.isnan().sum() == 0

        probs, _, _ = self.forward_step(x_0=probs, t=t1-1, transition_probs=transition_probs)

        return probs

    def train_step(self, x_0, attention_mask):
        """
        :param x_0:            Tensor[batch_size, maxlen]
        :param attention_mask: Tensor[batch_size, maxlen]

        x_t = one hot of ids at x
        p_t = probabilities of ids at x

        """
        # get random timestep t
        t = torch.randint(0, self.T-1, size=(1,))

        # turn Tensor[batch_size, maxlen] into Tensor[batch_size, maxlen, vocab_size]
        x_0 = F.one_hot(x_0, self.vocab_size).reshape(x_0.shape + (self.vocab_size,))

        # x_{t+1}, q(x_t | x_{t+1}), q(x_{t+1} | x_t)
        q_t, x_t1, transition_probs = self.forward_step(x_0, t)

        # predicted x_t from x_{t+1}
        p_t = self.backward_step(x_t1, t+1, attention_mask, transition_probs)

        loss = self.get_loss(probs=p_t, targets=x_0)
        loss = loss.mean()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
        self.optimizer.step()

        return loss.item()

    def get_loss(self, probs, targets, epsilon=1e-20):
        """
        kl divergence with probs

        compute the kl between two categorical distributions from their probabilities
        """
        assert probs.isnan().sum() == 0
        assert targets.isnan().sum() == 0

        kl = (probs * (torch.log(probs + epsilon) - torch.log(targets + epsilon))).sum(-1)

        # KL divergence should be positive, this helps with numerical stability
        loss = F.relu(kl)

        return loss

    def convert_to_text(self, x):
        """
        :param x: Tensor[batch_size, maxlen]
        :return: List[]
        """
        texts = []
        for ids in x:
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            text = ' '.join(tokens)
            texts.append(text)

        return texts

    def predict_text(self, x_t, attention_mask):
        """
        :param x_t: Tensor[batch_size, maxlen]
        :param attention_mask: Tensor[batch_size, maxlen]
        :return: texts: List[batch_size, maxlen]
        """

        for t in range(1, self.T)[::-1]:
            x_t = self.backward_step(x_t, t, attention_mask)
            x_t = categorical_sample(x_t)
            x_t = F.one_hot(x_t, self.vocab_size).reshape(x_t.shape + (self.vocab_size,))

        assert x_t.isnan().sum() == 0

        x_t = categorical_sample(x_t)
        texts = self.convert_to_text(x_t)

        return texts

    def save(self):
        torch.save(self.model.state_dict(), "saved/model")


if __name__ == "__main__":
    from transformers import BertTokenizer, BertForMaskedLM
    from models import DiffusionBERT

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = DiffusionBERT()

    test = DiffusionTrainer(T=100,
                            tokenizer=tokenizer,
                            model=model,
                            maxlen=128
                            )
    # posterior, sample, transition_probs = test.forward_step(
    #     x_0=torch.randint(low=0, high=30522, size=(2, 512)).to(torch.int64).cuda(),
    #     t=10
    # )

    loss = test.train_step(x_0=torch.randint(low=0, high=30522, size=(2, 512)).cuda(),
                           attention_mask=torch.ones((2, 512)).cuda())

    # print(posterior)
    # print(sample)
    # print(transition_probs.shape)

