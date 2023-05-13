

import torch
import torch.nn.functional as F
from torch.optim import Adam

import math
import random

from pytorch_pretrained_bert import BertForMaskedLM


def mutual_beta_schedule(timesteps):
    """
    linear schedule for betas
    """
    steps = torch.arange(timesteps)
    return 1. / (timesteps - steps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule for betas
    """
    steps = torch.arange(timesteps)
    return torch.cos((steps / timesteps + s) / (1 + s) * math.pi / 2)


def categorical_sample(logits):
    return torch.distributions.categorical.Categorical(logits=logits).sample()


class DiffusionBERT:

    def __init__(self,
                 T,
                 tokenizer,
                 vocab_size=30522,
                 maxlen=512,
                 mask_token_id=102,
                 word_freq_lambda=0.3,
                 device="cuda"):

        # general parameters
        self.T = T
        self.tokenizer = tokenizer

        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.device = device

        # precomputed variables
        self.betas = cosine_beta_schedule(timesteps=T)
        self.alphas = 1. - self.betas

        # equivalent to self.state[t] from original authors
        self.alphas_prod = torch.cumprod(self.alphas, dim=0)

        # diffusion bert variables
        self.mask_token_id = mask_token_id
        self.mask = F.one_hot(torch.tensor(mask_token_id, device=device), num_classes=vocab_size
                              ).unsqueeze(1).repeat(1, vocab_size).float()

        # word freq
        self.word_freq = torch.load(f'./word_freq/bert-base-uncased_lm1b.pt').to(torch.float32).cuda()
        # word_freq_lambda * sin wave from 0 to pi interval across time-steps
        # Tensor[T + 1]
        self.word_freq_lambda = word_freq_lambda * torch.sin(torch.arange(T + 1, device=self.device) / T * math.pi)

        # model
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)

    def noise_fn(self, x_0, t, word_freq_logits=None):
        """
        :param x_0:              Tensor[batch_size, maxlen]
        :param t:                int
        :param word_freq_logits: Tensor[batch_size, maxlen] (normalized frequency of each word [-0.5 to 0.5])

        :return:    Tensor[batch_size, maxlen]

        non_mask_prob = (alpha_prod + word_freq_probs) * x_0
        mask_prob = 1. - sum(non_mask_prob) + mask_token_id

        """
        p = self.alphas_prod[t]

        # get frequency of each word as another tensor and normalize between -0.5 to 0.5
        # then multiply by word_freq_lambda of current timestep
        if word_freq_logits is None:
            word_freq_logits = self.word_freq.repeat(x_0.size(0), 1).gather(1, x_0)
            word_freq_logits = word_freq_logits - word_freq_logits.mean(-1, keepdims=True)

        word_freq_probs = word_freq_logits * self.word_freq_lambda[t]

        p = torch.clip(p + word_freq_probs, 0., 0.999)

        # Tensor[batch_size, maxlen]
        non_mask_prob = p * x_0

        # mask_prob = 1. - sum(non_mask_prob) + mask_token_id
        mask_prob = 1. - non_mask_prob.sum(-1, keepdims=True) + non_mask_prob[..., self.mask_token_id].unsqueeze(-1)

        # concatenate ([:mask_token_id] + [mask_prob] + [mask_token_id + 1:]
        prob_t = torch.cat((
            non_mask_prob[..., :self.mask_token_id], mask_prob,
            non_mask_prob[..., self.mask_token_id + 1:]
        ), dim=-1)

        return prob_t

    def get_qt(self, x_0, t, epsilon=1e-20):
        """
        :param x_0
        :param t
        :param epsilon

        get qt given q0
        and get log probabilities
        """
        if t == 0:
            return x_0

        p = self.alphas_prod[t]

        # get frequency of each word as another tensor and normalize between -0.5 to 0.5
        # then multiply by word_freq_lambda of current timestep
        word_freq_logits = self.word_freq.repeat(x_0.size(0), 1).gather(1, x_0.argmax(-1))
        word_freq_logits = word_freq_logits - word_freq_logits.mean(-1, keepdims=True)
        word_freq_probs = word_freq_logits.unsqueeze(-1) * self.word_freq_lambda[t]

        p = torch.clip(p + word_freq_probs, 0., 0.999)

        # Tensor[batch_size, maxlen, vocab_size]
        non_mask_prob = p * x_0

        # mask_prob = 1. - sum(non_mask_prob) + mask_token_id
        mask_prob = 1. - non_mask_prob.sum(-1, keepdims=True) + non_mask_prob[..., self.mask_token_id].unsqueeze(-1)

        # concatenate ([:mask_token_id] + [mask_prob] + [mask_token_id + 1:]
        prob_t = torch.cat((
            non_mask_prob[..., :self.mask_token_id], mask_prob,
            non_mask_prob[..., self.mask_token_id + 1:]
        ), dim=-1)

        # prob_t cannot be negative or else it will create nan
        prob_t = torch.log(prob_t + epsilon)
        return prob_t

    def forward_qt(self, x_t1, t):
        """
        :param x_t1: floats specifying distribution over p(x_0)
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

    def forward_step(self, x_0, t, transition_probs=None, epsilon=1e-20):
        """
        :param x_0             : Tensor[batch_size, maxlen]
        :param t               : Tensor[]
        :param transition_probs: Tensor
        :param epsilon: float

        :return: posterior       : x_{t+1}
                 samples         : q(x_t | x_{t+1})
                 transition_probs: q(x_{t+1} | x_t)
        """
        # make x_0 one hot into Tensor[batch_size, maxlen, vocab_size]
        x_0 = F.one_hot(x_0, self.vocab_size).reshape(x_0.shape + (self.vocab_size,))

        # prob_t and next_prob_t are logits
        prob_t = self.get_qt(x_0, t)
        next_prob_t = self.get_qt(x_0, t+1)

        samples = categorical_sample(next_prob_t)
        # make samples one hot (also known as x_t1)
        samples = F.one_hot(samples, self.vocab_size).reshape(samples.shape + (self.vocab_size,))

        # if precomputed transition_probs not given recompute it
        if transition_probs is None:
            transition_probs = self.forward_qt(samples, t)

        # q(x_{t+1} | x_t)
        # normalize to sum to 1
        posterior = transition_probs * prob_t
        posterior = posterior / posterior.sum(dim=-1, keepdims=True)

        return posterior, samples, transition_probs

    def backward_step(self, x_t, t, target_mask, transition_probs=None):
        """
        :param x_t:
        :param t:
        :param transition_probs:
        :param target_mask:

        :return: Tensor[batch_size, maxlen, vocab_size]
                 probabilities for q(x_{t-1} | x_t)
        """
        probs = self.model(targets=x_t, timestep=t, target_mask=target_mask)

        qt_probs, _, _ = self.forward_step(x_0=probs, t=t-1, transition_probs=transition_probs)
        return qt_probs

    def train_step(self, x_0, target_mask):
        """
        :param batch: Tensor[batch_size, maxlen]

        pseudocode:
            - forward step to get q_t, x_t1, transition_probs
            - backward step to get p_t, probabilities from model
            - calculate loss

        q_t = x_t
        x_t1 = reverse_process

        """
        t = torch.randint(0, self.T, size=(1,))

        # q(x_{t+1} | x_0), q(x_t | x_{t+1}), q(x_{t+1} | x_t)
        q_t, x_t1, transition_probs = self.forward_step(x_0, t)
        p_t = self.backward_step(x_t1, t, transition_probs, target_mask)

        loss = self.get_loss(probs=p_t, targets=x_0)
        return loss

    def get_loss(self, probs, targets, epsilon=1e-20):
        """
        cross entropy with logits
        """
        assert probs.size()[:-1] == targets.size()

        probs = F.relu(probs)
        loss = -(torch.log(probs + epsilon).gather(-1, targets.unsqueeze(-1))).squeeze(-1)

        return loss


test = DiffusionBERT(T=300, tokenizer=None, maxlen=512)
posterior, sample, transition_probs = test.forward_step(
    x_0=torch.randint(low=0, high=30522, size=(2, 512)).to(torch.int64).cuda(),
    t=10
)

# print(posterior)
# print(sample)
print(transition_probs.shape)

