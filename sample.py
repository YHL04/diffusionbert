

import torch
import abc


class SampleClassBase(abc.ABC):
    def sample(self, logits, x_0):
        raise NotImplementedError

    def post_process_sample_in_prediction(self, sample, x_0):
        return sample


class Categorical(SampleClassBase):
    def sample(self, logits, x_0):
        return torch.distributions.categorical.Categorical(logits=logits).sample()
