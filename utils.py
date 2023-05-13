

import torch


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
