

import torch
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
from transformers import BertTokenizer

from diffusion import DiffusionBERT
from dataloader import DiffusionLoader
from utils import get_word_freq


def main(epochs=10,
         batch_size=32
         ):

    word_freq = get_word_freq()

    def process_fn_in_collate(wf):
        return wf - wf.mean()

    def collate_fn(batch_input):
        input_ids = [torch.tensor(d['input_ids']) for d in batch_input]
        attention_mask = [torch.tensor(d['attention_mask']) for d in batch_input]
        word_freq_logits = [process_fn_in_collate(word_freq.gather(0, torch.tensor(d['input_ids']))) for d in batch_input]
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        word_freq_logits = pad_sequence(word_freq_logits, batch_first=True)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'word_freq_logits': word_freq_logits
        }

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    diffusion_bert = DiffusionBERT(T=300, tokenizer=tokenizer)

    data = DiffusionLoader(tokenizer=tokenizer).my_load(task_name="lm1b", splits=["test"])
    dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=batch_size,
                                             collate_fn=collate_fn,
                                             )
    # train_loader = torch.utils.data.DataLoader(train_data,
    #                                            batch_size=args.batch_size,
    #                                            collate_fn=collate_fn,
    #                                            num_workers=4,
    #                                            pin_memory=True,
    #                                            sampler=train_sampler
    #                                            )

    for epoch in range(epochs):

        for i, batch in enumerate(tqdm(dataloader), 2):

            loss = diffusion_bert.train_step(
                x_0=batch["input_ids"],
                target_mask=batch["attention_mask"]
            )


if __name__ == "__main__":
    main()


"""
NEED TO KNOW:

DDP_main                ->  main training file

diffusion_word_freq     ->  create_discrete_diffusion_schedule (schedule)
                            MaskDiffusion (diffusion_instance?)
                            compute_kl_reverse_process (train step)
                            discrete_diffusion_elbo (test step to get metric)


DDP_main_conditional    ->  difference between main?
diffusion_condition     ->  difference between main?

compute_elbo                 ->  testing file
compute_metric               ->  utilities to compute metrics such as bleu
dataloader                   ->  load datasets
losses                       ->  kl divergence and cross entropy
predict                      ->  testing file
predict_downstream_condition ->  testing file
sample                       ->  sample classes (Categorical, WholeWordMasking)
utils                        ->  miscellaneous
word_freq                    ->  calculates frequency of each word in dataset and save
                                 Tensor[vocab_size,] as .pt

functions

compute_kl_reverse_process(
    x_start = x_0 = input_ids : Tensor[batch_size, maxlen],
    t = sample_t : Tensor[1,],
    denoise_fn : ,
    diffusion_instance,
    attention_mask,
    hybrid_lambda,
    predict_x0,
    word_freq_logits
)

terms

word_freq_logits = between -0.5 to 0.5 where each index that correspond to
                   a word has its frequency

batch = {
    'input_ids' : inputs_ids,
    'attention_mask' : attention_mask,
    'word_freq_logits' : word_freq_logits
}


"""

