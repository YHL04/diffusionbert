
import datetime

from transformers import BertTokenizer
from diffusion import DiffusionTrainer
from models import DiffusionBERT
from dataloader import get_dataloader


def main(epochs=10,
         batch_size=1
         ):
    dt = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt"
    file = open(f"logs/{dt}", "w")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = DiffusionBERT()
    model.train()

    diffusion_bert = DiffusionTrainer(T=300,
                                      tokenizer=tokenizer,
                                      model=model,
                                      maxlen=128)

    dataloader = get_dataloader(tokenizer=tokenizer,
                                batch_size=batch_size)

    for epoch in range(epochs):

        for i, batch in enumerate(dataloader):

            loss = diffusion_bert.train_step(
                x_0=batch["input_ids"].cuda(),
                attention_mask=batch["attention_mask"].cuda()
            )

            if i % 20 == 0:
                print("epoch {} loss {}".format(epoch, loss))
                file.write("{}\n".format(loss))
                file.flush()


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

