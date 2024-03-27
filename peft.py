from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from pathlib import Path
import torch
import time
import os
import argparse
import wandb
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from transformers import BitsAndBytesConfig

parser = argparse.ArgumentParser(description='train a GPT-2 model on Dolly')
parser.add_argument('-n', '--n-epochs', type=int, default=2, help='number of epochs')
parser.add_argument('-bs', '--batch-size', type=int, default=100, help='batch size')
parser.add_argument('-lr', '--learning-rate', type=float, default=5e-5, help='learning rate')
parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
parser.add_argument('--qlora', action='store_true', help='use QLoRA')
parser.add_argument('--lora', action='store_true', help='use LoRA')
parser.add_argument('--alpha', type=int, default=8, help='alpha for (Q)LoRA fine-tuning')
parser.add_argument('--r', type=int, default=4, help='rank for (Q)LoRA fine-tuning')

def preprocess(row, tokenizer=None): # row is a list because you're batching
    return tokenizer([' '.join(x) for x in zip(row['instruction'], row['response'])])

def group_texts(examples, block_size=128):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # split by chunks of block_size, drop the rest
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result['labels'] = result['input_ids'].copy()
    return result

def get_mem_usage(device):
    return f'{round(torch.cuda.memory_allocated(device) / 1048576, 2)} MB'

# adapted from https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing#scrollTo=gkIcwsSU01EB
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def main():
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    if args.qlora:
        nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        model = AutoModelForCausalLM.from_pretrained('gpt2', quantization_config=nf4_config, device_map='auto')
    else:
        model = AutoModelForCausalLM.from_pretrained('gpt2', device_map='auto')

    if args.lora or args.qlora:
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
                r=args.r,
                lora_alpha=args.alpha,
                target_modules=['c_proj', 'c_attn'],
                lora_dropout=0.05,
                bias='none',
                task_type='CAUSAL_LM',
        )
        model = get_peft_model(model, config)

    print_trainable_parameters(model)

    wandb.init()
    wandb.config.update(args)

    dataset = load_dataset('databricks/databricks-dolly-15k')

    tokenized = dataset.map(
        preprocess,
        batched=True,
        num_proc=4,
        remove_columns=dataset['train'].column_names,
        fn_kwargs={'tokenizer': tokenizer}
    )

    lm_dataset = tokenized.map(group_texts, batched=True, num_proc=4)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(lm_dataset['train'], shuffle=False, batch_size=args.batch_size, collate_fn=data_collator)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    num_training_steps = args.n_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with open('run.log', 'a', encoding='utf-8') as log_f:
        log_f.write(f'memory allocated after sending model to device: {get_mem_usage(device)}\n')
        progress_bar = tqdm(range(num_training_steps))

        model.train()
        for epoch in range(args.n_epochs):
            start = time.time()
            num_tokens = 0
            for batch in train_dataloader:
                num_tokens += batch.input_ids.size(0)
                batch = {k: v.to(device) for k, v in batch.items()}
                log_f.write(f'memory allocated after sending a batch to device: {get_mem_usage(device)}\n')

                outputs = model(**batch)
                log_f.write(f'memory allocated after forward pass: {get_mem_usage(device)}\n')
                loss = outputs.loss
                log_f.write(f'memory allocated after backward pass: {get_mem_usage(device)}\n')
                loss.backward(torch.ones_like(loss))

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            log_f.write(f'memory allocated at the end of an epoch: {get_mem_usage(device)}\n')
            time_per_epoch = round(time.time() - start, 2)
            wandb.log({'epoch': epoch, "training_loss": loss, 'time_per_epoch': time_per_epoch, 'tokens_per_second': num_tokens/time_per_epoch})
    wandb.finish()

if __name__ == '__main__':
    main()
