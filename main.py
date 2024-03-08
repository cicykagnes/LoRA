#pip install -q bitsandbytes datasets accelerate loralib
#pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git

import wandb
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model 
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from datasets import load_dataset
import numpy as np
import time
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm


block_size = 128
wandb.login()
# Initialize WandB
wandb.init(project="my_model",
           config={
               "learning_rate": 2e-4,
               "architecture": "GPT2",
               "dataset": "DOLLY",
               "epochs": 5,
           })
# DataSet
checkpoint = 'gpt2'
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
# DATA
dataset = load_dataset('databricks/databricks-dolly-15k')
def preprocess(row): # row is a list because you're batching
    return tokenizer([' '.join(x) for x in zip(row['instruction'], row['response'])])

tokenized = dataset.map(
    preprocess,
    batched=True,
    num_proc=4,
    #remove_columns=['context', 'category'], # just for debugging
    remove_columns=dataset['train'].column_names,
)

def group_texts(examples):
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

lm_dataset = tokenized.map(group_texts, batched=True, num_proc=4)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

train_dataloader = DataLoader(lm_dataset['train'], shuffle=False, batch_size=100, collate_fn=data_collator)

#MODEL
checkpoint = 'gpt2'
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

model = AutoModelForCausalLM.from_pretrained(
    checkpoint, 
    # load_in_8bit=True, 
   # device_map='auto',
)
# print(model)

def get_mem_usage(device):
    return f'{round(torch.cuda.memory_allocated(device) / 1048576, 2)} MB'


#LORA

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"], # from gpt2Attention
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

model.print_trainable_parameters()
epochs = 5

#output_dir = "lora-gpt2"
#training_args = TrainingArguments(
#    output_dir=output_dir,
#    learning_rate=2e-4,
#    num_train_epochs=5,
#    logging_dir=f"{output_dir}/logs",
#    logging_strategy="steps",
#    logging_steps=500,
#    save_strategy="no",
#    report_to="wandb",
#    run_name="my_model",
#    per_device_train_batch_size=2,
#)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False,
)
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

with open('run.log', 'a', encoding='utf-8') as log_f:
        log_f.write(f'memory allocated after sending model to device: {get_mem_usage(device)}\n')
        progress_bar = tqdm(range(num_training_steps))

        model.train()
        for epoch in range(num_epochs):
            start = time.time()
            num_tokens = 0
            for batch in train_dataloader:
                num_tokens += batch.input_ids.size(0)
                batch = {k: v.to(device) for k, v in batch.items()}
                log_f.write(f'memory allocated after sending a batch to device: {get_mem_usage(device)}\n')
                outputs = model(**batch)
                log_f.write(f'memory allocated after forward pass: {get_mem_usage(device)}\n')
                loss = outputs.loss
                loss.backward()
                log_f.write(f'memory allocated after backward pass: {get_mem_usage(device)}\n')

                #model.step()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            log_f.write(f'memory allocated at the end of an epoch: {get_mem_usage(device)}\n')
            time_per_epoch = round(time.time() - start, 2)
            wandb.log({'epoch': epoch, "training_loss": loss, 'time_per_epoch': time_per_epoch, 'tokens_per_second': num_tokens/time_per_epoch})
#wandb.finish()

#trainer = Trainer(
#    model=model,
#    args=training_args,
#    data_collator=data_collator,
#    train_dataset=lm_dataset['train'],
    #eval_dataset=tokenized_dataset['validation']
#)
##model.config.use_cache = False 

#trainer.train()
##eval_results = trainer.evaluate()
##print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss']))}")
wandb.finish()
