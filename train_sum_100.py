#!/usr/bin/env python
# coding: utf-8


from transformers import AutoTokenizer
from datasets import load_dataset, load_metric, load_from_disk, Dataset, DatasetDict
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
from statistics import mean
import os
from tqdm.auto import tqdm
import pandas as pd
import csv


# In[2]:


#os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[2]:


# In[5]:


token_path = 't5-large'
model_path = 'checkpoint-268000/'
tokenizer = AutoTokenizer.from_pretrained(token_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, max_length=512, num_beams=5)


# In[6]:


max_input_length = 512
max_target_length = 512

def preprocess_function(examples):
    prefix = "sum: "
    
    inputs = [prefix + str(doc) for doc in examples['input']]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels_input = [str(doc) for doc in examples['label']]
        labels = tokenizer(labels_input, max_length=max_target_length, truncation=True, padding=True)

    model_inputs['labels'] = labels['input_ids']
    #print(model_inputs)
    return model_inputs


# In[7]:


datasets = load_dataset('csv', data_files={'train':'train_pow100.csv', 'valid':'valid_pow100.csv'})

#datasets = datasets.applymap(str)
# In[8]:


tokenized_datasets = datasets.map(preprocess_function, batched=True, num_proc = 4)


# In[9]:


tokenized_datasets = tokenized_datasets.remove_columns(datasets["train"].column_names)


# In[10]:


import evaluate

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = [decoded_preds[i] == decoded_labels[i] for i in range(len(decoded_labels))]
    result_mean = sum(result) / len(result)
    
    return {'accuracy': round(result_mean, 4)}


# In[11]:


import wandb
run = wandb.init(
    # Set the project where this run will be logged
    project='sum_nums',
    name="t5-large_pow100",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 2e-4,
    })


# In[12]:


batch_size = 2
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
args = Seq2SeqTrainingArguments(
    "./checkpoints_t5_large_pow100",
    evaluation_strategy = "steps",
    learning_rate=2e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=10,
    warmup_steps = 100,
    predict_with_generate=True,
    metric_for_best_model='accuracy',
    eval_steps=4000,
    save_steps=2000,
    report_to='wandb',
    logging_dir='/logs2',
    ignore_data_skip=True,
    overwrite_output_dir=True
)


# In[13]:


trainer = Seq2SeqTrainer(model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# In[ ]:


trainer.train(resume_from_checkpoint=True)

wandb.finish()
