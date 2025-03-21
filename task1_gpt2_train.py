from datasets import load_dataset
from transformers import AutoTokenizer

import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

# Check if GPU is available
torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.cuda.get_device_properties(0))

#load data as dataset instance
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", streaming=False)  
print(dataset, "\n\n")

model = AutoModelForCausalLM.from_pretrained('gpt2', device_map="auto").to(device)
model.gradient_checkpointing_enable()
model.to(device)


#load gpt-2 tokenizer 
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# set eos_token as padding token to pad shorter sequences
tokenizer.pad_token = tokenizer.eos_token

#define tokenzing function
def tokenized(samples):
    
    inputs = tokenizer(samples["text"], truncation=True, padding="max_length", max_length=128)
    #print(inputs)
    inputs["labels"] = inputs["input_ids"].copy()

    return inputs

#get tokenized datasets
tokenized_datasets = dataset.map(tokenized, batched=True)
print(tokenized_datasets)

# Define training arguments
training_args = TrainingArguments(
    output_dir=r'C:\Users\kerem\Desktop\JOB\bmw-tasks\bmw_code\mnt\disks\disk1\results\output_dir',
    eval_strategy='epoch',
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    fp16=True,
    #save_strategy="epoch",
    #max_steps=5000,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=r'C:\Users\kerem\Desktop\JOB\bmw-tasks\bmw_code\output_dir\mnt\disks\disk1\logs'
)



# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)

# Train the model
trainer.train()

# save the model and tokenizer explicitly
model_output_dir = '/mnt/disks/disk1/results/model'

model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)



