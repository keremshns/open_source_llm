import torch
print(torch.__version__)
print(torch.cuda.is_available())

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from datasets import load_dataset

def tokenized(examples):
    inputs = tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

# Check GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

#Model to use
model_name = "deepseek-ai/deepseek-llm-7b-chat"

#Configure 4-bit quantization for efficient GPU usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16  # Use float16 for faster computation
)

#Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=bnb_config,  
    device_map="auto"
)

#Apply LoRA 
lora_config = LoraConfig(
    r=8,  # Low-rank adaptation size
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("DeepSeek Loaded with LoRA and 4-bit Precision!")

#To lower VRAM usage  --> Gradient checkpointing lowers VRAM usage by recomputing activations instead of storing them.
#model.gradient_checkpointing_enable()
#model.to(device)

#Load dataset and tokenize with tokenized function
dataset = load_dataset("imdb")
tokenized_datasets = dataset.map(tokenized, batched=True)

#Take a random subset from the dataset 
small_train_dataset = tokenized_datasets["train"].shuffle(seed=7).select(range(2500))
small_test_dataset = tokenized_datasets["test"].shuffle(seed=7).select(range(250))

#Print a sample tokenized text
print("Tokenized Sample:")
print("\n\n", small_train_dataset[0], "\n\n")


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir=r".\results",
    eval_strategy="epoch",
    learning_rate=3e-4,  #Lower learning rate for LoRA fine-tuning
    per_device_train_batch_size=1,  #Reduce batch size for memory efficiency
    gradient_accumulation_steps=4,  #Simulate larger batch size
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_steps=50,
    logging_dir=r".\logs",
    fp16=True,  #Mixed precision 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_test_dataset,
)
print("Trainer Initialized!")

print("Starting Training...")
trainer.train()  # Start the training process
print("Training Completed!")

trainer.save_model(r"./finetuned_deepseek")
tokenizer.save_pretrained(r"./finetuned_deepseek")
print("Model Saved Successfully!")