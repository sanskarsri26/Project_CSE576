#!/usr/bin/env python
# coding: utf-8

# In[4]:


from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch 

model_name = "microsoft/Phi-3.5-mini-instruct"
# Load model and tokenizer with quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare model for LoRA training
model = prepare_model_for_kbit_training(model)

# Update target modules based on inspection
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    target_modules=["qkv_proj", "o_proj"],  # Use valid module names
    lora_dropout=0.1,
    bias="none"
)

# Apply LoRA configuration to the model
model = get_peft_model(model, lora_config)

print("LoRA model prepared!")


# In[5]:


from datasets import Dataset

# Load training and evaluation data into Hugging Face Dataset format
train_dataset = Dataset.from_pandas(pd.read_csv("Datasets/train_data.csv"))
eval_dataset = Dataset.from_pandas(pd.read_csv("Datasets/eval_data.csv"))

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples["input"], text_target=examples["output"], truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

print("Data tokenized!")


# In[12]:


from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import DatasetDict

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Preprocess the dataset: tokenize inputs and labels with truncation and padding
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["input"],
        max_length=512,
        truncation=True,
        padding="max_length"  # Ensure consistent input length
    )
    labels = tokenizer(
        examples["output"],
        max_length=512,
        truncation=True,
        padding="max_length"  # Ensure consistent label length
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing to the datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Use a data collator designed for sequence-to-sequence tasks
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./lora_phi_model",
    eval_strategy="epoch",  # Updated from deprecated `evaluation_strategy`
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    fp16=True,  # Mixed precision training for faster performance
    push_to_hub=False,
)

# Initialize the Trainer with preprocessed datasets and custom data collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,  # Handles padding dynamically during training
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("lora_phi_model")
print("LoRA fine-tuned model saved!")


# In[ ]:




