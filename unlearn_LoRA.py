import sys
import logging
import os

import datasets
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import transformers
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig


logger = logging.getLogger(__name__)

base_model_path = "/base/path"
chosen_model_path = "chosen_model" if len(sys.argv) == 1 else sys.argv[1]

# Use "TOFU" somewhere in the chosen_model_path to use TOFU, otherwise LLM LAT Harmful will be used
if "TOFU" in chosen_model_path:
    checkpoint_path = f"{base_model_path}/{chosen_model_path}/"
    raw_dataset = load_dataset("locuslab/TOFU", "full")["train"]
    example_user_key = "question"
    example_assistant_key = "answer"
    output_dir = "/output/dir"

else:
    checkpoint_path = f"{base_model_path}/{chosen_model_path}/"
    raw_dataset = load_dataset("LLM-LAT/harmful-dataset")["train"]
    example_user_key = "prompt"
    example_assistant_key = "rejected"
    output_dir = "/output/dir"
    
###################
# Hyper-parameters
###################
training_config = {
    "bf16": True,
    "do_eval": False,
    "learning_rate": 5.0e-05,
    "log_level": "info",
    "logging_steps": 20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 3,
    "max_steps": -1,
    "output_dir": output_dir,
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": 4,
    "per_device_train_batch_size": 4,
    "remove_unused_columns": True,
    "save_steps": 100,
    "save_total_limit": 1,
    "seed": 0,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs":{"use_reentrant": False},
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.2,
    "max_seq_length": 2048,
    "packing": True,
    }

lora_config = {
"r": 16,  # Low-rank dimension
"lora_alpha": 32,  # Scaling factor
"target_modules": ["qkv_proj", "o_proj"],  # Target attention layers
"lora_dropout": 0.05,
"bias": "none"
}

train_conf = SFTConfig(**training_config)
peft_conf = LoraConfig(**lora_config)

###############
# Setup logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = train_conf.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process a small summary
logger.warning(
    f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
    + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
)
logger.info(f"Training/evaluation parameters {train_conf}")

################
# Model Loading
################

model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    trust_remote_code=False,
    use_cache=False,
    attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
    torch_dtype=torch.bfloat16,
    device_map=None
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

### LoRA peft model
model = get_peft_model(model, peft_conf)

##################
# Data Processing
##################
def apply_chat_template(
    example,
    tokenizer,
):
    messages = [
        {"content": example[example_user_key], "role": "user"},
        {"content": example[example_assistant_key], "role": "assistant"},
    ]

    example["text"] = tokenizer.apply_chat_template(messages, 
        tokenize=False, add_generation_prompt=False)
    return example

# Load the dataset and split
forget_split, test_split = raw_dataset.train_test_split(test_size=0.3, seed=42).values()
val_split, retain_split = test_split.train_test_split(test_size=0.5, seed=42).values() # retain_split is not used in this file (Remember the seed is 42)

column_names = list(raw_dataset.features)

# Apply chat template to the biased training set
processed_forget_dataset = forget_split.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    remove_columns=column_names,
    desc="Applying chat template to forget_split",
)

# Apply chat template to the biased test set
processed_val_dataset = val_split.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    remove_columns=column_names,
    desc="Applying chat template to val_split",
)


###########
# Training
###########

trainer = SFTTrainer(
    model=model,
    args=train_conf,
    train_dataset=processed_forget_dataset,
    eval_dataset=processed_val_dataset,
    tokenizer=tokenizer,
)
train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

#############
# Evaluation
#############
tokenizer.padding_side = 'left'
metrics = trainer.evaluate()
metrics["eval_samples"] = len(processed_val_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


# ############
# # Save model
# ############
trainer.save_model(train_conf.output_dir)
