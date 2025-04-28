import sys
import logging
import os

from datasets import load_dataset
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

base_model_path = "/base/dir"
chosen_model_path = "model" if len(sys.argv) == 1 else sys.argv[1]

# Use "TOFU" somewhere in the chosen_model_path to use TOFU, otherwise LLM LAT Harmful will be used
if "TOFU" in chosen_model_path:
    checkpoint_path = f"{base_model_path}/{chosen_model_path}/"
    output_dir = "/output/dir"

else:
    checkpoint_path = f"{base_model_path}/{chosen_model_path}/"
    output_dir = "/output/dir"



################
# Fine-tuned Model Loading
################
logger.info(f"Loading fine-tuned (biased) model from {checkpoint_path}...")
finetuned_model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    trust_remote_code=False,
    use_cache=False,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

################
# Original Model Loading
################
logger.info("Loading original model...")
original_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    trust_remote_code=True,
    use_cache=False,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)


################
# Task Vector Negation
################
logger.info("Performing task vector negation...")
original_state_dict = original_model.state_dict()
finetuned_state_dict = finetuned_model.state_dict()

for name in original_state_dict:
    if name in finetuned_state_dict:
        delta = finetuned_state_dict[name] - original_state_dict[name]
        original_state_dict[name] -= delta

original_model.load_state_dict(original_state_dict)

################
# Save the SKU model
################
original_model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.save_pretrained(output_dir)

logger.info(f"SKU model saved to {output_dir}")

