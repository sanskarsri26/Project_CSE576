import torch
import math
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import sys
from peft import PeftModel
import pandas as pd

torch.random.manual_seed(0)

base_model_path = "/base/dir"
chosen_model_path_1 = "model_1" if len(sys.argv) < 2 else sys.argv[1]
chosen_model_path_2 = "microsoft/Phi-3.5-mini-instruct" if len(sys.argv) < 3 else sys.argv[2]
model_path_2 = "microsoft/Phi-3.5-mini-instruct" if chosen_model_path_2 == "microsoft/Phi-3.5-mini-instruct" else f"{base_model_path}/{chosen_model_path_2}/"

# Use "TOFU" somewhere in the chosen_model_path to use TOFU, otherwise LLM LAT Harmful will be used
if "TOFU" in chosen_model_path_1:
    model_path_1 = f"{base_model_path}/{chosen_model_path_1}/"
    raw_dataset = load_dataset("locuslab/TOFU", "full")["train"]
    example_user_key = "question"
    example_assistant_key = "answer"

else:
    model_path_1 = f"{base_model_path}/{chosen_model_path_1}/"
    raw_dataset = load_dataset("LLM-LAT/harmful-dataset")["train"]
    example_user_key = "prompt"
    example_assistant_key = "rejected"


def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=False, 
    )

    if "LoRA" in model_path:
        model = PeftModel.from_pretrained(model, "/path/to/attention")
        model = model.merge_and_unload()
    return model

tokenizer = AutoTokenizer.from_pretrained(model_path_1)
##################
# Load and Process Dataset
##################
def apply_chat_template(
    example,
    tokenizer,
):
    messages = [
        {"content": example[example_user_key], "role": "user"}
    ]

    example["text"] = tokenizer.apply_chat_template(messages, 
        tokenize=False, add_generation_prompt=True)
    return example

# Load the dataset and split
forget_split, test_split = raw_dataset.train_test_split(test_size=0.3, seed=42).values()
val_split, retain_split = test_split.train_test_split(test_size=0.5, seed=42).values() # Remember the seed is 42
column_names = list(raw_dataset.features)

# Process forget dataset
processed_forget_dataset = forget_split.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    remove_columns=column_names,
    desc="Applying chat template to forget_split",
)

# Process retain dataset
processed_retain_dataset = retain_split.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    remove_columns=column_names,
    desc="Applying chat template to retain_split",
)

# Example message
message = processed_forget_dataset['text'][0]
print(message)

####################
# Generate message 
####################
def gen_msg(model, tokenizer, message, chosen_model_path):
    input_ids = tokenizer(message, return_tensors="pt").to(model.device)

    generation_args = {
        "max_new_tokens": 500,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "eos_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        output_ids = model.generate(**input_ids, **generation_args)

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Using model: {chosen_model_path}")
    print(generated_text)

##################
# Calculate Forgetting Score with optimized memory usage
##################
def calculate_logits_for_dataset(model, dataset, tokenizer):
    """Calculate logits for the entire dataset with a given model."""
    logits_list = []

    for sample in dataset:
        inputs = tokenizer(sample["text"], return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits_list.append(outputs.logits)

        # Clean up memory after processing each sample
        torch.cuda.empty_cache()

    return logits_list

def calculate_forgetting_scores(logits_model_1, logits_model_2, input_ids):
    """Calculate forgetting scores from logits of two models."""
    forgetting_scores = []

    # Calculate forgetting scores for each sample
    for logits_1, logits_2, input_id in zip(logits_model_1, logits_model_2, input_ids):
        # Remove the last token (because logits are for the next token prediction)
        logits_1 = logits_1[:, :-1]
        logits_2 = logits_2[:, :-1]

        # Prepare the labels (shifted input_ids)
        labels = input_id[:, 1:]

        # Calculate loss for model 1 (old model)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        loss_1 = loss_fn(logits_1.reshape(-1, logits_1.size(-1)), labels.reshape(-1)).mean()

        # Calculate loss for model 2 (new model)
        loss_2 = loss_fn(logits_2.reshape(-1, logits_2.size(-1)), labels.reshape(-1)).mean()

        # Convert losses to probabilities
        p_1 = math.exp(-loss_1.item())
        p_2 = math.exp(-loss_2.item())

        # Calculate the forgetting score
        forgetting_score = 1 - (p_2 / p_1)
        forgetting_scores.append(forgetting_score)

    return forgetting_scores

# Main function to calculate forgetting scores for an entire dataset
def calculate_forgetting_scores_over_dataset(model_path_1, model_path_2, tokenizer, dataset):
    """Calculate forgetting scores for the entire dataset using two models."""
    # Load the first model (to process the first set of logits)
    model_1 = load_model(model_path_1)

    # Calculate logits for the first model (old model)
    logits_model_1 = calculate_logits_for_dataset(model_1, dataset, tokenizer)

    # Clean up GPU memory after using the first model
    del model_1
    torch.cuda.empty_cache()

    # Load the second model (to process the second set of logits)
    model_2 = load_model(model_path_2)

    # Calculate logits for the second model (new model)
    logits_model_2 = calculate_logits_for_dataset(model_2, dataset, tokenizer)

    # Clean up GPU memory after using the second model
    del model_2
    torch.cuda.empty_cache()

    # Calculate forgetting scores using logits from both models
    forgetting_scores = calculate_forgetting_scores(logits_model_1, logits_model_2, [sample["input_ids"] for sample in dataset])

    return forgetting_scores

forgetting_scores = calculate_forgetting_scores_over_dataset(model_path_1, model_path_2, tokenizer, processed_forget_dataset)
df = pd.DataFrame({"forgetting_score": forgetting_scores})
print(df.head())