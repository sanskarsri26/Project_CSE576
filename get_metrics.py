import torch
import math
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import sys

torch.random.manual_seed(0)

model_old_path = "microsoft/Phi-3.5-mini-instruct"
base_model_path = "/base/dir"
chosen_model_path = "model" if len(sys.argv) == 1 else sys.argv[1]

truthfulQA_dataset = load_dataset("csv", data_files="TruthfulQA.csv")["train"]
truthfulQA_user_key = "Question"
truthfulQA_assistant_key = "Best Answer"
truthfulQA_column_names = list(truthfulQA_dataset.features)

# Use "TOFU" somewhere in the chosen_model_path to use TOFU, otherwise LLM LAT Harmful will be used
if "TOFU" in chosen_model_path:
    model_path = f"{base_model_path}/{chosen_model_path}/"
    raw_dataset = load_dataset("locuslab/TOFU", "full")["train"]
    example_user_key = "question"
    example_assistant_key = "answer"

else:
    model_path = f"{base_model_path}/{chosen_model_path}/"
    raw_dataset = load_dataset("LLM-LAT/harmful-dataset")["train"]
    example_user_key = "prompt"
    example_assistant_key = "rejected"

model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=False, 
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

##################
# Load and Process Dataset
##################
def apply_chat_template(
    example,
    tokenizer,
    user_key,
    assistant_key
):
    messages = [
        {"content": example[user_key], "role": "user"},
        {"content": example[assistant_key], "role": "assistant"},
    ]

    example["text"] = tokenizer.apply_chat_template(messages, 
        tokenize=False, add_generation_prompt=False)
    return example

# Load the dataset and split
forget_split, test_split = raw_dataset.train_test_split(test_size=0.3, seed=42).values()
_, retain_split = test_split.train_test_split(test_size=0.5, seed=42).values()

retain_column_names = list(raw_dataset.features)

# Process test dataset
processed_retain_dataset = retain_split.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer, "user_key": example_user_key, "assistant_key": example_assistant_key},
    num_proc=10,
    remove_columns=retain_column_names,
    desc="Applying chat template to retain_split",
)

# Process truthfulQA dataset
processed_truthfulQA_dataset = truthfulQA_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer, "user_key": truthfulQA_user_key, "assistant_key": truthfulQA_assistant_key},
    num_proc=10,
    remove_columns=truthfulQA_column_names,
    desc="Applying chat template to truthfulQA",
)

print(f"Getting metrics for {chosen_model_path}")
##################
# Perplexity Calculation
##################
def calculate_perplexity(model, tokenizer, dataset, num_samples=500):
    total_loss = 0
    count = 0
    dataset = dataset.select(range(min(num_samples, len(dataset))))  # Limit samples for efficiency
    
    for example in dataset["text"]:
        inputs = tokenizer(example, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        total_loss += loss
        count += 1

    avg_loss = total_loss / count
    perplexity = math.exp(avg_loss)
    return perplexity

retain_ppl = calculate_perplexity(model, tokenizer, processed_retain_dataset)
truthfulQA_ppl = calculate_perplexity(model, tokenizer, processed_truthfulQA_dataset)
print(f"Perplexity on test set: {retain_ppl:.2f}")
print(f"Perplexity on truthfulQA set: {truthfulQA_ppl:.2f}")

##################
# BLEU Score Calculation
##################
def calculate_bleu(model, tokenizer, dataset, num_samples=500):
    total_bleu = 0
    count = 0
    dataset = dataset.select(range(min(num_samples, len(dataset))))  # Limit samples for efficiency
    
    for example in dataset:
        # Tokenize and generate model output
        inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, max_length=512).to("cuda")
        
        # Generate predictions
        with torch.no_grad():
            output_ids = model.generate(inputs["input_ids"], max_length=512, num_return_sequences=1)
    
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_tokens = nltk.word_tokenize(generated_text)
        reference_tokens = [nltk.word_tokenize(example["text"])]  # List of references
        
        bleu_score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=SmoothingFunction().method1)
        total_bleu += bleu_score
        count += 1

    avg_bleu = total_bleu / count
    return avg_bleu




# Calculate BLEU score
retain_bleu_score = calculate_bleu(model, tokenizer, processed_retain_dataset)
truthfulQA_bleu_score = calculate_bleu(model, tokenizer, processed_truthfulQA_dataset)
print(f"Average BLEU Score on test set: {retain_bleu_score:.4f}")
print(f"Average BLEU Score on truthfulQA set: {truthfulQA_bleu_score:.4f}")