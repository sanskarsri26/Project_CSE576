# Mamba LoRA Unlearning – Bias Reduction on StereoSet

## Overview

This repository implements **machine unlearning** for debiasing a Mamba 0.13B (LoRA) language model using the **Partitioned Contrastive Gradient Unlearning (PCGU)** algorithm. Bias is measured using the **StereoSet** benchmark and qualitative prompt completion.

---

## Workflow

### 1. Data Preparation

- Place `wg.tsv` (WinoGender) and `stereoset_dev.json` (StereoSet) in the `data/` directory.
- Run preprocessing to generate grouped contrastive data:
    ```bash
    python scripts/preprocess_winogender.py
    ```

---

### 2. Fine-tuning

- Fine-tune Mamba-130M (LoRA) on WinoGender:
    ```bash
    python scripts/finetune_winogender.py
    ```
- Output: `models/finetuned_lora/`

---

### 3. PCGU Unlearning

- Apply **Partitioned Contrastive Gradient Unlearning**:
    ```bash
    python scripts/pcgu_unlearn.py
    ```
- Output: `models/debiased_lora/`

---

### 4. Evaluation

- **StereoSet Bias Score**:
    ```bash
    python scripts/evaluate_stereoset.py
    ```
- **Qualitative Prompt Comparisons**:
    ```bash
    python scripts/compare_bias_prompts.py
    ```

---

## Results

| Model Type                    | StereoSet Bias Score (%) | Notes                                     |
|-------------------------------|:-----------------------:|-------------------------------------------|
| Fine-tuned mamba 0.13B (LoRA) | 46.82                   | Before unlearning                         |
| **PCGU Unlearned**            | **46.73**               | Best debiasing, utility maintained better |
| PCGU (aggressive)             | 46.49                   | Hallucinations observed                   |

---

## Parameters

**Fine-tuning:**
- Epochs: 3
- Batch size: 8
- Learning rate: 5e-5

**PCGU Unlearning:**
- projection_steps: 30
- learning_rate: 2e-5
- lambda_reg: 0.15

---

## Dependencies

- torch
- transformers
- peft
- pandas
- tqdm

> **Note:**  
> This project is tested on an NVIDIA A100 GPU. Make sure you have CUDA drivers and compatible PyTorch installed.  
> If running on CPU, set device mapping and tensor types accordingly in the scripts.

> **GPU Setup:**  
> If using A100 and CUDA 11.8/12.x, ensure your PyTorch install matches your CUDA driver by running:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```
> Adjust the version (`cu121` for CUDA 12.1) according to your CUDA version.

Install all dependencies with:
```bash
pip install -r requirements.txt



