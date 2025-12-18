# LLM Topic Modeling

This repository contains code for training and evaluating various topic models, including LLM-based generative models and traditional baselines.

## Setup

1.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**

    Create a `.env` file in the root directory with the following keys:

    ```bash
    OPENAI_API_KEY=your_openai_key
    HF_TOKEN=your_huggingface_token
    HF_USERNAME=your_huggingface_username
    WANDB_API_KEY=your_wandb_key
    WANDB_ENTITY=your_wandb_entity
    ```

## Data Processing

Before training a generative topic model, you need to process the dataset to extract embeddings and next-token probabilities.

Use `process_dataset.py` to tokenize data, create vocabulary, and extract features using a causal LM.

### Usage

```bash
python process_dataset.py \
    --dataset fancyzhx/dbpedia_14 \
    --model_name baidu/ERNIE-4.5-0.3B-PT \
    --vocab_size 2000 \
    --batch_size 32 \
    --save_name dbpedia_processed
```

### Key Arguments

-   `--dataset`: Path to local dataset or HuggingFace repo ID (default: `fancyzhx/dbpedia_14`).
-   `--model_name`: Causal LM to use for embedding extraction (default: `baidu/ERNIE-4.5-0.3B-PT`).
-   `--vocab_size`: Size of the vocabulary (default: `2000`).
-   `--batch_size`: Batch size for processing (default: `32`).
-   `--save_name`: Name for the saved dataset (saved in `data/processed_data/`).
-   `--no_upload`: Skip uploading to HuggingFace Hub.
-   `--word_prob_method`: Method to compute word probabilities (`prefix` or `product`).

## Training Topic Models

Use `run_topic_model.py` to train and evaluate topic models.

### Supported Models

-   **Generative**: `generative` (LLM-based)
-   **Baselines**: `lda`, `prodlda`, `zeroshot`, `combined`, `etm`, `bertopic`, `fastopic`

### Usage

**Train a Generative Model:**

```bash
python run_topic_model.py \
    --model generative \
    --data_path user/dbpedia_processed \
    --num_topics 50 \
    --num_epochs 100 \
    --wandb_project dbpedia_topics
```

**Train a Baseline (e.g., LDA):**

```bash
python run_topic_model.py \
    --model lda \
    --data_path fancyzhx/dbpedia_14 \
    --num_topics 50
```

### Key Arguments

-   `--model`: Model type (choices: `generative`, `lda`, `prodlda`, `zeroshot`, `combined`, `etm`, `bertopic`, `fastopic`).
-   `--data_path`: Path to processed data (for generative) or raw data (for baselines). Can be a local path or HF repo.
-   `--num_topics`: Number of topics ($K$).
-   `--num_seeds`: Number of random seeds to run (default: `5`).
-   `--wandb_project`: Name of the W&B project for logging.
-   `--load_run_id_or_name`: Load a previous run from W&B for re-evaluation.

### Generative Model Specific Arguments

-   `--loss_weight`: Weight for the reconstruction loss.
-   `--sparsity_ratio`: Ratio for sparsity enforcement.
-   `--loss_type`: Loss function type (`KL` or `CE`).
-   `--temperature`: Softmax temperature.

## Project Structure

-   `process_dataset.py`: Script for data preprocessing and feature extraction.
-   `run_topic_model.py`: Main training and evaluation script.
-   `models/`: Implementations of various topic models.
-   `data/`: Data loading and processing utilities.
-   `evaluation/`: Metrics and evaluation scripts (Coherence, Diversity, etc.).
-   `llm/`: Templates and utilities for LLM interaction.
-   `configs/`: Configuration files (YAML).

## Re-evaluation

You can re-evaluate a previously trained model logged to W&B:

```bash
python run_topic_model.py \
    --load_run_id_or_name <run_id> \
    --wandb_project <project_name>
```
