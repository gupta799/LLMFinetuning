## Contents

The repository is organized as follows:

- **Supervised Fine-Tuning**: Fine-tuning pre-trained models on labeled datasets.
- **Unsupervised Fine-Tuning**: Fine-tuning without labels, focusing on pretext tasks.
- **Optimization with Optuna**: Hyperparameter optimization using `Optuna` to achieve better performance during model training.
- **Distributed Fine-Tuning**: Notebooks that explore distributed training approaches using model parallelism, DDP, and FSDP to efficiently scale training across multiple GPUs or nodes.

## Key Technologies

The following libraries and packages are heavily utilized in this project:

- [Transformers](https://github.com/huggingface/transformers) - Hugging Face's transformers library for implementing and fine-tuning pre-trained models.
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - For 8-bit optimizations, reducing memory footprint during training.
- [Optuna](https://optuna.org/) - A framework for hyperparameter optimization to tune the training process.
- **Model Parallelism**, **DDP**, and **FSDP** - Distributed training strategies to scale model fine-tuning across multiple GPUs.

## Notebooks

### 1. **Supervised Fine-Tuning Notebooks**
   - These notebooks focus on fine-tuning pre-trained models on supervised datasets.
   - Techniques explored include:
     - Cross-entropy loss for generation tasks.
     - Custom loss functions for specific tasks.
   - Example models used: `BERT`, `GPT-2`, `T5`.

### 2. **Unsupervised Fine-Tuning Notebooks**
   - Fine-tuning without labeled data, utilizing pretext tasks such as Causal language modeling.
   - Example models: `BART`, `BERT`.

### 3. **Optimization with Optuna**
   - These notebooks focus on optimizing hyperparameters (e.g., learning rates, batch size, weight decay) using the `Optuna` library.
   - Automated search strategies include:
     - Bayesian optimization
   - Example: Running `Optuna` trials to find the best hyperparameters for fine-tuning GPT-2 on a custom text corpus.

### 4. **Distributed Fine-Tuning**
   - Notebooks demonstrating how to scale fine-tuning across multiple GPUs or nodes using:
     - **Model Parallelism**: Splitting the model across multiple devices.
     - **DDP (Distributed Data Parallel)**: Parallelizing data batches across multiple devices for efficient gradient updates.
     - **FSDP (Fully Sharded Data Parallel)**: Sharding model states across devices to minimize memory usage during training.
   - These notebooks also explore mixed precision training to improve computational efficiency.

## Prerequisites

To get started with this repository, you'll need the following installed:

- Python 3.8+
- PyTorch (compatible version with your GPU setup)
- Hugging Face's `transformers` library
- `bitsandbytes` for memory-efficient 8-bit training
- `Optuna` for hyperparameter optimization
- NVIDIA GPUs (for distributed fine-tuning and parallelism techniques)
- Other dependencies listed in the `requirements.txt`
