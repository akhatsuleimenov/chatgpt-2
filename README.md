
# NanoGPT: A GPT-2 and GPT-3 Hybrid Model

This repository hosts the implementation of NanoGPT, a GPT-based model that combines features from OpenAI's GPT-2 and GPT-3, alongside the efficient training approaches inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy. NanoGPT is specifically designed to push the boundaries of learning efficiency and computational cost, achieving a significant milestone by surpassing the GPT-2 HellaSwag evaluation score (0.295) with just 10 billion tokens (score: 0.305), which is a 10x improvement in learning efficiency over the original models trained on much larger datasets.

## Model Overview

NanoGPT is fully implemented in PyTorch and supports both single-GPU and multi-GPU training configurations. It uses a subset of the [fineweb-edu](https://arxiv.org/pdf/2406.17557) dataset comprising 10 billion tokens, tailored to refine performance in educational content understanding and generation. This model is available on the Hugging Face Model Hub, which facilitates easy integration and usage.

### Key Achievements

- **Efficiency**: Achieves better performance than GPT-2 124M after training on 10 billion tokens and surpasses GPT-3 125M after 40 billion tokens.
- **Speed**: Training for one epoch on the 10B token subset takes approximately 3.5 hours on 8x A100-SMX4 40GB GPUs.

## Installation and Setup

To set up and run NanoGPT, follow these steps:

```bash
git clone https://github.com/akhatsuleimenov/chatgpt-2.git
cd chatgpt
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset

The fineweb-edu dataset is used as the dataset for training the GPT model as it is a large-scale, high quality dataset of educational content.

```bash
python fineweb.py
```

### Training

To initiate training, use the following commands based on your hardware setup:

```bash
python train.py # For single-GPU
torchrun --standalone --nproc_per_node=<number_of_GPUs> train.py # For multi-GPU
```

### Evaluation

Evaluate the model using the HellaSwag dataset by running:

```bash
python hellaswag.py
```

## License

This project is licensed under the terms of the MIT license.
