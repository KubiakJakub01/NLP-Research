# NLP-Research

This repository serves as a hub for various Natural Language Processing (NLP) research projects, experiments, and implementations, ranging from fundamental machine learning concepts to advanced speech and language processing models.

## Key Features

- **Core NLP & ML Basics**: Fundamental implementations of attention mechanisms, convolutions, RNNs, LSTMs, and more.
- **Speech & Language Processing (SLP)**:
    - **TTS**: Advanced Text-to-Speech models including **VITS**, **HiFi-GAN**, and **WaveNet**.
    - **ASR**: High-performance Automatic Speech Recognition with **Conformer**, **Faster-Whisper**, and HuggingFace integrations.
    - **Utilities**: Speaker Verification, Voice Activity Detection (VAD), and audio normalization.
- **Modern Architectures**: Implementations and experiments with **Transformers**, **Mamba** state-space models, and **U-Nets**.
- **Autograd Engine**: A minimal `micrograd` implementation for understanding backpropagation.
- **Production-Ready Tools**: Utilities for efficient training pipelines using PyTorch Lightning and data collators.

## Project Structure

The project is organized into several modules within the `nlp_research` package:

- `nlp_research/basics`: Fundamental ML/NLP building blocks (tokenization, linear regression, etc.).
- `nlp_research/slp`: Speech-related tasks (ASR, TTS, Mamba, Whisper integration).
- `nlp_research/nlp`: Core NLP architectures and decoding strategies.
- `nlp_research/trainers`: Training loops, data collators, and experiment management.
- `nlp_research/deep_ml`: General deep learning utilities and machine learning algorithms.
- `nlp_research/micrograd`: Minimal autograd engine implementation.
- `nlp_research/hrm`: Specialized hierarchical or research-specific models.
- `nlp_research/modules`: Reusable neural network components (e.g., summary mixing).

## Installation

To install and set up the project, you can use the `uv` package manager. Follow the steps below:

1. Make sure you have Python installed on your system. You can download it from the official Python website: [python.org](https://www.python.org/downloads/).

2. Install `uv` by running the following command in your terminal or command prompt:

   ```shell
   curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3. Clone the repository to your local machine:

   ```shell
   git clone https://github.com/KubiakJakub01/NLP-Research.git
   cd NLP-Research
    ```

4. Install the project dependencies using `uv`:

   ```shell
   uv sync
    ```

5. To activate the virtual environment, run the following command:

   ```shell
   source .venv/bin/activate
    ```

   Optionally you can add the following line to your .bashrc:

   ```shell
   alias activate="source .venv/bin/activate"
   ```

   Then you can activate the virtual environment by running:

   ```shell
   activate
   ```

