# brainformer
A transformer-based approach to predicting MEG (Magnetoencephalography) readings from EEG (Electroencephalography) sensory inputs. The project leverages multi-dimensional attention to predict over multi-channel timeseries data.

## Overview
Brainformer is a novel architecture that aims to bridge the gap between EEG and MEG neuroimaging techniques by learning to predict MEG signals from EEG inputs. The model employs a specialized transformer architecture with two-stage attention mechanisms to handle both temporal and spatial dependencies in brain activity data.

## Key Features
Two-Stage Attention: Implements a novel attention mechanism that processes both temporal and spatial dimensions of brain signals independently
Cross-Channel Communication: Enables information flow between different sensor channels through a router-based attention mechanism
Scalable Architecture: Handles variable-length sequences and different numbers of input/output channels
HuggingFace Integration: Full integration with the HuggingFace ecosystem for easy model sharing and deployment

## Architecture
The core architecture used for the task of translating EEG to MEG signals is a derivation of `CrossFormer`, a specialized transformer that implements:
  - Time-wise self-attention across sequence length
  - Channel-wise attention through a router mechanism
  - Cross-attention between EEG and MEG modalities

We extend the original crossformer architecture implementing optimizations for the task at hand, such as Pre-LN normalization. Further, we also extend the architecture to be encoder-decoder based and entirely sequence to sequence, to fully exploit the wealth of data available regarding coupled EEG and MEG recordings in the OpenNeuro.

![brainformer-v1](media/brainformer-v1.png)*Brainformer architecture overview: Two-stage attention mechanism for processing temporal and spatial dimensions of brain signals, with cross-attention between EEG and MEG modalities.*

While novel in the encoder-decoder formulation, Crossformer is partially derived from the Two-Stage Attention Layer introduced in [Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting](https://openreview.net/forum?id=vSVLM2j9eie).
In short, the Two-Stage Attention Layer processes data in two stages:
- Temporal stage: Captures dependencies across time
- Spatial stage: Models relationships between different sensor channels

## Installation

```bash
git clone https://github.com/fracapuano/brainformer.git && cd brainformer
pip install -e .
```

## Usage

### Data
The dataset used for training is derived from the OpenNeuro repository ([here](https://openneuro.org/datasets/ds000117)), but processed according to a custom defined processing pipeline. Multi-subject, cleaned and ready to train data are openly available on 🤗 Hugging Face at [`fracapuano/eeg2meg-medium-tokenized`](https://huggingface.co/datasets/fracapuano/eeg2meg-medium-tokenized). Evaluation is performed on a test set, derived from a held-out portion of the training dataset.

For context, processing the data is a fairly lenghty process due to its dimensionality. Each data point indeed represents a ~8min recording, sampled at 1100Hz, with 70 channels for the EEG modality and 306 channels for MEG.

### Training
Effort has been spent to make the custom transformer architecture defined in `model` compatible with Hugging Face's `Trainer` API, so as to allow for optimized training. This requires defining a `hf_transformer.py` wrapper around the custom `crossformer.py` module. Further, we define custom `hf_registermodel.py` and `hf_configs.py` files to be able to share the model weights and configurations with the open-source community. 

Lastly, we completely open-sourced our training runs, available on 🐝 Weights&Biases at [fracapuano/eeg2meg-sub1](https://wandb.ai/francescocapuano/eeg2meg-sub1?nw=nwuserfrancescocapuano).

```bash
python train.py --d_model 64 --n_heads 4 --n_layers 1 --sequence_length 256
```

Key parameters:
- `d_model`: Embedding dimension
- `n_heads`: Number of attention heads
- `n_layers`: Number of transformer layers
- `sequence_length`: Maximum sequence length to process

### Evaluation
```bash
python eval.py
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you use this code in your research, please cite:


<!-- @misc{capuano2024brainformer,
  title={Brainformer: A Transformer-based Approach to EEG-to-MEG Signal Translation},
  author={Capuano, Francesco and Ludington, William and Cinà, Gabriele and Zhang (Mingfang) Lucy},
  year={2024},
  journal={arXiv},
  howpublished={\url{https://arxiv.org/abs/XXXXXX}}
} -->