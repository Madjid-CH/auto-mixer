<h3 align="center">MixMAS</h3>

<p align="center">
  An official implementation for paper: "MixMAS: A Framework for Sampling-Based Mixer Architecture Search for Multimodal Fusion and Learning"
  <br/>
  <br/>

## Table Of Contents

* [About the Project](#about-the-project)
* [Built With](#built-with)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
* [Usage](#usage)
* [Hyperparameters](#hyperparameters)

## About The Project

![MixMAS](images/mixmas.png)

In this paper, we propose MixMAS, a framework for Sampling-Based Mixer Architecture Search for Multimodal Fusion and
Learning. Our framework automatically selects the adequate MLP-based architecture for a given multimodal machine
learning task (MML).

## Built With

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [PyTorch-Lightning](https://www.pytorchlightning.ai/index.html)

## Getting Started

### Prerequisites

python 3.11 or higher

### Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run.py
```

## Hyperparameters

We compare MixMAS's performance against M2-Mixer. The hyperparameters of M2-Mixer are as follows:

| Dataset   | Hidden Dim. | Patch Sizes         | Token Dim. | Channel Dim. | Blocks (modality 1/ modality 2 / Fusion) | Params (M) |
|-----------|-------------|---------------------|------------|--------------|------------------------------------------|------------|
| MM-IMDB   | 256         | 16 Image / 512 Text | 32         | 3072         | 4 / 4 / 2                                | 16.7       |
| AV-MNIST  | 128         | 14 Image / 56 Audio | 32         | 3072         | 4 / 4 / 2                                | 8.3        |
| MIMIC-III | 64          | 24 Time-series / -  | 16         | 64           | 1 / 2 / 1                                | 0.029      |

All the blocks in the M2-Mixer are MLP-Mixer blocks.
For MixMAS,
the hyperparameters are the same except for the type of the blocks that will be selected during the micro benchmarking.