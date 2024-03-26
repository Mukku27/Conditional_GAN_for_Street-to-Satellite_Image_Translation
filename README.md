## Conditional_GAN_for_Street-to-Satellite_Image_Translation
# Pix2Pix GAN for Maps Dataset 

This repository implements a Pix2Pix Generative Adversarial Network (GAN) for translating street maps (inputs) to their corresponding satellite images(targets) using a dataset from Kaggle ([https://www.kaggle.com/datasets/alincijov/pix2pix-maps](https://www.kaggle.com/datasets/alincijov/pix2pix-maps)). 

The code is inspired by the research paper "Image-to-Image Translation with Conditional Adversarial Networks" ([https://arxiv.org/abs/1611.07004](https://arxiv.org/abs/1611.07004)).

### Files

* `Discriminator.py`: Defines the Discriminator network architecture.
* `Generator.py`: Defines the Generator network architecture.
* `config.py`: Stores hyperparameters and configurations for the training process.
* `utils.py`: Includes utility functions for saving checkpoints, saving generated images, etc.
* `dataset.py`: Defines a custom PyTorch dataset class to load and pre-process the map images.
* `train.ipynb`: Implements the training loop for the GAN model.

### Dependencies

This code requires the following Python libraries:

* torch
* torch.nn
* torch.optim (or other optimizers)
* torchvision
* albumentations
* tqdm

### Running the Training Script

1. Install the required dependencies (refer to their respective documentation for installation instructions).
2. Download the Pix2Pix maps dataset from Kaggle and place it in a directory named `data/maps/`. Split the data into training and validation sets (modify paths in `train.ipynb` if needed).
3. Open `train.ipynb` in a Jupyter Notebook environment and execute the cells. The script will train the GAN model and save checkpoints and generated images.

**Note:** This is a basic implementation and might require adjustments depending on your specific hardware and dataset size. Consider adjusting hyperparameters in `config.py` for optimal performance.

### Getting Started with the Code

#### Network Architectures:

* `Discriminator.py`: The Discriminator network takes both the input image ( street maps) and the target image (satellite images) as input and outputs a probability that the pair is a real image (target and its corresponding map) or a fake image (generated map). It uses a series of convolutional layers with LeakyReLU activations.
* `Generator.py`: The Generator network takes the input image (street maps) as input and outputs a generated satellite image. It uses a U-Net like architecture with convolutional blocks for encoding and decoding the image features.

#### Training Loop:

* `train.ipynb`: This script defines and executes the training loop for the GAN. It involves:
    * Loading the training and validation datasets.
    * Defining the Discriminator and Generator models.
    * Initializing optimizers and loss functions (BCE with logits and L1 loss).
    * Using gradient scaling for mixed precision training.
    * Training loop with alternating updates for the Discriminator and Generator.
    * Saving model checkpoints and generated images periodically.

This is a high-level overview of the code. It's recommended to go through the code itself for a deeper understanding of the implementation details.
