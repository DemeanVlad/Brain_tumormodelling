# main.py

import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import os
import cv2
import seaborn as sns
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from models.generator import build_generator
from models.discriminator import build_discriminator
from utils import load_images, sample_images

# Define constants
NOISE_DIM = 100  
SEED = 42
WIDTH, HEIGHT, CHANNELS = 128, 128, 1
MAIN_DIR = "data/yes"

# Load images=
data, labels = load_images(MAIN_DIR)

# Select a subset of images
np.random.seed(SEED)
idxs = np.random.randint(0, len(data), 10)
X_train = data[idxs]

# Normalize and reshape the images
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = X_train.reshape(-1, WIDTH, HEIGHT, CHANNELS)

# Build generator and discriminator models
generator = build_generator()
discriminator = build_discriminator()

# Compile discriminator
discriminator.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5))

# Create GAN model
discriminator.trainable = False
gan_input = Input(shape=(NOISE_DIM,))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)
gan = Model(gan_input, gan_output, name="gan_model")
gan.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5))

# Generate random noise for sample images
noise = np.random.normal(0, 1, (10, NOISE_DIM))

# Sample and display generated images
sample_images(generator, noise, subplots=(2, 5), figsize=(15, 6), save=False)
