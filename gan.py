import numpy as np
import tensorflow as tf
from tqdm import tqdm
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, Reshape, Input, Conv2DTranspose
from keras.layers import Activation, LeakyReLU, BatchNormalization, Dropout
from models.generator import build_generator
from models.discriminator import build_discriminator
from gan import build_gan
import matplotlib.pyplot as plt


# Load or generate your training data
X_train = ...

# Define constants
BATCH_SIZE = 4
SEED = 42
STEPS_PER_EPOCH = 3750
WIDTH, HEIGHT, CHANNELS = 128, 128, 1
NOISE_DIM = 100

# Set random seed
np.random.seed(SEED)

# Build generator, discriminator, and GAN models
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Train the GAN
for epoch in range(10):
    for batch in tqdm(range(STEPS_PER_EPOCH)):
        # Generate noise
        noise = np.random.normal(0, 1, size=(BATCH_SIZE, NOISE_DIM))
        
        # Generate fake images
        fake_X = generator.predict(noise)
        
        # Select real images
        idx = np.random.randint(0, X_train.shape[0], size=BATCH_SIZE)
        real_X = X_train[idx]

        # Concatenate real and fake images
        X = np.concatenate((real_X, fake_X))

        # Create labels for discriminator
        disc_y = np.zeros(2 * BATCH_SIZE)
        disc_y[:BATCH_SIZE] = 1

        # Train discriminator
        d_loss = discriminator.train_on_batch(X, disc_y)
        
        # Create labels for generator
        y_gen = np.ones(BATCH_SIZE)

        # Train generator via GAN
        g_loss = gan.train_on_batch(noise, y_gen)

    print(f"EPOCH: {epoch + 1} Generator Loss: {g_loss:.4f} Discriminator Loss: {d_loss:.4f}")

    # Generate and display sample images
    noise = np.random.normal(0, 1, size=(10, NOISE_DIM))
    sample_images = generator.predict(noise)

    # Plot the generated images
    plt.figure(figsize=(20, 8))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(sample_images[i], cmap='gray')
        plt.axis('off')
        plt.title(f"Generated Image {i+1}")
    plt.show()
