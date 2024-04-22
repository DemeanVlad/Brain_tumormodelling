# utils.py

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Define constants
WIDTH, HEIGHT, CHANNELS = 128, 128, 1

def load_images(folder):
    """
    Load and preprocess images from the specified folder.
    """
    imgs = []
    labels = []
    for i in os.listdir(folder):
        img_dir = os.path.join(folder, i)
        try:
            img = cv2.imread(img_dir)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (WIDTH, HEIGHT))
            imgs.append(img)
            labels.append(1)  # Assign label 1 for all images
        except Exception as e:
            print(f"Error loading image {img_dir}: {str(e)}")
    
    return np.array(imgs), np.array(labels)

def sample_images(generator, noise, subplots, figsize=(22,8), save=False):
    """
    Generate and display sample images produced by the generator model.
    """
    generated_images = generator.predict(noise)
    plt.figure(figsize=figsize)
    
    for i, image in enumerate(generated_images):
        plt.subplot(subplots[0], subplots[1], i+1)
        if CHANNELS == 1:
            plt.imshow(image.reshape((WIDTH, HEIGHT)), cmap='gray')    
        else:
            plt.imshow(image.reshape((WIDTH, HEIGHT, CHANNELS)))
        if save:
            img_name = "gen" + str(i)
            plt.savefig(img_name)
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Other utility functions or definitions can go here
