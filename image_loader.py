import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Define the main directory containing the images
MAIN_DIR = "data/yes"

# Define constants
SEED = 42
WIDTH, HEIGHT, CHANNELS = 128, 128, 1

def load_images(folder):
    """
    Load images from the specified folder, preprocess them,
    and return as NumPy arrays along with corresponding labels.
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

# Load and preprocess the images
data, labels = load_images(MAIN_DIR)

# Generate random indices to select a subset of images
np.random.seed(SEED)
idxs = np.random.randint(0, len(data), 10)

# Select the corresponding images from the data
X_train = data[idxs]

# Normalize the images
X_train = (X_train.astype(np.float32) - 127.5) / 127.5

# Reshape the images
X_train = X_train.reshape(-1, WIDTH, HEIGHT, CHANNELS)

# Check the shape of the data
print("Shape of the data:", X_train.shape)

# Display the images
fig, axs = plt.subplots(2, 5, figsize=(15, 6))

for i, ax in enumerate(axs.flat):
    ax.imshow(X_train[i], cmap='gray')
    ax.axis('off')
    ax.set_title(f"Image {i+1}")

plt.tight_layout()
plt.show()
