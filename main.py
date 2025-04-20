import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
import albumentations as A
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import shutil
import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    Dropout,
    UpSampling2D,
    Concatenate,
    Layer,
    GlobalAveragePooling2D as KerasGlobalAveragePooling2D
)


base_dir = 'huggingface_dataset'
os.makedirs(base_dir, exist_ok=True)
os.makedirs(f'{base_dir}/plans', exist_ok=True)
os.makedirs(f'{base_dir}/walls', exist_ok=True)
os.makedirs(f'{base_dir}/model', exist_ok=True)

# Set parameters
IMG_HEIGHT = 512
IMG_WIDTH = 512
NUM_CLASSES = 2  # Wall and background
BATCH_SIZE = 4
EPOCHS = 50

# ============== DATA LOADING AND AUGMENTATION ==============

# def load_data(image_path, mask_path):
#     """Load and preprocess images and masks"""
#     # Read image
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
#     # Read mask
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
#     mask = np.expand_dims(mask, axis=-1)
    
#     # Normalize
#     img = img / 255.0
#     mask = mask / 255.0
    
#     # Binarize mask (threshold)
#     mask = (mask > 0.5).astype(np.float32)
    
#     return img, mask


# Data augmentation
def augment_data(img, mask):
    """Apply augmentations to images and masks"""
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.GaussianBlur(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
    ])
    
    transformed = transform(image=img, mask=mask)
    return transformed['image'], transformed['mask']

def load_data(image_dir, mask_dir, img_height, img_width):
    """Load image and mask data from specified directories."""
    images = []
    masks = []
    image_files = sorted(os.listdir(image_dir))  # Ensure consistent ordering
    mask_files = sorted(os.listdir(mask_dir))    # Ensure consistent ordering

    for img_file, mask_file in zip(image_files, mask_files):
        # Load images
        img_path = os.path.join(image_dir, img_file)
        img = load_img(img_path, target_size=(img_height, img_width))
        img = img_to_array(img)
        images.append(img)

        # Load masks
        mask_path = os.path.join(mask_dir, mask_file)
        mask = load_img(mask_path, target_size=(img_height, img_width), color_mode='grayscale')
        mask = img_to_array(mask)
        masks.append(mask)

    return np.array(images), np.array(masks)

def preprocess_input(images):
    """Normalize image data to be in the range [0, 1]"""
    return images.astype(np.float32) / 255.0

def preprocess_labels(masks):
    """
    Preprocess mask data.  Assumes masks are grayscale with 0 and 255 values.
    Converts masks to binary (0 and 1) and ensures correct shape and type.
    """
    masks = masks.astype(np.float32)
    masks = masks / 255.0  # Convert to 0 and 1
    masks = np.reshape(masks, (masks.shape[0], masks.shape[1], masks.shape[2], 1)) # Important for binary_crossentropy
    return masks

    
def DeepLabV3Plus(input_shape=(512, 512, 3), num_classes=1):
    """Create a DeepLabv3+ model"""

    # Input layer
    inputs = Input(shape=input_shape)

    # ResNet50 as backbone (encoder)
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    # Get features from different levels
    low_level_features = base_model.get_layer('conv2_block3_out').output
    x = base_model.output

    # ASPP (Atrous Spatial Pyramid Pooling)
    # 1x1 Conv
    aspp1 = Conv2D(256, 1, padding='same', use_bias=False)(x)
    aspp1 = BatchNormalization()(aspp1)
    aspp1 = Activation('relu')(aspp1)

    # 3x3 Conv rate 6
    aspp2 = Conv2D(256, 3, padding='same', dilation_rate=6, use_bias=False)(x)
    aspp2 = BatchNormalization()(aspp2)
    aspp2 = Activation('relu')(aspp2)

    # 3x3 Conv rate 12
    aspp3 = Conv2D(256, 3, padding='same', dilation_rate=12, use_bias=False)(x)
    aspp3 = BatchNormalization()(aspp3)
    aspp3 = Activation('relu')(aspp3)

    # 3x3 Conv rate 18
    aspp4 = Conv2D(256, 3, padding='same', dilation_rate=18, use_bias=False)(x)
    aspp4 = BatchNormalization()(aspp4)
    aspp4 = Activation('relu')(aspp4)

    # Image pooling - FIXED IMPLEMENTATION
    # Using Keras layers instead of direct TensorFlow ops
    aspp5 = KerasGlobalAveragePooling2D(keepdims=True)(x)
    aspp5 = Conv2D(256, 1, padding='same', use_bias=False)(aspp5)
    aspp5 = BatchNormalization()(aspp5)
    aspp5 = Activation('relu')(aspp5)
    
    # Use UpSampling2D with correct size instead of tf.image.resize
    # Get the spatial dimensions from x
    x_shape = tf.keras.backend.int_shape(x)
    height, width = x_shape[1], x_shape[2]
    
    # Calculate the upsampling size dynamically
    pool_height = 1  # Because of GlobalAveragePooling2D with keepdims=True
    pool_width = 1   # Because of GlobalAveragePooling2D with keepdims=True
    height_scale = height // pool_height
    width_scale = width // pool_width
    
    # Use UpSampling2D (a proper Keras layer) instead of ResizeLayer
    aspp5 = UpSampling2D(size=(height_scale, width_scale), interpolation='bilinear')(aspp5)

    # Concatenate ASPP features
    x = Concatenate()([aspp1, aspp2, aspp3, aspp4, aspp5])

    # 1x1 Conv after ASPP
    x = Conv2D(256, 1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # Get spatial dimensions of low_level_features for proper upsampling
    low_level_shape = tf.keras.backend.int_shape(low_level_features)
    low_level_height, low_level_width = low_level_shape[1], low_level_shape[2]
    
    # Calculate upsampling scale to match low_level_features dimensions
    x_shape = tf.keras.backend.int_shape(x)
    height_scale = low_level_height // x_shape[1]
    width_scale = low_level_width // x_shape[2]
    
    # Upsampling to match low_level_features dimensions
    x = UpSampling2D(size=(height_scale, width_scale), interpolation='bilinear')(x)

    # Process low level features
    low_level_features = Conv2D(48, 1, padding='same', use_bias=False)(low_level_features)
    low_level_features = BatchNormalization()(low_level_features)
    low_level_features = Activation('relu')(low_level_features)

    # Concatenate upsampled features with low level features (now they have same dimensions)
    x = Concatenate()([x, low_level_features])

    # Final convolutions
    x = Conv2D(256, 3, padding='same')(x)  # Removed activation, will add after BatchNorm
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(256, 3, padding='same')(x)  # Removed activation, will add after BatchNorm
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    # Upsampling to match input resolution
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    # Final 1x1 conv to get class predictions - CHANGED TO OUTPUT 1 CHANNEL
    outputs = Conv2D(num_classes, 1, activation='sigmoid')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model


# ============== TRAINING ==============

def train_model(X_train, X_val, Y_train, Y_val):
    """Train the DeepLabv3+ model"""
    
    # Create model
    model = DeepLabV3Plus(num_classes=1)
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)]
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            f'{base_dir}/model/deeplab_walls_best.h5',
            save_best_only=True,
            monitor='val_mean_io_u',
            mode='max'
        ),
        EarlyStopping(
            patience=10,
            monitor='val_mean_io_u',
            mode='max',
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            factor=0.1,
            patience=5,
            min_lr=1e-6,
            monitor='val_mean_io_u',
            mode='max'
        )
    ]
    
    # Train
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Save final model
    model.save(f'{base_dir}/model/deeplab_walls_final.h5')
    
    return model, history

# ============== VISUALIZATION AND PREDICTION ==============

def visualize_results(model, X_val, Y_val, num_samples=5):
    """Visualize prediction results"""
    
    # Select random samples
    indices = np.random.choice(range(len(X_val)), num_samples, replace=False)
    
    plt.figure(figsize=(15, 5*num_samples))
    
    for i, idx in enumerate(indices):
        # Original image
        plt.subplot(num_samples, 3, i*3+1)
        plt.imshow(X_val[idx])
        plt.title("Original Image")
        plt.axis('off')
        
        # Ground truth mask
        plt.subplot(num_samples, 3, i*3+2)
        plt.imshow(Y_val[idx, :, :, 0], cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')
        
        # Predicted mask
        pred_mask = model.predict(np.expand_dims(X_val[idx], axis=0))[0, :, :, 0]
        pred_mask = (pred_mask > 0.5).astype(np.float32)
        
        plt.subplot(num_samples, 3, i*3+3)
        plt.imshow(pred_mask, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{base_dir}/results_visualization.png')
    plt.show()
    
def predict_on_new_image(model, image_path):
    """Predict wall mask on a new image"""
    
    # Read and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_norm = img / 255.0
    
    # Predict
    pred_mask = model.predict(np.expand_dims(img_norm, axis=0))[0, :, :, 0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    
    # Visualize
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask, cmap='gray')
    plt.title("Predicted Wall Mask")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return pred_mask


def download_model(model_path):
    """Simulates downloading by copying model to a downloads folder"""
    os.makedirs('downloads', exist_ok=True)
    shutil.copy(model_path, 'downloads')
    print("Model copied to downloads folder.")


    
# ============== MAIN EXECUTION ==============

def main():
    """Main execution"""
    image_dir = f'{base_dir}/plans'
    mask_dir = f'{base_dir}/walls'
    images, masks = load_data(image_dir, mask_dir, IMG_HEIGHT, IMG_WIDTH)
    
    # Split dataset
    X_train, X_val, Y_train, Y_val = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )

    # Preprocess data
    X_train = preprocess_input(X_train)
    X_val = preprocess_input(X_val)
    Y_train = preprocess_labels(Y_train)
    Y_val = preprocess_labels(Y_val)

    # Train model
    model, history = train_model(X_train, X_val, Y_train, Y_val)

    # Visualize predictions
    print("Visualizing results...")
    visualize_results(model, X_val, Y_val)

    final_model_path = f'{base_dir}/model/deeplab_walls_final.h5'
    print("Training complete! Model saved to:", final_model_path)

    # Ask user if they want to download
    print("Do you want to download the trained model? (y/n)")
    download_choice = input()
    if download_choice.lower() == 'y':
        download_model(final_model_path)

if __name__ == "__main__":
    main()


