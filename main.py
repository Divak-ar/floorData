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

# Add this near the top of your script to enable memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    # For CPU training - limit memory usage
    # (needed because you're hitting CPU memory limits)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    

import tensorflow as tf
# Configure TensorFlow to use less memory
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)


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

def load_data(mask_dir, img_height, img_width):
    """Load only wall mask data and use it as both input and output"""
    masks = []
    mask_files = sorted(os.listdir(mask_dir))
    
    for mask_file in mask_files:
        # Load masks
        mask_path = os.path.join(mask_dir, mask_file)
        mask = load_img(mask_path, target_size=(img_height, img_width), color_mode='grayscale')
        mask = img_to_array(mask)
        masks.append(mask)
    
    return np.array(masks)

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
    """Create a DeepLabv3+ model with proper handling for grayscale input"""

    # For grayscale input, we need to convert to RGB in a Keras-compatible way
    if input_shape[2] == 1:
        # Create the input layer for grayscale
        inputs = Input(shape=input_shape)
        
        # Create a Lambda layer to duplicate channels (instead of tf.concat)
        def duplicate_channels(x):
            return tf.keras.backend.repeat_elements(x, 3, axis=-1)
            
        x = layers.Lambda(duplicate_channels)(inputs)
        
        # Create a standalone ResNet50 model with 3-channel input
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(input_shape[0], input_shape[1], 3))
        
        # Process the RGB-converted input through the base model
        features = base_model(x)
        
        # Get low level features using the functional API
        low_level_features = base_model.get_layer('conv2_block3_out')(x)
        
    else:
        # Normal RGB input case
        inputs = Input(shape=input_shape)
        base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
        features = base_model.output
        low_level_features = base_model.get_layer('conv2_block3_out').output

    # ASPP (Atrous Spatial Pyramid Pooling)
    # 1x1 Conv
    aspp1 = Conv2D(256, 1, padding='same', use_bias=False)(features)
    aspp1 = BatchNormalization()(aspp1)
    aspp1 = Activation('relu')(aspp1)

    # 3x3 Conv rate 6
    aspp2 = Conv2D(256, 3, padding='same', dilation_rate=6, use_bias=False)(features)
    aspp2 = BatchNormalization()(aspp2)
    aspp2 = Activation('relu')(aspp2)

    # 3x3 Conv rate 12
    aspp3 = Conv2D(256, 3, padding='same', dilation_rate=12, use_bias=False)(features)
    aspp3 = BatchNormalization()(aspp3)
    aspp3 = Activation('relu')(aspp3)

    # 3x3 Conv rate 18
    aspp4 = Conv2D(256, 3, padding='same', dilation_rate=18, use_bias=False)(features)
    aspp4 = BatchNormalization()(aspp4)
    aspp4 = Activation('relu')(aspp4)

    # Image pooling - FIXED IMPLEMENTATION
    aspp5 = KerasGlobalAveragePooling2D(keepdims=True)(features)
    aspp5 = Conv2D(256, 1, padding='same', use_bias=False)(aspp5)
    aspp5 = BatchNormalization()(aspp5)
    aspp5 = Activation('relu')(aspp5)
    
    # Use UpSampling2D with correct size
    feature_shape = tf.keras.backend.int_shape(features)
    height, width = feature_shape[1], feature_shape[2]
    aspp5 = UpSampling2D(size=(height, width), interpolation='bilinear')(aspp5)

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

    # Concatenate upsampled features with low level features
    x = Concatenate()([x, low_level_features])

    # Final convolutions
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    # Calculate the final upsampling factor to match the input resolution
    x_shape = tf.keras.backend.int_shape(x)
    # This is the key fix - ensure we upsample to exactly the input dimensions
    height_scale = input_shape[0] // x_shape[1]
    width_scale = input_shape[1] // x_shape[2]
    x = UpSampling2D(size=(height_scale, width_scale), interpolation='bilinear')(x)

    # Final 1x1 conv to get class predictions
    outputs = Conv2D(num_classes, 1, activation='sigmoid')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# ============== TRAINING ==============
def train_model(X_train, X_val, Y_train, Y_val):
    """Train the DeepLabv3+ model"""
    
    input_shape = X_train.shape[1:]
    
    # Create model
    model = DeepLabV3Plus(input_shape=input_shape, num_classes=1)
    
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
    """Predict wall mask on a new image and extract wall coordinates"""
    
    # Read and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_norm = img / 255.0
    
    # Predict
    pred_mask = model.predict(np.expand_dims(img_norm, axis=0))[0, :, :, 0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    
    # Extract wall coordinates
    wall_segments = extract_wall_coordinates(pred_mask)
    
    # Print wall coordinates
    print(f"\n=== Found {len(wall_segments)} wall segments ===")
    for i, wall in enumerate(wall_segments):
        print(f"Wall #{i+1} - Length: {wall['length']:.1f} pixels")
        print(f"  Coordinates: {wall['points']}")
    
    # Visualize
    plt.figure(figsize=(15, 8))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Floor Plan")
    plt.axis('off')
    
    # Predicted mask
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, cmap='gray')
    plt.title("Predicted Wall Mask")
    plt.axis('off')
    
    # Overlay with wall coordinates
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    
    # Draw the wall segments
    overlay_img = img.copy()
    for wall in wall_segments:
        # Draw lines connecting wall points
        points = wall['points']
        for i in range(len(points)-1):
            pt1 = (points[i]['x'], points[i]['y'])
            pt2 = (points[i+1]['x'], points[i+1]['y'])
            cv2.line(overlay_img, pt1, pt2, (255, 0, 0), 2)
        
        # Draw first point in green to mark start
        if len(points) > 0:
            cv2.circle(overlay_img, (points[0]['x'], points[0]['y']), 5, (0, 255, 0), -1)
    
    plt.imshow(overlay_img)
    plt.title(f"Extracted Wall Coordinates ({len(wall_segments)} segments)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Convert wall segments to JSON-friendly format for 3D visualization
    wall_data = {
        "walls": wall_segments,
        "imageWidth": IMG_WIDTH,
        "imageHeight": IMG_HEIGHT,
        "scale": 0.02  # 1 pixel = 2cm (example scale)
    }
    
    return pred_mask, wall_data

def save_wall_coordinates(wall_data, output_path="wall_coordinates.json"):
    """Save wall coordinates to JSON file for 3D visualization"""
    import json
    
    with open(output_path, 'w') as f:
        json.dump(wall_data, f, indent=2)
    
    print(f"Wall coordinates saved to {output_path}")

def download_model(model_path):
    """Simulates downloading by copying model to a downloads folder"""
    os.makedirs('downloads', exist_ok=True)
    shutil.copy(model_path, 'downloads')
    print("Model copied to downloads folder.")
    
def extract_wall_coordinates(pred_mask, min_area=10, min_length=15):
    """
    Extract wall coordinates from the predicted mask
    
    Args:
        pred_mask: Binary prediction mask (2D array)
        min_area: Minimum contour area to consider (filters noise)
        min_length: Minimum wall length to consider
        
    Returns:
        List of dictionaries containing wall coordinates
    """
    from skimage.morphology import skeletonize
    
    # Ensure we have a binary mask
    binary_mask = (pred_mask > 0).astype(np.uint8)
    
    # Clean the mask
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Get the centerlines via skeletonization
    skeleton = skeletonize(binary_mask > 0).astype(np.uint8) * 255
    
    # Use Hough transform to get line segments
    lines = cv2.HoughLinesP(
        skeleton, 
        rho=1, 
        theta=np.pi/180, 
        threshold=10, 
        minLineLength=min_length, 
        maxLineGap=10
    )
    
    wall_segments = []
    
    if lines is not None:
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            wall_segments.append({
                "id": i,
                "points": [
                    {"x": int(x1), "y": int(y1)},
                    {"x": int(x2), "y": int(y2)}
                ],
                "length": float(length)
            })
    
    return wall_segments


    
# ============== MAIN EXECUTION ==============
def main():
    """Main execution"""
 
    mask_dir = f'{base_dir}/walls'
    masks = load_data(mask_dir, IMG_HEIGHT, IMG_WIDTH)
    
    # Split dataset
    X_train, X_val, Y_train, Y_val = train_test_split(
        masks, masks, test_size=0.2, random_state=42
    )

    # Preprocess data
    X_train = preprocess_input(X_train)
    X_val = preprocess_input(X_val)
    Y_train = preprocess_labels(Y_train)
    Y_val = preprocess_labels(Y_val)
    
    # Modify the model to accept grayscale input (1 channel)
    # Create model with 1 channel input and 1 channel output
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)
    model = DeepLabV3Plus(input_shape=input_shape, num_classes=1)

    # Train model
    print("Training model on wall data...")
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

# Test on a single image
# if __name__ == "__main__":
#     from tensorflow.keras.models import load_model
    
#     # Check if the model file exists
#     model_path = 'huggingface_dataset/model/deeplab_walls_best.h5'
#     if not os.path.exists(model_path):
#         print(f"Model file not found at {model_path}. Please train the model first.")
#     else:
#         # Load trained model
#         model = load_model(model_path)
        
#         # Process test image
#         test_image_path = 'huggingface_dataset/walls/00000001.jpg'  # 
#         if not os.path.exists(test_image_path):
#             print(f"Test image not found at {test_image_path}")
#         else:
#             # Predict and extract wall coordinates
#             prediction, wall_data = predict_on_new_image(model, test_image_path)
            
#             # Save wall coordinates to file
#             save_wall_coordinates(wall_data)
            
#             print("\nYou can now use the wall_coordinates.json file for 3D visualization in your frontend.")




