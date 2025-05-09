# 🏠 Floor Plan Wall Detection using DeepLabV3+ 🧱

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 📝 Overview

This project implements a deep learning model for automatic wall detection in architectural floor plans using the DeepLabV3+ architecture. The model is trained on the [zimhe/pseudo-floor-plan-12k](https://huggingface.co/datasets/zimhe/pseudo-floor-plan-12k) dataset from HuggingFace.

![Floor Plan Example](huggingface_dataset/results_visualization.png)

### 🌟 Key Features

- **Multiple Detection Methods**: UNet, DeepLabV3+, and direct image processing
- **Wall Coordinate Extraction**: Converts wall masks to line coordinates
- **3D Visualization Ready**: Outputs coordinates in JSON format for 3D rendering

## 🔄 Model Comparison

| Model | Advantages | Use Case |
|-------|------------|----------|
| **UNet** (UnetModel.ipynb) | 🚀 Faster training<br>🎯 Better accuracy<br>🧠 Lower memory usage | **Recommended** for all wall detection tasks |
| **DeepLabV3+** (main.py) | 🔍 Higher capacity<br>🌐 Pre-trained backbone | Complex floor plans with many details |
| **Direct Processing** | ⚡ No training required<br>📊 Works on binary images | When you already have wall masks |

## 🛠️ Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/Divak-ar/floorData.git
cd floorData
```

### Step 2: Create a virtual environment
```bash
# For Windows
python -m venv .venv
.venv\Scripts\activate

# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

## 📊 Dataset

The project uses the pseudo-floor-plan-12k dataset from HuggingFace. To download the dataset:

```bash
python dataDownload.py
```

This will:
- Download 100 floor plan images and their corresponding wall masks
- Save them in the `huggingface_dataset` directory

## 🚀 Usage

### Training the model
```bash
python main.py
```

The training process:
1. Loads and preprocesses floor plan images and wall masks
2. Splits data into training and validation sets
3. Trains a DeepLabV3+ model with ResNet50 backbone
4. Saves the best model during training
5. Visualizes prediction results

## Using UNet Model (Recommended)
Open and run the `UnetModel.ipynb` notebook for better wall detection results.

### Inference on new images
```python
from main import predict_on_new_image
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('huggingface_dataset/model/deeplab_walls_best.h5')

```

## 🔧 Model Architecture

The wall detection model uses a DeepLabV3+ architecture with:

- ResNet50 backbone pre-trained on ImageNet
- Atrous Spatial Pyramid Pooling (ASPP) module
- Decoder with skip connections from encoder
- Binary segmentation output (walls vs. background)

## 📈 Results

The UNet model typically achieves:
- Higher IoU (Intersection over Union) for wall segmentation
- Better line detection accuracy
- Faster inference time

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.