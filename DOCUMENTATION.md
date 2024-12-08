# Chest X-Ray Pneumonia Detector - Detailed Documentation

## Project Overview

This project implements a deep learning model to detect pneumonia from chest X-ray images. It uses transfer learning with a pre-trained ResNet18 model and provides a user-friendly web interface for real-time predictions.

## Technical Implementation

### 1. Data Processing (`model.py`)

The project uses the Chest X-Ray Images dataset from Kaggle with the following specifications:

#### Dataset Structure
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

#### Image Preprocessing
- Resize to 224x224 pixels
- Convert to RGB format
- Normalize using ImageNet statistics:
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

### 2. Model Architecture

Based on ResNet18 with modifications:
- Pre-trained weights from ImageNet
- Modified final fully connected layer for binary classification
- Input shape: (3, 224, 224)
- Output shape: (2) [Normal, Pneumonia]

Training Parameters:
- Optimizer: Adam (lr=0.001)
- Loss Function: Cross Entropy Loss
- Batch Size: 32
- Epochs: 5
- Train/Val Split: 80/20

### 3. Web Interface (`app.py`)

Built using Gradio with:
- Drag-and-drop image upload
- Real-time predictions
- Probability scores for both classes
- Example images included

## Test Cases

### Example Images

1. Normal Case:
```
Location: example_images/normal1.jpeg
Expected: High probability for "Normal" class
Features:
- Clear lung fields
- No obvious consolidation
- Normal cardiac silhouette
```

2. Pneumonia Case:
```
Location: example_images/pneumonia1.jpeg
Expected: High probability for "Pneumonia" class
Features:
- Patchy consolidation
- Infiltrates visible
- Possible pleural effusion
```

### Image Requirements
- Format: JPEG/PNG
- Recommended size: At least 224x224 pixels
- Type: Chest X-ray (PA view)
- Quality: Clear, properly exposed radiograph

## Project Structure

```
xray_pneumonia_detector/
├── model.py           # Model architecture and training
├── app.py            # Gradio web interface
├── requirements.txt  # Project dependencies
├── example_images/   # Example X-ray images
├── README.md        # Project overview
└── DOCUMENTATION.md # Detailed documentation
```

## Development Process

1. Model Development
   - Implemented ResNet18 architecture
   - Added custom classification head
   - Applied transfer learning
   - Fine-tuned on medical images

2. Training Pipeline
   - Custom dataset class
   - Data augmentation
   - Model checkpointing
   - Validation monitoring

3. Interface Development
   - Image upload handling
   - Prediction visualization
   - Example case integration

## Model Performance Metrics

```
Validation Metrics:
- Accuracy: ~90%
- Sensitivity: ~92%
- Specificity: ~88%
- F1 Score: ~90%
```

## Hosted Application

The Chest X-Ray Pneumonia Detector is now available online. Access it at:

[https://huggingface.co/spaces/TyJensen/xray-pneumonia-detector](https://huggingface.co/spaces/TyJensen/xray-pneumonia-detector)

This deployment allows users to upload X-ray images and receive instant predictions on the likelihood of pneumonia.

## Future Improvements

1. Model Enhancements
   - Try different architectures (DenseNet, EfficientNet)
   - Implement ensemble methods
   - Add data augmentation techniques

2. Interface Features
   - Add heatmap visualization
   - Include batch processing
   - Provide detailed analysis reports

3. Additional Capabilities
   - Multi-class classification
   - Severity scoring
   - Integration with DICOM format

## Troubleshooting

Common issues and solutions:

1. CUDA Out of Memory
```python
# Reduce batch size in model.py
batch_size = 16  # Default: 32
```

2. Image Loading Issues
```python
# Check image format
supported_formats = ['.jpeg', '.jpg', '.png']
```

3. Model Loading Error
```python
# Ensure correct model path
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
```

## References

1. Dataset: [Kaggle Chest X-Ray Images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
2. ResNet Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
3. Transfer Learning: [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## Usage Examples

### Command Line
```bash
# Train model
python model.py

# Start web interface
python app.py
```

### Python API
```python
from model import create_model
from PIL import Image

# Load model
model = create_model()
model.load_state_dict(torch.load('best_model.pth'))

# Make prediction
image = Image.open('example.jpg')
prediction = predict(image)
