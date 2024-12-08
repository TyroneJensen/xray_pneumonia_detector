# Chest X-Ray Pneumonia Detector

A deep learning application that analyzes chest X-ray images to detect pneumonia. Built with PyTorch and Gradio.

## Features

- Upload and analyze chest X-ray images
- Real-time pneumonia detection
- Probability scores for normal and pneumonia cases
- Interactive web interface
- Pre-trained ResNet18 model fine-tuned on medical images

## Quick Start

1. Clone the repository
2. Download the dataset from [Kaggle Chest X-Ray Images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Train the model:
```bash
python model.py
```
5. Start the web interface:
```bash
python app.py
```
6. Open http://localhost:7860 in your browser

## Technology Stack

- Python 3.11
- PyTorch (Deep Learning)
- Torchvision (Image Processing)
- Gradio (Web Interface)
- ResNet18 (Pre-trained Model)

## Model Architecture

- Base: ResNet18 (pre-trained on ImageNet)
- Modified for binary classification
- Input size: 224x224 RGB images
- Output: Binary classification (Normal/Pneumonia)

## Dataset

Uses the Chest X-Ray Images (Pneumonia) dataset from Kaggle:
- 5,863 X-Ray images (JPEG)
- 2 categories: Normal and Pneumonia
- Training and testing sets

## Performance

- Validation Accuracy: ~90%
- Real-time inference
- Balanced precision and recall

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
