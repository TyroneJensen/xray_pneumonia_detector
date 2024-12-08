import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import create_model
import torch.nn.functional as F

# Load the trained model
def load_model():
    model = create_model()
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(image):
    model = load_model()
    
    # Convert to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Preprocess the image
    input_tensor = preprocess_image(image)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        
    return {
        "Normal": float(probabilities[0]),
        "Pneumonia": float(probabilities[1])
    }

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    examples=[
        ["example_images/normal1.jpeg"],
        ["example_images/pneumonia1.jpeg"]
    ],
    title="Chest X-Ray Pneumonia Detector",
    description="Upload a chest X-ray image to detect the presence of pneumonia."
)

if __name__ == "__main__":
    iface.launch()
