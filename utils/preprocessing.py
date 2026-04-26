from torchvision import transforms
from PIL import Image

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),   # match training size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    transform = get_transform()
    return transform(image).unsqueeze(0)  # add batch dimension
