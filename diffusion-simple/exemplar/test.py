import torch
import clip
from PIL import Image


model_cnn, preprocess_cnn = clip.load("RN50")


model_vit, preprocess_vit = clip.load("ViT-B/32")


device = "cuda" if torch.cuda.is_available() else "cpu"
model_cnn = model_cnn.to(device)
model_vit = model_vit.to(device)


image = Image.open("sample_image.png")
image_input_cnn = preprocess_cnn(image).unsqueeze(0).to(device)
image_input_vit = preprocess_vit(image).unsqueeze(0).to(device)

with torch.no_grad():
    image_features_cnn = model_cnn.encode_image(image_input_cnn)
    image_features_vit = model_vit.encode_image(image_input_vit)


image_features_cnn = image_features_cnn / image_features_cnn.norm(dim=1, keepdim=True)
image_features_vit = image_features_vit / image_features_vit.norm(dim=1, keepdim=True)

print(f"CNN : {image_features_cnn.shape}")
print(f"ViT : {image_features_vit.shape}")