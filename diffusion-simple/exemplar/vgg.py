import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchinfo import summary 

def load_and_preprocess_image(image_path, size=(224, 224)):

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor, image

def extract_vgg19_features(input_image, layers_to_extract=None):
    vgg_model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    vgg_model.eval()

    if layers_to_extract is None:
        layers_to_extract = [8, 17, 26, 35]
    
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output
        return hook
    
    hooks = []
    for layer_idx in layers_to_extract:
        layer = vgg_model.features[layer_idx]
        hooks.append(layer.register_forward_hook(get_features(f"layer_{layer_idx}")))
    
    with torch.no_grad():
        vgg_model(input_image)
    
    for hook in hooks:
        hook.remove()
    
    return features    

# def extract_vgg19_features(input_image, layers_to_extract=None):

#     vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
#     vgg19.eval()
    
#     # summary(vgg19, input_size=(1, 3, 224, 224))

#     if layers_to_extract is None:
#         layers_to_extract = []
    

#     feature_maps = {}

#     # print(vgg19.features)
    
#     with torch.no_grad():
#         x = input_image
#         for i, layer in enumerate(vgg19.features):
#             x = layer(x)
#             if i in layers_to_extract:
#                 feature_maps[f"layer_{i}"] = x
    
#     return feature_maps

def visualize_feature_maps(feature_maps, num_features=4):

    plt.figure(figsize=(15, 10))
    
    for i, (layer_name, feature_map) in enumerate(feature_maps.items()):

        for j in range(min(num_features, feature_map.size(1))):
            plt.subplot(len(feature_maps), num_features, i * num_features + j + 1)
            plt.imshow(feature_map[0, j].cpu().numpy(), cmap='viridis')
            plt.title(f"{layer_name} - ch{j}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('VGG_feature_maps.png')
    plt.show()

def main(image_path="sample_image.png"):

    image_tensor, original_image = load_and_preprocess_image(image_path)
    
    selected_layers = [8, 17, 26, 35]
    
    feature_maps = extract_vgg19_features(image_tensor, selected_layers)
    
    print("Feature map sizes")
    for layer_name, feature_map in feature_maps.items():
        print(f"{layer_name}: {feature_map.shape}")
    
    visualize_feature_maps(feature_maps)
    
        

if __name__ == "__main__":

    main("sample_image.png") 