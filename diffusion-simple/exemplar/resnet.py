import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path, preprocess, device):
    image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    return image

def extract_clip_rn50_features(model, input_image, layers_to_extract=None):
    visual_model = model.visual  # ResNet50 část CLIPu

    if layers_to_extract is None:
        layers_to_extract = ['relu3', 'layer1', 'layer2', 'layer3']

    feature_maps = {}

    def get_activation(name):
        def hook(model, input, output):
            feature_maps[name] = output
        return hook

    hooks = []
    for name in layers_to_extract:
        if name == 'relu3':
            layer = visual_model.relu3
        else:
            layer = getattr(visual_model, name)
        hooks.append(layer.register_forward_hook(get_activation(name)))

    _ = visual_model(input_image)

    for hook in hooks:
        hook.remove()

    return feature_maps

def visualize_feature_maps(feature_maps, num_features=4):
    plt.figure(figsize=(15, 10))

    for i, (layer_name, feature_map) in enumerate(feature_maps.items()):
        for j in range(min(num_features, feature_map.size(1))):
            plt.subplot(len(feature_maps), num_features, i * num_features + j + 1)
            plt.imshow(feature_map[0, j].cpu().detach().numpy(), cmap='viridis')
            plt.title(f"{layer_name} - ch{j}")
            plt.axis('off')

    plt.tight_layout()
    plt.savefig('RESNET_feature_maps.png')
    plt.show()

def main(image_path="sample_image.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device)

    image_tensor = load_and_preprocess_image(image_path, preprocess, device)
    selected_layers = ['relu3', 'layer1', 'layer2', 'layer3']

    feature_maps = extract_clip_rn50_features(model, image_tensor, selected_layers)

    print("Feature map sizes")
    for layer_name, feature_map in feature_maps.items():
        print(f"{layer_name}: {feature_map.shape}")

    visualize_feature_maps(feature_maps)

if __name__ == "__main__":
    main("sample_image.png")