import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Load and preprocess image
def load_image(img_path, max_size=400):
    image = Image.open(img_path).convert('RGB')

    # Resize keeping aspect ratio
    size = min(max(image.size), max_size)

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image

# Load your content and style image
content = load_image("image.jpg")
style = load_image("painting.jpg")

# Load pretrained VGG19 model
vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content = content.to(device)
style = style.to(device)
vgg.to(device)

# Define layers to use
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # content
            '28': 'conv5_1'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Calculate Gram Matrix for style comparison
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Extract features from content and style
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# Initialize target image
target = content.clone().requires_grad_(True).to(device)

# Style transfer weights
style_weights = {
    'conv1_1': 1.0,
    'conv2_1': 0.75,
    'conv3_1': 0.2,
    'conv4_1': 0.2,
    'conv5_1': 0.2
}
content_weight = 1e4
style_weight = 1e2

# Optimizer
optimizer = optim.Adam([target], lr=0.003)

# Run style transfer
steps = 200
for i in range(steps):
    target_features = get_features(target, vgg)

    # Content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

    # Style loss
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        style_feature = style_features[layer]

        target_gram = gram_matrix(target_feature)
        style_gram = gram_matrix(style_feature)

        layer_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        style_loss += layer_loss / (target_feature.shape[1] * target_feature.shape[2])

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"Step {i}, Total loss: {total_loss.item()}")

# Convert tensor to image
final_img = target.to("cpu").clone().detach().squeeze()
final_img = transforms.ToPILImage()(final_img)

final_img.save("styled_output.jpg")

