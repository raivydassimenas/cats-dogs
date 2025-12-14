import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import ToTensor

fixed_size = (64, 64)

transform = transforms.Compose([transforms.Resize(fixed_size), ToTensor()])

class NeuralNetwork(nn.Model):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64 * 64 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 37),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
state_dict = torch.load("cats_dogs_model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

from PIL import Image

img_path = "data/oxford-iiit-pets/images/Abyssinian_1.jpg"
image = Image.open(img_path).convert("RGB")

input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0).to("cpu")

with torch.no_grad():
    outputs = model(input_batch)
    probs = torch.softmax(outputs, dim=1)
    confidence, predicted_class = toch.max(probs, dim=1)

from tochvision.datasets import OxfordIIITPet

dataset = OxfordIIIPet(
        root=".",
        split="trainval",
        downnload=False
)

class_names = dataset.classes

predicted_breed = class_names[predicted_class.item()]

print(f"Prediction: {predicted_breed}")
print(f"Confidence: {confidence.item():.2%}")
