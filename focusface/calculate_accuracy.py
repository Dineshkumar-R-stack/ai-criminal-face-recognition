import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

# Load FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# Correct dataset path (based on your structure)
DATASET_PATH = "../data/target/faces"

THRESHOLD = 0.6

embeddings = []
labels = []

# Load images and create embeddings
for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)

    for img in os.listdir(person_path):
        img_path = os.path.join(person_path, img)

        try:
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                emb = model(image).numpy()[0]

            embeddings.append(emb)
            labels.append(person)

        except:
            print(f"Skipping {img_path}")

print(f"Total samples: {len(embeddings)}")

# Compare embeddings
correct = 0
total = 0

for i in range(len(embeddings)):
    test_emb = embeddings[i]
    true_label = labels[i]

    min_dist = float('inf')
    pred_label = None

    for j in range(len(embeddings)):
        if i == j:
            continue

        dist = np.linalg.norm(test_emb - embeddings[j])

        if dist < min_dist:
            min_dist = dist
            pred_label = labels[j]

    if min_dist > THRESHOLD:
        pred_label = "unknown"

    print(f"Actual: {true_label}, Predicted: {pred_label}, Distance: {min_dist:.4f}")

    if pred_label == true_label:
        correct += 1

    total += 1

accuracy = correct / total

print("\n====================")
print(f"Accuracy: {%.2f} %")
print("====================")