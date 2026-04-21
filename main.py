import PIL.ImageShow
import torch
from models.CNN_4 import CNN4
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN4()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

transform = A.Compose([
    A.LongestMaxSize(224),
    A.PadIfNeeded(224, 224, border_mode=0),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    PIL.ImageShow.show(image)
    image = np.array(image)

    image = transform(image=image)["image"]
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()

    return pred

label = predict("data/images/Abyssinian_102.jpg")
class_names = ['Abyssinian', 'american bulldog', 'american pitbull', 'basset', 'beagle', 'Bengal', 'Birman', 'Bombay', 'boxer', 'British', 'chihuahua', 'Egyptian', 'cocker_spaniel', 'english', 'german', 'great', 'havanese', 'japanese', 'keeshond', 'leonberger', 'Maine', 'miniature', 'newfoundland', 'Persian', 'pomeranian', 'pug', 'Ragdoll', 'Russian', 'saint', 'samoyed', 'scottish', 'shiba', 'Siamese', 'Sphynx', 'staffordshire', 'wheaten', 'yorkshire']
print("Predicted class:", label,"-", class_names[label])