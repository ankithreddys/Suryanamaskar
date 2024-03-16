import os
import random
from PIL import Image
from torchvision.io import read_image
import torchvision.transforms as transforms
import re



def augmentation(path):
    img = read_image(path)
    transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation((0,360)),
        transforms.ColorJitter()]
    )

    img = transformations(img)
    return img

    
    
    
    
    
    
    

if __name__ == "__main__":
    for root, dirs, files in os.walk('C:/Users/ankit/Music/suryanamaskar/data_suryanamaskar', topdown=True):
        print(root)
        for file in files:
            path = os.path.join(root, file)
            file_name, _ = os.path.splitext(file)
            for i in range(10):
                img = augmentation(path)
                new_filename = f"{file_name}_aug_{i+1}.png"
                img.save(os.path.join(root, new_filename))

