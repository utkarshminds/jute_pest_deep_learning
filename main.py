import keras
import tensorflow
import matplotlib.pyplot as plt
import os
from PIL import Image  

#python3 -m pip install --upgrade pip
#python3 -m pip install --upgrade Pillow

folder_list_train = os.listdir('jute+pest+dataset\Jute_Pest_Dataset\Jute_Pest_Dataset\train')
folder_list_test = os.listdir('jute+pest+dataset\Jute_Pest_Dataset\Jute_Pest_Dataset\test')
folder_list_val = os.listdir('jute+pest+dataset\Jute_Pest_Dataset\Jute_Pest_Dataset\val')

Xtrain = []
ytrain = []
Xval = []
yval = []
Xtest = []
ytest = []

def reading_images_train(folder_name):
    for filename in os.listdir(folder_name):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_name, filename)
            img = Image.open(image_path)
            img_resized = img.resize((640,640), Image.LANCZOS)
            Xtrain.append(img_resized)
            ytrain.append(folder_name)

def reading_images_val(folder_name):
    for filename in os.listdir(folder_name):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_name, filename)
            img = Image.open(image_path)
            img_resized = img.resize((640,640), Image.LANCZOS)
            Xval.append(img_resized)
            yval.append(folder_name)

def reading_images_test(folder_name):
    for filename in os.listdir(folder_name):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_name, filename)
            img = Image.open(image_path)
            img_resized = img.resize((640,640), Image.LANCZOS)
            Xtest.append(img_resized)
            ytest.append(folder_name)


for name in folder_list_train:
        base_path = 'jute+pest+dataset\Jute_Pest_Dataset\Jute_Pest_Dataset\train'
        folder_name = os.path.join(base_path, name)
        reading_images_train(folder_name)

for name in folder_list_test:
        base_path = 'jute+pest+dataset\Jute_Pest_Dataset\Jute_Pest_Dataset\test'
        folder_name = os.path.join(base_path, name)
        reading_images_test(folder_name)

for name in folder_list_val:
        base_path = 'jute+pest+dataset\Jute_Pest_Dataset\Jute_Pest_Dataset\val'
        folder_name = os.path.join(base_path, name)
        reading_images_val(folder_name)