import keras
import tensorflow
import matplotlib.pyplot as plt
import os
from PIL import Image  
import numpy

#python3 -m pip install --upgrade pip
#python3 -m pip install --upgrade Pillow

folder_list_train = os.listdir('D:\Python_Course\DL\day3-4hrs\jute_pest_deep_learning\dataset\Jute_Pest_Dataset\Jute_Pest_Dataset\\test')
folder_list_test = os.listdir('D:\Python_Course\DL\day3-4hrs\jute_pest_deep_learning\dataset\Jute_Pest_Dataset\Jute_Pest_Dataset\\test')
folder_list_val = os.listdir('D:\Python_Course\DL\day3-4hrs\jute_pest_deep_learning\dataset\Jute_Pest_Dataset\Jute_Pest_Dataset\\val')

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
            #original -640,640
            img_resized = img.resize((64,64), Image.LANCZOS)
            Xtrain.append(img_resized)
            ytrain.append(folder_name)

def reading_images_val(folder_name):
    for filename in os.listdir(folder_name):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_name, filename)
            img = Image.open(image_path)
            img_resized = img.resize((64,64), Image.LANCZOS)
            Xval.append(img_resized)
            yval.append(folder_name)

def reading_images_test(folder_name):
    for filename in os.listdir(folder_name):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_name, filename)
            img = Image.open(image_path)
            img_resized = img.resize((64,64), Image.LANCZOS)
            Xtest.append(img_resized)
            ytest.append(folder_name)


for name in folder_list_train:
        print(name)
        if name == 'Beet Armyworm' or name == 'Black Hairy':
            base_path = 'dataset\Jute_Pest_Dataset\Jute_Pest_Dataset\\train'
            folder_name = os.path.join(base_path, name)
            reading_images_train(folder_name)

for name in folder_list_test:
        print(name)
        if name == 'Beet Armyworm' or name == 'Black Hairy':
            base_path = 'dataset\Jute_Pest_Dataset\Jute_Pest_Dataset\\test'
            folder_name = os.path.join(base_path, name)
            reading_images_test(folder_name)

for name in folder_list_val:
        print(name)
        if name == 'Beet Armyworm' or name == 'Black Hairy':
            base_path = 'dataset\Jute_Pest_Dataset\Jute_Pest_Dataset\\val'
            folder_name = os.path.join(base_path, name)
            reading_images_val(folder_name)


print('train data', len(Xtrain))
print('val data', len(Xval))
print('test data', len(Xtest))

Xtrain = numpy.array(Xtrain)
Xtest = numpy.array(Xtest)
Xval = numpy.array(Xval)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
ytrain = le.fit_transform(ytrain)
ytest = le.fit_transform(ytest)
yval = le.fit_transform(yval)

#normalization - to bring 0 to 1
Xtrain = Xtrain/255.0
Xval = Xval/255.0
Xtest = Xtest/255.0

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Normalization, Dense, Dropout

juteAnn = Sequential()
juteAnn.add(Flatten())
juteAnn.add(Dense(units=128, activation='relu'))
juteAnn.add(Dense(units=128, activation='relu'))
juteAnn.add(Dense(units=1, activation='sigmoid'))
juteAnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
mc = ModelCheckpoint(filepath='bestmodel.keras', monitor='val_loss', save_best_only=True)
lr = ReduceLROnPlateau(monitor='val_loss', patience=3)

history = juteAnn.fit(Xtrain, ytrain, validation_data=(Xval, yval), epochs=10, callbacks=[mc, lr, es])

from tensorflow.keras.models import load_model

bestmodel = load_model('bestmodel.keras')

bestmodel.evaluate(Xtest, ytest)

plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()

'''
https://stackoverflow.com/questions/68776790/model-predict-classes-is-deprecated-what-to-use-instead
'''

ytest_pred = numpy.argmax(bestmodel.predict(Xtest), axis=1)