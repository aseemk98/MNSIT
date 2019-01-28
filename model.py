import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
import seaborn as sns
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D,Dense,Dropout,MaxPool2D,Flatten
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


#train and testing data
y_train = data["label"]
x_train = data.drop(labels='label',axis=1)

#check frequency of each label
g = sns.countplot(y_train)



# grayscale normalization
x_train = x_train/255.0
test = test/255.0

#resizing
x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

#categorical splitting 
y_train = to_categorical(y_train,num_classes=10)


np.random.seed(2)
random_seed=2
#train test split
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.1,random_state=random_seed)


#CNN Model

#Conv2D*2 -> MaxPooling -> Dense   (with drop out)

model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(filters=64,kernel_size=(5,5),strides=(1,1),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),activation='relu'))
model.add(Conv2D(filters=64,kernel_size=(5,5),strides=(1,1),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
#model.add(Dense(124,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer=Adam(), loss='categorical_crossentropy',metrics=['accuracy'])

datagen = ImageDataGenerator(
        featurewise_center=False,
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False
        
        )

datagen.fit(x_train)

batch_size=80

history = model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),epochs=50,steps_per_epoch=batch_size)

results = model.predict(test)

results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("result.csv",index=False)








