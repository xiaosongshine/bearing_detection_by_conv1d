
import keras
from scipy.io import loadmat
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
import math
import os
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import numpy as np

MANIFEST_DIR = "Bear_data/train.csv"
Batch_size = 20
Long = 792
Lens = 640
#把标签转成oneHot
def convert2oneHot(index,Lens):
    hot = np.zeros((Lens,))
    hot[int(index)] = 1
    return(hot)

def xs_gen(path=MANIFEST_DIR,batch_size = Batch_size,train=True,Lens=Lens):

    img_list = pd.read_csv(path)
    if train:
        img_list = np.array(img_list)[:Lens]
        print("Found %s train items."%len(img_list))
        print("list 1 is",img_list[0,-1])
        steps = math.ceil(len(img_list) / batch_size)    # 确定每轮有多少个batch
    else:
        img_list = np.array(img_list)[Lens:]
        print("Found %s test items."%len(img_list))
        print("list 1 is",img_list[0,-1])
        steps = math.ceil(len(img_list) / batch_size)    # 确定每轮有多少个batch
    while True:
        for i in range(steps):

            batch_list = img_list[i * batch_size : i * batch_size + batch_size]
            np.random.shuffle(batch_list)
            batch_x = np.array([file for file in batch_list[:,1:-1]])
            batch_y = np.array([convert2oneHot(label,10) for label in batch_list[:,-1]])

            yield batch_x, batch_y

TEST_MANIFEST_DIR = "Bear_data/test_data.csv"

def ts_gen(path=TEST_MANIFEST_DIR,batch_size = Batch_size):

    img_list = pd.read_csv(path)

    img_list = np.array(img_list)[:Lens]
    print("Found %s train items."%len(img_list))
    print("list 1 is",img_list[0,-1])
    steps = math.ceil(len(img_list) / batch_size)    # 确定每轮有多少个batch
    while True:
        for i in range(steps):

            batch_list = img_list[i * batch_size : i * batch_size + batch_size]
            #np.random.shuffle(batch_list)
            batch_x = np.array([file for file in batch_list[:,1:]])
            #batch_y = np.array([convert2oneHot(label,10) for label in batch_list[:,-1]])

            yield batch_x



TIME_PERIODS = 6000
def build_model(input_shape=(TIME_PERIODS,),num_classes=10):
    model = Sequential()
    model.add(Reshape((TIME_PERIODS, 1), input_shape=input_shape))
    model.add(Conv1D(16, 8,strides=2, activation='relu',input_shape=(TIME_PERIODS,1)))

    model.add(Conv1D(16, 8,strides=2, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(64, 4,strides=2, activation='relu',padding="same"))
    model.add(Conv1D(64, 4,strides=2, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(256, 4,strides=2, activation='relu',padding="same"))
    model.add(Conv1D(256, 4,strides=2, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(512, 2,strides=1, activation='relu',padding="same"))
    model.add(Conv1D(512, 2,strides=1, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    """model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))"""
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    return(model)

Train = True

if __name__ == "__main__":
    if Train == True:
        train_iter = xs_gen()
        val_iter = xs_gen(train=False)

        ckpt = keras.callbacks.ModelCheckpoint(
            filepath='best_model.{epoch:02d}-{val_loss:.4f}.h5',
            monitor='val_loss', save_best_only=True,verbose=1)

        model = build_model()
        opt = Adam(0.0002)
        model.compile(loss='categorical_crossentropy',
                    optimizer=opt, metrics=['accuracy'])
        print(model.summary())

        model.fit_generator(
            generator=train_iter,
            steps_per_epoch=Lens//Batch_size,
            epochs=50,
            initial_epoch=0,
            validation_data = val_iter,
            nb_val_samples = (Long - Lens)//Batch_size,
            callbacks=[ckpt],
            )
        model.save("finishModel.h5")
    else:
        test_iter = ts_gen()
        model = load_model("best_model.49-0.00.h5")
        pres = model.predict_generator(generator=test_iter,steps=math.ceil(528/Batch_size),verbose=1)
        print(pres.shape)
        ohpres = np.argmax(pres,axis=1)
        print(ohpres.shape)
        #img_list = pd.read_csv(TEST_MANIFEST_DIR)
        df = pd.DataFrame()
        df["id"] = np.arange(1,len(ohpres)+1)
        df["label"] = ohpres
        df.to_csv("submmit.csv",index=None)
        test_iter = ts_gen()
        for x in test_iter:
            x1 = x[0]
            break
        plt.plot(x1)
        plt.show()

