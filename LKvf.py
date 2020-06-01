import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image

##############################################################
''' Image preprocessing:
- grayscale
- resize
'''
print("[INFO] loading images...")

image_data = np.loadtxt("train-data.csv", delimiter="\n", dtype=str)

for i in range(0,len(image_data)):
    if (i+1)%3 != 0:
        image_data[i] = image_data[i].replace(',','')
        image_data[i] = image_data[i].replace('\\','/')

#print image_data[i]
#print(image_data)

inputs = []
outputs = []

x = 0
while x < len(image_data)-4:
    #read in both input images
    img1 = Image.open(image_data[x]).convert(mode='L')
    img2 = Image.open(image_data[x+ 1]).convert(mode='L')
    #stack them together vertically
    vis = np.concatenate((img1, img2), axis=0) #vertical stacking with axis=0, horiz is axis=1
    #resize the new stacked image
    vis.resize((28,28))
    #vis = img_to_array(vis)
    inputs.append(vis)
    #print (image_data[x+2])
    label = [float(i) for i in image_data[x+2].split(',') if i.isdigit() ]
    outputs.append(label)

    x+=3

#more preprocessing
# scale the raw pixel intensities to the range [0, 1]
inputs = np.array(inputs, dtype="float") / 255.0
outputs = np.array(outputs)
print("input shape:", inputs.shape)
print("output shape: ", outputs.shape)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(inputs, outputs,
test_size=0.25)
print(trainX.shape)

# convert the labels from integers to vectors
trainX = np.reshape(trainX, (trainX.shape[0],28,28,-1))
print("trainx shape:", trainX.shape)
print("trainy shape: ", trainY.shape)

testX = np.reshape(testX, (testX.shape[0],28,28,-1))
print("testx shape:", testX.shape)
print("testy shape:", testY.shape) #lol
#trainY = to_categorical(trainY, num_classes=None)
#testY = to_categorical(testY, num_classes=None)


##############################################################
# initialize the model
bs = 10
print("[INFO] compiling model...")
model = Sequential()
model.add(Conv2D(32, (5, 5), data_format='channels_last',
input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
print("[x ]")

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
print("[xx ]")

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape)
# the model so far outputs 3D feature maps (height, width, features)
print("[xxx ]")

model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors
print(model.output_shape)
model.add(Dense(64))
model.add(Activation('relu'))
print("[xxxx ]")

model.add(Dense(32))
model.add(Activation('relu'))
print("[xxxxx ]")

model.add(Dropout(0.5))
model.add(Dense(6))
model.add(Activation('sigmoid')) #sigmoid
print("[xxxxxx ]")

model.compile(loss='binary_crossentropy', optimizer='sgd',
metrics=['accuracy']) #mean_squared_error, optimizer sgd
print("[xxxxxxx]")


##################################################################
print("[INFO] fitting model...")
# Fit the model
history = model.fit(trainX,trainY, epochs=100, batch_size=bs, verbose=1)

#record the history of the fitted model to graph accuracies and stuff later
pickle.dump(history.history, open('save4.p','wb'))


##################################################################
#print "[INFO] running predictions..."
# calculate predictions
#y_pred= model.predict(testX)

print("[INFO] evaluating model...")
score_train = model.evaluate(trainX, trainY, batch_size=bs, verbose=1)
print("Accuracy for trained set: ", score_train[1]*100)

score_test = model.evaluate(testX, testY, batch_size=bs, verbose=1)
print("Accuracy for test set: ", score_test[1]*100)

##################################################################

# save the model to disk
print("[INFO] saving network weights...")

# serialize model to YAML
model_yaml = model.to_yaml()
with open("model4.yaml", "w") as yaml_file:yaml_file.write(model_yaml)

# serialize weights to HDF5
model.save_weights("model4.h5")
print("Saved model to disk")


###############################################################
print("Yay!")