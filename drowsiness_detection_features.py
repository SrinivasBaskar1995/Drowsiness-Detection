import keras
from keras.layers import TimeDistributed, Dense, Dropout, LSTM
import load_data_features
import random
import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.isdir("../chkp/"):
    os.makedirs("../chkp/")

#load train data
train_data = load_data_features.load(True,False)
train_data = list(zip(train_data[0], train_data[1]))
random.shuffle(train_data)
x_train, y_train = zip(*train_data)

#number of features
num_features = len(x_train[0][0])

#number of classes
try:
    output_classes = len(y_train[0])
    loss_fn = 'categorical_crossentropy'
    metric = "accuracy"
except:
    output_classes = 1
    loss_fn = 'mse'
    metric = "mse"
    
#load test data
test_data = load_data_features.load(False,False)

#function to regularize a give array
def regularize(array):
    temp = []
    for item in array:
        temp.append(item/sum(array))
    return temp

#models
#fully connected layer
def build_fcnet(shape=(num_features)):
    model = keras.Sequential()
    
    model.add(keras.layers.Dense(1024, input_shape=shape, activation='sigmoid'))
    
    return model

def build_cnn(shape, nbout=output_classes):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), input_shape=shape))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(32, (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(64, (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Flatten())
    model.add(Dense(10))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))
    return model

def action_model_cnn(n, img_height, img_width, img_ch, nbout=output_classes):

    cnn = build_cnn((img_height, img_width, img_ch))
    
    model = keras.Sequential()
    model.add(TimeDistributed(cnn, input_shape=(n, img_height, img_width, img_ch)))
    # here, you can also use GRU
    model.add(LSTM(512))
    
    #output layer
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(.5))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(nbout, activation='tanh'))
    return model
	
def action_model(shape=(5, num_features), nbout=output_classes):
    fcnet = build_fcnet(shape[1:])
	#cnn = build_cnn(shape)
    
    model = keras.Sequential()
    model.add(TimeDistributed(fcnet, input_shape=shape))
    # here, you can also use GRU
    model.add(LSTM(512))
    
    #output layer
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(.5))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(nbout, activation='tanh'))
    return model

model = action_model_cnn()
optimizer = keras.optimizers.Adam(0.0005)
model.compile(
    optimizer,
    loss_fn,
    metrics=[metric,"accuracy"]
)

EPOCHS=50
callbacks = [
    keras.callbacks.ReduceLROnPlateau(verbose=1),
    keras.callbacks.ModelCheckpoint(
        '../chkp/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        verbose=1),
]

x_train = np.asarray(x_train).reshape((len(x_train),5, num_features))
y_train = np.asarray(y_train)

#train
history = model.fit(x_train, y_train, batch_size=32,
    validation_split=0.33,
    verbose=1,
    epochs=EPOCHS,
    callbacks=callbacks
)

#output accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("training_accuracy.png")
plt.show()

#output loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("training_loss.png")
plt.show()

#test with test dataset
x_test = np.asarray(test_data[0]).reshape((len(test_data[0]),5, num_features))
y_test = np.asarray(test_data[1])

print(model.metrics_names,model.evaluate(x_test, y_test, batch_size=32))

#save model to file
model_json = model.to_json()
with open("./model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("./model.h5")
print("Saved model to disk")