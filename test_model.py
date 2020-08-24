import keras
from keras.layers import TimeDistributed, Dense, Dropout, LSTM
import load_data_features
import numpy as np

#loading test data
test_data = load_data_features.load(False,False)

#number of features
num_features = len(test_data[0][0][0])

#number of output classes
try:
    output_classes = len(test_data[1][0])
    loss_fn = 'categorical_crossentropy'
    metric = "accuracy"
except:
    output_classes = 1
    loss_fn = 'mse'
    metric = "mse"

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

model = action_model()
optimizer = keras.optimizers.Adam(0.0005)
model.compile(
    optimizer,
    'mse',
    metrics=['accuracy']
)

x_test = np.asarray(test_data[0]).reshape((len(test_data[0]),5, num_features))
y_test = np.asarray(test_data[1])

#load model
model.load_weights("../final_weights/acc-0.82_tanh.hdf5")

#find accuracy
count=0
results = model.evaluate(x_test,y_test, batch_size=32)
print("accuracy ",results[1])