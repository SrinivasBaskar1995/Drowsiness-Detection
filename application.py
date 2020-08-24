from imutils import build_montages
import keras
from keras.layers import TimeDistributed, Dense, Dropout, LSTM
import numpy as np
import cv2
import utility
import math
import time

num_features = 8
output_classes = 1

#function to calculate euclidean distance between two points
def dist(a,b):
    return math.sqrt(math.pow(a[0]-b[0],2)+math.pow(a[1]-b[1],2))

#function to calculate the features of left eye
def features_left(landmarks):
    EAR = (dist(landmarks[38],landmarks[42])+dist(landmarks[39],landmarks[41]))/(2*dist(landmarks[37],landmarks[40]))
    MAR = dist(landmarks[63],landmarks[67])/dist(landmarks[61],landmarks[65])
    area = float(math.pow(dist(landmarks[38],landmarks[41])/2,2))*np.pi
    perimeter = dist(landmarks[37],landmarks[38])+dist(landmarks[38],landmarks[39])+dist(landmarks[39],landmarks[40])+dist(landmarks[40],landmarks[41])+dist(landmarks[41],landmarks[42])+dist(landmarks[42],landmarks[37])
    PUC = (4*np.pi*area)/(math.pow(perimeter,2))
    MOE = MAR/EAR
    return EAR,MAR,PUC,MOE

#function to calculate the features of right eye
def features_right(landmarks):
    EAR = (dist(landmarks[44],landmarks[48])+dist(landmarks[45],landmarks[47]))/(2*dist(landmarks[43],landmarks[46]))
    MAR = dist(landmarks[63],landmarks[67])/dist(landmarks[61],landmarks[65])
    area = float(math.pow(dist(landmarks[44],landmarks[47])/2,2))*np.pi
    perimeter = dist(landmarks[43],landmarks[44])+dist(landmarks[44],landmarks[45])+dist(landmarks[45],landmarks[46])+dist(landmarks[46],landmarks[47])+dist(landmarks[47],landmarks[48])+dist(landmarks[48],landmarks[43])
    PUC = (4*np.pi*area)/(math.pow(perimeter,2))
    MOE = MAR/EAR
    return EAR,MAR,PUC,MOE

#fully connected layer
def build_fcnet(shape=(num_features)):
    model = keras.Sequential()
    
    model.add(keras.layers.Dense(1024, input_shape=shape, activation='sigmoid'))
    #model.add(Dropout(.2))
    
    return model

#model
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
    #model.add(Dropout(.2))
    model.add(Dense(nbout, activation='tanh'))
    return model

model = action_model()
optimizer = keras.optimizers.Adam(0.0005)
model.compile(
    optimizer,
    'mse',
    metrics=['accuracy']
)

#load model from file
model.load_weights("../final_weights/acc-0.82_tanh.hdf5")

#variables for video output
window_name = 'drowsiness detector'
font = cv2.FONT_HERSHEY_SIMPLEX
org = (100, 100)
fontScale = 1
color = (255, 0, 0)
thickness = 2

#for landmark detection
face_landmarks = utility.face_landmarks()

#capture video
vidcap = cv2.VideoCapture(0)
print("calibrating...")
time.sleep(2)

#show images as video in a window
success, image = vidcap.read()
image_new = cv2.putText(image, "calibrating", org, font,fontScale, color, thickness, cv2.LINE_AA)
(h, w) = image_new.shape[:2]
montages = build_montages([image_new], (w,h), (1, 1))
for montage in montages:
    cv2.imshow("Drowsiness detector",montage)
key = cv2.waitKey(1) & 0xFF

#calculate amount of rotation needed
rotation = 0
for i in range(4):
    array_landmarks = face_landmarks.extract_face_landmarks(image)
    if len(array_landmarks) != 0:
        break
    else:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotation+=1

#calculate mean and std of every feature from the first 3 frames
count=0
initial_EAR_left = []
initial_PUC_left = []
initial_MOE_left = []
initial_EAR_right = []
initial_PUC_right = []
initial_MOE_right = []
initial_MAR = []
while True:
    if count==3:
        mean_EAR_left = np.mean(initial_EAR_left)
        std_EAR_left = np.std(initial_EAR_left)
        mean_PUC_left = np.mean(initial_PUC_left)
        std_PUC_left = np.std(initial_PUC_left)
        mean_MOE_left = np.mean(initial_MOE_left)
        std_MOE_left = np.std(initial_MOE_left)
        
        mean_EAR_right = np.mean(initial_EAR_right)
        std_EAR_right = np.std(initial_EAR_right)
        mean_PUC_right = np.mean(initial_PUC_right)
        std_PUC_right = np.std(initial_PUC_right)
        mean_MOE_right = np.mean(initial_MOE_right)
        std_MOE_right = np.std(initial_MOE_right)
        
        mean_MAR = np.mean(initial_MAR)
        std_MAR = np.std(initial_MAR)
        
        if std_EAR_left!=0 and std_PUC_left!=0 and std_MOE_left!=0 and std_EAR_right!=0 and std_PUC_right!=0 and std_MOE_right!=0 and std_MAR!=0:
            break
        else:
            count=0
            initial_EAR_left = []
            initial_PUC_left = []
            initial_MOE_left = []
            initial_EAR_right = []
            initial_PUC_right = []
            initial_MOE_right = []
            initial_MAR = []
    success, image = vidcap.read()
    image_new = cv2.putText(image, "calibrating", org, font,fontScale, color, thickness, cv2.LINE_AA)
    (h, w) = image_new.shape[:2]
    montages = build_montages([image_new], (w,h), (1, 1))
    for montage in montages:
        cv2.imshow("Drowsiness detector",montage)
    key = cv2.waitKey(1) & 0xFF
    for i in range(rotation):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    array_landmarks = face_landmarks.extract_face_landmarks(image)
    if len(array_landmarks) != 0:
        landmarks = array_landmarks[0]
        
        EAR_left,MAR,PUC_left,MOE_left = features_left(landmarks)
                    
        EAR_right,MAR,PUC_right,MOE_right = features_right(landmarks)
        
        initial_EAR_left.append(EAR_left)
        initial_PUC_left.append(PUC_left)
        initial_MOE_left.append(MOE_left)
        initial_EAR_right.append(EAR_right)
        initial_PUC_right.append(PUC_right)
        initial_MOE_right.append(MOE_right)
        initial_MAR.append(MAR)
        count += 1

data_point = []
text = ""
classes = ["alert","slightly drowsy","drowsy"]

#real time detection
while True:
    success, image = vidcap.read()
    array_landmarks = face_landmarks.extract_face_landmarks(image)
    if len(array_landmarks) != 0:
        landmarks = array_landmarks[0]
        
        #features
        EAR_left,MAR,PUC_left,MOE_left = features_left(landmarks)
        EAR_right,MAR,PUC_right,MOE_right = features_right(landmarks)
        
        EAR_left_norm = (EAR_left - mean_EAR_left)/std_EAR_left
        PUC_left_norm = (PUC_left - mean_PUC_left)/std_PUC_left
        MOE_left_norm = (MOE_left - mean_MOE_left)/std_MOE_left
        
        EAR_right_norm = (EAR_right - mean_EAR_right)/std_EAR_right
        PUC_right_norm = (PUC_right - mean_PUC_right)/std_PUC_right
        MOE_right_norm = (MOE_right - mean_MOE_right)/std_MOE_right
        
        MAR_norm = (MAR - mean_MAR)/std_MAR
        
        if len(data_point)<5:
            data_point.append([(EAR_left+EAR_right)/2,(PUC_left+PUC_right)/2,(MOE_left+MOE_right)/2,MAR,(EAR_left_norm+EAR_right_norm)/2,(PUC_left_norm+PUC_right_norm)/2,(MOE_left_norm+MOE_right_norm)/2,MAR_norm])
        elif len(data_point)==5:
            data_point_np = np.asarray(data_point).reshape((1,5,len(data_point[0])))
            
            #prediction
            label = model.predict(data_point_np)
            
            #classification
            if label[0]<0.5:
                text = classes[2]
            elif label[0]>=0.5:
                text = classes[0]

            print(text)
            
            #show images as video in a window
            image_new = cv2.putText(image, text, org, font,fontScale, color, thickness, cv2.LINE_AA)
            (h, w) = image_new.shape[:2]
            montages = build_montages([image_new], (w,h), (1, 1))
            for montage in montages:
                cv2.imshow("Drowsiness detector",montage)
            key = cv2.waitKey(1) & 0xFF
            
            data_point.pop(0)
            data_point.append([(EAR_left+EAR_right)/2,(PUC_left+PUC_right)/2,(MOE_left+MOE_right)/2,MAR,(EAR_left_norm+EAR_right_norm)/2,(PUC_left_norm+PUC_right_norm)/2,(MOE_left_norm+MOE_right_norm)/2,MAR_norm])