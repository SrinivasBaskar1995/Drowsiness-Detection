import cv2
import utility
import os

#class to extract landmarks
face_landmarks = utility.face_landmarks()
path = '../frames'
entries = os.listdir(path)

#reading stored frames and extracting landmarks, which is then saved to a txt file
for entry in entries:
    data = []
    labels = []
    print(entry)
    sub_entries = os.listdir(path+'/'+entry)
    for sub_entry in sub_entries:
        print(sub_entry)        
        filenames = os.listdir(path+'/'+entry+'/'+sub_entry)
        filenames.sort()
        
        for filename in filenames:
            print(filename)
            
            file = path+'/'+entry+'/'+sub_entry+'/'+filename
            
            image = cv2.imread(file)
            array_landmarks = face_landmarks.extract_face_landmarks(image)
            if len(array_landmarks) != 0:
                landmarks = array_landmarks[0]
                temp = [filename]
                for landmark in landmarks:
                    temp.append(landmark)
                data.append(temp)
                labels.append(int(sub_entry))

    #saving the extracted landmarks to a text file
    
    if not os.path.isdir("../data_txt/"):
        os.makedirs(path)
    
    f = open('../data_txt/data_landmarks-'+entry+'.txt', 'w')
    for i in range(len(data)):
        f.write(str(data[i][0]))
        f.write(",")
        for j in range(1,len(data[i])):
            f.write(str(data[i][j][0])+","+str(data[i][j][1]))
            f.write(",")
        f.write(str(labels[i]))
        f.write("\n")
    f.close()