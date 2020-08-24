#file to convert video data into frames with a distance of 3sec between them
import cv2
import utility
import os

face_landmarks = utility.face_landmarks()
labels = []
path = '../data'
entries = os.listdir(path)

for entry in entries:
    print(entry)
    sub_entries = os.listdir(path+'/'+entry)
    for sub_entry in sub_entries:
        print(sub_entry)
        for filename in os.listdir(path+'/'+entry+'/'+sub_entry):
            print(filename)
            path_frames = '../frames/'+entry+'/'+sub_entry
            if not os.path.isdir(path):
                os.makedirs(path)
            
            file = path+'/'+entry+'/'+sub_entry+'/'+filename
            vidcap = cv2.VideoCapture(file)
            
            #skip the first 100 frames as many videos have the subject moving around in the beginning
            for i in range(100):
                success, image = vidcap.read()
                
            #record the fps
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            count = 1
            frame_number=1
            success, image = vidcap.read()
            rotation = 0
            
            #check if the orientation of image is right and if not how much rotation is needed
            for i in range(4):
                array_landmarks = face_landmarks.extract_face_landmarks(image)
                if len(array_landmarks) != 0:
                    break
                else:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    rotation+=1
                    
            #start capturing frames
            success, image = vidcap.read()
            while success:
                #rotate the frames to the right orientation
                for i in range(rotation):
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    
                #extract landmarks
                array_landmarks = face_landmarks.extract_face_landmarks(image)
                
                if len(array_landmarks) != 0:
                    #save image to the path
                    cv2.imwrite(path_frames+'/'+filename[:filename.index(".")]+'-'+str(count)+'.jpg',image)
                    count += 1
                    #skip frames equivalent to 3 sec
                    for skip in range(3*int(fps)):
                        frame_number+=1
                        success, image = vidcap.read()
                        if not success:
                            break
                else:
                    #if no landmarks is able to be detected then move to next frame
                    print("no landmarks",filename,frame_number)
                    cv2.imwrite("test.jpg",image)
                    frame_number+=1
                    success, image = vidcap.read()
            vidcap.release()