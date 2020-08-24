# import the necessary packages
from imutils import face_utils
import imutils
import dlib
import cv2

#class to extract and return the landmarks from a image
class face_landmarks():
    
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
        
    def extract_face_landmarks(self,image):
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        landmarks = []
        
        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            landmarks.append(shape)
                
        return landmarks
		
