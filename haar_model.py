import numpy as np
import cv2

assets_dir = './assets/'

class HaarModel:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(assets_dir + 'haar/haarcascade_frontalface_alt.xml')

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        return detected_faces

    def detect_and_bbimg(self, img):
        '''
        Detects faces and draws a bb around them
        Receives image and returns the bounding boxe coords 
        and an image of the bounding boxes
        '''
        detected_faces = self.detect(img)
        
        # Draw the boxes
        for (x,y,w,h) in detected_faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 4)
            

        return detected_faces, img
