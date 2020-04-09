import numpy as np
import cv2
from crosshair import Targeter
from haar_model import HaarModel

# Video source - can be camera index number given by 'ls /dev/video*
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Make a targeting object
targeter = Targeter(frame.shape)

# Make a face detection model
model = HaarModel()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Detect faces and draw bbs
    detected_faces, frame = model.detect_and_bbimg(frame)
    # Draw crosshair on frame
    targeter.track(detected_faces)
    frame = targeter.draw_crosshair(frame)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
