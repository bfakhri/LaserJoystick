# Original code from: https://riptutorial.com/opencv/example/21401/get-image-from-webcam

import numpy as np
import cv2

# Video source - can be camera index number given by 'ls /dev/video*
# or can be a video file, e.g. '~/Video.avi'
cap = cv2.VideoCapture(0)

### Set camera options ###
## Set the resolution
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
## Set any other webcam settings: https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-set
#cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    #cv2.imshow('frame',gray)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
