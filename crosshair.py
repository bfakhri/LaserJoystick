import pygame
import threading
import cv2
import numpy as np

# Initialize pygame/joysticks
pygame.init()
pygame.joystick.init()

# Get count of joysticks
joystick_count = pygame.joystick.get_count()

# Check to make sure joysticks are connected
assert joystick_count > 0

class Targeter:
    '''
    Targets an object
    '''
    def __init__(self, img_shape):
        self.asset_dir = './assets/'
        self.pos = np.array([img_shape[0]/2, img_shape[1]/2])
        self.crosshair_img = cv2.imread(self.asset_dir+'crosshair_2.png', cv2.IMREAD_UNCHANGED)/255.0
        cv2.imshow('OG CH: ', self.crosshair_img)
        # Crosshair size as a proportion of the image
        self.crosshair_scale = 0.15
        self.sensitivity = 40
        self.ch_size = int(np.max(img_shape)*self.crosshair_scale)
        self.ch_rs = cv2.resize(self.crosshair_img, (self.ch_size, self.ch_size), cv2.INTER_LINEAR)
        self.img_shape = img_shape
        cv2.imshow('RS CH: ', self.ch_rs)
        cv2.waitKey(-1)

        # Joystick settings
        self.deadzone = 0.2
        self.mapping = {0: 1, 1: 0, 3: 1, 4: 0}

        # Start the polling thread
        poll_thread = threading.Thread(target=self.poll_joystick)
        poll_thread.start()

    def poll_joystick(self):
        '''
        Runs in a loop, polling the joystick for information and updating vars
        '''
        # Assumes the xbox 360 controller is 0
        # Manage the polling rate
        clock = pygame.time.Clock()
        while(True):
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    done=True # Flag that we are done so we exit this loop
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            axes = joystick.get_numaxes()
            name = joystick.get_name()
            for i in range( axes ):
                axis = joystick.get_axis( i )
                if(abs(axis) > self.deadzone):
                    try:
                        self.pos[self.mapping[i]] += axis*self.sensitivity
                    except KeyError:
                        print('Unknown Key Mapping!: ', i)
                    # Keep crosshair in boundaries of image
                    self.pos = np.clip(self.pos, 0, self.img_shape[:2])
            clock.tick(20)



    def draw_crosshair(self, img):
        '''
        Draws a crosshair at the position self.pos
        '''
        # Save properties to convert it back later
        old_dtype = img.dtype
        old_img_max = np.max(img)
        img = img.astype(np.float32)/old_img_max

        # Offset half of the crosshair width/height
        mid_offset = self.ch_size/2
        # Round the position
        pos = self.pos.astype(np.int32)
        # Find boundaries relative to the input image (unbounded)
        y_min = int(pos[0]-mid_offset) 
        y_max = int(pos[0]+mid_offset) 
        x_min = int(pos[1]-mid_offset) 
        x_max = int(pos[1]+mid_offset) 
        # Bound where the crosshair will be on real image
        y_min_real = int(np.clip(y_min, 0, img.shape[0]-1))
        y_max_real = int(np.clip(y_max, 0, img.shape[0]-1))
        x_min_real = int(np.clip(x_min, 0, img.shape[1]-1))
        x_max_real = int(np.clip(x_max, 0, img.shape[1]-1))
        # Find how much we clip off
        y_min_clip = int(abs(y_min - y_min_real))
        y_max_clip = int(abs(y_max - y_max_real))
        x_min_clip = int(abs(x_min - x_min_real))
        x_max_clip = int(abs(x_max - x_max_real))

        # Crop crosshair if it goes over the image boundaries
        crosshair = self.ch_rs[y_min_clip:self.ch_size-y_max_clip, x_min_clip:self.ch_size-x_max_clip]
        # Remove crosshair portion of image
        img[y_min_real:y_max_real, x_min_real:x_max_real] *= (1.0 - crosshair[..., -1, np.newaxis])
        # Add crosshair
        img[y_min_real:y_max_real, x_min_real:x_max_real] += crosshair[..., :3]

        return (img*old_img_max).astype(old_dtype)


if __name__ == '__main__':
    targeter = Targeter((100,100))
    input()
    pygame.quit ()



