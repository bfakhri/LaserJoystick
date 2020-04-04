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
        self.pos = [img_shape[0]/2, img_shape[1]/2]
        self.crosshair_img = cv2.imread(self.asset_dir+'crosshair.png', cv2.IMREAD_UNCHANGED)
        # Crosshair size as a proportion of the image
        self.crosshair_scale = 0.025
        ch_size = int(np.max(img_shape)*self.crosshair_scale)
        self.ch_rs = cv2.resize(self.crosshair_img, (ch_size, ch_size))

        # Joystick settings
        self.deadzone = 0.2

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
                #print('Name {} Axis {} value: {:>6.3f}'.format(name, i, axis) )
                print(self.pos)
                if(abs(axis) > self.deadzone):
                    self.pos[i] += axis
            clock.tick(40)
            print('')



    def draw_crosshair(self, img):
        '''
        Draws a crosshair at the position self.pos
        '''
        # Offset half of the crosshair width/height
        mid_offset = self.ch_size/2
        # Find where the crosshair will be on real image
        y_min = self.pos[0]-mid_offset
        y_max = self.pos[0]+mid_offset
        x_min = self.pos[1]-mid_offset
        x_max = self.pos[1]+mid_offset
        # Find how much of the ch we have to clip off
        y_min_clip = abs(y_min - np.clip(y_min, 0, img.shape[0]-1))
        y_max_clip = abs(y_max - np.clip(y_max, 0, img.shape[0]-1))
        x_min_clip = abs(x_min - np.clip(x_min, 0, img.shape[1]-1))
        x_max_clip = abs(x_max - np.clip(x_max, 0, img.shape[1]-1))
        # Crop crosshair if it goes over the image boundaries
        crosshair = self.ch_rs[y_min_clip:y_max_clip, x_min_clip:x_max_clip]
        # Remove crosshair portion of image
        img[y_min:y_max, x_min:x_max] *= 1.0 - crosshair[..., -1]
        # Add crosshair
        img[y_min:y_max, x_min:x_max] += crosshair[..., :3]

        return img


if __name__ == '__main__':
    targeter = Targeter((100,100))
    input()
    pygame.quit ()



