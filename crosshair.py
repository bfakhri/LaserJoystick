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
        # Crosshair size as a proportion of the image
        self.crosshair_scale = 0.15
        self.sensitivity = 40
        self.ch_size = int(np.max(img_shape)*self.crosshair_scale)
        self.ch_rs = cv2.resize(self.crosshair_img, (self.ch_size, self.ch_size), cv2.INTER_LINEAR)
        self.img_shape = img_shape

        # Joystick settings
        self.deadzone = 0.2
        self.joystick_mapping = {0: 1, 1: 0, 3: 1, 4: 0}

        # Tracking settings
        self.snapped = False
        self.snap_toggle_button = 5
        self.snap_loss_num = 10 # num frames to lose tracking from
        self.snap_loss_cnt = 0 # num frame where object has been lost
        self.snap_max_dist = 0.25 # Distance b/t crosshair and object before snap is lost (proportion of img dims)
        # shape (bs, 4) -> (bs, (x,y,w,y))
        self.last_bbs = []

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
        done = False
        while(not done):
            # Event detection
            for event in pygame.event.get(): # User did something
                if event.type == pygame.JOYBUTTONDOWN:
                    print("Joystick button pressed.")
                    # Check buttons
                    buttons = joystick.get_numbuttons()
                    if(joystick.get_button(self.snap_toggle_button) == 1):
                        self.snap()

                if event.type == pygame.JOYBUTTONUP:
                    print("Joystick button released.")
                if event.type == pygame.QUIT: # If user clicked close
                    done=True # Flag that we are done so we exit this loop
            # Poll the joysticks (dual analog)
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            axes = joystick.get_numaxes()
            name = joystick.get_name()
            for i in range( axes ):
                axis = joystick.get_axis( i )
                if(abs(axis) > self.deadzone):
                    try:
                        self.pos[self.joystick_mapping[i]] += axis*self.sensitivity
                    except KeyError:
                        print('Unknown Key Mapping!: ', i)
                    # Keep crosshair in boundaries of image
                    self.pos = np.clip(self.pos, 0, self.img_shape[:2])

            # Enable/disable tracking

            clock.tick(20)

    def snap(self, enforce_max_dist=False):
        '''
        Snaps to the closest bounding box 
        '''
        if(len(self.last_bbs) > 0):
            # Convert (x,y,w,h) to centroids (x,y)
            box_centers = self.last_bbs[:, 0:2] + self.last_bbs[:, 2:4]/2 
            distances = np.sqrt(np.sum((box_centers-self.pos[::-1])**2, axis=-1))
            closest = np.argmin(distances) 
            if(not enforce_max_dist):
                self.pos = box_centers[closest, ::-1]
            elif(distances[closest] < self.snap_max_dist*np.min(self.img_shape[:2])):
                self.pos = box_centers[closest, ::-1]
            else:
                print('Nearst box too far away to track')
            self.snapped = True
        else:
            print('Failed to find a box')
            self.snapped = False

    def track(self, boxes):
        '''
        Tracks objects from bounding boxes of form (x, y, w, h)
        '''
        #if(self.snapped):
        if(len(boxes) > 0):
            self.last_bbs = np.array(boxes)
            #print(self.last_bbs.shape)
            self.snap_loss_cnt = 0 
            self.snap(enforce_max_dist=True)
        else:
            self.snap_loss_cnt += 1
            if(self.snap_loss_cnt > self.snap_loss_num):
                self.snapped = False




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



