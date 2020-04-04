import pygame

# Initialize the joysticks
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
        self.crosshair_img = cv2.imread(self.asset_dir+'crosshair.png')
        # Crosshair size as a proportion of the image
        self.crosshair_scale = 0.025
        ch_size = int(np.max(img.shape)*self.crosshair_scale)
        self.ch_rs = cv2.resize(self.crosshair_img, (ch_size, ch_size))


    def draw_crosshair(self, img):
        '''
        Draws a crosshair at the position self.pos
        '''
        # Offset half of the crosshair width/height
        mid_offset = self.ch_size/2
        # Find where the crosshair will be on real image
        y_min = np.clip(self.pos[0]-mid_offset, 0, img.shape[0]-1)
        y_max = np.clip(self.pos[0]+mid_offset, 0, img.shape[0]-1)
        x_min = np.clip(self.pos[1]-mid_offset, 0, img.shape[1]-1)
        x_max = np.clip(self.pos[1]+mid_offset, 0, img.shape[1]-1)
        # Crop crosshair if it goes over the image boundaries
        crosshair = self.ch_rs[
        # Remove crosshair portion of image
        img[y_min:y_max, x_min:x_max] *= 1.0 - self.ch_rs[..., -1]
        # Add crosshair
        img[y_min:y_max, x_min:x_max] += self.ch_rs[..., :3]

        return img





