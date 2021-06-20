import numpy as np
import random
from PIL import Image

def mask_sketch_basic(array):
    '''
    input: 128 x 256 x 1 numpy.array
    output: masked array
    '''
    #generate a random number
    random_num = random.random()
    left_mask = np.ones((128, 128, 1))
    right_mask = np.ones([128,128, 1])

    bottom = 3 * 128 // 7 
    if random_num < 0.4:
        return array
    elif random_num>=0.4 and random_num < 0.8:
        #mask the left
        start_width = np.random.randint(low=0, high=84)
        start_height = np.random.randint(low=bottom, high=(bottom+20))
        width_len = np.random.randint(low=54, high=128)
        height_len = np.random.randint(low=54, high=(128-bottom))

        end_width = min(start_width+width_len, 128)
        end_height = min(start_height+height_len, 128)

        right_mask[start_height:end_height,start_width:end_width ,:] = 0
    else:
        #mask the right
        start_width = np.random.randint(low=0, high=84)
        start_height=np.random.randint(low=0, high=10)
        width_len = np.random.randint(low=48, high=128)
        height_len = np.random.randint(low=48, high=bottom)

        end_width = min(start_width + width_len, 128)
        end_height = min(start_height + height_len, bottom)
        right_mask[start_height:end_height,start_width:end_width,:] = 0

    mask = np.concatenate([left_mask, right_mask], axis=1)

    return array * mask

def mask_sketch(batch_array):
    batch_size = batch_array.shape[0]
    for i in range(batch_size):
        batch_array[i] = mask_sketch_basic(batch_array[i])

    return batch_array

if __name__ == '__main__':
    batch_array = np.random.rand(2, 128, 256, 1)

    batch_array = mask_sketch(batch_array)

    img = Image.fromarray((batch_array[0,:,:,0]*255).astype(np.uint8))
    img.save('test.jpg')
