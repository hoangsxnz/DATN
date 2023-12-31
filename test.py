from tensorflow import keras
from keras.models import load_model
import keras.backend as K
from tqdm import tqdm
import os 
from PIL import Image as im
import numpy as np
from dataset import read_image, show_example, read_image_ori
from config import SAVE_FOLDER, SHOW_IMAGE,TEST_PATH
import cv2 
from loss.focalloss import focal_loss

model = load_model('weights\\att_unet.h5', custom_objects={"K": K}, compile=False)
save_folder = SAVE_FOLDER
test_images = [os.path.join(TEST_PATH, f'{x}') for x in os.listdir(
TEST_PATH)]

color_dict = {0: (0,   0, 0),
              1: (0, 255,   0),
              2: (255, 0,   0)}

def rgb_to_onehot(rgb_arr, color_dict):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    arr = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
    return arr


def onehot_to_rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in color_dict.keys():
        output[single_layer==k] = color_dict[k]
    return np.uint8(output)
colors = np.array([[ 0,   0, 0],
                       [ 0, 255,   0],
                       [   0, 0,   255]])
for image in tqdm(test_images):
    '''
    We need segent 3 classes:
        + 0 if the pixel is part of the image background (denoted by black color);
        + 1 if the pixel is part of a non-neoplastic polyp (denoted by green color);
        + 2 if the pixel is part of a neoplastic polyp (denoted by red color).
    '''
    save_name = os.path.basename(image).replace('.jpeg','.png')
    print(save_name)
    original_image = cv2.imread(image)
    h, w, c = original_image.shape
    x = read_image(image)
    p = model.predict(np.expand_dims(x, axis=0))[0]  #256,256,3
    show_img = False 
    if show_img == True: #False 
        p = np.argmax(p, axis=-1) # 995,1280
        rgb = np.zeros((*p.shape, 3))
        for label, color in enumerate(colors):
            rgb[p == label] = color
        show_example(x, rgb)
    else:
        p = cv2.resize(p.astype(np.float32), ( w , h ), cv2.INTER_CUBIC) #995,1280,3
        p = np.argmax(p, axis=-1) # 995,1280
        # print(np.unique(p))
        # p_rgb = onehot_to_rgb(p,color_dict)
        rgb = np.zeros((*p.shape, 3)).astype(np.uint8)
        for label, color in enumerate(colors):
            rgb[p == label] = color
        cv2.imwrite(save_folder+'/'+save_name, rgb)
    