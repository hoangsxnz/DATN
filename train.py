import numpy as np
import tensorflow as tf
import os
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.training.tracking.tracking import Asset
from dataset import *
from config import *
from models.AttentionUnet import *
from models.AttentionUnetefficientb0 import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

images, masks = load_data(IMAGE_PATH,MASK_PATH)
print(f'Amount of images: {len(images)}')

train_x, valid_x, train_y, valid_y = train_test_split(images, masks, test_size=0.2, random_state=42)
print(f'Training: {len(train_x)} - Validation: {len(valid_x)}')


train_step = len(train_x)//BATCH_SIZE
if len(train_x) % BATCH_SIZE != 0:
    train_step += 1

valid_step = len(valid_x)//BATCH_SIZE
if len(valid_x) % BATCH_SIZE != BATCH_SIZE:
    valid_step += 1
    
print(f'{train_step} - {valid_step}')

train_dataset = convert2TfDataset(train_x, train_y, BATCH_SIZE)
valid_dataset = convert2TfDataset(valid_x, valid_y, BATCH_SIZE)

def augment_image(image, label):
    # Apply your augmentation logic using tf.image functions
    augmented_image = tf.image.random_flip_left_right(image)
    augmented_image = tf.image.random_flip_up_down(augmented_image)
    augmented_image = tf.image.random_brightness(augmented_image, 0.2)
    augmented_image = tf.image.random_contrast(augmented_image, 0.8, 1.2)
    augmented_image = tf.image.random_saturation(augmented_image, 5, 10)
    # augmented_image = tf.image.random_crop(augmented_image, (WIDTH, HEIGHT))

    return augmented_image, label

# Apply augmentation to the training dataset
train_dataset = train_dataset.map(augment_image)
valid_dataset = valid_dataset.map(augment_image)

# Optionally, you may want to prefetch and shuffle the augmented dataset
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=len(train_x))

if (MODEL_SELECTION == "att_unet"): 
    print("Using Attention UNet model")
    att_unet = build_att_unet()
    att_unet.summary()
    att_unet,callbacks = compile_att_unet(att_unet)
    H = att_unet.fit(train_dataset,
                validation_data=valid_dataset,
                steps_per_epoch=train_step,
                validation_steps=valid_step,
                epochs=40,
                callbacks=callbacks)
    
elif (MODEL_SELECTION == "unet_att_eff0"): 
    print("Using Attention UNet B0 model ")
    r2_unet = build_att_unet_eff0()
    r2_unet.summary()
    r2_unet,callbacks = compile_att_unet_eff0(r2_unet)
    H = r2_unet.fit(train_dataset,
                validation_data=valid_dataset,
                steps_per_epoch=train_step,
                validation_steps=valid_step,
                epochs=40,
                callbacks=callbacks)
    
else:
    raise AssertionError("Model not supported")

fig = plt.figure()
numOfEpoch = 40
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['acc'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_acc'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()