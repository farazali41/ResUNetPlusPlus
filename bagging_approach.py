### Bagging Approach

## Load Module
import os
import time
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
from tensorflow.keras.utils import CustomObjectScope
# from metrics import dice_coef, dice_loss, miou_coef, miou_loss
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from segmentation_metrics import dice_coe,dice_hard_coe,iou_coe

## Define image parser
def parse_image(img_path, image_size):
    image_rgb = cv2.imread(img_path, 1)
    h, w, _ = image_rgb.shape
    if (h == image_size) and (w == image_size):
        pass
    else:
        image_rgb = cv2.resize(image_rgb, (image_size, image_size))
    image_rgb = image_rgb/255.0
    return image_rgb

def parse_mask(mask_path, image_size):
    mask = cv2.imread(mask_path, -1)
    h, w = mask.shape
    if (h == image_size) and (w == image_size):
        pass
    else:
        mask = cv2.resize(mask, (image_size, image_size))
    mask = np.expand_dims(mask, -1)
    mask = mask/255.0

    return mask

# from metrics import dice_coef, dice_loss, miou_coef, miou_loss
## define custom evaluation metrics
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def miou_coef(y_true, y_pred):
  y_true_f = tf.keras.layers.Flatten()(y_true)
  y_pred_f = tf.keras.layers.Flatten()(y_pred)
  intersection = tf.reduce_sum(y_true_f * y_pred_f)
  union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)-intersection
  iou = (intersection + smooth) / (union + smooth)
  return iou

def miou_loss(y_true, y_pred):
    return 1.0 - miou_coef(y_true, y_pred)

def mask_to_3d(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


# Set path to test dataset
# TEST_DATASET_PATH = "./test_images"
TEST_DATASET_PATH = "new_data/Kvasir-SEG/test/images"
GROUND_TRUTH_PATH = "new_data/Kvasir-SEG/test/masks/"
# MASK_PATH = "./mask"
PREDICTED_MASK_PATH = "new_data/Kvasir-SEG/test/predicted_masks"
Results = "new_data/bagging_results"
model_path = "files_before_tpu/miou.h5"

# Load Keras model
with CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef,'miou_loss':miou_loss,'miou_coef':miou_coef}):
  resunet_pp = tf.keras.models.load_model("files/models/resunet_pp_miou_3.h5")

with CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef,'miou_loss':miou_loss,'miou_coef':miou_coef}):
  sepv_conv_rmsprop = tf.keras.models.load_model("files/models/sepv_conv_rmsprop.h5")

with CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef,'miou_loss':miou_loss,'miou_coef':miou_coef}):
  daspp = tf.keras.models.load_model("files/models/daspp.h5")

## List for each evaluation metrics
time_taken = []
precision = [] #Precision()
p = tf.keras.metrics.Precision()
recall = [] #Recall()
r = tf.keras.metrics.Recall()
mean_io_u = [] #MeanIoU(num_classes=2)
m = tf.keras.metrics.MeanIoU(num_classes=2)
smooth = 1.
image_size = 256
jaccard = list()
f1 = list()
loss = list()

## Loop over each images
for image_name in os.listdir(TEST_DATASET_PATH):

    # Load the test image
    image_path = os.path.join(TEST_DATASET_PATH, image_name)
    image = cv2.imread(image_path)
    H, W, _ = image.shape
    # image = cv2.resize(image, (256, 256))
    # image = np.expand_dims(image, axis=0)
    image = parse_image(image_path, 256)
    # print(image.shape)
    # print(np.expand_dims(image, axis=0).shape)

    # Start time
    start_time = time.time()

    ## Prediction
    mask1 = resunet_pp.predict(np.expand_dims(image, axis=0))[0]
    mask2 = sepv_conv_rmsprop.predict(np.expand_dims(image, axis=0))[0]
    mask3 = daspp.predict(np.expand_dims(image, axis=0))[0]

    # End timer
    end_time = time.time() - start_time

    time_taken.append(end_time)
    print("**********************************************************")
    # print(GROUND_TRUTH_PATH + image_name)
    ground_truth_mask = parse_mask(GROUND_TRUTH_PATH + image_name, image_size)
    print("{} - {:.10f}".format(image_name, end_time))

    sep_line = np.ones((image_size, 10, 3)) * 255

    ## Set decision criteria/boundry
    mask1 = mask1 > 0.5
    mask1 = mask1.astype(np.float32)
    mask2 = mask2 > 0.5
    mask2 = mask2.astype(np.float32)
    mask3 = mask3 > 0.5
    mask3 = mask3.astype(np.float32)

    ## Apply Bagging Approach on 3 models
    mask = np.round((mask1+mask2+mask3)/3, decimals = 0)

    ## Evaluate Metrics
    loss.append(miou_loss(ground_truth_mask, mask))
    r.update_state(ground_truth_mask, mask)
    recall.append(r.result().numpy())
    p.update_state(ground_truth_mask, mask)
    precision.append(p.result().numpy())
    m.update_state(ground_truth_mask, mask)
    mean_io_u.append(m.result().numpy())
    f1.append(dice_coef(ground_truth_mask,mask).numpy())
    jaccard.append(miou_coef(ground_truth_mask,mask).numpy())


    mask = mask * 255.0
    mask = cv2.resize(mask, (H, W))

    ## Add Channels to mask to append it with original image
    mask1 = mask_to_3d(mask1)
    mask2 = mask_to_3d(mask2)
    mask3 = mask_to_3d(mask3)
    mask = mask_to_3d(mask)
    ground_truth_mask = mask_to_3d(ground_truth_mask)

    mask_path = os.path.join(PREDICTED_MASK_PATH, image_name)
    cv2.imwrite(mask_path, mask)

    ## Order : original image, ground_truth, predict_mask1 , predict_mask2 , predict_mask3 , predict_mask_bagging
    all_images = [image * 255, sep_line, ground_truth_mask * 255, sep_line, mask1 * 255 , sep_line , mask2 * 255 , sep_line , mask3 * 255 , sep_line , mask]
    # print("++++++++++++++++++++++++++++++++++++")
    # print(image.shape)
    # print(ground_truth_mask.shape)
    # print(mask1.shape)
    # print(mask2.shape)
    # print(mask3.shape)
    # print(mask.shape)
    cv2.imwrite(f"{Results}/{image_name}.png", np.concatenate(all_images, axis=1))


## Print mean of evaluation metrics
mean_time_taken = np.mean(time_taken)
mean_fps = 1/mean_time_taken
print("******************************************")
print("Mean FPS: ", mean_fps)
print("Mean loss: ", np.mean(loss))
print("Mean recall: ", np.mean(recall))
print("Mean precision: ", np.mean(precision))
print("Mean dice_coef: ", np.mean(f1))
print("Mean mean_io_u: ", np.mean(mean_io_u))
print("Mean miou_coef: ", np.mean(jaccard))