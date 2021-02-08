## TPU Setup

import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from data_generator import DataGen
from unet import Unet
from resunet import ResUnet
# from m_resunet_modified import ResUnetPlusPlus
from resunet_pp_modified import ResUnetPlusPlus
from metrics import dice_coef, dice_loss, miou_coef, miou_loss


# Distribution strategies
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

if __name__ == "__main__":
    ## Path
    file_path = "sm/files/"
    model_path = "sm/files/unet_resnet_miou.h5"
    model_name = "unet_resnet_miou"

    ## Create files folder
    try:
        os.mkdir("files")
    except:
        pass

    train_path = "new_data/Kvasir-SEG/train/"
    valid_path = "new_data/Kvasir-SEG/valid/"

    ## Training
    train_image_paths = glob(os.path.join(train_path, "images", "*"))
    train_mask_paths = glob(os.path.join(train_path, "masks", "*"))
    train_image_paths.sort()
    train_mask_paths.sort()

    # train_image_paths = train_image_paths[:2000]
    # train_mask_paths = train_mask_paths[:2000]

    ## Validation
    valid_image_paths = glob(os.path.join(valid_path, "images", "*"))
    valid_mask_paths = glob(os.path.join(valid_path, "masks", "*"))
    valid_image_paths.sort()
    valid_mask_paths.sort()

    ## Parameters
    image_size = 256
    batch_size = 8
    lr = 1e-4
    epochs = 10

    train_steps = len(train_image_paths)//batch_size
    valid_steps = len(valid_image_paths)//batch_size

    print("train steps:", train_steps , "valid_steps", valid_steps)

    ## Generator
    train_gen = DataGen(image_size, train_image_paths, train_mask_paths, batch_size=batch_size)
    valid_gen = DataGen(image_size, valid_image_paths, valid_mask_paths, batch_size=batch_size)

    loss = sm.losses.bce_jaccard_loss
    metrics = [sm.metrics.iou_score]

    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    with strategy.scope():
      if os.path.exists(model_path):
        # with CustomObjectScope({'loss': sm.losses.bce_jaccard_loss, 'metrics': sm.metrics.iou_score}):
        model = load_model(model_path, custom_objects={'loss': loss , 'metrics':metrics})
      else:
        
        model = sm.Unet('seresnet18', classes=1, activation='sigmoid')
        model.compile('Adam',loss=loss,    metrics=metrics,    )
    
    csv_logger = CSVLogger(f"{file_path}{model_name}_{batch_size}_{epochs}.csv", append=False)
    checkpoint = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    callbacks = [csv_logger, checkpoint, reduce_lr, early_stopping]
   
    model.fit_generator(
        train_gen,
        validation_data=valid_gen,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        epochs=50,
        callbacks=callbacks
        )
    
# !python3 run_modified.py