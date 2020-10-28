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
from m_resunet import ResUnetPlusPlus
from metrics import dice_coef, dice_loss, miou_coef, miou_loss


if __name__ == "__main__":
    ## Path
    file_path = "files/"
    model_path_resunetpp = "files/resunetplusplus.h5"
    model_path_unet = "files/unet.h5"

    ## Create files folder
    try:
        os.mkdir("files")
    except:
        pass

    train_path = "new_data/Kvasir-SEG/train/"
    valid_path = "new_data/Kvasir-SEG/valid/"
    # train_path = "new_data/kvasir_segmentation_dataset/train/"
    # valid_path = "new_data/kvasir_segmentation_dataset/valid/"

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
    epochs = 2

    train_steps = len(train_image_paths)//batch_size
    valid_steps = len(valid_image_paths)//batch_size
    print("train_steps ",train_steps,'\nvalid_steps ',valid_steps)
    ## Generator
    train_gen = DataGen(image_size, train_image_paths, train_mask_paths, batch_size=batch_size)
    valid_gen = DataGen(image_size, valid_image_paths, valid_mask_paths, batch_size=batch_size)

    ## Unet
    # arch = Unet(input_size=image_size)
    # model = arch.build_model()

    ## ResUnet
    # arch = ResUnet(input_size=image_size)
    # model = arch.build_model()
    print("******************Training ResUNet++******************")
    ## ResUnet++
    arch = ResUnetPlusPlus(input_size=image_size)
    resunetpp = arch.build_model()

    optimizer = Nadam(lr)
    metrics = [Recall(), Precision(), dice_coef, MeanIoU(num_classes=2),miou_coef]
    resunetpp.compile(loss=miou_loss, optimizer=optimizer, metrics=metrics)

    csv_logger = CSVLogger(f"{file_path}resunetpp_mIoU_{batch_size}.csv", append=False)
    checkpoint = ModelCheckpoint(model_path_resunetpp, verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    callbacks = [csv_logger, checkpoint, reduce_lr, early_stopping]

    resunetpp.fit_generator(train_gen,
            validation_data=valid_gen,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            epochs=epochs,
            callbacks=callbacks)
    
    print("******************Training UNet******************")

    ## UNet
    arch = Unet(input_size=image_size)
    unet = arch.build_model()

    optimizer = Nadam(lr)
    metrics = [Recall(), Precision(), dice_coef, MeanIoU(num_classes=2),miou_coef]
    unet.compile(loss=miou_loss, optimizer=optimizer, metrics=metrics)

    csv_logger = CSVLogger(f"{file_path}unet_mIoU_{batch_size}.csv", append=False)
    checkpoint = ModelCheckpoint(model_path_unet, verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    callbacks = [csv_logger, checkpoint, reduce_lr, early_stopping]

    unet.fit_generator(train_gen,
            validation_data=valid_gen,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            epochs=epochs,
            callbacks=callbacks)

# !python3 run.py