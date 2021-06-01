
import os
import numpy as np
import collections
import sys
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Adam, Nadam, SGD , RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from data_generator import DataGen
from unet import Unet
from resunet import ResUnet
# from m_resunet import ResUnetPlusPlus
from m_resunet_modified import ResUnetPlusPlus #daspp
# from resunet_pp_modified import ResUnetPlusPlus #sepv_conv
from metrics import dice_coef, dice_loss, miou_coef, miou_loss


if __name__ == "__main__":
    print(sys.argv[0]) # prints python_script.py
    print(sys.argv[1]) # prints var1
    
    
    seed_value = sys.argv[1]

    ## Path
    file_path = "files/"
    model_path = "files/unet_cvc2"+str(seed_value)+".h5"
    model_name = "unet_cvc2"+str(seed_value)

    

    ## Create files folder
    try:
        os.mkdir("files")
    except:
        pass

    train_path = "new_data/CVC-ClinicDB/train/"
    valid_path = "new_data/CVC-ClinicDB/valid/"

    ## Training
    train_image_paths = glob(os.path.join(train_path, "images", "*"))
    train_mask_paths = glob(os.path.join(train_path, "masks", "*"))
    train_image_paths.sort()
    train_mask_paths.sort()

    print(len(train_image_paths), len(train_mask_paths))

    np.random.seed(int(seed_value))

    bts_train_image_paths = list(np.random.choice(train_image_paths , size = int(len(train_image_paths)*0.5) , replace = True))
    bts_train_mask_paths = [i.replace("images","masks") for i in bts_train_image_paths] #list(np.random.choice(train_mask_paths , size = 10 , replace = True))
    print(len(bts_train_image_paths), len(bts_train_mask_paths))
    
    a = collections.Counter(bts_train_image_paths)
    print("bts_train_image_paths")
    print(sorted(a.items(), key=lambda item: item[1], reverse = True)[:5])
    
    a = collections.Counter(bts_train_mask_paths)
    print("bts_train_mask_paths")
    print(sorted(a.items(), key=lambda item: item[1], reverse = True)[:5])
    

    ## Validation
    valid_image_paths = glob(os.path.join(valid_path, "images", "*"))
    valid_mask_paths = glob(os.path.join(valid_path, "masks", "*"))
    valid_image_paths.sort()
    valid_mask_paths.sort()

    ## Parameters
    image_size = 256
    batch_size = 4
    lr = 1e-4
    epochs = 50

    train_steps = len(bts_train_image_paths)//batch_size
    valid_steps = len(valid_image_paths)//batch_size

    print("train steps:", train_steps , "valid_steps", valid_steps)

    ## Generator
    train_gen = DataGen(image_size, bts_train_image_paths, bts_train_mask_paths, batch_size=batch_size)
    valid_gen = DataGen(image_size, valid_image_paths, valid_mask_paths, batch_size=batch_size)

    ## Unet
    arch = Unet(input_size=image_size)
    model = arch.build_model()

    optimizer = Nadam(lr)
    metrics = [Recall(), Precision(), dice_coef, MeanIoU(num_classes=2),miou_coef]
    model.compile(loss=miou_loss, optimizer=optimizer, metrics=metrics)

    csv_logger = CSVLogger(f"{file_path}{model_name}_{batch_size}_{epochs}_seed{str(seed_value)}.csv", append=False)
    checkpoint = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, min_lr=1e-7, verbose=1)
    callbacks = [csv_logger, checkpoint, reduce_lr, early_stopping]

    model.fit_generator(train_gen,
            validation_data=valid_gen,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            epochs=epochs,
            callbacks=callbacks)