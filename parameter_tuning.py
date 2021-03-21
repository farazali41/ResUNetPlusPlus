## Final Script for HypterParameter Tuning

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Adam, Nadam, SGD, Adadelta, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from data_generator import DataGen
from unet import Unet
from resunet import ResUnet
from m_resunet import ResUnetPlusPlus
# from m_resunet_modified import ResUnetPlusPlus
# from resunet_pp_modified import ResUnetPlusPlus
from metrics import dice_coef, dice_loss, miou_coef, miou_loss


import sys
print(sys.argv[0]) # prints python_script.py
print(sys.argv[1]) # prints var1

batch = int(sys.argv[1])

file_path = "files/"
model_path = "files/sepv_tuning.h5"
model_name = "sepv"

## Create files folder
try:
    os.mkdir("files")
except:
    pass

train_path = "tuning_data/Kvasir-SEG/train/"
valid_path = "tuning_data/Kvasir-SEG/valid/"

## Training
train_image_paths = glob(os.path.join(train_path, "images", "*"))
train_mask_paths = glob(os.path.join(train_path, "masks", "*"))
train_image_paths.sort()
train_mask_paths.sort()

## Validation
valid_image_paths = glob(os.path.join(valid_path, "images", "*"))
valid_mask_paths = glob(os.path.join(valid_path, "masks", "*"))
valid_image_paths.sort()
valid_mask_paths.sort()



HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-3, 1e-4]))
# HP_BATCH_SIZE = hp.HParam('batch_size', hp.RealInterval(4, 8))
# HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete(4, 8))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['Nadam', 'sgd','RMSprop', 'Adadelta']))

METRIC_MIOU = 'miou_coef'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_LEARNING_RATE, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_MIOU, display_name='miou_coef')],
  )


## Parameters
image_size = 256
batch_size = batch
# lr = 1e-4
epochs = 20

train_steps = len(train_image_paths)//batch_size
valid_steps = len(valid_image_paths)//batch_size

print("train steps:", train_steps , "valid_steps", valid_steps)



## Generator
train_gen = DataGen(image_size, train_image_paths, train_mask_paths, batch_size=batch_size)
valid_gen = DataGen(image_size, valid_image_paths, valid_mask_paths, batch_size=batch_size)


def train_test_model(hparams):
  arch = Unet(input_size=image_size)
  model = arch.build_model()

  metrics = [dice_coef ,miou_coef]
  model.compile(loss=miou_loss, optimizer=hparams[HP_OPTIMIZER], metrics=metrics)

  model.fit(train_gen, epochs=epochs) # Run with 1 epoch to speed things up for demo purposes
  _, dice , miou = model.evaluate(valid_gen)

  return miou

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    metric = train_test_model(hparams)
    tf.summary.scalar(METRIC_MIOU, metric, step=1)


session_num = 0

for learning_rate in HP_LEARNING_RATE.domain.values:
  for optimizer in HP_OPTIMIZER.domain.values:
    hparams = {
        HP_LEARNING_RATE: learning_rate,
        HP_OPTIMIZER: optimizer,
    }
    run_name = "run-%d" % session_num
    print('--- Starting trial: %s' % run_name)
    print({h.name: hparams[h] for h in hparams})
    run('logs/hparam_tuning/' + run_name, hparams)
    session_num += 1