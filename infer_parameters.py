import os
import sys
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from data_generator import *
from metrics import dice_coef, dice_loss, miou_coef, miou_loss
from segmentation_metrics import dice_coe,dice_hard_coe,iou_coe
# from weighted_hausdorff_loss import weighted_hausdorff_distance
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Adam, Nadam, SGD , RMSprop

from keras import backend as K
from keras_flops import get_flops


def mask_to_3d(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

if __name__ == "__main__":
    print(sys.argv[0]) # prints python_script.py
    print(sys.argv[1]) # prints var1
    
    
    model_path = sys.argv[1]
    
    save_path = "result"
    test_path = "new_data/CVC-ClinicDB/test/"

    image_size = 256
    batch_size = 1

    test_image_paths = glob(os.path.join(test_path, "images", "*"))
    test_mask_paths = glob(os.path.join(test_path, "masks", "*"))
    test_image_paths.sort()
    test_mask_paths.sort()

    ## Create result folder
    try:
        os.mkdir(save_path)
    except:
        pass

    ## Model
    with CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef,'miou_loss':miou_loss,'miou_coef':miou_coef,
        'dice_coe':dice_coe,'dice_hard_coe':dice_hard_coe,'iou_coe':iou_coe#, 'weighted_hausdorff_distance':weighted_hausdorff_distance
    }):
        model = load_model(model_path)
    
    ## Added for '<' not supported between instances of 'function' and 'str'
    lr = 1e-4
    optimizer = Nadam(lr)
    metrics = [Recall(), Precision(), dice_coef, MeanIoU(num_classes=2),miou_coef,dice_coe,dice_hard_coe,iou_coe#,weighted_hausdorff_distance
    ]
    model.compile(loss=miou_loss, optimizer=optimizer, metrics=metrics)
    print("model compiled successfully")

    ## Test
    print("Test Result: ")
    test_steps = len(test_image_paths)//batch_size
    test_gen = DataGen(image_size, test_image_paths, test_mask_paths, batch_size=batch_size)
    print(len(test_gen))
    model.evaluate_generator(test_gen, steps=test_steps, verbose=1)
    flops = get_flops(model, batch_size=1)
    # print(f"FLOPS: {flops / 10 ** 6:.03} G")
    print(flops)
    model.summary()

    ## Generating the result
    # for i, path in tqdm(enumerate(test_image_paths), total=len(test_image_paths)):
    #     image = parse_image(test_image_paths[i], image_size)
    #     mask = parse_mask(test_mask_paths[i], image_size)

    #     predict_mask = model.predict(np.expand_dims(image, axis=0))[0]
    #     predict_mask = (predict_mask > 0.5) * 255.0

    #     sep_line = np.ones((image_size, 10, 3)) * 255

    #     mask = mask_to_3d(mask)
    #     predict_mask = mask_to_3d(predict_mask)

    #     all_images = [image * 255, sep_line, mask * 255, sep_line, predict_mask]
    #     cv2.imwrite(f"{save_path}/{i}.png", np.concatenate(all_images, axis=1))
# !python3 infer.py
# https://colab.research.google.com/drive/1TTtNwEF4TjWQoXdvjpWky-dnXN1zFu13