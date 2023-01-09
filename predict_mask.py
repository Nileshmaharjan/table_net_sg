import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config
import os
from utils import (
    get_data_loaders,
    load_checkpoint,
    save_checkpoint,
    display_metrics,
    write_summary,
    compute_metrics,
    get_TableMasks,
    fixMasks,
    display,
    display_predicted_and_fixed

)
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from model import TableNet
from pytorch_model_summary import summary
import tensorboard
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw

model = TableNet(encoder = 'densenet', use_pretrained_model = True, basemodel_requires_grad = True)

print("Model Architecture and Trainable Paramerters")
print("="*50)
print(summary(model, torch.zeros((1, 3, 1024, 1024)), show_input=False, show_hierarchical=False))

model = TableNet(encoder = 'densenet', use_pretrained_model = True, basemodel_requires_grad = True)
model = model.to(config.DEVICE)


#load checkpoint
_,_,_ = load_checkpoint(torch.load("densenet_config_batch_size_4_model_checkpoint.pth.tar"), model)


processed_data = pd.read_csv(config.DATAPATH_TEST)
_, test_data  = train_test_split(processed_data, test_size = 0.2, random_state = config.SEED)


def createFixedMask(bboxes, dim):
    image = Image.new("RGB", dim)
    mask = ImageDraw.Draw(image)
    for each_list in bboxes:
        mask.rectangle(each_list, fill=255)
    image_array = np.array(image)
    image = Image.fromarray(image_array[:, :, 0])

    return image


for i in range(1):
    print("Test Image :",test_data.iloc[i, 0])
    test_img = np.array(Image.open(test_data.iloc[i, 0]))
    test_table = np.array(Image.open(test_data.iloc[i, 1]))
    table_out = get_TableMasks(test_img, model)

    outputs = fixMasks(test_img, table_out)

    table_out = table_out * 255


    table_out = np.reshape(table_out, (1024,1024))
    # a = np.arange(6).reshape((3, 2))
    print(table_out.shape)

    # backtorgb = cv2.cvtColor(table_out, cv2.COLOR_GRAY2RGB)
    stacked_predicted_mask = np.stack([table_out, table_out, table_out], axis = 2)

    cv2.imwrite('predicted_mask.png', table_out)
    cv2.imwrite('stacked_predicted_mask.png', table_out)
    #     predicted_mask_img = cv2.imread('predicted_mask.png')
    #     print(predicted_mask_img)

    #     display(test_img, test_table, table_out,title = 'Original')

    outputs = fixMasks(test_img, table_out)
    dim = (1024, 1024)
    fixed_mask = createFixedMask(outputs[1], dim)

#     display_predicted_and_fixed(test_img, table_out, fixed_mask)


# #     display(test_img, table_out, title = 'Test')
