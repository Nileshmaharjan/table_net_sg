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
    display

)
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from model import TableNet
from pytorch_model_summary import summary
import tensorboard
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytesseract
from io import StringIO

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

model = TableNet(encoder='densenet', use_pretrained_model=True, basemodel_requires_grad=True)
model = model.to(config.DEVICE)

# load checkpoint
_, _, _ = load_checkpoint(torch.load("densenet_config_batch_size_2_model_checkpoint.pth.tar"), model)

processed_data = pd.read_csv(config.DATAPATH)
_, test_data = train_test_split(processed_data, test_size=0.2, random_state=config.SEED,
                                stratify=processed_data.hasTable)

test_img = r'marmot_processed\image_v2\10.1.1.36.8492_1.jpg'
test_table = r'marmot_processed\table_mask\10.1.1.36.8492_1_table_mask.png'
test_col = r'marmot_processed\col_mask\10.1.1.36.8492_1_col_mask.png'
test_image = np.array(Image.open('../' + test_img))
test_img = np.array(Image.open('../' + test_img))

test_table = np.array(Image.open('../' + test_table))
test_col = np.array(Image.open('../' + test_col))

# display(test_img, test_table, test_col, title='Original')

table_out, column_out = get_TableMasks(test_img, model)
print('here')

# display(test_img, table_out, column_out, title='Test')

outputs = fixMasks(test_img, table_out, column_out)

# copied from ipynb - start
image, table_boundRect, col_boundRects = outputs


#
# # draw bounding boxes of Table Coordinates
# color = (0, 255, 0)
# thickness = 4
#
# t_image = image.copy()
# for x, y, w, h in table_boundRect:
#     t_image = cv2.rectangle(t_image, (x, y), (x + w, y + h), color, thickness)
#
# t_image_table = t_image.copy()
#
# # Fix Column Coordinates
#
# t_image = t_image.copy()
# for c_bbox in col_boundRects:
#     for x, y, w, h in c_bbox:
#         t_image = cv2.rectangle(t_image, (x, y), (x + w, y + h), color, thickness)
#
# t_image_column = t_image.copy()
#
# f, ax = plt.subplots(1, 3, figsize=(15, 8))
# ax[0].imshow(test_image,  cmap='gray')
# ax[0].set_title(f'Image')
# ax[1].imshow(t_image_table, cmap='gray')
# ax[1].set_title(f'Table Fixed')
# ax[2].imshow(t_image_column, cmap='gray')
# ax[2].set_title(f'Column Fixed')
# plt.show()
# copied from ipynb - end

def getPredictions(image, table_boundRect):
    # Get output of multiple Tables in an Image and save it to csv

    for i, (x, y, w, h) in enumerate(table_boundRect):
        image_crop = image[y:y + h, x:x + w]
        plt.figure()
        plt.imshow(image_crop)
        plt.show()
        data = pytesseract.image_to_string(image_crop)
        df = pd.read_csv(StringIO(data), sep=r'\|', lineterminator=r'\n', engine='python')
        df.to_csv(f'Table_{i + 1}.csv')
        print(df)


getPredictions(test_image, table_boundRect)
