import torch
import random
import numpy as np
import os
import pandas as pd
import config
import cv2
import matplotlib.pyplot as plt
from dataset import ImageFolder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

TRANSFORM = A.Compose([
    # ToTensor --> Normalize(mean, std)
    A.Normalize(
        mean=[0.485],
        std=[0.229],
        max_pixel_value=255,
    ),
    # A.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225],
    #     max_pixel_value=255,
    # ),
    ToTensorV2()
])


def seed_all(SEED_VALUE=config.SEED):
    random.seed(SEED_VALUE)
    os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed(SEED_VALUE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_data_loaders(data_path_train=config.DATAPATH_TRAIN, data_path_test=config.DATAPATH_TEST):
    df_train = pd.read_csv(data_path_train)
    df_test = pd.read_csv(data_path_test)
    # train_data, test_data = train_test_split(df_train, test_size=0.2, random_state=config.SEED, stratify=df_train.hasTable)
    train_data = df_train
    test_data = df_test

    train_dataset = ImageFolder(train_data, isTrain=True, transform=None)
    test_dataset = ImageFolder(test_data, isTrain=False, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader


# Checkpoint
def save_checkpoint(state, filename="model_checkpoint.pth.tar"):
    torch.save(state, filename)
    print("Checkpoint Saved at ", filename)


def load_checkpoint(checkpoint, model, optimizer=None):
    print("Loading checkpoint...")
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    last_epoch = checkpoint['epoch']
    tr_metrics = checkpoint['train_metrics']
    te_metrics = checkpoint['test_metrics']
    return last_epoch, tr_metrics, te_metrics


def write_summary(writer, tr_metrics, te_metrics, epoch):
    writer.add_scalar("Table loss/Train", tr_metrics['table_loss'], global_step=epoch)
    writer.add_scalar("Table loss/Test", te_metrics['table_loss'], global_step=epoch)

    writer.add_scalar("Table Acc/Train", tr_metrics['table_acc'], global_step=epoch)
    writer.add_scalar("Table Acc/Test", te_metrics['table_acc'], global_step=epoch)

    writer.add_scalar("Table F1/Train", tr_metrics['table_f1'], global_step=epoch)
    writer.add_scalar("Table F1/Test", te_metrics['table_f1'], global_step=epoch)

    writer.add_scalar("Table Precision/Train", tr_metrics['table_precision'], global_step=epoch)
    writer.add_scalar("Table Precision/Test", te_metrics['table_precision'], global_step=epoch)

    writer.add_scalar("Table Recall/Train", tr_metrics['table_recall'], global_step=epoch)
    writer.add_scalar("Table Recall/Test", te_metrics['table_recall'], global_step=epoch)

    # writer.add_scalar("Column loss/Train", tr_metrics['column_loss'], global_step=epoch)
    # writer.add_scalar("Column loss/Test", te_metrics['column_loss'], global_step=epoch)
    #
    # writer.add_scalar("Column Acc/Train", tr_metrics['col_acc'], global_step=epoch)
    # writer.add_scalar("Column Acc/Test", te_metrics['col_acc'], global_step=epoch)
    #
    # writer.add_scalar("Column F1/Train", tr_metrics['col_f1'], global_step=epoch)
    # writer.add_scalar("Column F1/Test", te_metrics['col_f1'], global_step=epoch)
    #
    # writer.add_scalar("Column Precision/Train", tr_metrics['col_precision'], global_step=epoch)
    # writer.add_scalar("Column Precision/Test", te_metrics['col_precision'], global_step=epoch)
    #
    # writer.add_scalar("Column Recall/Train", tr_metrics['col_recall'], global_step=epoch)
    # writer.add_scalar("Column Recall/Test", te_metrics['col_recall'], global_step=epoch)


def display_metrics(epoch, tr_metrics, te_metrics):
    nl = '\n'

    print(f"Epoch: {epoch} {nl}\
            Table Loss -- Train: {tr_metrics['table_loss']:.3f} Test: {te_metrics['table_loss']:.3f}{nl}\
            Table Acc -- Train: {tr_metrics['table_acc']:.3f} Test: {te_metrics['table_acc']:.3f}{nl}\
            Table F1 -- Train: {tr_metrics['table_f1']:.3f} Test: {te_metrics['table_f1']:.3f}{nl}\
            Table Precision -- Train: {tr_metrics['table_precision']:.3f} Test: {te_metrics['table_precision']:.3f}{nl}\
            Table Recall -- Train: {tr_metrics['table_recall']:.3f} Test: {te_metrics['table_recall']:.3f}{nl}\
            {nl}\-")
    # Col Loss -- Train: {tr_metrics['column_loss']:.3f} Test: {te_metrics['column_loss']:.3f}{nl}\
    # Col Acc -- Train: {tr_metrics['col_acc']:.3f} Test: {te_metrics['col_acc']:.3f}{nl}\
    # Col F1 -- Train: {tr_metrics['col_f1']:.3f} Test: {te_metrics['col_f1']:.3f}{nl}\
    # Col Precision -- Train: {tr_metrics['col_precision']:.3f} Test: {te_metrics['col_precision']:.3f}{nl}\
    # Col Recall -- Train: {tr_metrics['col_recall']:.3f} Test: {te_metrics['col_recall']:.3f}{nl}\


def compute_metrics(ground_truth, prediction, threshold=0.5):
    # https://stackoverflow.com/a/56649983

    ground_truth = ground_truth.int()
    prediction = (torch.sigmoid(prediction) > threshold).int()

    TP = torch.sum(prediction[ground_truth == 1] == 1)
    TN = torch.sum(prediction[ground_truth == 0] == 0)
    FP = torch.sum(prediction[ground_truth == 1] == 0)
    FN = torch.sum(prediction[ground_truth == 0] == 1)

    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (FP + TP + 1e-4)
    recall = TP / (FN + TP + 1e-4)
    f1 = 2 * precision * recall / (precision + recall + 1e-4)

    metrics = {
        'acc': acc.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }

    return metrics


def display(img, table, predicted_mask, title='Original'):
    f, ax = plt.subplots(1, 3, figsize=(15, 8))
    ax[0].imshow(img)
    ax[0].set_title(f'{title} Image')
    ax[1].imshow(table)
    ax[1].set_title(f'{title} Table Mask')
    ax[2].imshow(predicted_mask)
    ax[2].set_title(f'Predicted Table Mask')
    plt.show()


def display_predicted_and_fixed(img, predicted_mask, fixed_mask):
    f, ax = plt.subplots(1, 3, figsize=(15, 8))
    ax[0].imshow(img)
    ax[0].set_title(f'Original Image')
    ax[1].imshow(predicted_mask)
    ax[1].set_title(f'Predicted Table Mask')
    ax[2].imshow(fixed_mask)
    ax[2].set_title(f'Fixed Table Mask')
    plt.show()


def display_everything_1(org_img, predicted_mask):
    f, ax = plt.subplots(1, 2, figsize=(15, 8))
    ax[0].imshow(org_img)
    ax[0].set_title(f'Original Image')
    ax[1].imshow(predicted_mask)
    ax[1].set_title(f'Predicted Table Mask')
    plt.show()


def display_everything_2(fixed_mask, fixed_mask_original_image):
    f, ax = plt.subplots(1, 2, figsize=(15, 8))
    ax[0].imshow(fixed_mask)
    ax[0].set_title(f'Predicted mask')
    ax[1].imshow(fixed_mask_original_image)
    ax[1].set_title(f'Predicted mask original')
    plt.show()


def get_TableMasks(test_img, model, transform=TRANSFORM, device=config.DEVICE):
    image_stack = np.stack([test_img, test_img, test_img], axis=2)
    image = transform(image=image_stack)["image"]
    # get predictions
    model.eval()
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)
        # with torch.cuda.amp.autocast():
        # table_out, column_out = model(image)
        table_out = model(image)
        table_out = torch.sigmoid(table_out)
        # column_out = torch.sigmoid(column_out)

    # remove gradients

    table_out = (table_out.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0) > 0.5).astype(int)
    # column_out = (column_out.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0) > 0.5).astype(int)
    return table_out


#     return table_out, column_out


def is_contour_bad(c):
    # ref: https://www.pyimagesearch.com/2015/02/09/removing-contours-image-using-python-opencv/

    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # the contour is 'bad' if it is not a rectangle
    return not len(approx) == 4


def convert_to_pascal_voc_format(bbox):
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]

    xmin = x
    ymin = y
    xmax = x + w
    ymax = y + h
    new_bbox = [xmin, ymin, xmax, ymax]

    return new_bbox


def fixMasks(image, table_mask):
    table_mask = table_mask.reshape(1024, 1024).astype(np.uint8)
    # column_mask = column_mask.reshape(1024, 1024).astype(np.uint8)

    # get contours of the mask to get number of tables
    contours, table_heirarchy = cv2.findContours(table_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    table_contours = []

    for c in contours:
        # if the contour is bad, draw it on the mask
        # if not is_contour_bad(c):
        if cv2.contourArea(c) > 2000:
            table_contours.append(c)

    if len(table_contours) == 0:
        return None

    table_boundRect = [None] * len(table_contours)
    for i, c in enumerate(table_contours):
        polygon = cv2.approxPolyDP(c, 3, True)
        table_boundRect[i] = cv2.boundingRect(polygon)

    # table bounding Box
    table_boundRect.sort()

    # image = image[..., 0].reshape(1024, 1024).astype(np.uint8)
    image = image[..., 0].astype(np.uint8)

    # draw bounding boxes
    color = (0, 255, 0)
    thickness = 4

    for x, y, w, h in table_boundRect:
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    corrected_bbox = []
    for bbox in table_boundRect:
        a = convert_to_pascal_voc_format(bbox)
        corrected_bbox.append(a)

    return image, corrected_bbox
