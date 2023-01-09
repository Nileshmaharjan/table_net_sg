import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import config
import os
from utils import (
    get_data_loaders,
    load_checkpoint,
    save_checkpoint,
    display_metrics,
    write_summary,
    compute_metrics,
    seed_all
)
from loss import TableNetLoss
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from model import TableNet
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary

import sys
import warnings

warnings.filterwarnings("ignore")


def train_on_epoch(data_loader, model, optimizer, loss, scaler, threshold=0.5):
    table_loss, table_acc, table_precision, table_recall, table_f1 = [], [], [], [], []
    loop = tqdm(data_loader, leave=True)

    for batch_idx, img_dict in enumerate(loop):
        image = img_dict["image"].to(config.DEVICE)
        table_image = img_dict["table_image"].to(config.DEVICE)

        with torch.cuda.amp.autocast():
            table_out = model(image)
            t_loss = loss(table_out, table_image)

        table_loss.append(t_loss.item())

        # backward
        optimizer.zero_grad()
        scaler.scale(t_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = sum(table_loss) / len(table_loss)
        loop.set_postfix(loss=mean_loss)

        cal_metrics_table = compute_metrics(table_image, table_out, threshold)

        table_f1.append(cal_metrics_table['f1'])
        table_precision.append(cal_metrics_table['precision'])
        table_acc.append(cal_metrics_table['acc'])
        table_recall.append(cal_metrics_table['recall'])

        metrics = {
            'table_loss': np.mean(table_loss),
            'table_f1': np.mean(table_f1),
            'table_precision': np.mean(table_precision),
            'table_recall': np.mean(table_recall),
            'table_acc': np.mean(table_acc),
        }

    return metrics


def test_on_epoch(data_loader, model, loss, threshold=0.5, device=config.DEVICE):
    table_loss, table_acc, table_precision, table_recall, table_f1 = [], [], [], [], []

    model.eval()
    with torch.no_grad():
        loop = tqdm(data_loader, leave=True)

        for batch_idx, img_dict in enumerate(loop):
            image = img_dict["image"].to(device)
            table_image = img_dict["table_image"].to(device)

            with torch.cuda.amp.autocast():
                table_out = model(image)
                t_loss = loss(table_out, table_image)

            table_loss.append(t_loss.item())
            mean_loss = sum(table_loss) / len(table_loss)
            loop.set_postfix(loss=mean_loss)

            cal_metrics_table = compute_metrics(table_image, table_out, threshold)

            table_f1.append(cal_metrics_table['f1'])
            table_precision.append(cal_metrics_table['precision'])
            table_acc.append(cal_metrics_table['acc'])
            table_recall.append(cal_metrics_table['recall'])

    metrics = {
        'table_loss': np.mean(table_loss),
        'table_f1': np.mean(table_f1),
        'table_precision': np.mean(table_precision),
        'table_recall': np.mean(table_recall),
        'table_acc': np.mean(table_acc)
    }

    model.train()
    return metrics


if __name__ == '__main__':
    # if torch.cuda.is_available():
    #     dev = "cuda:0"
    # else:
    #     dev = "cpu"

    seed_all(SEED_VALUE=config.SEED)
    checkpoint_name = 'densenet_config_batch_size_4_model_checkpoint.pth.tar'
    model = TableNet(encoder='densenet', use_pretrained_model=True, basemodel_requires_grad=True)

    print("Model Architecture and Trainable Paramerters")
    print("=" * 50)
    print(summary(model, torch.zeros((1, 3, 1024, 1024)), show_input=False, show_hierarchical=True))

    model = model.to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    loss = TableNetLoss()
    scaler = torch.cuda.amp.GradScaler()
    train_loader, test_loader = get_data_loaders(data_path_train=config.DATAPATH_TRAIN,  data_path_test=config.DATAPATH_TEST)

    nl = '\n'

    # load checkpoint
    if os.path.exists(checkpoint_name):
        last_epoch, tr_metrics, te_metrics = load_checkpoint(torch.load(checkpoint_name), model)
        last_table_f1 = te_metrics['table_f1']

        print("Loading Checkpoint")
        display_metrics(last_epoch, tr_metrics, te_metrics)
        print()
    else:
        last_epoch = 0
        last_table_f1 = 0.

    # Train Network
    print("Training Model\n")
    writer = SummaryWriter(
        f"runs/TableNet/densenet/config_4_batch_{config.BATCH_SIZE}_LR_{config.LEARNING_RATE}_encoder_train"
    )

    # for early stopping
    i = 0

    for epoch in range(last_epoch + 1, config.EPOCHS):
        print("=" * 30)
        print('Here')
        start = time.time()

        tr_metrics = train_on_epoch(train_loader, model, optimizer, loss, scaler, 0.5)
        te_metrics = test_on_epoch(test_loader, model, loss, threshold=0.5)

        write_summary(writer, tr_metrics, te_metrics, epoch)

        end = time.time()

        display_metrics(epoch, tr_metrics, te_metrics)

        if last_table_f1 < te_metrics['table_f1']:
            print('Not early stopping')
            i = 0
            last_table_f1 = te_metrics['table_f1']
            last_table_f1 = te_metrics['table_f1']

            checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                          'train_metrics': tr_metrics, 'test_metrics': te_metrics}
            save_checkpoint(checkpoint, checkpoint_name)
        else:
            print('About to early stop')
            i += 1

        if i == 25:
            print("Early Stopping")
            break
