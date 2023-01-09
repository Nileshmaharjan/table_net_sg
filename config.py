import torch

SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.00001
EPOCHS = 300
BATCH_SIZE = 4
WEIGHT_DECAY = 0.008
DATAPATH_TRAIN = 'C:/Users/user/Projects/table_net_samsung_pytorch/combine_augmented_train.csv'
DATAPATH_TEST = 'C:/Users/user/Projects/table_net_samsung_pytorch/combine_augmented_test.csv'
