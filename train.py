import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.optim as optim

from scripts.utils import Logger, AccuracyMetric, Metric
from scripts.dataset import SYNTH90Dataset, collate_fn
from scripts.config import *
from scripts.model import CRNN
from scripts.engine import Engine


from dotenv import load_dotenv
load_dotenv('.env')

def main():
    DATA_PATH = os.getenv('DATA_PATH')

    logger = Logger()
    log = logger()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = T.Compose([
        T.ColorJitter(brightness=0.5),
        T.ToTensor()
    ])

    train_dataset = SYNTH90Dataset(DATA_PATH, mode='train', transform=train_transform)
    val_dataset = SYNTH90Dataset(DATA_PATH, mode='val')
    test_dataset = SYNTH90Dataset(DATA_PATH, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=8)

    OUTPUT_SIZE = len(SYNTH90Dataset.LABEL2CHAR) + 1

    metric = Metric(SYNTH90Dataset.LABEL2CHAR)

    model = CRNN(NUM_LAYERS, OUTPUT_SIZE)
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CTCLoss()

    engine = Engine(model, optimizer, criterion, EPOCHS, metric, device)
    engine.training(train_loader, val_loader)

    checkpoints = {
        'CHAR2LABEL':SYNTH90Dataset.CHAR2LABEL,
        'LABEL2CHAR':SYNTH90Dataset.LABEL2CHAR,
        'model':{
            'state_dict':engine.model.state_dict(),
            'architecture':model,
            'image_size':IMAGE_SIZE,
            'output_size':OUTPUT_SIZE,
            'num_layers':NUM_LAYERS
        },
        'optimizer':{
            'state_dict':engine.optimizer.state_dict(),
            'optimizer':optimizer
        }
    }

    engine.save_model(checkpoints, nameModel='1.0.0.pth')

if __name__ == "__main__":
    main()