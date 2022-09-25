import os
import logging
import collections
from tqdm import tqdm

import torch

class Engine(object):
    def __init__(self, model, optimizer, criterion, epochs, metric=None, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.device = device
        self.metric = metric

    def training(self, trainLoader, valLoader=None):
        logging.info(f"Training the model with {self.epochs} epochs and {self.device} Device")
        for epoch in range(self.epochs):
            self.model.train()
            trainLoss = 0
            pbar = tqdm(trainLoader, total=len(trainLoader))
            for image, target in pbar:
                image, target = image.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(image)

                loss = self.calc_loss(logits, target)

                loss.backward()

                self.optimizer.step()

                trainLoss += loss.item() * image.size(0)

                if self.metric is not None:
                    accuracy = self.metric(target, logits)
                    pbar.set_postfix({'Epoch' : epoch+1, 'Train Loss': loss.item(), 'Train Accuracy':accuracy*100})
                else:
                    pbar.set_postfix({'Epoch' : epoch+1, 'Train Loss': loss.item()})

            trainLoss = trainLoss / len(trainLoader.dataset)

            logging.info(f"Epoch : {epoch + 1} <=====> Training Loss : {trainLoss:.2f}")

            if valLoader is not None:
                self.testing(valLoader, epoch)

    
    def testing(self, loader, epoch=None):
        result = collections.defaultdict(list)
        valLoss = 0
        self.model.eval()
        pbar = tqdm(loader, total=len(loader))
        with torch.no_grad():
            for image, target in pbar:
                image, target = image.to(self.device), target.to(self.device)

                logits = self.model(image)

                loss = self.calc_loss(logits, target)

                if (self.metric is not None) and (epoch is not None):
                    accuracy = self.metric(target, logits)
                    result['accuracy'].append(accuracy)
                    pbar.set_postfix({'Epoch':epoch+1, 'Validation Loss':loss.item(), 'Validation Accuracy': accuracy * 100})
                
                elif epoch is not None:
                    pbar.set_postfix({'Epoch':epoch+1, 'Validation Loss':loss.item()})
                
                else:
                    pbar.set_postfix({'Loss':loss.item()})


                result['logits'].append(logits)
                result['target'].append(target)

                valLoss += loss.item() * image.size(0)
        
        valLoss = valLoss  / len(loader.dataset)
        
        if epoch is not None:
            logging.info(f"Epoch : {epoch+1} <=====> Validation Loss: {valLoss:.2f}")
        else:
            logging.info(f"Validation Loss : {valLoss:.2f}")
        
        return result, valLoss

    def calc_loss(self, logits, target):
        T, N, _ = logits.size()
        S = target.size(1)

        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int32)
        target_lengths = torch.full(size=(N,), fill_value=S, dtype=torch.int32)

        loss = self.criterion(logits.log_softmax(2), target, input_lengths, target_lengths)

        return loss

    def save_model(self, checkpoint, nameModel, path='models'):
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(checkpoint, os.path.join(path, nameModel))
        logging.info(f"Save the model with name '{nameModel}'")
    