import os 
import sys
import string

from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from scripts.config import IMAGE_SIZE

class SYNTH90Dataset(Dataset):
    CHARS = string.ascii_letters \
        + string.punctuation \
        + string.digits \
        + ' ' + '★' + '€' + '©' + '°' + '£' + 'É' + '¡' + '¢' + '•' + '®' + 'Ç' + '¥' + '\n' + '—' + '\t'

    CHAR2LABEL = {char: i+1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, path, mode='train', image_binary=False, transform=None):
        self.path = path
        self.mode = mode
        self.payloads = self._load_file(path, mode)
        self.image_binary = image_binary
        self.transform = transform

    def __len__(self):
        return len(self.payloads)

    def __getitem__(self, index):
        try:
            image, target = self._extract_payloads(index)
        except:
            image, target = self._extract_payloads(index+1)

        return image, target

    @staticmethod
    def _load_file(path, mode):
        if mode == 'train':
            annotation = os.path.join(path, 'annotation_train.txt')
        elif mode == 'val':
            annotation = os.path.join(path, 'annotation_val.txt')
        elif mode == 'test':
            annotation = os.path.join(path, 'annotation_test.txt')
        else:
            sys.exit("Mode not defined!")

        with open(annotation, 'r') as f:
            payload = f.readlines()

        return payload

    def _find_target(self, payload):
        return " ".join(payload.split("_")[1:-1])

    def _extract_payloads(self, index):
        payload = self.payloads[index]
        image_path = os.path.join(self.path, payload.split()[0][2:])
        image = Image.open(image_path).convert('L').resize(IMAGE_SIZE)

        if self.image_binary:
            image = image.point(lambda x: 0 if x < 128 else 255, '1')

        if self.transform:
            image = self.transform(image)

        else:
            image = T.ToTensor()(image)

        target = self._find_target(payload)
        target = torch.tensor([self.CHAR2LABEL[c] for c in target], dtype=torch.int32)

        return image, target


def collate_fn(batch):
    (image, target) = list(zip(*batch))
    image_pad = torch.nn.utils.rnn.pad_sequence(image, batch_first=True)
    target_pad = torch.nn.utils.rnn.pad_sequence(target, batch_first=True)
    return image_pad, target_pad
