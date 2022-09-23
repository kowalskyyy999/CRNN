import torch 
import torch.nn as nn 
import torchvision

class Bidirectional(nn.Module):
    def __init__(self, inputs, hidden, num_layers, out):
        super(Bidirectional, self).__init__()
        self.rnn = nn.GRU(inputs, hidden, num_layers=num_layers, bidirectional=True)
        self.embedding = nn.Linear(hidden*2, out)

    def forward(self, x):
        rec, _ = self.rnn(x)
        out = self.embedding(rec)
        return out


class CRNN(nn.Module):
    def __init__(self, num_layers, out, device='cpu'):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 256, 9, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        
        # self.linear = nn.Linear(15872, 256)
        self.linear = nn.Linear(6656, 256)  # For size Image = (212, 46)
        # self.linear = nn.Linear(9216, 256)
        self.bn1 = nn.BatchNorm2d(256)
        self.rnn = Bidirectional(256, 1024, num_layers, out)
        self.device = device

    def forward(self, x):
        out = self.cnn(x)
        N, C, W, H = out.size()
        out = out.view(N, -1, H)    # out size: [N, C*W, H]
        out = out.permute(0, 2, 1)  # out size: [N, H, C*W]

        # weight = nn.Parameter(torch.FloatTensor(256, out.size()[-1]).to(self.device)) # Weight size: [256, C*W]
        # out = F.linear(out, weight=weight)  # out size: [N, H, 256]

        out = self.linear(out)
        out = out.permute(1, 0, 2)          # out size: [H, N, 256]
        out = self.rnn(out)                 # out size: [H, N, OUTPUT_SIZE]

        return out

# class CRNN(nn.Module):
#     def __init__(self, num_layers, output_size, input_channel=1):
#         super(CRNN, self).__init__()
#         self.init_conv = self.create_init_conv(input_channel)
#         self.backbone = self.create_backbone()
#         self.bidirectional = Bidirectional(256, 1024, num_layers, output_size)
#         self.linear = nn.Linear(512, 256)

#     def forward(self, image):
#         out = self.backbone(self.init_conv(image))
#         N, C, W, H = out.size()
#         out = out.view(N, -1, H)
#         out = out.permute(0, 2, 1)

#         out = self.linear(out)
#         out = out.permute(1, 0, 2)
#         out = self.bidirectional(out)
        
#         return out

#     @staticmethod
#     def create_backbone():
#         resnet18 = torchvision.models.resnet18()
#         backbone = [resnet18.bn1, resnet18.relu, 
#                     resnet18.maxpool, resnet18.layer1, 
#                     resnet18.layer2, resnet18.layer3, resnet18.layer4]
#         return nn.Sequential(*backbone)
    
#     @staticmethod
#     def create_init_conv(input_channel):
#         return nn.Sequential(
#             nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
#         )

if __name__ == "__main__":
    import os
    from dataset import SYNTH90Dataset, collate_fn
    from torch.utils.data import DataLoader
    from ctc_decoder import CTC_Decoder
    from dotenv import load_dotenv
    from utils import *

    load_dotenv('.env')

    DATA_PATH = os.getenv('DATA_PATH')

    dataset = SYNTH90Dataset(DATA_PATH, mode='train')
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    decoder = CTC_Decoder(SYNTH90Dataset.LABEL2CHAR, mode='greedy')
    metric = Metric(SYNTH90Dataset.LABEL2CHAR)
    image, target = next(iter(loader))

    model = CRNN(2, 109)

    outs = model(image)

    # preds = decoder(out)
    accuracy = metric(target, outs)
    print(accuracy)
    outs = outs.permute(1, 0, 2)

    targets = []
    acc = 0
    for t, out in zip(target, outs):
        t = t.cpu().numpy()
        result = ''
        for i in t:
            if i > 0:
                result += SYNTH90Dataset.LABEL2CHAR[i]
        preds = decoder._greedy_decoder(out.log_softmax(1))
        acc += AccuracyMetric(result, preds)
    
    print(acc/image.size(0))

    

    def _greedy_decoder(self, log_probs):
        out = log_probs.argmax(1).detach().cpu().numpy()
        preds = self.decode_tensor(out)
        return preds


