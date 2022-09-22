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
    def __init__(self, num_layers, output_size, input_channel=1):
        super(CRNN, self).__init__()
        self.init_conv = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.backbone = self.create_backbone()
        self.bidirectional = Bidirectional(256, 1024, num_layers, output_size)
        self.linear = nn.Linear(512, 256)

    def forward(self, image):
        out = self.backbone(self.init_conv(image))
        N, C, W, H = out.size()
        out = out.view(N, -1, H)
        out = out.permute(0, 2, 1)

        out = self.linear(out)
        out = out.permute(1, 0, 2)
        out = self.bidirectional(out)
        
        return out

    @staticmethod
    def create_backbone():
        resnet18 = torchvision.models.resnet18()
        backbone = [resnet18.bn1, resnet18.relu, 
                    resnet18.maxpool, resnet18.layer1, 
                    resnet18.layer2]
        return nn.Sequential(*backbone)

