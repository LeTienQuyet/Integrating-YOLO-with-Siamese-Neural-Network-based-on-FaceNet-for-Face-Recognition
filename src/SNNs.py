from facenet_pytorch import InceptionResnetV1

import torch
import torch.nn as nn

class SiameseNeuralNetwork(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SiameseNeuralNetwork, self).__init__()
        self.base_model = InceptionResnetV1(pretrained="vggface2")
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_features=512, out_features=1)

    def forward(self, x1, x2):
        x1 = self.base_model(x1)
        x2 = self.base_model(x2)
        x = torch.abs(x1-x2)
        x = self.dropout(x)
        x = self.fc(x)
        return x