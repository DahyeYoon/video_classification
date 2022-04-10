import torch
import torch.nn as nn
from collections import OrderedDict

class C3D(nn.Module):
    """
    C3D network.
    """

    def __init__(self, num_classes, pretrained=None):
        super(C3D, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)), nn.ReLU(),nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))

        self.layer2 = nn.Sequential(nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)), nn.ReLU(),nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))

        self.layer3a = nn.Sequential(nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)), nn.ReLU())

        self.layer3b = nn.Sequential(nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)), nn.ReLU(), nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))

        self.layer4a = nn.Sequential(nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)), nn.ReLU())

        self.layer4b = nn.Sequential(nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)), nn.ReLU(),nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))

        self.layer5a = nn.Sequential(nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)), nn.ReLU())

        self.layer5b = nn.Sequential(nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)), nn.ReLU(),nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)))

        self.fc6 = nn.Sequential(nn.Linear(8192, 4096), nn.ReLU(),nn.Dropout(p=0.5))
        self.fc7 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5))
        self.fc8 = nn.Linear(4096, num_classes)

        self._init_weight()

        if pretrained:
            self._load_pretrained_weights(pretrained)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3a(out)
        out = self.layer3b(out)
        out = self.layer4a(out)
        out = self.layer4b(out)
        out = self.layer5a(out)
        out = self.layer5b(out)
        out = out.view(-1, 8192) #out.reshape(-1, 8192)
        out = self.fc6(out)
        out = self.fc7(out)
        out = self.fc8(out)

        return out


    def _load_pretrained_weights(self, pretrained_model_path):
         checkpoint = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)   
         c3d_state_dict=iter(list(self.state_dict().keys()))
         state_dict = OrderedDict()


         premodel_last_layer_name=list(checkpoint.items())[-1][0].split('.')
         for k, v in checkpoint.items():
            name=c3d_state_dict.__next__()
            if k.split('.')[:-1]==premodel_last_layer_name[:-1]: 
                state_dict[name] = self.state_dict()[name]
            else: state_dict[name] = v
         self.load_state_dict(state_dict)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                # m.bias.data.zero_()

    def __get_model_name__(self):
        return "C3D"
