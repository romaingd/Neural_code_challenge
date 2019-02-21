import torch
import torch.nn as nn
from torch.autograd import Variable
import math


cuda = torch.cuda.is_available()


###############################################################################
#                                                                             #
#                                  1D models                                  #
#                                                                             #
###############################################################################


class VGG_1D(nn.Module):

    def __init__(self, features, num_classes, base=64, arch='vgg'):
        super(VGG_1D, self).__init__()
        self.arch = arch
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(base * 4 * 6, base * 4),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(base * 4, base * 2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(base * 2, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers_vgg_1d(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


base = 8
cfg = {         # TODO Fix configs (nb of poolings)
    'custom': [base, 'M', base*2, 'M', base*4, 'M'],
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vggcustom(**kwargs):
    model = VGG_1D(make_layers_vgg_1d(cfg['custom'], batch_norm=True), arch='vggcustom', base=base, **kwargs)
    return model

def vgg13bn1d(**kwargs):
    model = VGG_1D(make_layers_vgg_1d(cfg['B'], batch_norm=True), arch='vgg13bn1d', base=base, **kwargs)
    return model

def vgg16bn1d(**kwargs):
    model = VGG_1D(make_layers_vgg_1d(cfg['D'], batch_norm=True), arch='vgg16bn1d', **kwargs)
    return model

def vgg19bn1d(**kwargs):
    model = VGG_1D(make_layers_vgg_1d(cfg['E'], batch_norm=True), arch='vgg19bn1d', **kwargs)
    return model



class LSTMClassifier(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes, batch_size):
        super(LSTMClassifier, self).__init__()
        self.arch = 'lstm'
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_dir = 2 if bidirectional else 1
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional
            )

        self.hidden2label = nn.Sequential(
            nn.Linear(hidden_dim*self.num_dir, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_dim, num_classes),
        )

        # self.hidden = self.init_hidden()

    def init_hidden(self, local_batch_size):
        if cuda:
            h0 = Variable(torch.zeros(self.num_layers*self.num_dir, local_batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(self.num_layers*self.num_dir, local_batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers*self.num_dir, local_batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(self.num_layers*self.num_dir, local_batch_size, self.hidden_dim))
        # if cuda:
        #     h0 = Variable(torch.zeros(self.num_layers*self.num_dir, self.batch_size, self.hidden_dim).cuda())
        #     c0 = Variable(torch.zeros(self.num_layers*self.num_dir, self.batch_size, self.hidden_dim).cuda())
        # else:
        #     h0 = Variable(torch.zeros(self.num_layers*self.num_dir, self.batch_size, self.hidden_dim))
        #     c0 = Variable(torch.zeros(self.num_layers*self.num_dir, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, x): # x is (batch_size, 1, 200), permute to (200, batch_size, 1)
        x = x.permute(2, 0, 1)
        local_batch_size = x.size()[1]
        # See: https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426/2
        lstm_out, (h, c) = self.lstm(x, self.init_hidden(local_batch_size))
        y  = self.hidden2label(lstm_out[-1])
        return y