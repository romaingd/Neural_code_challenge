import numpy as np
import pandas as pd

from preprocessing import load_data, preprocess_data
from classification import classify

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.optim as optim

from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.model_selection import GroupShuffleSplit

import math
import os
import glob
import shutil
import itertools


# Parameters
data_folder = './data/'
isi_folder = './features/isi/'
submission_folder = './submissions/cnn/'

perform_evaluation = True

compute_submission = False

cuda = torch.cuda.is_available()
print('Training on ' + ('GPU' if cuda else 'CPU'))



############
# Define classifier #
############

class VGG(nn.Module):

    def __init__(self, features, num_classes, base=64, arch='vgg'):
        super(VGG, self).__init__()
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

def make_layers(cfg, batch_norm=False):
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
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [base, 'M', base*2, 'M', base*4, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
# cfg = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

def vgg13bn(**kwargs):
    model = VGG(make_layers(cfg['B'], batch_norm=True), arch='vgg13bn', base=base, **kwargs)
    return model

# def vgg13bn(**kwargs):
#     model = VGG(make_layers(cfg['B'], batch_norm=True), arch='vgg13bn', **kwargs)
#     return model

def vgg16bn(**kwargs):
    model = VGG(make_layers(cfg['D'], batch_norm=True), arch='vgg16bn', **kwargs)
    return model

def vgg19bn(**kwargs):
    model = VGG(make_layers(cfg['E'], batch_norm=True), arch='vgg19bn', **kwargs)
    return model



#
# Load data #
#

# Usual procedure
x_tr, x_te, y_tr = load_data(
    features_folder=isi_folder,
    data_folder=data_folder
)

preprocessing_steps = []
resampling_steps = []
x_tr, x_te, groups_tr, y_tr = preprocess_data(
    x_tr,
    x_te,
    y_tr=y_tr,
    preprocessing_steps=preprocessing_steps,
    resampling_steps=resampling_steps
)


# CNN specific pre-processing
class TimeSeriesDataset(Dataset):
    def __init__(self, data, target, norm, is_train):
        self.data = data
        self.target = target
        self.norm = norm
        self.is_train = is_train

    def __len__(self):
        return(self.data.shape[0])

    def __getitem__(self, idx):
        data_idx = np.expand_dims(self.data[idx, :], axis=0)
        target_idx = self.target[idx]
        return(data_idx, target_idx)


# Group-wise splitter
# Group-wise splitter
splitter = GroupShuffleSplit(n_splits=5, test_size=0.33,
                             random_state=42)
train_idx, test_idx = next(splitter.split(x_tr, y_tr, groups_tr))

train_data = TimeSeriesDataset(x_tr.values[train_idx], y_tr.values[train_idx].ravel(), False, True)
test_data = TimeSeriesDataset(x_tr.values[test_idx], y_tr.values[test_idx].ravel(), False, False)




#
# Training procedure #
#

def run_trainer(experiment_path, model_path, model, train_loader, test_loader, get_acc, resume, batch_size, num_epoch):

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    def save_checkpoint(state, is_best, filename=model_path+'checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, model_path+'model_best.pth.tar')
    def get_last_checkpoint(model_path):
        fs = sorted([f for f in os.listdir(model_path) if 'Epoch' in f], key=lambda k: int(k.split()[1]))
        return model_path+fs[-1] if len(fs) > 0 else None
    
    start_epoch = 0
    best_res = 0
    lrcurve = []
    conf_mats = []
    resume_state = get_last_checkpoint(model_path) if resume else None
    if resume_state and os.path.isfile(resume_state):
        print("=> loading checkpoint '{}'".format(resume_state))
        checkpoint = torch.load(resume_state)
        start_epoch = checkpoint['epoch']+1
        best_res = checkpoint['val_acc']
        lrcurve = checkpoint['lrcurve']
        conf_mats = checkpoint['conf_mats']
        model.load_state_dict(checkpoint['state_dict'])
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_state, checkpoint['epoch']))
    else:
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., 2.]).cuda())
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5) # optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)

    def train(epoch):
        model.train()
        total, total_correct = 0., 0.
        train_conf_mats = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.float()), Variable(target.long())
            if cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            _, correct, num_instance, conf_mat = get_acc(output, target)
            total_correct += correct
            total += num_instance
            train_conf_mats.append(conf_mat)
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Acc: {:.2f}%/{:.2f}%'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item(),
                    # 100. * batch_idx / len(train_loader), loss.data[0],
                    100. * correct / num_instance, 100. * total_correct / total ))
        
        return 100. * total_correct / total, np.dstack(train_conf_mats).sum(axis=2)

    def test():
        model.eval()
        test_loss = 0.
        total, total_correct = 0., 0.
        test_conf_mats = []
        preds = []
        class_correct = [0, 0]
        class_total = [0, 0]
        for data, target in test_loader:
            data, target = Variable(data.float(), volatile=True), Variable(target.long())
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # test_loss += criterion(output, target).data[0] # sum up batch loss
            test_loss += criterion(output, target).data.item() # sum up batch loss
            
            pred, correct, num_instance, conf_mat = get_acc(output, target)
            total_correct += correct
            total += num_instance
            test_conf_mats.append(conf_mat)

            # print('Average prediction: {:.3f} - Cohen Kappa Score {:.3f}'.format(
            #     np.mean(pred), get_cohen_kappa_score(output, target)
            # ))
            _, predicted = torch.max(output, 1)
            c = (target == predicted).squeeze()
            for i in range(target.size()[0]):
                label = target[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
            preds.append(pred)

        test_acc = 100. * total_correct / total
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, total_correct, total,
            test_acc))
        # print("Class correct", class_correct)
        # print("Class total", class_total)
        # print("Class accuracy", [class_correct[i] / class_total[i] for i in range(2)])
        print('Cohen kappa score', get_cohen_kappa_score_from_confmat(np.dstack(test_conf_mats).sum(axis=2)))
        print('\n')


        return np.hstack(preds), test_acc, np.dstack(test_conf_mats).sum(axis=2)


    for epoch in range(start_epoch, num_epoch):
        is_best = False

        train_acc, train_conf = train(epoch)
        preds, val_acc, val_conf = test()
        
        # print("Training Confmat: ")
        # print(train_conf)
        print("Testing Confmat: ")
        print(val_conf)
        # print("Number of Predictions Made: ")
        # print(preds.shape)
        
        lrcurve.append((train_acc, val_acc))
        conf_mats.append((train_conf, val_conf))
        # scheduler.step(val_loss)

        if val_acc > best_res:
            best_res = val_acc
            is_best = True

        save_checkpoint({
                'epoch': epoch,
                'arch': model.arch,
                'state_dict': model.cpu().state_dict(),
                'train_acc':train_acc,
                'val_acc': val_acc,
                'optimizer' : optimizer.state_dict(),
                'lrcurve':lrcurve,
                'train_conf':train_conf,
                'val_conf':val_conf,
                'conf_mats':conf_mats,
                'test_predictions':preds,
            }, is_best,
            model_path+"Epoch %d Acc %.4f.pt"%(epoch, val_acc))

        if cuda:
            model.cuda()
            
    return lrcurve, conf_mats


def run_experiment(experiment_path, train_data, test_data, model_root, models, norm, get_acc, resume=False, num_epoch=10):
    
    exp_result = {}
    for batch_size, model in models:
        print("Running %s" % model.arch)
        
        print('Loading Data..')
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
        model_path = os.path.join(
            experiment_path, 
            model_root,
            'norm' if norm else 'nonorm',
            model.arch) + '/'
        lrcurve, conf_mats = run_trainer(experiment_path, model_path, model, train_loader, test_loader, get_acc, resume, batch_size, num_epoch)
        exp_result[model.arch] = {'lrcurve':lrcurve, 'conf_mats':conf_mats}
        
    return exp_result



def get_models(): # tuples of (batch_size, model)
    return [
        (1024, vgg13bn(num_classes=2))
    ]

def get_acc(output, target):
    # takes in two tensors to compute accuracy
    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct = pred.eq(target.data.view_as(pred)).cpu().sum()
    conf_mat = confusion_matrix(pred.cpu().numpy(), target.data.cpu().numpy(), labels=range(2))
    return np.squeeze(pred.cpu().numpy()), correct, target.size(0), conf_mat

def get_cohen_kappa_score(output, target):
    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    return(cohen_kappa_score(pred.cpu().numpy(), target.data.cpu().numpy()))

def get_cohen_kappa_score_from_confmat(confusion, weights=None):
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    if weights is None:
        w_mat = np.ones([n_classes, n_classes], dtype=np.int)
        w_mat.flat[:: n_classes + 1] = 0
    elif weights == "linear" or weights == "quadratic":
        w_mat = np.zeros([n_classes, n_classes], dtype=np.int)
        w_mat += np.arange(n_classes)
        if weights == "linear":
            w_mat = np.abs(w_mat - w_mat.T)
        else:
            w_mat = (w_mat - w_mat.T) ** 2
    else:
        raise ValueError("Unknown kappa weighting type.")

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return(1 - k)


#
# Run parameters #
#

experiments = [
    {
        'experiment_path':'roll', 
        'train_data': train_data,
        'test_data': test_data,
        'model_root':'model', 
        'models':get_models(),
        'norm':False,
        'get_acc': get_acc,
        'resume':False,  
        'num_epoch':23
    }
]

for experiment in experiments:
    exp_log = run_experiment(**experiment)
    print(exp_log)