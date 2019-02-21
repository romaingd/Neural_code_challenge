
import os
import shutil

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from deeplearning.metrics import get_cohen_kappa_score_from_confmat


cuda = torch.cuda.is_available()
print('Training on ' + ('GPU' if cuda else 'CPU'))


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
