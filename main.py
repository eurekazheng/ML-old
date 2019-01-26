import models
from config import DefaultConfig
from data import DatasetPrep
from data.DatasetImpl import DiaretDataset
from utils import Visualizer
import numpy as np
import torch as t
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms as T
from torchnet import meter

opt = DefaultConfig()
#  DatasetPrep.fashionDatasetPrep(root=opt.data_root)
viz = Visualizer(opt.env)
assert viz.check_connection()
viz.close()
lm = eval_lm = meter.AverageValueMeter()
cm = eval_cm = meter.ConfusionMeter(5)
criterion = t.nn.CrossEntropyLoss()


def train(**kwargs):
    opt.parse(kwargs)
    #  Instantialize model
    model = getattr(models, opt.model)()
    print('====================================================')
    print('CURRENT MODEL:')
    print(model)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    #  Instantialize train and eval dataset and dataloader
    trans = T.Compose([
        T.Grayscale(),
        T.Resize(512),
        T.CenterCrop(512),
        T.ToTensor()
    ])
    train_data = DiaretDataset(img_root=opt.img_root, label_root=opt.label_root,
                               transforms=trans, mode='train')
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    eval_data = DiaretDataset(img_root=opt.img_root, label_root=opt.label_root,
                              transforms=trans, mode='eval')
    eval_dataloader = DataLoader(eval_data, opt.batch_size,
                                 shuffle=True,
                                 num_workers=opt.num_workers)
    #  Specify criterion and optimizer
    optimizer = t.optim.Adam(model.parameters(),
                             lr=opt.lr,
                             weight_decay=opt.weight_decay)
    #  Train
    print('====================================================')
    print('TRAINING...')
    for epoch in range(opt.max_epoch):
        lm.reset()
        cm.reset()
        for i, (img, label) in enumerate(train_dataloader):
            input = Variable(img)
            target = Variable(label)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            lm.add(loss.item() / len(img))
            cm.add(output.data, target.data)
            #  Visualize loss, accuracy, and confusion matrix
            if i % opt.print_freq == opt.print_freq - 1:
                # print(label)
                viz.plot('Train Loss', lm.value()[0])
                acc = np.matrix.trace(cm.value()) / np.sum(cm.value())
                viz.plot('Train Accuracy', acc)
                viz.plotcm(cm.value(), prefix='Train')
                print(('epoch {e} batch {b}: ' +
                      'Train Loss: {tl}, Train Accuracy: {ta}').format(
                          e = epoch,
                          b = i,
                          tl = lm.value()[0],
                          ta = acc))
        eval(model, eval_dataloader, epoch)
        #  model.save()


def eval(model, eval_dataloader, epoch):
    model.eval()
    eval_lm.reset()
    eval_cm.reset()
    for i, (img, label) in enumerate(eval_dataloader):
        eval_input = Variable(img)
        eval_target = Variable(label)
        eval_output = model(eval_input)
        eval_loss = criterion(eval_output, eval_target)
        eval_lm.add(eval_loss.item() / len(img))
        eval_cm.add(eval_output.data, eval_target.data)
        if i % opt.print_freq == opt.print_freq - 1:
            viz.plot('Evaluation Loss', eval_lm.value()[0])
            eval_acc = np.matrix.trace(eval_cm.value()) / np.sum(eval_cm.value())
            viz.plot('Evaluation Accuracy', eval_acc)
            viz.plotcm(eval_cm.value(), prefix='Eval')
            print(('epoch {e} batch {b}: ' +
                  'Eval Loss: {el}, Eval Accuracy: {ea}').format(
                      e = epoch,
                      b = i,
                      el = eval_lm.value()[0],
                      ea = eval_acc))
    model.train()


def test(**kwargs):
    pass


def help():
    print('help')

if __name__ == '__main__':
    import fire
    fire.Fire()
    train()
