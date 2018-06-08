import models
from config import DefaultConfig
from data import DatasetPrep
from data.DatasetImpl import FashionDataset
from utils import Visualizer
import numpy as np
import torch as t
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter

opt = DefaultConfig()
#  DatasetPrep.fashionDatasetPrep(root=opt.data_root)
viz = Visualizer(opt.env)
assert viz.check_connection()
viz.close()
lm = eval_lm = meter.AverageValueMeter()
cm = eval_cm = meter.ConfusionMeter(10)
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
    train_data = FashionDataset(root=opt.data_root, mode='train')
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    eval_data = FashionDataset(root=opt.data_root, mode='eval')
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
                viz.plot('Train Loss', lm.value()[0])
                acc = np.matrix.trace(cm.value()) / np.sum(cm.value())
                viz.plot('Train Accuracy', acc)
                viz.plotcm(cm.value(), prefix='Train')
        eval(model, eval_dataloader)
        #  model.save()


def eval(model, eval_dataloader):
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
            acc = np.matrix.trace(eval_cm.value()) / np.sum(eval_cm.value())
            viz.plot('Evaluation Accuracy', acc)
            viz.plotcm(eval_cm.value(), prefix='Eval')
    model.train()


def test(**kwargs):
    pass


def help():
    print('help')

if __name__ == '__main__':
    import fire
    fire.Fire()
    train()
