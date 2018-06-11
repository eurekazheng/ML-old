import os
from skimage import io
import torchvision.datasets.mnist as mnist
from torchvision import transforms as T
from PIL import Image
import csv

def fashionDatasetPrep(root):
    train_set = (
        mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
    )
    eval_set = (
        mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
    )
    print("train_set :", train_set[0].size())
    print("eval_set :", eval_set[0].size())
    for mode in ('train', 'eval'):
        data_path = os.path.join(root, mode)
        with open(os.path.join(root, mode + '.csv'), 'w') as f:
            if (not os.path.exists(data_path)):
                os.makedirs(data_path)
            for i, (img, label) in enumerate(zip(eval_set[0], eval_set[1])):
                img_path = os.path.join(data_path, str(i) + '.jpg')
                io.imsave(img_path, img.numpy())
                f.write(img_path + ' ' + str(label.item()) + '\n')


def diaretDatasetPrep():
    for mode in ('train', 'eval'):
        with open('C:\\Users\\HU SHIHE\\Desktop\\' + mode + '.csv') as csvfile:
            data = csv.reader(csvfile, delimiter=',')
            for line in data:
                img = Image.open(os.path.join('C:\\Users\\HU SHIHE\\Downloads\\train', line[0] + '.jpeg'))
                trans = T.Compose([
                    T.Grayscale(),
                    T.Resize(512),
                    T.CenterCrop(512),
                ])
                img = trans(img)
                img.save(os.path.join('C:\\Users\\HU SHIHE\\ML\\data\\dataset\\' + mode, line[0] + '.jpeg'))


def selectSample():
    trans = T.Compose([
        T.Grayscale(),
        T.Resize(512),
        T.CenterCrop(512),
    ])
    num = (100, 25)
    with open(os.path.join('C:\\Users\\HU SHIHE\\Desktop\\trainLabels.csv'), 'r') as csvfile:
        data = list(csv.reader(csvfile, delimiter=','))
        csvfile.close()
    for i, mode in enumerate(('train', 'eval')):
        with open(os.path.join('C:\\Users\\HU SHIHE\\Desktop\\', mode + '.csv'), 'w', newline='') as writefile:
            writer = csv.writer(writefile, delimiter=',')
            for j in range(5):
                img_fn = list(filter(lambda x: int(x[1]) == j, data))
                upper = min(num[i], len(img_fn))
                img_fn = img_fn[: (1-2*i)*upper : 1-2*i]
                for l in img_fn:
                    writer.writerow(l)
                img_fn = [l[0] for l in img_fn]
                for fn in img_fn:
                    img = Image.open(os.path.join('C:\\Users\\HU SHIHE\\Downloads\\train', fn + '.jpeg'))
                    img = trans(img)
                    img.save(os.path.join('C:\\Users\\HU SHIHE\\ML\\data\\dataset\\' + mode, fn + '.jpeg'))


selectSample()
