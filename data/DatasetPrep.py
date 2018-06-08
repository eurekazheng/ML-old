import os
from skimage import io
import torchvision.datasets.mnist as mnist


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
