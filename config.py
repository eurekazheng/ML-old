import warnings


class DefaultConfig(object):
    env = 'main'  # visdom 环境
    model = 'FashionNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    data_root = 'data/dataset/'
    label_root = 'data/dataset/'
    load_model_path = None  # 'checkpoints/model.pth'  加载预训练的模型的路径，为None代表不加载

    batch_size = 64  # batch size
    use_gpu = False  # use GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 5  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 10
    lr = 1e-3  # initial learning rate
    lr_decay = 0.99  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has no attribute %s" % k)
            setattr(self, k, v)
        print('====================================================')
        print('CURRENT CONFIG:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))
