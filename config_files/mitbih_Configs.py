
class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128
        self.features_len = 27
        self.dropout = 0.35

        self.num_classes = 3
        self.semi_rate = 0.1

        # training configs
        self.train_epochs = 1 # 100
        self.fine_epochs = 1 # 40 

        # optimizer parameters
        self.batch_size = 128 # 512
        self.lr = 3e-6
        self.beta1 = 0.9
        self.beta2 = 0.99
        # data parameters
        self.drop_last = True

        self.Context_Cont = Context_Cont_configs()
        self.augmentation = augmentations()

class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.6 # [50]=1.6
        self.jitter_ratio = 0.7 # [50]=0.7;
        self.max_seg = 5


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.05
        self.use_cosine_similarity = True