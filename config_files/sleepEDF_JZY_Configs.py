
class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 3
        self.final_out_channels = 128
        self.features_len = 127
        self.dropout = 0.35

        self.num_classes = 5
        self.semi_rate = 0.1

        # training configs
        self.train_epochs = 40 # 100
        self.fine_epochs = 1 # 40 

        # optimizer parameters
        self.batch_size = 128
        self.lr = 3e-4
        self.beta1 = 0.9
        self.beta2 = 0.99

        # data parameters
        self.drop_last = True

        self.Context_Cont = Context_Cont_configs()
        self.augmentation = augmentations()
        

class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 5


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.5
        self.use_cosine_similarity = True