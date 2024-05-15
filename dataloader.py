import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from augmentations import DataTransform


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        # X_train = X_train[0:80000]
        # y_train = y_train[0:80000]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        # make sure the Channels in second dim
        if X_train.shape.index(min(X_train.shape)) != 1:
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            print('Entry isinstance')
            self.x_data = torch.from_numpy(X_train)
            self.y_data = y_train.long()
            print("X_train:", self.x_data.shape)
            print("X_label:", self.y_data.shape)

        else:
            print('no isinstance')
            self.x_data = X_train
            self.y_data = y_train
            print("X_train:", self.x_data.shape)
            print("X_label:", self.y_data.shape)

        self.len = X_train.shape[0]
        # if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            # self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def samples_statstics(dataset, if_train = False,  imbalance_sampling=False, class_im_r= 1 ):
    # data_path = '/data/JiaoxueDeng/YJNwork_dir/TS-TCC-main/data/sleepEDF_eog/'
    # train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    labels = dataset['labels']
    samples = dataset['samples']
    # [tensor(6111), tensor(1623), tensor(10570), tensor(3910), tensor(5211)]
    num_of_labels = int(torch.max(labels) + 1)
   
    counts = [(labels == i).sum() for i in range(num_of_labels)]
    ratio = [(labels == i).sum()*100/labels.shape[0] for i in range(num_of_labels)]
    print('sample_size', samples.shape)
    print('label_size', labels.shape)
    print("样本数量", counts)
    print("样本比例", ratio)
    print()

    if if_train:
        new_dataset = {}
        if imbalance_sampling:

            max_idx = counts.index(np.max(counts))
            min_idx = counts.index(np.min(counts))

            num_of_max_class = counts[max_idx]
            num_of_min_class = counts[min_idx]

            new_samples = []
            new_labels = []
            for label_idx in range(num_of_labels):
                cur_samples = samples[labels == label_idx]
                cur_labels = labels[labels == label_idx]

                # if label_idx==max_idx:
                    # cur_samples = cur_samples[:num_of_min_class*40]
                    # cur_labels = cur_labels[:num_of_min_class*40]
                # if label_idx == min_idx:
                #     cur_samples = cur_samples[:int(num_of_max_class/class_im_r)]
                #     cur_labels = cur_labels[:int(num_of_max_class/class_im_r)]

                new_samples.append( cur_samples )
                new_labels.append( cur_labels )
            
            new_samples = torch.cat(new_samples, dim=0)
            new_labels = torch.cat(new_labels, dim=0)

            num_of_labels = int(torch.max(new_labels) + 1)
            counts = [(new_labels == i).sum() for i in range(num_of_labels)]
            print("resample 样本数量", counts)
            print()
            new_dataset['samples'] = new_samples
            new_dataset['labels'] = new_labels

            return new_dataset
        else:
            return dataset
    else:
        return dataset


def data_generator(batch_size, data_path, configs, training_mode, seed):

    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    train_dataset = samples_statstics(train_dataset, if_train = True, imbalance_sampling=True, class_im_r = seed)
    
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    valid_dataset = samples_statstics(valid_dataset)

    test_dataset = torch.load(os.path.join(data_path, "test.pt"))
    test_dataset = samples_statstics(test_dataset)

    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader