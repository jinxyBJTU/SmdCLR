import numpy as np
import torch


def DataTransform(sample, config):

    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

    return weak_aug, strong_aug


def jitter(x, sigma): #扰动 加一些噪声 sigma=0.8
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma): #缩放 乘上一些噪声 sigma=1.1
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))#(1, 3000)
    ai = []
    for i in range(x.shape[1]): # total channels
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return torch.from_numpy(np.concatenate((ai), axis=1))


def permutation(x, max_segments, seg_mode="random"):
    # print('warning')

    orig_steps = np.arange(x.shape[2])
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    # print(x.shape)
    # print(num_segs)
    # print(num_segs.shape)
    # print('sdadasda')
    # exit()
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])

            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[:,warp]
        else:
            ret[i] = pat
    
    return torch.from_numpy(ret)
