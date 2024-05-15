from matplotlib import pyplot as plt
import torch
import random
import numpy as np
import pandas as pd
import os
import sys
import logging
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
from shutil import copy
from scipy.stats import norm

def generate_indicating_matrix(label):
    N = len(label)
    # 将标签向量扩展为大小为Nx1的矩阵
    expanded_label = label.unsqueeze(1)
    # 利用广播生成相似性矩阵
    indicating_matrix = expanded_label == expanded_label.t()
    return indicating_matrix

def copy_Files(destination, data_type):
    destination_dir = os.path.join(destination, "model_files")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("trainer/trainer.py", os.path.join(destination_dir, "trainer.py"))
    copy(f"config_files/{data_type}_Configs.py",
         os.path.join(destination_dir, f"{data_type}_Configs.py"))
    copy("dataloader/augmentations.py",
         os.path.join(destination_dir, "augmentations.py"))
    copy("dataloader/dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"models/model.py", os.path.join(destination_dir, f"model.py"))
    copy("models/loss.py", os.path.join(destination_dir, "loss.py"))
    copy("models/TC.py", os.path.join(destination_dir, "TC.py"))


def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def _calc_metrics(pred_labels, true_labels, log_dir, home_path):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    labels_save_path = os.path.join(log_dir, "labels")
    os.makedirs(labels_save_path, exist_ok=True)
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)

    r = classification_report(true_labels, pred_labels,
                              digits=6, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df = df * 100

    # save classification report
    exp_name = os.path.split(os.path.dirname(log_dir))[-1]
    training_mode = os.path.basename(log_dir)
    file_name = f"{exp_name}_{training_mode}_classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    # save confusion matrix
    cm_file_name = f"{exp_name}_{training_mode}_confusion_matrix.torch"
    cm_Save_path = os.path.join(home_path, log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def draw_neg(total_big_ratios, totla_loss, savepath):
    fig = plt.figure(dpi=500)
    batch_length = len( total_big_ratios)
    batch_x = np.linspace(0, batch_length-1, num=batch_length, dtype='int32')

    epoch_length = len(totla_loss)
    epoch_x = np.linspace(0, epoch_length-1, num=epoch_length, dtype='int32')


    ax1 = plt.subplot2grid((1,1),(0,0),colspan=1,rowspan=1)
    ax1.plot(epoch_x, total_big_ratios, color = 'orange', label = 'negatieves') 
    # ax1.legend()
    ax1.set_xlabel('epoch')  # x轴变量名称
    ax1.set_ylabel('numbers')  # x轴变量名称

    ax2 = ax1.twinx()
    ax2.plot(epoch_x, totla_loss, color = 'red', label = 'loss') 
    # ax2.legend()
    ax2.set_ylabel('Loss_value')  # x轴变量名称
    
    # ax2 = plt.subplot2grid((2,1),(1,0),colspan=1,rowspan=1)

    fig.savefig('{}negatives_big.jpg'.format(savepath))  # 图片保存
    plt.clf()

# def draw_Pic(total_big_ratios, totla_loss, history, savepath):
#     # 画一条线
#     titles = ['acc', 'loss', 'f1']
#     draw_neg(total_big_ratios,totla_loss,savepath)
#     for i in range(3):
#         plt.title(titles[i])  # 图片标题
#         plt.xlabel('epoch')  # x轴变量名称
#         plt.ylabel(titles[i])  # y轴变量名称
#         plt.plot(history[i], label="x")  # 画出 a_line 线  label="x": 图中左上角示例
#         plt.legend()  # 画出曲线图标
#         plt.savefig('{}{}.jpg'.format(savepath, titles[i]))  # 图片保存
#         plt.clf()

def draw_3_2_norm(selected_epochs, total_losses, total_labels, savePath):
    # 3: (aug), (same_class), (diff_class)
    # 2: (minority) (majority)
    savePath = os.path.join(savePath, '3_2_norm')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    titles = ['encoder_loss', 'projection_loss']

    for epoch_idx in selected_epochs:
        epoch_losses = total_losses[epoch_idx-1]
        epoch_labels = total_labels[epoch_idx-1]

        epoch_labels = torch.cat(epoch_labels).numpy()      # 51200

        encoder_loss = [l[0] for l in epoch_losses]
        projection_loss = [l[1] for l in epoch_losses]

        encoder_loss = np.concatenate(encoder_loss, axis=0)
        projection_loss = np.concatenate(projection_loss, axis=0)

        losses = [encoder_loss, projection_loss]
        for title_idx in [0,1]:
            epoch_losses = losses[title_idx]
            epoch_losses = (epoch_losses-np.min(epoch_losses))/(np.max(epoch_losses) - np.min(epoch_losses))
            total_loss, same_loss, diff_loss = epoch_losses[:,0], epoch_losses[:,1], epoch_losses[:,2]
        
            num_of_labels = np.max(epoch_labels)+1
            counts = [(epoch_labels == i).sum() for i in range(num_of_labels)]
            print('Class numbers in training set', counts)
            max_idx = counts.index(np.max(counts))
            min_idx = counts.index(np.min(counts))
            print('max_idx',max_idx)
            print('min_idx',min_idx)

            plt.figure(dpi=500) # figsize=(10,7), 
            num_bins = 50 #直方图柱子的数量
            names = ['aug', 'same', 'dif']
            idx = 0
            for sampels in [total_loss, same_loss, diff_loss]:
                max_class_samples = sampels[epoch_labels==max_idx]
                min_class_samples = sampels[epoch_labels==min_idx]

                mu_max = np.mean(max_class_samples) #计算均值
                mu_min = np.mean(min_class_samples) #计算均值
                sigma_max =np.std(max_class_samples) #计算标准差
                sigma_min =np.std(min_class_samples) #计算标准差

                n_max, bins_max, patches_max = plt.hist(max_class_samples, num_bins, density=True, edgecolor="black", alpha=0.2, label= names[idx]+' majority') # facecolor='blue',
                n_min, bins_min, patches_min = plt.hist(min_class_samples, num_bins, density=True, edgecolor="black", alpha=0.2, label= names[idx]+' minority')

                #直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象
                y_max = norm.pdf(bins_max, mu_max, sigma_max) #拟合一条最佳正态分布曲线y
                y_min = norm.pdf(bins_min, mu_min, sigma_min) #拟合一条最佳正态分布曲线y

                plt.plot(bins_max, y_max, '--', label =  names[idx]+' majority pdf') #绘制y的曲线
                plt.plot(bins_min, y_min, '--', label =  names[idx]+' minority pdf') #绘制y的曲线
                idx +=1

            plt.xlabel('Loss') #绘制x轴
            plt.ylabel('Probability') #绘制y轴
            plt.title(titles[title_idx])
            plt.legend(fontsize = 8 )
            plt.subplots_adjust(left=0.15) #左边距
            plt.savefig(savePath + f'/{epoch_idx}_{titles[title_idx]}.jpg')  # 图片保存

def draw_1_2_unnorm(selected_epochs, total_losses, total_labels, savePath, title = ''):
    # 1: (aug)
    # 2: (minority) (majority)
    savePath = os.path.join(savePath, '1_2_unnorm')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    titles = ['encoder_loss', 'projection_loss']
    
    for epoch_idx in selected_epochs:
        epoch_losses = total_losses[epoch_idx-1]
        epoch_labels = total_labels[epoch_idx-1]

        # epoch_losses = (epoch_losses-np.min(epoch_losses))/(np.max(epoch_losses) - np.min(epoch_losses))
        num_of_labels = np.max(epoch_labels)+1
        counts = [(epoch_labels == i).sum() for i in range(num_of_labels)]
        print('Class numbers in training set', counts)
        # max_idx = counts.index(np.max(counts))
        # min_idx = counts.index(np.min(counts))
        max_idx = 2
        min_idx = 1
        print('max_idx',max_idx)
        print('min_idx',min_idx)
       
        plt.figure(dpi=500) # figsize=(10,7), 
        num_bins = 50 #直方图柱子的数量
        sampels = epoch_losses[:,0]
        max_class_samples = sampels[epoch_labels==max_idx]
        min_class_samples = sampels[epoch_labels==min_idx]

        mu_max = np.mean(max_class_samples) #计算均值
        mu_min = np.mean(min_class_samples) #计算均值
        sigma_max = np.std(max_class_samples) #计算标准差
        sigma_min = np.std(min_class_samples) #计算标准差

        n_max, bins_max, patches_max = plt.hist(max_class_samples, num_bins, density=True, edgecolor="black", alpha=0.2, label= 'majority') # facecolor='blue',
        n_min, bins_min, patches_min = plt.hist(min_class_samples, num_bins, density=True, edgecolor="black", alpha=0.2, label= 'minority')

        #直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象
        y_max = norm.pdf(bins_max, mu_max, sigma_max) #拟合一条最佳正态分布曲线y
        y_min = norm.pdf(bins_min, mu_min, sigma_min) #拟合一条最佳正态分布曲线y

        plt.plot(bins_max, y_max, '--', label =  'majority pdf') #绘制y的曲线
        plt.plot(bins_min, y_min, '--', label =  'minority pdf') #绘制y的曲线

        plt.xlabel('Loss') #绘制x轴
        plt.ylabel('Probability') #绘制y轴
        plt.title('rnsda')
        plt.legend(fontsize = 8 )
        plt.subplots_adjust(left=0.15) #左边距
        plt.savefig(savePath + f'/{epoch_idx}_fuckyou.jpg')  # 图片保存

def draw_tnse(embeds, test_trues, savePath):

    def plot_embedding_2d(X, Y, savePath):
        """Plot an embedding X with the class label y colored by the domain d."""
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        num_of_labels = np.max(test_trues)+1
        counts = [(test_trues == i).sum() for i in range(num_of_labels)]
        print('Class numbers in training set', counts)
        max_idx = counts.index(np.max(counts))
        min_idx = counts.index(np.min(counts))
    
        # Plot colors numbers
        plt.figure(figsize=(10,10))
        ax = plt.subplot(111)

        for label_idx in [min_idx, max_idx]:
            sampels = X[Y==label_idx]
            plt.scatter(sampels[:, 0], sampels[:, 1], label='class_'+str(label_idx))

        plt.xticks([]), plt.yticks([])
        plt.legend()
        plt.title('tsne')
        
        plt.savefig(savePath + '/tsne.png')

    embeds = np.concatenate(embeds, axis=0)
    test_trues = test_trues.astype(int)
    
    from sklearn.manifold import TSNE
    tsne2d = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne_2d = tsne2d.fit_transform(embeds)
    
    plot_embedding_2d(X_tsne_2d[:,0:2], test_trues, savePath)

def draw_loss_batch_epoch(total_losses, total_labels, savePath, batch_size = 2*128 ):
    epoch_labels = total_labels[0]
    epoch_labels = torch.cat(epoch_labels).numpy()    
    num_of_labels = np.max(epoch_labels)+1
    counts = [(epoch_labels == i).sum() for i in range(num_of_labels)]
    max_idx = counts.index(np.max(counts))
    min_idx = counts.index(np.min(counts))

    majority_losses = []
    minority_losses = []

    color_maj = 'red'
    color_min = 'orange'

    for epoch_idx in range(len(total_losses)):
        epoch_losses = total_losses[epoch_idx-1]
        epoch_labels = total_labels[epoch_idx-1]

        epoch_labels = torch.cat(epoch_labels).numpy()              # 51200
        epoch_losses = np.concatenate(epoch_losses, axis=0)

        # epoch_aug_loss, _, _ = epoch_losses[:,0], epoch_losses[:,1], epoch_losses[:,2]
        epoch_aug_loss= epoch_losses[:,0]

        num_of_batch = epoch_aug_loss.shape[0]//batch_size
        for batch_idx in range(num_of_batch+1): 
            if (batch_idx+1)<= num_of_batch*batch_size:
                batch_aug_loss = epoch_aug_loss[batch_idx*batch_size:(batch_idx+1)*batch_size]
                batch_labels = epoch_labels[batch_idx*batch_size:(batch_idx+1)*batch_size]
            else:
                batch_aug_loss = epoch_aug_loss[batch_idx*batch_size:]
                batch_labels = epoch_labels[batch_idx*batch_size:]

            maj_loss = batch_aug_loss[batch_labels==max_idx]
            min_loss = batch_aug_loss[batch_labels==min_idx]

            majority_losses.append(np.mean(maj_loss))
            minority_losses.append(np.mean(min_loss))

        fig = plt.figure(dpi=500)
        total_batch_length = len( majority_losses)
        total_batch_x = np.linspace(0, total_batch_length-1, num=total_batch_length, dtype='int32')

        ax1 = plt.subplot2grid((3,1),(0,0),colspan=1,rowspan=1)
        ax1.plot(total_batch_x, majority_losses, color= color_maj, label = 'majority_'+str(max_idx) ) 
        ax1.plot(total_batch_x, minority_losses, color= color_min, label = 'minority_'+str(min_idx) ) 
       
        ax1.set_xlabel('epoch')  # x轴变量名称
        ax1.set_ylabel('loss value')  # x轴变量名称
        ax1.legend()
        ax1.set_title('Loss value for different classes')

        ax2 = plt.subplot2grid((3,1),(1,0),colspan=1,rowspan=1)
        ax2.plot(total_batch_x, majority_losses, color= color_maj, label = 'majority_'+str(max_idx) ) 
        ax2.set_xlabel('epoch')  # x轴变量名称
        ax2.set_ylabel('loss value')  # x轴变量名称
        ax2.legend()

        ax3 = plt.subplot2grid((3,1),(2,0),colspan=1,rowspan=1)
        ax3.plot(total_batch_x, minority_losses, color= color_min, label = 'minority_'+str(min_idx) )
        ax3.set_xlabel('epoch')  # x轴变量名称
        ax3.set_ylabel('loss value')  # x轴变量名称
        ax3.legend()
        
        fig.savefig(savePath + '/loss_with_batch.jpg')  # 图片保存

def draw_uncers(total_uncers, total_labels, savePath):
    
    epoch_uncers = total_uncers[-1]

    epoch_labels = total_labels[-1]
    epoch_labels = torch.cat(epoch_labels).numpy()    
    num_of_labels = np.max(epoch_labels)+1
    counts = [(epoch_labels == i).sum() for i in range(num_of_labels)]
    max_idx = counts.index(np.max(counts))
    min_idx = counts.index(np.min(counts))

    def plot_embedding_2d(X, Y, savePath):
        """Plot an embedding X with the class label y colored by the domain d."""
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
    
        # Plot colors numbers
        plt.figure(figsize=(10,10))
        ax = plt.subplot(111)

        for label_idx in range(num_of_labels):
            sampels = X[Y==label_idx]
            plt.scatter(sampels[:, 0], sampels[:, 1], label='class_'+str(label_idx))

        plt.xticks([]), plt.yticks([])
        plt.legend()
        plt.title('uncers')
        
        plt.savefig(savePath + '/uncers.png')

    epoch_uncers = torch.cat(epoch_uncers, dim=0).detach().cpu().numpy()    
    
    from sklearn.manifold import TSNE
    tsne2d = TSNE(n_components=2, init='pca', random_state=0)
    epoch_uncers = tsne2d.fit_transform(epoch_uncers)
    
    plot_embedding_2d(epoch_uncers[:,0:2], epoch_labels, savePath)

def draw_Pic(total_losses, total_labels, savePath):
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    # selected_epochs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    selected_epochs = [1, 10, 20, 30, 40]

    # draw_3_2_norm(selected_epochs, total_losses, total_labels, savePath)
    draw_1_2_unnorm(selected_epochs, total_losses, total_labels, savePath)

    # draw_tnse(embeds, test_trues, savePath)

    # draw_loss_batch_epoch( total_losses, total_labels, savePath)