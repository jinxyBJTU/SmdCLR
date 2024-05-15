from sklearn.model_selection import KFold
from torch.optim import SGD
from torch.optim import Adam
from trainer import Trainer_mine
from model import base_Model
from dataloader import data_generator
import os
import sys
from tqdm.std import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path
from args import get_args
from printScore import PrintScore
from utils import AverageMeter, draw_Pic

sys.path.append(Path.cwd())
sys.path
# either sleepedf or sleepedf_eog

label_idx = 0
# moco 的K K
K, k = 32768, 2000
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_eps = 1e-8
lr = 0.0001
weight_decay = 1e-3

def main(args):

    if 'sleepEDF_JZY' in args.data_path:
        from config_files.sleepEDF_JZY_Configs import Config
        type_names = ['Wake','N1','N2','N3','REM']
    elif 'EDF-153' in args.data_path:
        from config_files.sleepEDF_153_Configs import Config
        type_names = ['Wake','N1','N2','N3','REM']
    elif 'ECG' in args.data_path:
        from config_files.ECG_Configs import Config
        type_names = ['AF','Normal','Other','Noisy']
    elif 'HAR' in args.data_path:
        from config_files.HAR_Configs import Config
        type_names = ['Walking','Upstairs','Downstairs','Standing','Sitting','Lying']
    elif 'TUSZ_12' in args.data_path:
        from config_files.tusz_12_Configs import Config
        type_names = ['CF','GN','AB','CT']
    elif 'mitbih' in args.data_path:
        from config_files.mitbih_Configs import Config
        type_names = ['N','S','V']
    else:
        print('New Dataset !!!')
        exit()

    configs = Config()
    device = torch.device( 'cuda:{}'.format(args.cuda) )
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train_dl, valid_dl, test_dl = data_generator(configs.batch_size, args.data_path, configs, args.training_mode, args.seed)
    
    
    dataset = args.data_path.split('/')[-2]  

    base_path = f"results/{args.model_name}_semi{configs.semi_rate}/{dataset}/"
        
    result_save_path = base_path + 'results/' + f'{args.seed}_batch{configs.batch_size}_temp{configs.Context_Cont.temperature}'
    log_save_path = base_path + 'log/' + f'{args.seed}_batch{configs.batch_size}_temp{configs.Context_Cont.temperature}_'
    model_save_path = base_path  + 'model_pretrained/' +  f'{args.seed}_batch{configs.batch_size}_temp{configs.Context_Cont.temperature}_pretrain{configs.train_epochs}.pkl'

    if not os.path.exists(base_path + 'log/'):
        os.makedirs(base_path + 'log/')
    if not os.path.exists(base_path + 'results/'):
        os.makedirs(base_path + 'results/')
    
    model = base_Model(args, configs).to(device)
    model_optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.sgd_momentum, weight_decay=args.weight_decay)
    # Trainer
    total_losses, total_labels = Trainer_mine(model, model_optimizer, train_dl, valid_dl, test_dl, device, args, configs, "self_supervised")
    # save_dict = {"model": model}  # "args": args
    # torch.save(save_dict, model_save_path)
    # print("save pretrain model to ", model_save_path)
    
    # configs.num_classes = num_of_classes    
    fine_tune_model = base_Model(args, configs, fine_tune=True).to(device)
    fine_tune_model_optimizer = SGD(fine_tune_model.parameters(), lr=args.lr, momentum=args.sgd_momentum, weight_decay=args.weight_decay)
    
    total_embeddings, trues, preds, logits = Trainer_mine(fine_tune_model, fine_tune_model_optimizer, train_dl, valid_dl, test_dl, device, args, configs, "fine_tune", pre_trained_model = model)
    
    PrintScore(type_names, trues, preds, logits, savePath=log_save_path, average='macro')

    draw_Pic(total_losses, total_labels, savePath = log_save_path )
    
    np.savez_compressed(
            result_save_path,
            train_losses = total_losses,
            train_labels = total_labels,
            preds = preds,
            trues = trues,
            logits = logits,
            total_embeddings = total_embeddings, 
            )
    print(result_save_path)


if __name__ == '__main__':
    main(get_args())
