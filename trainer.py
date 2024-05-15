from loss import NTXentLoss
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import os
import sys
from augmentations import DataTransform
import time

sys.path.append("..")
sys.path.append("...")

def Trainer_mine(model, model_optimizer, train_dl, valid_dl, test_dl, device, args, config, training_mode, pre_trained_model = None):
    # Start training
    print("{} started ....".format(training_mode))

    criterion = nn.CrossEntropyLoss(reduction='none')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    if training_mode != "self_supervised":
        for name, param in model.named_parameters():
            if name not in ['classifier.weight', 'classifier.bias']:
                param.requires_grad = False

        state_dict = pre_trained_model.state_dict()
        msg = model.load_state_dict(state_dict, strict=False)

    if training_mode == 'self_supervised':
        n_epochs = config.train_epochs 
    else:
        n_epochs = config.fine_epochs
        
    total_losses, total_labels = [], []
    for epoch in range(1, n_epochs + 1):
        # Train and validate
       
        training_time, train_loss, train_acc, epoch_losses, epoch_labels = model_train(model, model_optimizer, criterion, train_dl, config, device, training_mode, args)
        
        testing_time, valid_loss, valid_acc, _, _,_,_  = model_evaluate( model, valid_dl, device, training_mode)

        # use scheduler in all other modes.
        if training_mode != 'self_supervised':
            scheduler.step(valid_loss)

        total_losses.append( epoch_losses )
        total_labels.append( epoch_labels )
        
        if training_mode == 'self_supervised':
            print(f'Pretraining Epoch : {epoch} | '
                f'Train Loss: {train_loss:.4f}\t| Train Time: {training_time:<7.2f}s | Test Time: {testing_time:<7.2f}s')
        else:
            print(f'Finetune Epoch : {epoch} | '
                  f'Train Accuracy: {train_acc:2.4f}\t| Valid Accuracy: {valid_acc:2.4f}\t| Train Time  : {training_time:<7.2f}s | Test Time  : {testing_time:<7.2f}s')

    # os.makedirs(os.path.join(experiment_log_dir,
    #             "saved_models"), exist_ok=True)
    # chkpoint = {'model_state_dict': model.state_dict(
    # ), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    # torch.save(chkpoint, os.path.join(
    #     experiment_log_dir, "saved_models", f'ckp_last.pt'))

    # no need to run the evaluation for self-supervised mode.
    if training_mode != "self_supervised":
        # evaluate on the test set
        print('\nEvaluate on the Test set:')
        _, test_loss, test_acc, total_embeddings, preds, trues, logits = model_evaluate( model, test_dl, device, training_mode)
        print(
            f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')
        return total_embeddings, trues, preds, logits
    else:
        print("\n################## Training is Done! #########################")
        return total_losses, total_labels

def get_labeled_samples(labels, ratio = 0.02):
        
        num_of_labels = torch.max(labels)+1
        each_ratio = ratio/num_of_labels
        batchsize = labels.shape[0]
        oversampling_numbers = int(batchsize * each_ratio) 
        
        all_idx = []
        for cur_idx in range(num_of_labels):  
            cur_samples_idx = [idx for idx, x in enumerate(labels) if x == cur_idx]
            num_of_cur_samples = len(cur_samples_idx)
            if num_of_cur_samples == 0:
                continue
            if num_of_cur_samples<oversampling_numbers:
                all_idx.extend(cur_samples_idx)
            else:
                all_idx.extend(cur_samples_idx[:oversampling_numbers])

        all_idx = all_idx + [ x+batchsize for x in all_idx ]
        return all_idx

def model_train(model, model_optimizer, classfication_criterion, train_loader, config, device, training_mode, args):
    total_loss = []
    total_acc = []
    total_embeds = []
    model.train()

    epoch_losses = []
    epoch_labels = []
    total_time = 0

    mse_criterion = nn.MSELoss()  # 定义损失函数

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = DataTransform(data.cpu(), config)
       
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        batch_size = data.shape[0]
        
        model_optimizer.zero_grad()
        start_time = time.time()

        if training_mode == "self_supervised":
           
            encoder_sim_matrix, encoder_embeddings, projection_embeddings, encoder_classification_results, projection_classification_results = model(aug1, aug2)      

            nt_xent_criterion = NTXentLoss(device, batch_size, config.Context_Cont.temperature, config.Context_Cont.use_cosine_similarity)

            # same_class_mask = generate_indicating_matrix(labels.repeat(2))
            # if 'proj_multidic' in args.model_name:
            #     loss, _ = nt_xent_criterion( projection_embeddings, encoder_sim_matrix)
            # else:
            #     loss, _ = nt_xent_criterion( projection_embeddings)      
            
            if 'proj_insdic' in args.model_name and 'enc_multidic' not in args.model_name:
                loss, _ = nt_xent_criterion( projection_embeddings ) 
            elif 'enc_multidic' in args.model_name:
                loss, _ = nt_xent_criterion( encoder_embeddings, encoder_sim_matrix) 
            else:
                print('Wrong Loss!!!')
                exit()
 
            if 'proj_insdic' in args.model_name:
                loss_proj, _ = nt_xent_criterion( projection_embeddings)
                loss += loss_proj
                     
            labeled_idx = get_labeled_samples(labels, config.semi_rate) # 2B

            if 'enc_semi' in args.model_name:
                encoder_classification_loss = classfication_criterion(encoder_classification_results[labeled_idx], labels.repeat(2)[labeled_idx])
                # loss += encoder_classification_loss
                # loss += torch.mean(encoder_classification_loss)
            if 'proj_semi' in args.model_name:
                projection_classification_loss = classfication_criterion(projection_classification_results[labeled_idx], labels.repeat(2)[labeled_idx])
                # loss += projection_classification_loss
                # loss += torch.mean(projection_classification_loss)

            selec_loss = torch.stack([encoder_classification_loss, projection_classification_loss],dim=-1)
            selec_labels = labels.repeat(2)[labeled_idx]
            
            epoch_losses.append( selec_loss.cpu().detach().numpy() ) #  .cpu().detach().numpy()
            epoch_labels.append( selec_labels.cpu().detach().numpy() ) 

        else:
            predictions, _ = model(data)
            # loss = classfication_criterion(predictions, labels)
            loss = torch.mean(classfication_criterion(predictions, labels))
            total_acc.append( labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        end_time = time.time()
        total_time += (end_time - start_time)

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
        epoch_losses = np.concatenate(epoch_losses, axis=0)
        epoch_labels = np.concatenate(epoch_labels, axis=0)
    else:
        total_acc = torch.tensor(total_acc).mean()
        
    return total_time, total_loss, total_acc, epoch_losses, epoch_labels


def model_evaluate(model, test_dl, device, training_mode):
    model.eval()
    total_loss = []
    total_acc = []
    total_embeddings = []
    total_time = 0

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])
    logits = []

    with torch.no_grad():
        for data, labels in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                start_time = time.time()
                output = model(data)
                end_time = time.time()
                total_time += (end_time - start_time)

                predictions, embeddings = output
                output_logits = torch.softmax(predictions, dim=-1)
                
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())
                total_embeddings.append(embeddings.cpu().numpy())

                # get the index of the max log-probability
                pred = predictions.max(1, keepdim=True)[1]
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())
                logits.append(output_logits.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
        total_acc = torch.tensor(total_acc).mean()  # average acc
        total_embeddings = np.concatenate(total_embeddings,axis=0)
        
        logits = np.concatenate(logits, axis=0)
        return total_time, total_loss, total_acc, total_embeddings, outs, trgs, logits
    else:
        total_acc = 0
        total_loss = 0
        return total_time, total_loss, total_acc, [], [],[],[]
    

