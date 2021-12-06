import os
import copy
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import typing as Dict


def init_log_loss(last_log_loss_csv, num_models=None):
    last_best_avg_loss_all = np.inf
    if last_log_loss_csv is None:
        if num_models is None:
            raise ValueError('Missing num_models argument.')
        log_loss = {'Epoch': []}
        for model_idx in range(num_models):
            log_loss['train_loss_' + str(model_idx)] = []
            log_loss['val_loss_' + str(model_idx)] = []
        log_loss['train_loss_avg'] = []
        log_loss['val_loss_avg'] = []
    else:
        last_log_loss_df = pd.read_csv(last_log_loss_csv)
        last_log_loss = {col: list(last_log_loss_df[col]) for col in last_log_loss_df.columns}

        last_best_avg_loss_all = max(last_log_loss['val_loss_avg'])
        log_loss = last_log_loss

    return (log_loss, last_best_avg_loss_all)


def update_log_loss(log_loss, epoch, phase, avg_loss_each, avg_loss_all):
    if phase == 'train':
        log_loss['Epoch'].append(epoch)
        for model_idx in range(len(avg_loss_each)):
            log_loss['train_loss_' + str(model_idx)].append(avg_loss_each[model_idx])
        log_loss['train_loss_avg'].append(avg_loss_all)
    else:
        for model_idx in range(len(avg_loss_each)):
            log_loss['val_loss_' + str(model_idx)].append(avg_loss_each[model_idx])
        log_loss['val_loss_avg'].append(avg_loss_all)

    return log_loss


def single_training_loop(models, optimizer, loss_func, dataloader, device, save_dir, scheduler=None, num_epochs=5,
                         last_epoch=0, last_log_loss_csv: str = None):
    pass


def mul_model_training_loop(models, optimizer, loss_func, dataloader, device, save_dir, scheduler=None, num_epochs=5,
                            last_epoch=0, last_log_loss_csv: str = None):
    # TODO:
    #   what if loss_func_reduction != none
    #   use sample weight or not
    if loss_func.reduction != 'none':
        loss_func.reduction = 'none'
    num_models = len(models)

    # Model Init
    for model_idx in range(num_models):
        models[model_idx] = models[model_idx].to(device)

    # Load the log_loss and also the last_best_avg_loss_all
    # if there last_log_loss_csv not exist best_avg_loss_all = np.inf
    log_loss, best_avg_loss_all = init_log_loss(last_log_loss_csv, num_models)

    # Init essential value
    outputs, sample_sum_loss, sample_avg_loss = [0] * num_models, 0, 0

    # Start epoch
    for epoch in range(last_epoch + 1, num_epochs + 1):
        print('-' * 50)
        print("Epoch {}/{}".format(epoch, num_epochs - 1))

        # Start phase training or validation
        for phase in ['train', 'val']:
            epoch_time = time.time()
            # avg_loss_each (list): contain avg sample_loss per epoch of each model
            # num_sample_each (list): number sample of each value in epoch
            avg_loss_each = [0] * num_models
            num_sample_each = [0] * num_models

            for model_idx in range(num_models):
                if phase == 'train':
                    models[model_idx].train()
                else:
                    models[model_idx].eval()

            # Start training
            for inputs, labels, sample_weight in tqdm(dataloader[phase], desc="Epoch {} {}".format(epoch, phase),
                                                      unit="batch"):
                inputs = inputs.to(device)
                optimizer.zero_grad()

                # Training each model
                for model_idx in range(num_models):
                    labels[model_idx] = labels[model_idx].to(device)
                    sample_weight[model_idx] = sample_weight[model_idx].to(device)
                    num_sample = sample_weight[model_idx].sum()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs[model_idx] = models[model_idx](inputs)
                        # Compute sample_loss
                        sample_loss = loss_func(outputs[model_idx], labels[model_idx])
                        sample_sum_loss = (sample_loss * sample_weight[model_idx]).sum()
                        sample_avg_loss = sample_sum_loss / num_sample if num_sample != 0 else sample_sum_loss

                        if phase == 'train':
                            sample_avg_loss.backward()

                    # These variable support to compute average sample_loss of each model in this epoch
                    avg_loss_each[model_idx] += sample_sum_loss
                    num_sample_each[model_idx] += num_sample

                # Apply optimizer to all models
                if phase == 'train':
                    optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # Compute average sample_loss of each model in epoch
            for model_idx in range(num_models):
                if num_sample_each[model_idx] != 0:
                    avg_loss_each[model_idx] /= num_sample_each[model_idx]
                else:
                    avg_loss_each[model_idx] = 0

            # Compute average sample_loss of all models
            # avg_loss_all (float): is the avg sample_loss of all model in each epoch
            avg_loss_all = sum(avg_loss_each) / num_models
            epoch_time = time.time() - epoch_time

            log_loss = update_log_loss(log_loss=log_loss, epoch=epoch, phase=phase,
                                       avg_loss_each=avg_loss_each, avg_loss_all=avg_loss_all)

            print('{} Loss: {:.4f}, {} Latency: {:.0f}m{:.0f}s'.format(phase, avg_loss_all, phase,
                                                                       epoch_time / 60, epoch_time % 60))

            # Save model and log_loss file
            if phase == 'val':
                if avg_loss_all < best_avg_loss_all:
                    best_avg_loss_all = avg_loss_all
                    # Save the best model
                    for model_idx in range(num_models):
                        torch.save(models[model_idx],
                                   os.path.join(save_dir, 'best_model_{}.pt'.format(model_idx)))
                # Save models
                for model_idx in range(num_models):
                    torch.save(models[model_idx],
                               os.path.join(save_dir, 'model_{}_epoch_{}.pt'.format(model_idx, epoch)))
                # Save the log_loss
                print(log_loss)
                log_loss = {key: [val.detach().cpu().numpy() if torch.is_tensor(val) else val for val in vals]
                            for key, vals in log_loss.items()}
                pd.DataFrame(log_loss).to_csv(os.path.join(save_dir, 'log_loss.csv'), index=False)

    print('Best Val Acc: {:.4f}'.format(best_avg_loss_all))

    return models


if __name__ == '__main__':
    model = torch.load(r'D:\Machine Learning Project\5kCompliance\5kCompliance\best_model_0.pt',
                       map_location=torch.device('cpu'))
    torch.save(model.state_dict(), r'D:\Machine Learning Project\5kCompliance\5kCompliance\best_model_weight.pt')
    print(model)

    # print(init_log_loss(r'D:\Machine Learning Project\5kCompliance\5kCompliance\log_loss.csv'))
