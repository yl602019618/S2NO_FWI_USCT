import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def visual_train_breast(csv_file_path,save_path):
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        fig, axes = plt.subplots(2, 3, figsize=(10, 8))
        axes = axes.flatten()  # Flattening the axes to loop easily
        metrics = ['loss_epoch', 'loss_step', 'lr-AdamW', 'step', 'val_loss_breast','exp_loss_breast']
        titles = ['Loss (Epoch) vs Step', 'Loss (Step) vs Step', 'Learning Rate (AdamW) vs Step', 'Step Progression', 'Validation Loss vs Step','exp_loss_breast' ]
        y_labels = ['Loss (Epoch)', 'Loss (Step)', 'Learning Rate', 'Step', 'Validation Breast','Exp_loss_breast']

        for i, metric in enumerate(metrics):
            if metric == 'step':
                # If the metric is 'step', plot it against itself or a similar visualization
                axes[i].plot(df['step'], df['step'], '.', label='Step')
            else:
                if metric in df.keys().tolist():
                    axes[i].plot(df['step'], df[metric], '.', label=metric)

            axes[i].set_title(titles[i])
            axes[i].set_xlabel('Step')
            axes[i].set_ylabel(y_labels[i])
            axes[i].legend()
        plt.tight_layout()
        # Display the plot
        plt.savefig(os.path.join(save_path,'train_log_breast.png'))
        plt.close(fig)
    else:
        print(csv_file_path)
        print('No Path for log')

def visual_train_arm(csv_file_path,save_path):
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        fig, axes = plt.subplots(2, 3, figsize=(10, 8))
        axes = axes.flatten()  # Flattening the axes to loop easily
        metrics = ['loss_epoch', 'loss_step', 'lr-AdamW', 'step', 'val_loss_arm','exp_loss_arm']
        titles = ['Loss (Epoch) vs Step', 'Loss (Step) vs Step', 'Learning Rate (AdamW) vs Step', 'Step Progression', 'Validation Loss vs Step','exp_loss_arm' ]
        y_labels = ['Loss (Epoch)', 'Loss (Step)', 'Learning Rate', 'Step', 'Validation Arm','Exp_loss_arm']

        for i, metric in enumerate(metrics):
            if metric == 'step':
                # If the metric is 'step', plot it against itself or a similar visualization
                axes[i].plot(df['step'], df['step'], '.', label='Step')
            else:
                if metric in df.keys().tolist():
                    axes[i].plot(df['step'], df[metric], '.', label=metric)

            axes[i].set_title(titles[i])
            axes[i].set_xlabel('Step')
            axes[i].set_ylabel(y_labels[i])
            axes[i].legend()
        plt.tight_layout()
        # Display the plot
        plt.savefig(os.path.join(save_path,'train_log_arm.png'))
        plt.close(fig)
    else:
        print(csv_file_path)
        print('No Path for log')


def visual_train_limb(csv_file_path,save_path):
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        fig, axes = plt.subplots(2, 3, figsize=(10, 8))
        axes = axes.flatten()  # Flattening the axes to loop easily
        metrics = ['loss_epoch', 'loss_step', 'lr-AdamW', 'step', 'val_loss_limb','exp_loss_limb']
        titles = ['Loss (Epoch) vs Step', 'Loss (Step) vs Step', 'Learning Rate (AdamW) vs Step', 'Step Progression', 'Validation Loss vs Step','exp_loss_limb' ]
        y_labels = ['Loss (Epoch)', 'Loss (Step)', 'Learning Rate', 'Step', 'Validation Limb','Exp_loss_limb']

        for i, metric in enumerate(metrics):
            if metric == 'step':
                # If the metric is 'step', plot it against itself or a similar visualization
                axes[i].plot(df['step'], df['step'], '.', label='Step')
            else:
                if metric in df.keys().tolist():
                    axes[i].plot(df['step'], df[metric], '.', label=metric)

            axes[i].set_title(titles[i])
            axes[i].set_xlabel('Step')
            axes[i].set_ylabel(y_labels[i])
            axes[i].legend()
        plt.tight_layout()
        # Display the plot
        plt.savefig(os.path.join(save_path,'train_log_limb.png'))
        plt.close(fig)
    else:
        print(csv_file_path)
        print('No Path for log')


def visual_train(csv_file_path,save_path):
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        fig, axes = plt.subplots(2, 3, figsize=(10, 8))
        axes = axes.flatten()  # Flattening the axes to loop easily
        metrics = ['loss_epoch', 'loss_step', 'lr-AdamW', 'step', 'val_loss','exp_loss']
        titles = ['Loss (Epoch) vs Step', 'Loss (Step) vs Step', 'Learning Rate (AdamW) vs Step', 'Step Progression', 'Validation Loss vs Step','exp_loss' ]
        y_labels = ['Loss (Epoch)', 'Loss (Step)', 'Learning Rate', 'Step', 'Validation Loss','Exp_loss']

        for i, metric in enumerate(metrics):
            if metric == 'step':
                # If the metric is 'step', plot it against itself or a similar visualization
                axes[i].plot(df['step'], df['step'], '.', label='Step')
            else:
                if metric in df.keys().tolist():
                    axes[i].plot(df['step'], df[metric], '.', label=metric)

            axes[i].set_title(titles[i])
            axes[i].set_xlabel('Step')
            axes[i].set_ylabel(y_labels[i])
            axes[i].legend()
        plt.tight_layout()
        # Display the plot
        plt.savefig(os.path.join(save_path,'train_log.png'))
        plt.close(fig)
    else:
        print(csv_file_path)
        print('No Path for log')
