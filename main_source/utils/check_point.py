import torch
import os

def load_checkpoint():
    return True


def save_model(net_state_dict, epoch, test_acc, save_folder, filename):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    path = os.path.join(save_folder, filename)
    torch.save({
        'epoch': epoch,
        'test_acc': test_acc,
        'net_state_dict': net_state_dict},
        path)
