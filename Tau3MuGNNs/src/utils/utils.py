import torch
import random
import numpy as np


def load_checkpoint(model, optimizer, log_path, device):
    print(f'[INFO] Loading checkpoint from {log_path.name}')
    checkpoint = torch.load(log_path / 'model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_recall = checkpoint['best_val_recall@1kHz']
    best_val_auroc = checkpoint['best_val_auroc']
    return (start_epoch, [best_val_recall, best_val_auroc])


def save_checkpoint(
    model, optimizer, log_path, epoch, 
    best_val_recall=None, best_val_auroc=None
):
    print(f'[INFO] Saving checkpoint to {log_path.name}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_recall@1kHz' : best_val_recall,
        'best_val_auroc' : best_val_auroc,
    }, log_path / 'model.pt')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def add_cuts_to_config(config, cut_id):
    if cut_id is None:
        return config
    config['data']['cut'] = f'{cut_id}'
    return config
