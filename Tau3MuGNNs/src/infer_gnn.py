import yaml
from pathlib import Path
from utils import get_data_loaders, load_checkpoint, log_epoch, Criterion, add_cuts_to_config
import torch
from models import Model
from tqdm import tqdm
import pandas as pd

cuda_id = 0
log_name = '05_16_2022_22_23_30-GNN_full_dR_1'  # log id of the saved model to load
setting = log_name.split('-')[1]

config = yaml.safe_load(Path(f'./configs/{setting}.yml').open('r'))
device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')
log_path = Path(config['data']['log_dir']) / log_name

if len(log_name.split('-')) > 2:
    cut_id = log_name.split('-')[2]
    config = add_cuts_to_config(config, cut_id)
    
data_loaders, x_dim, edge_attr_dim, dataset = get_data_loaders(setting, config['data'], config['optimizer']['batch_size'])

df = pd.read_pickle(dataset.get_df_save_path())
assert len(df) == len(dataset)

model = Model(x_dim, edge_attr_dim, config['data']['virtual_node'], config['model']).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['lr'])
load_checkpoint(model, optimizer, log_path, device)
criterion = Criterion(config['optimizer'])

torch.no_grad()
def eval_one_batch(data, model, criterion):
    model.eval()

    clf_logits = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch, data=data)
    loss, loss_dict = criterion(clf_logits.sigmoid(), data.y)
    return loss_dict, clf_logits.data.cpu()


def run_one_epoch(data_loader, epoch, phase, device, model, criterion):
    loader_len = len(data_loader)
    run_one_batch = eval_one_batch
    phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

    all_loss_dict, all_clf_logits, all_clf_labels, all_sample_idx = {}, [], [], []
    pbar = tqdm(data_loader, total=loader_len)
    for idx, data in enumerate(pbar):
        loss_dict, clf_logits = run_one_batch(data.to(device), model, criterion)

        desc = log_epoch(epoch, phase, loss_dict, clf_logits, data.y.data.cpu(), batch=True)
        for k, v in loss_dict.items():
            all_loss_dict[k] = all_loss_dict.get(k, 0) + v
        all_clf_logits.append(clf_logits), all_clf_labels.append(data.y.data.cpu()), all_sample_idx.append(data.sample_idx.data.cpu())

        if idx == loader_len - 1:
            all_clf_logits, all_clf_labels, all_sample_idx = torch.cat(all_clf_logits), torch.cat(all_clf_labels), torch.cat(all_sample_idx)
            for k, v in all_loss_dict.items():
                all_loss_dict[k] = v / loader_len
            desc, auroc, recall, avg_loss = log_epoch(epoch, phase, all_loss_dict, all_clf_logits, all_clf_labels, False, None)
        pbar.set_description(desc)

    return avg_loss, auroc, recall, all_clf_logits, all_sample_idx
    
clf_probs, all_sample_idx = [], []
for phase in ['train', 'valid', 'test']:
    avg_loss, auroc, recall, clf_logits, sample_idx = run_one_epoch(data_loaders[phase], 999, phase, device, model, criterion)
    clf_probs.append(clf_logits.sigmoid())
    all_sample_idx.append(sample_idx)
clf_probs = torch.cat(clf_probs)
all_sample_idx = torch.cat(all_sample_idx)

scores = pd.DataFrame({'sample_idx': all_sample_idx, 'probs': clf_probs.reshape(-1)})
scores = scores.sort_values('sample_idx').reset_index(drop=True)

scores.to_pickle(dataset.get_df_save_path().parent / f'{setting}-scores.pkl')

