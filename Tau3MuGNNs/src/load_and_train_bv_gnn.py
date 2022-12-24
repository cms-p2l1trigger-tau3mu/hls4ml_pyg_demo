import shutil
import torch
import yaml
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from models import Model, BV_Model
from models.bv_model import convertBnToBvbn
from utils import Criterion, Writer, log_epoch, load_checkpoint, save_checkpoint, set_seed, get_data_loaders, add_cuts_to_config
from models.custom_bv import getCustomQuantizer
from models.bv_model import changeBnToStaticQuantizer
import os

class Tau3MuGNNs:
    def __init__(self, config, device, log_path, setting):
        self.config = config
        self.device = device
        self.log_path = log_path
        self.writer = Writer(log_path)

        self.data_loaders, x_dim, edge_attr_dim, _ = get_data_loaders(setting, config['data'], config['optimizer']['batch_size'])


        target_norm_fract_bitwidth = config['model']["norm_ap_fixed_fract"]
        # self.norm_fract_bitwidths = [i for i in range(target_norm_fract_bitwidth, 14, 2)]
        self.norm_fract_bitwidths = [i for i in range(target_norm_fract_bitwidth, 16+1, 2)] # 12+1 bc we want to include 12
        self.norm_fract_bitwidths.reverse() # reverse so bigger bitwidth comes first
        # self.norm_fract_bitwidths = [4] # for testing
        self.norm_int_bitwidth = config['model']["norm_ap_fixed_int"]
        print(f"[INFO] Target int bitwidth: {self.norm_int_bitwidth}")
        # change the fract bitwidth the model will be initialized with 
        config['model']["norm_ap_fixed_int"] = self.norm_fract_bitwidths.pop(0)
        print(f"[INFO] Current norm fract bitwidth: {config['model']['norm_ap_fixed_int']}") 
        self.model = BV_Model(x_dim, edge_attr_dim, config['data']['virtual_node'], config['model'])
        self.load_model_pth = Path(config['model']['saved_model_path'])
        state_dict = torch.load(self.load_model_pth / 'model.pt', map_location=self.device)
        self.state_dict = state_dict
        # self.load_model_best_val_recall = state_dict['best_val_recall@1kHz']
        # print(f"state_dict['best_val_recall@1kHz']: {state_dict['best_val_recall@1kHz']}")
        # convert the torch bn into quant bn
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.model = convertBnToBvbn(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['optimizer']['lr'])
        self.criterion = Criterion(config['optimizer'])
        _ = save_checkpoint(
                        self.model, self.optimizer, self.log_path, 
                        epoch = 0, best_val_recall=state_dict['best_val_recall@1kHz'],
                        best_val_auroc = state_dict['best_val_auroc']
        )
        self.model.to(self.device)
        

        
        print(f'[INFO] Number of trainable parameters: {sum(p.numel() for p in self.model.parameters())}')
        print(f"[INFO] load_and_train_bv_gnn.py")

    @torch.no_grad()
    def eval_one_batch(self, data):
        self.model.eval()

        clf_probs = self.model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch, data=data)
        loss, loss_dict = self.criterion(clf_probs, data.y)
        return loss_dict, clf_probs.data.cpu()

    def train_one_batch(self, data):
        self.model.train()

        clf_probs = self.model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch, data=data)
        loss, loss_dict = self.criterion(clf_probs, data.y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_dict, clf_probs.data.cpu()

    def run_one_epoch(self, data_loader, epoch, phase):
        loader_len = len(data_loader)
        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

        all_loss_dict, all_clf_probs, all_clf_labels = {}, [], []
        pbar = tqdm(data_loader, total=loader_len)
        # break_len = 50
        for idx, data in enumerate(pbar):
            # if idx == break_len:
            #     break

            loss_dict, clf_probs = run_one_batch(data.to(self.device))

            
            desc = log_epoch(epoch, phase, loss_dict, clf_probs, data.y.data.cpu(), batch=True)
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v
            all_clf_probs.append(clf_probs), all_clf_labels.append(data.y.data.cpu())

            if idx == loader_len - 1:
            # if idx == break_len - 1:
                # print(f"Phase: {phase}")
                all_clf_probs, all_clf_labels = torch.cat(all_clf_probs), torch.cat(all_clf_labels)
                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len
                desc, auroc, recall, avg_loss = log_epoch(epoch, phase, all_loss_dict, all_clf_probs, all_clf_labels, False, self.writer)
                print(desc)
            pbar.set_description(desc)
            

        return avg_loss, auroc, recall

    def train(self):
        print(f"pwd: {os.path.abspath(os.getcwd())}")
        start_epoch = 0
        # start_epoch = self.state_dict['epoch']
        # best_epoch = 0
        best_val_recall = 0 #self.state_dict['best_val_recall@1kHz']
        best_val_auroc = 0
        # quantizer_change_interval = 3*self.config['eval']['test_interval']
        quantizer_change_interval = 25 #150

        if self.config['optimizer']['resume']:
            start_epoch, ckpt_data = load_checkpoint(self.model, self.optimizer, self.log_path, self.device)
            best_val_recall, best_val_auroc = ckpt_data
        for epoch in range(start_epoch, self.config['optimizer']['epochs'] + 1):
            if ((epoch % quantizer_change_interval == 0) and (epoch !=0) and
            (len(self.norm_fract_bitwidths) !=0)):
                # # keep the original load path if no improvement, since nothing is save on log path
                # if (best_val_recall < self.load_model_best_val_recall):
                #     log_path = self.load_model_pth  
                # else :
                #     log_path = self.log_path
                log_path = self.log_path
                _ = load_checkpoint(self.model, self.optimizer, log_path, self.device)# load last best model
                fract_bitwidth = self.norm_fract_bitwidths.pop(0)
                print(f"changing norm fract bitwidth to {fract_bitwidth}") 
                quantizer_dict = getCustomQuantizer(self.norm_int_bitwidth, fract_bitwidth)
                self.model = changeBnToStaticQuantizer(self.model, quantizer_dict).to(self.device)
                # reset best_val_recall
                best_val_recall = 0
            self.run_one_epoch(self.data_loaders['train'], epoch, 'train')

            if epoch % self.config['eval']['test_interval'] == 0:
                valid_res = self.run_one_epoch(self.data_loaders['valid'], epoch, 'valid')
                test_res = self.run_one_epoch(self.data_loaders['test'], epoch, 'test')
                if valid_res[-1] >= best_val_recall:
                    best_val_recall, best_test_recall, best_epoch = valid_res[-1], test_res[-1], epoch
                    best_val_auroc = valid_res[-2]
                    save_checkpoint(
                        self.model, self.optimizer, self.log_path, 
                        epoch, best_val_recall=best_val_recall,
                        best_val_auroc = best_val_auroc
                    )
                    

            # self.writer.add_scalar('best/best_epoch', best_epoch, epoch)
            # self.writer.add_scalar('best/best_val_recall', best_val_recall, epoch)
            # self.writer.add_scalar('best/best_test_recall', best_test_recall, epoch)
            print('-' * 50)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train Tau3MuGNNs')
    parser.add_argument('--setting', type=str, help='experiment settings', default='GNN_half_dR_1')
    parser.add_argument('--cut', type=str, help='cut id', default=None)
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu', default=3)
    args = parser.parse_args()
    setting = args.setting
    cuda_id = args.cuda
    cut_id = args.cut
    print(f'[INFO] Running {setting} on cuda {cuda_id} with cut {cut_id}')

    torch.set_num_threads(5)
    set_seed(42)
    time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    config_path = f'./configs/{setting}.yml' # normal path
    config = yaml.safe_load(Path(config_path).open('r'))
    
    config = add_cuts_to_config(config, cut_id)
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')

    log_cut_name = '' if cut_id is None else f'-{cut_id}'
    log_name = f'{time}-{setting}{log_cut_name}' if not config['optimizer']['resume'] else config['optimizer']['resume']

    log_path = Path(config['data']['log_dir']) / log_name
    log_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, log_path / 'config.yml')

    print(f"config yml: {config}")
    Tau3MuGNNs(config, device, log_path, setting).train()


if __name__ == '__main__':
    import os
    # os.chdir('./src')
    print(f"current program pid: {os.getpid()}")
    main()
