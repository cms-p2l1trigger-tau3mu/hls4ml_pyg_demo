import shutil
import torch
import yaml
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from models import PruneModel, Model
from utils import Criterion, Writer, log_epoch, load_checkpoint, save_checkpoint, set_seed, get_data_loaders, add_cuts_to_config
import torch.nn.utils.prune as prune
import numpy as np

# def countNonZeroWeights(model):
#     """
#     function taken from
#     https://github.com/ben-hawks/pytorch-jet-classify/blob/master/slider_autoencoder/ae_iter_prune.py
#     """
#     nonzero = total = 0
#     for name, p in model.named_parameters():
#         tensor = p.data.cpu().numpy()
#         nz_count = np.count_nonzero(tensor)
#         total_params = np.prod(tensor.shape)
#         nonzero += nz_count
#         total += total_params
#         print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
#     print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
#     return nonzero

def countNonZeroWeights(model):
    nonzero = total = 0
    layer_count_alive = {}
    layer_count_total = {}
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        layer_count_alive.update({name: nz_count})
        layer_count_total.update({name: total_params})
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    return nonzero, total, layer_count_alive, layer_count_total

def prune_model(model, amount, mask, method=prune.L1Unstructured, device = "cpu"):
    model.to('cpu')
    model.mask_to_device('cpu')
    for name, module in model.named_modules():  # re-apply current mask to the model
        if isinstance(module, torch.nn.Linear):
            prune.custom_from_mask(module, "weight", mask[name])
            module_parameters = list(module.named_parameters()) # for debugging
            module_named_buffers = list(module.named_buffers()) # for debugging

    # parameters_to_prune = [
    #     (model.node_encoder, 'weight'),
    #     (model.edge_encoder, 'weight'),
    # ]
    parameters_to_prune = [
        (model.node_encoder, 'weight'),
        (model.edge_encoder, 'weight'),
        (model.fc_out, 'weight')
    ]
    for idx in range(len(model.mlps)):
        mlp = model.mlps[idx]
        parameters_to_prune.append(((mlp.fc1, 'weight')))
        parameters_to_prune.append(((mlp.fc2, 'weight')))

    prune.global_unstructured(  # global prune the model
        parameters_to_prune,
        pruning_method=method,
        amount=amount,
    )
    # # Connections to outputs are pruned at half of the rate of the rest of the network
    # parameters_to_prune = [(model.fc_out, 'weight')]
    # prune.global_unstructured(  # global prune the model
    #     parameters_to_prune,
    #     pruning_method=method,
    #     amount=amount/2,
    # )

    for name, module in model.named_modules():  # make pruning "permanant" by removing the orig/mask values from the state dict
        if isinstance(module, torch.nn.Linear):
            torch.logical_and(module.weight_mask, mask[name],
                              out=mask[name])  # Update progress mask
            prune.remove(module, 'weight')  # remove all those values in the global pruned model

    model.to(device)
    model.mask_to_device(device)
    return model

class Tau3MuGNNs:

    def __init__(self, config, device, log_path, setting):
        self.config = config
        self.device = device
        self.log_path = log_path
        self.writer = Writer(log_path)

        self.data_loaders, x_dim, edge_attr_dim, _ = get_data_loaders(setting, config['data'], config['optimizer']['batch_size'])

        # take ~10% of the "original" value each time, until last few iterations, reducing to ~1.2% original network size
        # self.prune_value_set = [0.0, 0.10, 0.111, .125, .143, .166, .20, .25, .333, .50, .666, .766]
        self.prune_value_set = [0.0, 0.5, 0.8, .964, .981]
        self.prune_value_set.append(0)  # Last 0 is so the final iteration can fine tune before testing
        common_dim = self.config['model']['out_channels']
        self.prune_mask = {
            "node_encoder": torch.ones(common_dim, x_dim),
            "edge_encoder": torch.ones(common_dim, edge_attr_dim),
            "fc_out": torch.ones(1, common_dim)
        }
        # make masks for mlp blocks
        # for idx in range(self.config['model']['n_layers']):
        #     mlp_mask = {
        #         "fc1" :torch.ones(2*common_dim, common_dim),
        #         "fc2" :torch.ones(common_dim, 2*common_dim)
        #     }
        #     self.prune_mask[f"mlp_{idx}"] = mlp_mask
        for idx in range(self.config['model']['n_layers']):
            self.prune_mask[f"mlps.{idx}.fc1"] = torch.ones(2*common_dim, common_dim)
            self.prune_mask[f"mlps.{idx}.fc2"] = torch.ones(common_dim, 2*common_dim)
        
        self.model = PruneModel(self.prune_mask, x_dim, edge_attr_dim, config['data']['virtual_node'], config['model'])
        # self.model = Model(x_dim, edge_attr_dim, config['data']['virtual_node'], config['model'])

        # save initial state_dict for lottery pruning
        self.init_sd = self.model.state_dict()
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['optimizer']['lr'])
        self.criterion = Criterion(config['optimizer'])
        
        

        print(f'[INFO] Number of trainable parameters: {sum(p.numel() for p in self.model.parameters())}')
        
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
                all_clf_probs, all_clf_labels = torch.cat(all_clf_probs), torch.cat(all_clf_labels)
                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len
                desc, auroc, recall, avg_loss = log_epoch(epoch, phase, all_loss_dict, all_clf_probs, all_clf_labels, False, self.writer)
                print(desc)
            pbar.set_description(desc)

        return avg_loss, auroc, recall

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        start_epoch = 0
        best_val_recall = 0
        # prune_epoch_interval = self.config["model"]["prune_interval"]
        prune_epoch_interval = 200
        print(f"[INFO] prune_epoch_interval: {prune_epoch_interval}")
        prune_value_idx = 0
        if self.config['optimizer']['resume']:
            start_epoch, ckpt_data = load_checkpoint(self.model, self.optimizer, self.log_path, self.device)
            best_val_recall, _ = ckpt_data

        # self.model.mask_to_device(self.device)
        

        for epoch in range(start_epoch, self.config['optimizer']['epochs'] + 1):
            time_to_prune = epoch%prune_epoch_interval==0 and self.config["model"]["lottery"]
            
            time_to_prune =  time_to_prune and (prune_value_idx < len(self.prune_value_set)) # if already at last prune_value, then skip

            if time_to_prune:
                print("~~~~~!~!~!~!~!~!~Resetting Model!~!~!~!~!~!~~~~~\n\n")
                print("Resetting Model to Inital State dict with masks applied. Verifying via param count.\n\n")
                self.model.load_state_dict(self.init_sd)
                self.model.update_masks(self.prune_mask)
                self.model.to(self.device)
                self.model.mask_to_device(self.device)
                self.model.force_mask_apply()
                countNonZeroWeights(self.model)


            self.run_one_epoch(self.data_loaders['train'], epoch, 'train')

            if epoch % self.config['eval']['test_interval'] == 0:
                valid_res = self.run_one_epoch(self.data_loaders['valid'], epoch, 'valid')
                test_res = self.run_one_epoch(self.data_loaders['test'], epoch, 'test')
                # print(f"valid_res: {valid_res}")
                end_prune_value_idx = len(self.prune_value_set)
                if ((valid_res[-1] >= best_val_recall) and 
                (prune_value_idx == end_prune_value_idx)):
                    best_val_recall, best_test_recall, best_epoch = valid_res[-1], test_res[-1], epoch
                    best_val_auroc = valid_res[-2]
                    save_checkpoint(
                        self.model, self.optimizer, self.log_path, epoch,
                        best_val_recall = best_val_recall,
                        best_val_auroc = best_val_auroc
                    )

            if time_to_prune:  # if using lottery ticket method, reset all weights to first initalized vals
                
                # now prune the model
                prune_value = self.prune_value_set[prune_value_idx]
                prune_value_idx += 1 # increment for next time
                # Prune for next iter
                if prune_value > 0:
                    self.model = prune_model(self.model, prune_value, self.prune_mask, device = self.device)
                    # Plot weight dist
                    # filename = path.join(options.outputDir, 'weight_dist_{}b_e{}_{}.png'.format(nbits, epoch_counter, time))
                    print("Post Pruning: ")
                    # print(f"self.model: {self.model}")
                    pruned_params,_,_,_ = countNonZeroWeights(self.model)
                    # plot_kernels(model,
                    #                             text=' (Pruned ' + str(base_params - pruned_params) + ' out of ' + str(
                    #                             base_params) + ' params)',
                    #                             output=filename)
                    print(f"[INFO] pruned_params: {pruned_params}")
            self.writer.add_scalar('best/best_epoch', best_epoch, epoch)
            self.writer.add_scalar('best/best_val_recall', best_val_recall, epoch)
            self.writer.add_scalar('best/best_test_recall', best_test_recall, epoch)
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
