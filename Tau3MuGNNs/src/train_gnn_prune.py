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
import copy

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
        if ("convs" in name) or ("bn" in name):
            continue
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




class Tau3MuGNNs:

    def __init__(self, config, device, log_path, setting):
        self.config = config
        self.device = device
        self.log_path = log_path
        self.writer = Writer(log_path)

        self.data_loaders, x_dim, edge_attr_dim, _ = get_data_loaders(setting, config['data'], config['optimizer']['batch_size'])

        # take ~10% of the "original" value each time, until last few iterations, reducing to ~1.2% original network size
        # self.prune_value_set = [0.0, 0.10, 0.111, .125, .143, .166, .20, .25, .333, .50, .666, .766]
        self.prune_value_set = [0.0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9]
        self.prune_value_set.append(0)  # Last 0 is so the final iteration can fine tune before testing
        common_dim = self.config['model']['out_channels']
        self.prune_mask = {
            "weight": {
                "node_encoder": torch.ones(common_dim, x_dim),
                "edge_encoder": torch.ones(common_dim, edge_attr_dim),
                "fc_out": torch.ones(1, common_dim)
            },
            "bias": {
                "node_encoder": torch.ones(common_dim),
                "edge_encoder": torch.ones(common_dim),
                "fc_out": torch.ones(1)
            }
        }
        # make masks for mlp blocks
        for idx in range(self.config['model']['n_layers']):
            self.prune_mask["weight"][f"mlps.{idx}.fc1"] = torch.ones(2*common_dim, common_dim)
            self.prune_mask["weight"][f"mlps.{idx}.fc2"] = torch.ones(common_dim, 2*common_dim)

            self.prune_mask["bias"][f"mlps.{idx}.fc1"] = torch.ones(2*common_dim)
            self.prune_mask["bias"][f"mlps.{idx}.fc2"] = torch.ones(common_dim)
        
        self.model = PruneModel(self.prune_mask, x_dim, edge_attr_dim, config['data']['virtual_node'], config['model'])

        # save initial state_dict for lottery pruning
        self.init_sd = self.model.state_dict()
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['optimizer']['lr'])
        self.criterion = Criterion(config['optimizer'])
        
        # for kl divergence residual node pruning
        self.kl_prune_counter = np.zeros(common_dim)

        print(f'[INFO] Number of trainable parameters: {sum(p.numel() for p in self.model.parameters())}')




    # @torch.no_grad()
    # def eval_one_batch(self, data):
    #     self.model.eval()

    #     clf_probs = self.model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch, data=data)
    #     loss, loss_dict = self.criterion(clf_probs, data.y)
    #     return loss_dict, clf_probs.data.cpu()

    # def train_one_batch(self, data):
    #     self.model.train()

    #     clf_probs = self.model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch, data=data)
    #     loss, loss_dict = self.criterion(clf_probs, data.y)

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss_dict, clf_probs.data.cpu()

    @torch.no_grad()
    def eval_one_batch(self, model, data):
        model.eval()

        clf_probs = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch, data=data)
        loss, loss_dict = self.criterion(clf_probs, data.y)
        return loss_dict, clf_probs.data.cpu()

    def train_one_batch(self, model, data):
        model.train()
        model.to(self.device)
        model.mask_to_device(self.device)
        # print(f"train model device: {model.node_encoder.weight.device}")
        # print(f"train mask device: {model.m_node_encoder_weight.device}")
        # print(f"data device: {data.x.device}")
        # print(f"data device: {data.y.device}")
        clf_probs = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch, data=data)
        # print(f"clf_probs device: {clf_probs.device}")
        loss, loss_dict = self.criterion(clf_probs, data.y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_dict, clf_probs.data.cpu()

    def run_one_epoch(self, data_loader, epoch, phase, break_len = None):
        loader_len = len(data_loader)
        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

        all_loss_dict, all_clf_probs, all_clf_labels = {}, [], []
        pbar = tqdm(data_loader, total=loader_len)
        # break_len = 50
        for idx, data in enumerate(pbar):
            if break_len is not None:
                if idx == break_len:
                    break
            loss_dict, clf_probs = run_one_batch(self.model, data.to(self.device))

            desc = log_epoch(epoch, phase, loss_dict, clf_probs, data.y.data.cpu(), batch=True)
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v
            all_clf_probs.append(clf_probs), all_clf_labels.append(data.y.data.cpu())

            last_idx = (loader_len - 1) if break_len is None else (break_len - 1)
            if idx == last_idx:
                all_clf_probs, all_clf_labels = torch.cat(all_clf_probs), torch.cat(all_clf_labels)
                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len
                desc, auroc, recall, avg_loss = log_epoch(epoch, phase, all_loss_dict, all_clf_probs, all_clf_labels, False, self.writer)
                print(desc)
            pbar.set_description(desc)

        return avg_loss, auroc, recall

    def run_one_epoch_kl_prune(self, model, data_loader, break_len = None):
        all_clf_probs = []
        for idx, data in enumerate(data_loader):
            if break_len is not None:
                if idx == break_len:
                    break
            _, clf_probs = self.eval_one_batch(model, data.to(self.device))
            all_clf_probs.append(clf_probs)
        return torch.cat(all_clf_probs)

    def getKLDivergence(self, data_loader, new_model,  break_len = None):
        # get clf_probs of the original
        new_model.to(self.device)
        new_model.mask_to_device(self.device)
        self.model.to(self.device)
        self.model.mask_to_device(self.device)
        original_clf_probs = self.run_one_epoch_kl_prune(
            self.model, data_loader, break_len=break_len
        ).flatten()
        new_clf_probs = self.run_one_epoch_kl_prune(
            new_model, data_loader, break_len=break_len
        ).flatten()
        new_model.to('cpu')
        new_model.mask_to_device('cpu')
        self.model.to('cpu')
        self.model.mask_to_device('cpu')
        
        # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html 
        # states that the input is assumed to be log probability, whereas target is not
        return torch.nn.KLDivLoss(reduction='batchmean')(torch.log(new_clf_probs), original_clf_probs)
        # kl_div_arr = torch.zeros(len(original_clf_probs))
        # # original_clf_probs is p_i, and new_clf_probs is q_i
        # kl_div_arr += original_clf_probs *(torch.log(original_clf_probs)- torch.log(new_clf_probs))
        # kl_div_arr += (1-original_clf_probs)*(torch.log(1-original_clf_probs)- torch.log(1-new_clf_probs))
        # return torch.mean(kl_div_arr)
        

    @torch.no_grad()
    def getKLPrunedModelNMask(self, res_prune_idxs):
        """
        Asummes the self.model is torch prune model, having weight_mask 
        and weight as its attributes
        """
        # deepcopying the prune model doesn't work, so just load the state dict
        new_model = PruneModel(
            self.prune_mask, self.model.x_dim, self.model.edge_attr_dim, 
            self.model.virtual_node, self.model.model_config
        )
        for name, module in new_model.named_modules():  # re-apply current mask to the model
            if isinstance(module, torch.nn.Linear):
                # prune.custom_from_mask(module, "weight", mask[name])
                prune.custom_from_mask(module, "weight", self.prune_mask["weight"][name])
                prune.custom_from_mask(module, "bias", self.prune_mask["bias"][name])
        new_model.load_state_dict(self.model.state_dict())
        # new_model = copy.deepcopy(self.model)
        new_model.to('cpu')
        new_model.mask_to_device('cpu')
        # new_prune_mask = copy.deepcopy(self.prune_mask)
        # prune residual nodes related to the res_prune_idx
        common_dim = self.model.out_channels
        pruning_condition = torch.zeros(common_dim, dtype=torch.bool)
        pruning_condition[res_prune_idxs] = True
        M = torch.eye(common_dim)
        M[pruning_condition, pruning_condition] = 0
        # new_model.node_encoder.weight.data = torch.mm(M, new_model.node_encoder.weight.data)
        # new_model.edge_encoder.weight.data = torch.mm(M, new_model.edge_encoder.weight.data)
        # new_prune_mask["weight"]["node_encoder"] = torch.mm(M, new_prune_mask["weight"]["node_encoder"])
        # new_prune_mask["weight"]["edge_encoder"] = torch.mm(M, new_prune_mask["weight"]["edge_encoder"])
        new_model.node_encoder.weight_mask = torch.mm(M, new_model.node_encoder.weight_mask)
        new_model.node_encoder.weight = torch.mm(M, new_model.node_encoder.weight)
        new_model.edge_encoder.weight_mask = torch.mm(M, new_model.edge_encoder.weight_mask)
        new_model.edge_encoder.weight = torch.mm(M, new_model.edge_encoder.weight)
       
        # # zeroing the biases doesn't matter if we zero the first layer of mlps blocks,
        # # but just for the numbers ??? 
        # new_model.node_encoder.bias.data[res_prune_idx] = 0
        # new_model.edge_encoder.bias.data[res_prune_idx] = 0
        for idx in range(len(new_model.mlps)):
            mlp_block = new_model.mlps[idx]
            # mlp_block.fc1.weight.data = torch.mm(mlp_block.fc1.weight.data, M)
            # mlp_block.fc2.weight.data = torch.mm(M, mlp_block.fc2.weight.data)
            # new_prune_mask["weight"][f"mlps.{idx}.fc1"] = torch.mm(new_prune_mask["weight"][f"mlps.{idx}.fc1"], M)
            # new_prune_mask["weight"][f"mlps.{idx}.fc2"] = torch.mm(M, new_prune_mask["weight"][f"mlps.{idx}.fc2"])
            mlp_block.fc1.weight_mask = torch.mm(mlp_block.fc1.weight_mask, M)
            mlp_block.fc1.weight = torch.mm(mlp_block.fc1.weight, M)
            mlp_block.fc2.weight_mask = torch.mm(M, mlp_block.fc2.weight_mask)
            mlp_block.fc2.weight = torch.mm(M, mlp_block.fc2.weight)
        # new_model.fc_out.weight.data = torch.mm(new_model.fc_out.weight.data, M)
        # new_prune_mask["weight"]["fc_out"] = torch.mm(new_prune_mask["weight"]["fc_out"], M)
        new_model.fc_out.weight_mask = torch.mm(new_model.fc_out.weight_mask, M)
        new_model.fc_out.weight = torch.mm(new_model.fc_out.weight, M)

        # new_model.update_masks(new_prune_mask)
        # # new_model.to(self.device)
        # # new_model.mask_to_device(self.device)
        # new_model.force_mask_apply()
        # return new_model, new_prune_mask
        return new_model

    def prune_model(self, amount, mask, 
        method=prune.L1Unstructured, device = "cpu", additional_method = "vanilla"
        ):
        self.model.to('cpu')
        self.model.mask_to_device('cpu')
        # this loop changes torch model to torch prune model, with weight not being a parameter
        for name, module in self.model.named_modules():  # re-apply current mask to the model
            if isinstance(module, torch.nn.Linear):
                # prune.custom_from_mask(module, "weight", mask[name])
                prune.custom_from_mask(module, "weight", mask["weight"][name])
                prune.custom_from_mask(module, "bias", mask["bias"][name])
                module_parameters = list(module.named_parameters()) # for debugging
                module_named_buffers = list(module.named_buffers()) # for debugging

        res_prune_amount_lim = 0.1#0.3
        fc_out_amount = amount/4

        if (additional_method == "vanilla") or (additional_method == "residual"):
            # prune only fc_out
            # Connections to outputs are pruned at third of the rate of the rest of the network
            parameters_to_prune = [(self.model.fc_out, 'weight')] # NOTE: not sure if we want to prune that one bias term

            # enforce the amount limit
            fc_out_amount = fc_out_amount if fc_out_amount < res_prune_amount_lim else res_prune_amount_lim

            prune.global_unstructured(  
                parameters_to_prune,
                pruning_method=method,
                amount=fc_out_amount,
            )
            print(f"[INFO] fc out prune amount: {fc_out_amount}")

        print(f"[INFO] {additional_method} additional pruning method")
        if additional_method == "vanilla":
            pass
        elif additional_method == "residual":
            #find which nodes of fc_out are pruned, and propagate that to all residual connections
            fc_out_weight_zeros = (self.model.fc_out.weight_mask == 0).flatten()
            # obtain a mask diagonal matrix
            M = torch.eye(len(fc_out_weight_zeros))
            M[fc_out_weight_zeros, fc_out_weight_zeros] = 0
            # and update the weight and weight_mask so that the residual nodes are masked
            self.model.node_encoder.weight_mask = torch.mm(M, self.model.node_encoder.weight_mask)
            self.model.node_encoder.weight = torch.mm(M, self.model.node_encoder.weight)
            print(f"node encoder weight and weight mask same? {torch.all(self.model.node_encoder.weight_mask == (self.model.node_encoder.weight != 0))}")
            self.model.edge_encoder.weight_mask = torch.mm(M, self.model.edge_encoder.weight_mask)
            self.model.edge_encoder.weight = torch.mm(M, self.model.edge_encoder.weight)
            test_node_weight = self.model.node_encoder.weight.clone()
            test_edge_weight = self.model.edge_encoder.weight.clone()
            test_node_weight_mask = self.model.node_encoder.weight_mask.clone()
            
            test_mlp_weight = None
            for idx in range(len(self.model.mlps)):
                mlp_block = self.model.mlps[idx]
                mlp_block.fc1.weight_mask = torch.mm(mlp_block.fc1.weight_mask, M)
                mlp_block.fc1.weight = torch.mm(mlp_block.fc1.weight, M)
                mlp_block.fc2.weight_mask = torch.mm(M, mlp_block.fc2.weight_mask)
                mlp_block.fc2.weight = torch.mm(M, mlp_block.fc2.weight)
                test_mlp_weight = mlp_block.fc2.weight
        
        elif additional_method == "kl":    
            # use kl divergence residual channel pruning method shown in https://openaccess.thecvf.com/content_CVPR_2020/papers/Luo_Neural_Network_Pruning_With_Residual-Connections_and_Limited-Data_CVPR_2020_paper.pdf
            kl_break_len = self.config["model"]["kl_method_break_len"]
            # get residual node/channel indexes to prune

            # get amount of nodes to prune
            total_n_nodes_to_prune = fc_out_amount if fc_out_amount < res_prune_amount_lim else res_prune_amount_lim
            total_n_nodes_to_prune = int(total_n_nodes_to_prune*self.config['model']['out_channels'])
            print(f"total_n_nodes_to_prune: {total_n_nodes_to_prune}")
            
            pruned_idxs = np.argwhere(self.kl_prune_counter==100).flatten() # val of hundred means it's pruned
            n_nodes_to_prune = total_n_nodes_to_prune - len(pruned_idxs)
            if n_nodes_to_prune > 0:
                # grab all node idx that aren't pruned yet
                if len(pruned_idxs) ==0:
                    prunable_idxs = np.arange(len(self.kl_prune_counter))
                else:
                    prunable_idxs = np.argpartition(self.kl_prune_counter, -len(pruned_idxs))[:-len(pruned_idxs)]
                
                for prunable_idx in prunable_idxs:
                    new_model = self.getKLPrunedModelNMask([prunable_idx])
                    kl_div = self.getKLDivergence(
                        self.data_loaders['train'], new_model, break_len=kl_break_len
                    )
                    # print(f"kl_div: {kl_div}")
                    # update prune counter
                    self.kl_prune_counter[prunable_idx] = kl_div

                # get the lowest kl div nodes and prune
                idxs_to_prune = np.argpartition(self.kl_prune_counter, n_nodes_to_prune)[:n_nodes_to_prune]
                # prune 
                self.model = self.getKLPrunedModelNMask([idxs_to_prune])
                # update prune counter after pruning
                self.kl_prune_counter[idxs_to_prune] = 100
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['optimizer']['lr'])
                self.criterion = Criterion(self.config['optimizer'])

            print(f"kl pruning test: {torch.all(self.model.node_encoder.weight.data[self.kl_prune_counter==100,:] == 0)}")
            # get the best percent of 

        else:
            raise NotImplementedError


        # now global prune
        parameters_to_prune = [
            (self.model.node_encoder, 'weight'),
            (self.model.edge_encoder, 'weight'),
            # (self.model.fc_out, 'weight'),
            (self.model.node_encoder, 'bias'),
            (self.model.edge_encoder, 'bias'),
            # (self.model.fc_out, 'bias')
        ]
        for idx in range(len(self.model.mlps)):
            mlp = self.model.mlps[idx]
            parameters_to_prune.append(((mlp.fc1, 'weight')))
            parameters_to_prune.append(((mlp.fc2, 'weight')))
            parameters_to_prune.append(((mlp.fc1, 'bias')))
            parameters_to_prune.append(((mlp.fc2, 'bias')))

        prune.global_unstructured(  # global prune the model
            parameters_to_prune,
            pruning_method=method,
            amount=amount,
        )
        print(f"[INFO] global prune amount: {amount}")

        for name, module in self.model.named_modules():  # make pruning "permanant" by removing the orig/mask values from the state dict
            if isinstance(module, torch.nn.Linear):
                # torch.logical_and(module.weight_mask, mask[name],
                #                   out=mask[name])  # Update progress mask
                # name_mask = mask["weight"][name]
                test_mask = module.weight_mask.clone()
                torch.logical_and(module.weight_mask, mask["weight"][name],
                                out=mask["weight"][name])  # Update progress mask
                torch.logical_and(module.bias_mask, mask["bias"][name],
                                out=mask["bias"][name])  # Update progress mask
                # print(f'update mask different?: {torch.sum(test_mask != mask["weight"][name])}')
                # print(f'update mask same?: {torch.sum(test_mask == mask["weight"][name])}')
                prune.remove(module, 'weight')  # remove all those values in the global pruned model
                prune.remove(module, 'bias')  # remove all those values in the global pruned model

        
        # print(f"test_node_weight {torch.all(test_node_weight == model.node_encoder.weight.data)}")
        # print(f"test_node_weight_mask {torch.all(test_node_weight_mask == (model.node_encoder.weight.data != 0))}")
        # print(f"test_edge_weight {torch.all(test_edge_weight == model.edge_encoder.weight.data)}")
        # print(f"test_mlp_weight {torch.all(test_mlp_weight == model.mlps[-1].fc2.weight.data)}")
        # sd = model.state_dict() # for debugging
        self.model.to(device)
        self.model.mask_to_device(device)
        return self.model


    def train(self):
        torch.autograd.set_detect_anomaly(True)
        start_epoch = 0
        best_val_recall = 0
        # prune_epoch_interval = self.config["model"]["prune_interval"]
        prune_epoch_interval = 250
        print(f"[INFO] prune_epoch_interval: {prune_epoch_interval}")
        prune_value_idx = 0
        if self.config['optimizer']['resume']:
            start_epoch, ckpt_data = load_checkpoint(self.model, self.optimizer, self.log_path, self.device)
            best_val_recall, _ = ckpt_data

        self.model.update_masks(self.prune_mask)
        self.model.to(self.device)
        self.model.mask_to_device(self.device)
        self.model.force_mask_apply()
        prune_value = self.prune_value_set[prune_value_idx]

        for epoch in range(start_epoch, self.config['optimizer']['epochs'] + 1):
            time_to_prune = epoch%prune_epoch_interval==0 and self.config["model"]["lottery"]
            
            time_to_prune =  time_to_prune and (prune_value_idx < len(self.prune_value_set)) # if already at last prune_value, then skip
            
            break_len = None
            # break_len = 50
            self.run_one_epoch(self.data_loaders['train'], epoch, 'train', break_len=break_len)
            if epoch % self.config['eval']['test_interval'] == 0:
                valid_res = self.run_one_epoch(self.data_loaders['valid'], epoch, 'valid', break_len=break_len)
                test_res = self.run_one_epoch(self.data_loaders['test'], epoch, 'test', break_len=break_len)
                # print(f"valid_res: {valid_res}")
                # end_prune_value_idx = len(self.prune_value_set)
                # if ((valid_res[-1] >= best_val_recall) and 
                # (prune_value_idx == end_prune_value_idx)):
                if (valid_res[-1] >= best_val_recall):
                    best_val_recall, best_test_recall, best_epoch = valid_res[-1], test_res[-1], epoch
                    best_val_auroc = valid_res[-2]
                    prune_value_str = f"{prune_value:.3g}".replace('.', "_")
                    save_path = Path(str(self.log_path) + f"/prune_val{prune_value_str}")
                    save_path.mkdir(parents=True, exist_ok=True)
                    save_checkpoint(
                        self.model, self.optimizer, save_path, epoch,
                        best_val_recall = best_val_recall,
                        best_val_auroc = best_val_auroc
                    )
                    print(f"saving model at {save_path}")

            if time_to_prune:  # if using lottery ticket method, reset all weights to first initalized vals
                best_val_recall = 0 # reset best_val_recall

                # now prune the model
                prune_value = self.prune_value_set[prune_value_idx]
                print(f"[INFO] new prune value: {prune_value}")
                prune_value_idx += 1 # increment for next time
                # Prune for next iter
                if prune_value > 0:
                    print("Pre Pruning: ")
                    countNonZeroWeights(self.model)
                    test_prune_node_encoder_mask = self.prune_mask["weight"]["node_encoder"].clone()
                    self.model = self.prune_model(
                        prune_value, self.prune_mask, 
                        device = self.device, additional_method=self.config["model"]["additional_prune_method"]
                    )
                    # print(f"mask same?: {torch.all(test_prune_node_encoder_mask == self.prune_mask['weight']['node_encoder'])}")
                    # Plot weight dist
                    # filename = path.join(options.outputDir, 'weight_dist_{}b_e{}_{}.png'.format(nbits, epoch_counter, time))
                    print("Post Pruning: ")
                    countNonZeroWeights(self.model)
                    # print(f"self.model: {self.model}")
                    # plot_kernels(model,
                    #                             text=' (Pruned ' + str(base_params - pruned_params) + ' out of ' + str(
                    #                             base_params) + ' params)',
                    #                             output=filename)
                    print("~~~~~!~!~!~!~!~!~Resetting Model!~!~!~!~!~!~~~~~\n\n")
                    print("Resetting Model to Inital State dict with masks applied. Verifying via param count.\n\n")
                    self.model.load_state_dict(self.init_sd)
                    self.model.update_masks(self.prune_mask)
                    self.model.to(self.device)
                    self.model.mask_to_device(self.device)
                    self.model.force_mask_apply()
                    countNonZeroWeights(self.model)
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
