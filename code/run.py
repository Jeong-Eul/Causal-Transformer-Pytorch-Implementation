# """Here is an initial version of the VITAL. The scripts will be further refined in the future, after paper acceptance. """

import wandb
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import hydra
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from src.models.utils import AlphaRise, FilteringMlFlowLogger
from src.data.cancer_sim.dataset import SyntheticCancerDataset, SyntheticCancerDatasetCollection
from src.models.ct import CT
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)

        
@hydra.main(version_base="1.1", config_name=f'config.yaml', config_path='./config/')
def main(args: DictConfig):
    
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100000"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['WANDB_SILENT']="true"
    
    # Non-strict access to fields
    OmegaConf.set_struct(args, False) # 없는 키를 만들어도 됨
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True) # config 안에서 두 수를 더하는 커스텀 함수 지원
    
    wandb_use = True
    
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='Cancer', choices=['Cancer', 'MIMIC', 'eICU']) #
    # parser.add_argument('--withmissingratio', default=False, help='if True, missing ratio ranges from 0 to 0.5; if False, missing ratio =0') #
    # parser.add_argument('--splittype', type=str, default='random', choices=['random', 'age', 'gender'], help='only use for P12 and P19')
    # parser.add_argument('--reverse', default=False, help='if True,use female, older for tarining; if False, use female or younger for training') #
    # parser.add_argument('--feature_removal_level', type=str, default='no_removal', choices=['no_removal', 'set', 'sample', 'lab'],
    #                     help='use this only when splittype==random; otherwise, set as no_removal') #
    # parser.add_argument('--predictive_label', type=str, default='mortality', choices=['mortality', 'LoS'],
    #                     help='use this only with P12 dataset (mortality or length of stay)')


    # # Model configuration
    # parser.add_argument('--enc_in', type=int, default=34, help='encoder input size')
    # parser.add_argument('--num_tokens', type=int, default=8)
    # parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    # parser.add_argument('--n_heads', type=int, default=1, help='num of heads')
    # parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    # parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model') # LLAMA, GPT2, BERT, MAMBA
    # parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768, MAMBA:768
    # parser.add_argument('--llm_layers', type=int, default=12)
    
    # args, unknown = parser.parse_known_args()
    
    # set seed
    torch.manual_seed(args.exp.seed)
    torch.cuda.manual_seed(args.exp.seed)
    torch.cuda.manual_seed_all(args.exp.seed)
    np.random.seed(args.exp.seed)

    arch = 'CT'
    model_path = './models/'
    os.makedirs(model_path, exist_ok=True)
    
    dataset = 'Cancer'
    print('Dataset used: ', dataset)
    
    if dataset == 'Cancer':
        
        dataset_collection = SyntheticCancerDatasetCollection(name='tumor_generator', coeff=args.dataset.coeff, chemo_coeff=args.dataset.coeff,
                                                                radio_coeff=args.dataset.coeff, seed=args.exp.seed, num_patients={'train': 10000, 'val': 1000, 'test': 1000,},
                                                                window_size=15, lag=0, max_seq_length=60, projection_horizon=5, cf_seq_mode='sliding_treatment', val_batch_size=512,treatment_mode='multiclass')
        
        dataset_collection.process_data_multi()
        args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
        args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
        args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
        args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]
        
        print(dataset_collection.train_f.data['outputs'].shape[0], dataset_collection.val_f.data['outputs'].shape[0], dataset_collection.test_cf_one_step.data['outputs'].shape[0], dataset_collection.test_cf_treatment_seq.data['outputs'].shape[0])
        
        model_cfg = args.model.multi
        model_kwargs = {k: v for k, v in model_cfg.items() if k != '_target_'}

        model = CT(**model_kwargs, args=args, dataset_collection=dataset_collection)
        
        model_cfg = args.model.multi
        trn_batch_size = model_cfg.batch_size
        val_batch_size = model.hparams.dataset.val_batch_size
        learning_rate = model_cfg.optimizer.learning_rate
        
        train_loader = DataLoader(dataset_collection.train_f, batch_size=trn_batch_size, shuffle=True)
        val_loader = DataLoader(dataset_collection.val_f, batch_size=val_batch_size, shuffle=False)
        
        # elif dataset == 'MIMIC':
        #     split_path = '/splits/phy19_split' + str(split_idx) + '_new.npy'

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)     
            
        if wandb_use:
            # wandb.login(key=str('0126f71b25a3ecd1e32ed0a83047073475ee9cea'))
            # config = wandb.config
            wandb.init(name=f'CF-{dataset}',
                        project='Causal Transformer', 
                        config={'Dataset':dataset, 'Domain coeff': args.dataset.coeff, 'Learning Rate':learning_rate, 'seed': args.exp.seed})

        
        # 파라미터 분리
        named_params = dict(model.named_parameters())
        treatment_param_names = ['br_treatment_outcome_head.' + name for name in model.br_treatment_outcome_head.treatment_head_params]

        treatment_params = [p for n, p in named_params.items() if any(n.startswith(tpn) for tpn in treatment_param_names)]
        non_treatment_params = [p for n, p in named_params.items() if all(not n.startswith(tpn) for tpn in treatment_param_names)]

        optimizer_main =  torch.optim.Adam(non_treatment_params, lr=learning_rate)
        optimizer_adv =  torch.optim.Adam(treatment_params, lr=learning_rate)
        
        best_rmse_val = 100.0
        patience = 0
        
        if wandb_use:
            wandb.watch(model)

        for epoch in range(args.exp.max_epochs):
            start = time.time()
            model.train()
            loss_main_total, mse_total, bce_conf_total, bce_adv_total = 0, 0, 0, 0
            num_batches = 0
            
            for batch in train_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Step 1: outcome + confuse (optimizer_idx == 0)
                optimizer_main.zero_grad()
                treatment_pred, outcome_pred, _ = model(batch)

                mse_loss = ((outcome_pred - batch['outputs'])**2 * batch['active_entries']).sum() / batch['active_entries'].sum()
                bce_loss_conf = model.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='confuse')
                bce_loss_conf = (batch['active_entries'].squeeze(-1) * bce_loss_conf).sum() / batch['active_entries'].sum()
                loss_main = mse_loss + model.alpha * bce_loss_conf

                loss_main.backward()
                optimizer_main.step()

                # Step 2: adversarial classifier (optimizer_idx == 1)
                optimizer_adv.zero_grad()
                treatment_pred_adv, _, _ = model(batch, detach_treatment=True)
                bce_loss_adv = model.bce_loss(treatment_pred_adv, batch['current_treatments'].double(), kind='predict')
                bce_loss_adv = (batch['active_entries'].squeeze(-1) * bce_loss_adv).sum() / batch['active_entries'].sum()
                loss_adv = model.alpha * bce_loss_adv

                loss_adv.backward()
                optimizer_adv.step()
                
                loss_main_total += loss_main.item()
                mse_total += mse_loss.item()
                bce_conf_total += bce_loss_conf.item()
                bce_adv_total += bce_loss_adv.item()
                num_batches += 1
                    
        
            if wandb_use:
                wandb.log({
                "dc_train_loss": loss_main_total / num_batches,
                "mse_train": mse_total / num_batches,
                "nt_train": bce_conf_total / num_batches,
                "adv_train": bce_adv_total / num_batches,
                "epoch": epoch
            })
        
        
            """Validation"""  
            model.eval()
            
            loss_val_total, loss_bce, loss_mse = 0, 0, 0
            num_batches = 0
            
            all_preds = []
            
            unscale = model.hparams.exp.unscale_rmse
            percentage = model.hparams.exp.percentage_rmse
            dataset = dataset_collection.val_f
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    treatment_pred, outcome_pred, _ = model(batch)
                    
                    all_preds.append(outcome_pred.cpu())

                    mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)
                    bce_loss = model.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='confuse')
                    
                    # Masking for shorter sequences
                    # Attention! Averaging across all the active entries (= sequence masks) for full batch
                    bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch['active_entries'].sum()
                    mse_loss = (batch['active_entries'] * mse_loss).sum() / batch['active_entries'].sum()
                    loss = bce_loss + mse_loss
                    
                    loss_val_total += loss.item()
                    loss_bce += bce_loss.item()
                    loss_mse += mse_loss.item()
                    num_batches += 1
                
                
                all_preds = torch.cat(all_preds).numpy()
        
                if unscale:
                    output_stds, output_means = dataset.scaling_params['output_stds'], dataset.scaling_params['output_means']
                    outputs_unscaled = all_preds * output_stds + output_means

                    # Batch-wise masked-MSE calculation is tricky, thus calculating for full dataset at once
                    mse = ((outputs_unscaled - dataset.data['unscaled_outputs']) ** 2) * dataset.data['active_entries']
                else:
                    # Batch-wise masked-MSE calculation is tricky, thus calculating for full dataset at once
                    mse = ((all_preds - dataset.data['outputs']) ** 2) * dataset.data['active_entries']
                    
                mse_orig = mse.sum(0).sum(-1) / dataset.data['active_entries'].sum(0).sum(-1)
                mse_orig = mse_orig.mean()
                rmse_normalised_orig = np.sqrt(mse_orig) / dataset.norm_const
                
                
                if percentage:
                    rmse_normalised_orig *= 100.0
                    
                if wandb_use:
                    wandb.log({
                    "valid_loss": loss_val_total / num_batches,
                    "mse_valid": loss_mse / num_batches,
                    "nt_valid": loss_bce / num_batches,
                    "RMSE_val":rmse_normalised_orig,
                    "epoch": epoch
                })
                    
                    
                if rmse_normalised_orig < best_rmse_val:
                    best_rmse_val = rmse_normalised_orig
                    print(
                        "**[S] Epoch %d, rmse_val: %.4f**" % (
                        epoch, rmse_normalised_orig))
                    
                    patience = 0
                    torch.save(model.state_dict(), model_path + arch + '.pt')
                    
                else:
                    patience += 1
                    if patience >= 150:
                        print('early stopping triggered')
                        break
            end = time.time()
            time_elapsed = end - start
            print('Total Time elapsed: %.3f mins' % (time_elapsed / 60.0))
                
                        
    
        """testing"""
        print('Start Testing...')

        model.load_state_dict(torch.load('/Users/DAHS/Desktop/Causal Transformer-Pytorch Implementation/code/multirun/2025-06-06/21-31-44/0/models/CT.pt'))
        print('Inference ready')
        
        
        if hasattr(dataset_collection, 'test_cf_one_step'):  # Test one_step_counterfactual rmse
            print('Start inference for one-step counterfactual prediction...')
            
            tst_loader = DataLoader(dataset_collection.test_cf_one_step, batch_size=val_batch_size, shuffle=False)
            one_step_counterfactual = True
            
            all_preds = []

            unscale = model.hparams.exp.unscale_rmse
            percentage = model.hparams.exp.percentage_rmse
            dataset = dataset_collection.test_cf_one_step
            
            model.eval()    
            with torch.no_grad():
                for batch in tst_loader:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    treatment_pred, outcome_pred, _ = model(batch)
                    
                    all_preds.append(outcome_pred.cpu())

                    mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)
                    bce_loss = model.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='confuse')
                    
                    # Masking for shorter sequences
                    # Attention! Averaging across all the active entries (= sequence masks) for full batch
                    bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch['active_entries'].sum()
                    mse_loss = (batch['active_entries'] * mse_loss).sum() / batch['active_entries'].sum()
                    loss = bce_loss + mse_loss
                
                all_preds = torch.cat(all_preds).numpy()
                
                np.save('/Users/DAHS/Desktop/Causal Transformer-Pytorch Implementation/Buffer/one_step_cf.npy', all_preds)
                    
                if unscale:
                    output_stds, output_means = dataset.scaling_params['output_stds'], dataset.scaling_params['output_means']
                    outputs_unscaled = all_preds * output_stds + output_means

                    # Batch-wise masked-MSE calculation is tricky, thus calculating for full dataset at once
                    mse = ((outputs_unscaled - dataset.data['unscaled_outputs']) ** 2) * dataset.data['active_entries']
                else:
                    # Batch-wise masked-MSE calculation is tricky, thus calculating for full dataset at once
                    mse = ((all_preds - dataset.data['outputs']) ** 2) * dataset.data['active_entries']
                    
                mse_orig = mse.sum(0).sum(-1) / dataset.data['active_entries'].sum(0).sum(-1)
                mse_orig = mse_orig.mean()
                rmse_normalised_orig = np.sqrt(mse_orig) / dataset.norm_const
                
                if percentage:
                    rmse_normalised_orig *= 100.0
                
                if one_step_counterfactual:
                    # Only considering last active entry with actual counterfactuals
                    num_samples, time_dim, output_dim = dataset.data['active_entries'].shape
                    last_entries = dataset.data['active_entries'] - np.concatenate([dataset.data['active_entries'][:, 1:, :],
                                                                                    np.zeros((num_samples, 1, output_dim))], axis=1)
                    if unscale:
                        mse_last = ((outputs_unscaled - dataset.data['unscaled_outputs']) ** 2) * last_entries
                    else:
                        mse_last = ((all_preds - dataset.data['outputs']) ** 2) * last_entries

                    mse_last = mse_last.sum() / last_entries.sum()
                    rmse_normalised_last = np.sqrt(mse_last) / dataset.norm_const
                    
                if percentage:
                    rmse_normalised_last *= 100.0
                    
                    
        if hasattr(dataset_collection, 'test_cf_treatment_seq'):  # Test one_step_counterfactual rmse
            
            print('Start inference for n-step counterfactual prediction...')
            
            unscale = model.hparams.exp.unscale_rmse
            percentage = model.hparams.exp.percentage_rmse
            dataset = dataset_collection.test_cf_treatment_seq
            
            predicted_outputs = np.zeros((len(dataset), model.hparams.dataset.projection_horizon, model.dim_outcome))
            tst_loader = DataLoader(dataset, batch_size=val_batch_size, shuffle=False)
            
            model.eval()    
            with torch.no_grad():
                
                for t in range(model.hparams.dataset.projection_horizon + 1):    
                    
                    tst_loader = DataLoader(dataset, batch_size=val_batch_size, shuffle=False)
                    all_preds = []
                    
                    for batch in tst_loader:
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                        _, outcome_pred, _ = model(batch)
                        
                        all_preds.append(outcome_pred.cpu())
                        
                    all_preds = torch.cat(all_preds).numpy()
                            
                    for i in range(len(dataset)):
                        split = int(dataset.data['future_past_split'][i])
                        
                        if t < model.hparams.dataset.projection_horizon:
                            dataset.data['prev_outputs'][i, split + t, :] = all_preds[i, split - 1 + t, :]
                        if t > 0:
                            predicted_outputs[i, t - 1, :] = all_preds[i, split - 1 + t, :]
                            
                    
                if unscale:
                    output_stds, output_means = dataset.scaling_params['output_stds'], dataset.scaling_params['output_means']
                    outputs_unscaled = predicted_outputs * output_stds + output_means

                    # Batch-wise masked-MSE calculation is tricky, thus calculating for full dataset at once
                    mse = ((outputs_unscaled - dataset.data_processed_seq['unscaled_outputs']) ** 2) * dataset.data_processed_seq['active_entries']
                else:
                    # Batch-wise masked-MSE calculation is tricky, thus calculating for full dataset at once
                    mse = ((predicted_outputs - dataset.data_processed_seq['outputs']) ** 2) * dataset.data_processed_seq['active_entries']
                    
                nan_idx = np.unique(np.where(np.isnan(dataset.data_processed_seq['outputs']))[0])
                not_nan = np.array([i for i in range(predicted_outputs.shape[0]) if i not in nan_idx])

                # Calculation like in original paper (Masked-Averaging over datapoints (& outputs) and then non-masked time axis)
                mse_orig = mse[not_nan].sum(0).sum(-1) / dataset.data_processed_seq['active_entries'][not_nan].sum(0).sum(-1)
                rmses_normalised_orig = np.sqrt(mse_orig) / dataset.norm_const

                if percentage:
                    rmses_normalised_orig *= 100.0
                
                test_rmses = {f'{k+2}-step': v for (k, v) in enumerate(rmses_normalised_orig)}
                
        print('------------------------------------------')
        print('One step prediction RMSE = %.2f' % (rmse_normalised_last))
        for k, v in test_rmses.items():
            print(k+' prediction RMSE = %.2f' % (test_rmses[k]))


if __name__ == "__main__":
    main()