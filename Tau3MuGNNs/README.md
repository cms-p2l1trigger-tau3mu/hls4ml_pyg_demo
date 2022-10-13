# Tau3MuGNNs
This repo is to reproduce the results of the Tau3MuGNNs.

# Use Gilbreth
## 0. connect VPN
To ssh to the Gilbreth cluster, one should first establish a VPN connection to `webvpn.purdue.edu`. Tutorials on how to do this can be found [here](https://www.itap.purdue.edu/newsroom/2020/200316-webvpn.html).

Basically, you need to first install `Cisco AnyConnect` and then connect to `webvpn.purdue.edu`, where the username is `purdue_id` (e.g. `miao61`) and the password is `[pin],push`. The `[pin]` is the 4-digit PIN for [BoilerKey Two-Factor Authentication](https://www.purdue.edu/apps/account/BoilerKey/).

## 1. ssh to Gilbreth
```
ssh purdue_id@gilbreth.rcac.purdue.edu
```
Then, it will ask for a password, which is also `[pin],push` for the BoilerKey.

## 2. change directory
I was recommended to use the `scratch` directory instead of the default home directory.
```
cd /scratch/gilbreth/purdue_id/
```

## 3. setup ssh key for GitHub
Since our repo is private, we need to setup a ssh key for GitHub. [Here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) provides a tutorial on how to do this.

Then, clone the repo:
```
git clone git@github.com:cms-p2l1trigger-tau3mu/Tau3MuGNNs.git
cd Tau3MuGNNs/
git checkout gnn-siqi
```


## 4. install anaconda
Run:
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
sh Anaconda3-2021.11-Linux-x86_64.sh
```
It will ask where to install, rememeber install it at `/scratch/gilbreth/purdue_id/anaconda3`. It may take a while to install. Type `yes` for all options. After installation, activate `conda` command by `source ~/.bashrc` everythime logging the server.

Now we should be at the `base` environment. Type `python`, we shall see a version of `Python 3.9.7`. Then type `quit()` to quit python, and start installing packages.

## 5. create conda environment and install packages

First, create a conda environment and activate it.
```
conda create --name tau3mu python=3.9
conda activate tau3mu
```

Install dependencies:
```
conda install -y pytorch=1.10.0 torchvision cudatoolkit=11.3 -c pytorch
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1 torch-geometric==2.0.3 -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install -r requirements.txt
```

In case a lower CUDA version is required, please use the following command to install dependencies:
```
conda install -y pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1 torch-geometric==2.0.3 -f https://data.pyg.org/whl/torch-1.9.0+cu102.html
pip install -r requirements.txt
```



# Get the code
The branch `master` branch is created by Daniel to process `.root` files. I use branch `gnn-siqi` to develop my models, and this branch contains the latest code. `check-siqi` is the branch with simplified code, which was trying to make the code more readable, but it is deprecated now.

To get the code, one can do:
```
git clone git@github.com:cms-p2l1trigger-tau3mu/Tau3MuGNNs.git
cd Tau3MuGNNs/
git checkout gnn-siqi
```

# Get the data
To run the code, we need to put those `.pkl` files in the right place.

Specifically, one has to put `DsTau3muPU0_MTD.pkl`, `DsTau3muPU200_MTD.pkl`, `MinBiasPU200_MTD.pkl` under `$ProjectDir/data/raw`. When running the code, those dataframes will be processed according to the specific setting, and the processed files will be saved under `$ProjectDir/data/processed-[setting]-[cut_id]`. In this project, for simplicity I call `SingalPU0, SingalPU200, BkgPU200` as `pos0, pos200, neg200` respectively.

Please note that the processed files may take up lots of disk space (5 gigabytes+), and when processing them it may also take up lots of memory (10 gigabytes+).

# Train a model
To train a Decision Tree, one can refer to the `./src/decision_tree.ipynb` notebook.

I provide `8` settings for training GNNs, and the corresponding configurations can be found in `$ProjectDir/src/configs/`. To train a GNN with a specific setting, one can do:

```
cd ./src
python train_gnn.py --setting [setting_name] --cuda [GPU_id] --cut [cut_id]
```

`GPU_id` is the id of the GPU to use. To use CPU, please set it to `-1`. `cut_id` is the id of the cut to use. Its default value is `None` and can be set to `cut1` or `cut1+2`. Note that when some cut is used, the `pos_neg_ratio` may need to be adjusted because many positive samples will be dropped.


The `setting_name` can be chosen from the following settings:
| setting_name | Description |
| ------------ | ----------- |
| GNN_full_dR_1 | (1) full detector; (2) construct graphs based on dR; (3) use hits from station 1 (current best setting)|
| GNN_full_dR_2 | (1) full detector; (2) construct graphs based on dR; (3) use hits from station 1 & 2  |
| GNN_full_dR_4 | (1) full detector; (2) construct graphs based on dR; (3) use hits from station 1 & 2 & 3 & 4  |
| GNN_full_FC_1 | (1) full detector; (2) construct fully-connected (complete) graphs; (3) use hits from station 1 |
| GNN_full_dR_1_mix | (1) full detector; (2) construct graphs based on dR; (3) use hits from station 1; (4) positive samples: construct a sample by mixing signal hits from a SignalPU0 sample and background hits from a BkgPU200 sample, negative samples: BkgPU200 |
| GNN_full_dR_1_mix_check | (1) full detector; (2) construct graphs based on dR; (3) use hits from station 1; (4) positive samples: construct a sample by mixing signal hits from a SignalPU0 sample and background hits from a BkgPU200 sample, negative samples: SignalPU200 |
| GNN_half_dR_1 | (1) half detector; (2) construct graphs based on dR; (3) use hits from station 1 |
| GNN_half_dR_1_check | (1) half detector; (2) construct graphs based on dR; (3) use hits from station 1; (4) positive samples: non-tau endcap in SignalPU200, negative samples: BkgPU200 (half detector) |

We provide detailed documentation for each option in the setting in `./src/configs/GNN_full_dR_1.yml`. One can play with those options and try new settings.

One thing to notice is that if you have had processed files for a specific setting, even then you change some options in the config file, the processed files will not be changed in the next run. So, if you want to change the options in a config file, you need to delete the corresponding processed files first. This is because the code will search `.pt` files given the `setting_name`; if it finds any `.pt` files under `$ProjectDir/data/processed-[setting_name]-[cut_id]`, it will assume that the processed files for the specified setting are already there and will not re-process data with the new options.

# Workflow of the code

1. Class `Tau3MuDataset` in `$ProjectDir/src/utils/dataset` is used to build datasets that can be used to train pytorch models. The code will first call this class to process dataframes, including graph building, node/edge feature generations, dataset splits, etc. After this process, the fully processed data shall be saved on the disk.

2. Then the model will be trained by the class `Tau3MuGNNs` in `train_gnn.py`, and during the training some metrics will show on the progress bar.

3. The trained model will be saved into `$ProjectDir/data/logs/[time_step-setting_name-cut_id]/model.pt`, where `[time_step-setting_name-cut_id]` is the log id for this model and will be needed to load the model later.


# Training Logs
Standard output provides basic training logs, while more detailed logs and interpretation visualizations can be found on tensorboard:
```
tensorboard --logdir=$ProjectDir/data/logs
```

In case you are using Gilbreth, you can use the following command to open tensorboard:
```
unset PYTHONPATH
tensorboard --logdir=$ProjectDir/data/logs --bind_all
```

# Inference
To save scores of each model configuration , we create a new folder `$ProjectDir/data/scores`. All inference scores will be saved here by running `./src/infer_gnn.ipynb`.

The folder will contain multiple subfolders, one for each *data setting*. For example, `scores/raw_cut1` will be one folder, representing a dataset using raw data and cut1; `scores/mix_cut1+2` would represent a dataset using mixed data and cut1+2.

In each subfolder, say `scores/raw_cut1`, we will have a `raw_cut1.pkl` file for reference, which contains the raw features of the dataset (both positive and negative data) and its row idx is the sample_idx. Then, we may have multiple score files, one for each *model setting*, where there are three columns: `sample_idx`, `probs`, and `phase`. For example, we may have `GNN_full_dR_1-scores.pkl`, which contains the scores of the model trained with the setting `GNN_full_dR_1`, and we may have another file, say `GNN_full_dR_4-scores.pkl` for a model trained with four stations.

To accommodate the half detector case, the `probs` column is a dict for each sample_idx. The key of it is the endcap_id and the value of it is the corresponding probs, where `+1` means positive endcap, `-1` means negative endcap, and `0` means the model uses full detector to make the prediction. Sometimes we might want to study only test sample, so the `phase` column (also a dict like the `probs` column) would tell us which endcap is used in which phase.

Finally, for each *data setting*, we can combine the results from `raw_cut1.pkl`, `GNN_half_dR_1-scores.pkl` and `GNN_full_dR_4-scores.pkl` to get a comprehensive score file containing all the scores and raw features. (The script for this is not provided, as it is just a couple of lines and can be done once needed.)


# File Structure

```
.
├── data
│   ├── logs                                # logs for training models
│   ├── raw                                 # store raw .pkl files
│   ├── scores                              # store inference scores
│   ├── processed-GNN_full_dR_1-cut1        # store processed files for each setting
│   ├── processed-GNN-full_dR_2-cut1+2      # store processed files for each setting
│   └── ...
├── README.md
├── requirements.txt
└── src
    ├── train_gnn.py                        # train GNNs
    ├── desicion_tree.ipynb                 # train a decision tree
    ├── configs                             # configs for different settings
    │   ├── GNN_full_dR_1.yml
    │   ├── GNN_full_dR_2.yml
    │   └── ...
    ├── models
    │   ├── __init__.py
    │   ├── gen_conv.py                     # define GNN conv layers
    │   └── model.py                        # define GNN models
    └── utils
        ├── __init__.py
        ├── dataset.py                      # dataset class
        ├── logger.py                       # logger functions
        ├── loss.py                         # loss functions
        ├── root2df.py                      # convert root file to dataframe
        └── utils.py                        # utility functions
```