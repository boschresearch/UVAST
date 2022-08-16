# UVAST: Unified Fully and Weakly Supervised Temporal Action Segmentation via Sequence to Sequence Translation

Official PyTorch implementation of the ECCV 2022 paper:
[UVAST: Unified Fully and Weakly Supervised Temporal Action Segmentation via Sequence to Sequence Translation](https://arxiv.org/tbd) 
The code allows users to reproduce and extend the results of our method. Please cite the paper when reporting, reproducing or extending the results.

## Overview

This repository implements UVAST, a method for temporal action segmentation via sequence-to-sequence translation.
Details regarding the required environment, datasets, training scripts and pretrained models can be found below.
 <br />
  <br />
   <br />

![ECCV2022](https://user-images.githubusercontent.com/25614955/179375271-1dd0eb92-45fa-42d9-8a37-8bd86bcee014.gif)


## Enviroment
Pytorch == `1.10.0+cu102`, 
torchvision == `0.11.1`, 
python == `3.9.7`, 
CUDA==`10.2`

### Enviroment Setup
Install the required libaries as follows:

``` python
conda clean -a -y
conda create -n uvast python=3.9.7 numpy
conda activate uvast
conda install  --insecure pytorch=1.10.0 torchvision=0.11.1 torchaudio=0.10.0 cudatoolkit=11.3.1  -c pytorch
python -c "import torch; print(torch.__version__)"
conda install -c conda-forge tqdm
conda install -c conda-forge matplotlib
conda install -c conda-forge einops
conda install -c conda-forge torchinfo
conda install -c anaconda pandas
conda install -c conda-forge tensorboardx
conda install -c anaconda ipykernel
conda install ipython
conda install pip
```

Clone this repository

``` python
git clone https://github.com/boschresearch/uvast.git
cd uvast
```
## Datasets
Use [this link](https://zenodo.org/record/3625992#.YsMSBdLMJhF) to download the features and the ground truth labels for the GTEA, 50Salads and Breakfast datasets (~30GB).

Extract `data.zip` so that `data` is placed inside the `uvast` folder. Otherwise you need to modify the `--data_root` flag to point to the `data` folder.

### Compute mean durations
To compute the mean durations to be used in FIFA/Viterbi call (and optionally pass `--data_root`):
``` python
python compute_mean_dur.py
```

## Training
We train the model in a two stages process:
In the first stage, we train the encoder and the transcript decoder using Eq. (6) from the paper (and without the alignment decoder).
In the second stage, we optionally train the alignment decoder (alternatively, Viterbi or FIFA can be used to compute durations). 

All training scrips for all three datasets are provided with the [pretrained_models](pretrained_models). For each of the scripts you need to specify the `--data_root` (if `data` is not placed inside the `uvast` folder) and `--experiment_path` to specify where to save the models. 
For more information regarding the flags, please look into  the information for each flag in `run.py`.
An example training script for the first split of the `gtea` dataset is provided below:

<strong>Training script for stage 1:</strong>
``` python
python run.py --use_cuda --step_size 800 --dataset gtea --split 1 --split_segments --use_pe_tgt --do_framewise_loss --do_framewise_loss_g --framewise_loss_g_apply_nothing --do_segwise_loss --do_segwise_loss_g --segwise_loss_g_apply_logsoftmax --do_crossattention_action_loss_nll
```

<strong>Training script for stage 2:</strong>
``` python
python run.py --use_cuda --dataset gtea --split 1 --split_segments --use_pe_tgt --use_alignment_dec --do_crossattention_dur_loss_ce --aug_rnd_drop --pretrained_model pretrained_models/gtea/split1/gtea_split1_stage1.model
```

Note that for this stage you need to specify the pretrained model from the first stage via the `--pretrained_model` flag.


## Evaluation

While the transcript decoder predicts the sequence of actions in the video, we propose three different approaches for predicting durations: a learnable alignment decoder (stage 2), or FIFA/Viterbi.
All evaluation scripts along with pretrained model are provided in [pretrained_models](pretrained_models).
To run the inference code the flag `--inference_only` needs to be added as well as `--path_inference_model` to point to the model to be evaluated.

An example script for testing a model is provided below:

<strong>Evaluate Alignment Decoder:</strong>
``` python
python run.py --use_cuda --dataset gtea --split 1 --path_inference_model pretrained_models/gtea/split1/gtea_split1_stage2.model --inference_only --split_segments --use_pe_tgt --use_alignment_dec 
```            

<strong>Evaluate with Viterbi:</strong>
``` python
python run.py --use_cuda --dataset gtea --split 1 --use_viterbi --viterbi_sample_rate 1 --path_inference_model pretrained_models/gtea/split1/gtea_split1_stage1.model --inference_only --split_segments --use_pe_tgt
```

<strong>Evaluate with FIFA:</strong>
``` python
python run.py --use_cuda --dataset gtea --split 1 --use_fifa --fifa_init_dur --path_inference_model pretrained_models/gtea/split1/gtea_split1_stage2.model --inference_only --split_segments --use_pe_tgt --use_alignment_dec
```

  
## Citation
If this code is useful in your research we would kindly ask you to cite our paper.
```
@inproceedings{uvast2022ECCV,
 title={Unified Fully and Timestamp Supervised Temporal Action Segmentation via Sequence to Sequence Translation},
 author={Nadine Behrmann and S. Alireza Golestaneh and Zico Kolter and Juergen Gall and Mehdi Noroozi},
 booktitle={ECCV},
 year={2022}
}
```

##  License

This project is open-sourced under the AGPL-3.0 license. See the [License](LICENSE) file for details.

For a list of other open source components included in this project, see the file [3rd-party-licenses.txt](3rd-party-licenses.txt).

## Purpose of the Project
This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be
maintained nor monitored in any way.

## Contact
Please feel free to open an issue or contact us personally if you have questions, need help, or need explanations.
Write to one of the following email addresses, and maybe put the other in the cc:

isalirezag@gmail.com

nadine.behrmann@de.bosch.com

