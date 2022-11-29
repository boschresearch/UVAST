# Training and Evaluating the Models

<strong>Pretrained models will be released soon. </strong>

All training and evaluation scripts and pretrained models can be found in [gtea](gtea), [50salads](50salads) and [breakfast](breakfast).

## Training the models
### Stage 1:
To train the encoder and transcript decoder (Eq. (6) in the paper) run the following [command (gtea split 1)](gtea/gtea_train_stage1_scripts.sh):

`python run.py --use_cuda --step_size 800 --dataset gtea --split 1 --split_segments --use_pe_tgt --do_framewise_loss --do_framewise_loss_g --framewise_loss_g_apply_nothing --do_segwise_loss --do_segwise_loss_g --segwise_loss_g_apply_logsoftmax --do_crossattention_action_loss_nll`

### Stage 2:
To train the alignment decoder run the following [command (gtea split 1)](gtea/gtea_train_stage2_scripts.sh):

`python run.py --use_cuda --dataset gtea --split 1 --split_segments --use_pe_tgt --use_alignment_dec --do_crossattention_dur_loss_ce --aug_rnd_drop --pretrained_model pretrained_models/gtea/gtea_split1_stage1.model`

## Evaluating the models
### Evaluate Transcript Decoder:
To evaluate the predicted transcript run the [command (gtea split 1)](gtea/gtea_test_stage1_scripts.sh):
`python run.py --use_cuda --dataset gtea --split 1 --path_inference_model pretrained_models/gtea/gtea_split1_stage1.model --inference_only --split_segments --use_pe_tgt`

### Evaluate Alignment Decoder:
To evaluate the alignment decoder run the [command (gtea split 1)](gtea/gtea_test_stage2_scripts.sh):
`python run.py --use_cuda --dataset gtea --split 1 --path_inference_model pretrained_models/gtea/gtea_split1_stage2.model --inference_only --split_segments --use_pe_tgt --use_alignment_dec`

### Evaluate with Viterbi:
To evaluate the model with Viterbi run the [command (gtea split 1)](gtea/gtea_test_viterbi_scripts.sh):
`python run.py --use_cuda --dataset gtea --split 1 --use_viterbi --viterbi_sample_rate 1 --path_inference_model pretrained_models/gtea/gtea_split1_stage1.model --inference_only --split_segments --use_pe_tgt`

### Evaluate with FIFA:
To evaluate the model with FIFA run the [command (gtea split 1)](gtea/gtea_test_fifa_scripts.sh):
`python run.py --use_cuda --dataset gtea --split 1 --use_fifa --fifa_init_dur --path_inference_model pretrained_models/gtea/gtea_split1_stage2.model --inference_only --split_segments --use_pe_tgt --use_alignment_dec`
