# split 1
python run.py --use_cuda --save_args \
            --dataset 'gtea' --split 1 \
            --split_segments \
            --use_pe_tgt \
            --use_alignment_dec --do_crossattention_dur_loss_ce \
            --aug_rnd_drop \
            --exp_name 'gtea_stage2' \
            --experiment_path save_models \ 
            --data_root data \ 
	    --seed 3565244205 \
	    --pretrained_model pretrained_models/gtea/split1/gtea_split1_stage1.model

# split 2
python run.py --use_cuda --save_args \
            --dataset 'gtea' --split 2 \
            --split_segments \
            --use_pe_tgt \
            --use_alignment_dec --do_crossattention_dur_loss_ce \
            --aug_rnd_drop \
            --exp_name 'gtea_stage2' \
            --experiment_path save_models \ 
            --data_root data \ 
            --seed 3565244205 \
            --pretrained_model pretrained_models/gtea/split2/gtea_split2_stage1.model

# split 3
python run.py --use_cuda --save_args \
            --dataset 'gtea' --split 3 \
            --split_segments \
            --use_pe_tgt \
            --use_alignment_dec --do_crossattention_dur_loss_ce \
            --aug_rnd_drop \
            --exp_name 'gtea_stage2' \
            --experiment_path save_models \ 
            --data_root data \ 
            --seed 3565244205 \
            --pretrained_model pretrained_models/gtea/split3/gtea_split3_stage1.model

# split 4
python run.py --use_cuda --save_args \
            --dataset 'gtea' --split 4 \
            --split_segments \
            --use_pe_tgt \
            --use_alignment_dec --do_crossattention_dur_loss_ce \
            --aug_rnd_drop \
            --exp_name 'gtea_stage2' \
            --experiment_path save_models \ 
            --data_root data \ 
            --seed 3565244205 \
            --pretrained_model pretrained_models/gtea/split4/gtea_split4_stage1.model
