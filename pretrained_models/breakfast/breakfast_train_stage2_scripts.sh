# split 1
python run.py --use_cuda --save_args \
            --dataset 'breakfast' --split 1 \
            --split_segments \
            --use_pe_tgt \
            --use_alignment_dec --do_crossattention_dur_loss_ce  \
            --aug_rnd_drop \
            --exp_name 'breakfast_stage2' \
            --experiment_path save_models \ 
            --data_root data \ 
            --seed 1543417815 \
	    --pretrained_model pretrained_models/breakfast/split1/breakfast_split1_stage1.model

# split 2
python run.py --use_cuda --save_args \
            --dataset 'breakfast' --split 2 \
            --split_segments \
            --use_pe_tgt \
            --use_alignment_dec --do_crossattention_dur_loss_ce  \
            --aug_rnd_drop \
            --exp_name 'breakfast_stage2' \
            --experiment_path save_models \ 
            --data_root data \ 
            --seed 1543417815 \
            --pretrained_model pretrained_models/breakfast/split2/breakfast_split2_stage1.model

# split 3
python run.py --use_cuda --save_args \
            --dataset 'breakfast' --split 3 \
            --split_segments \
            --use_pe_tgt \
            --use_alignment_dec --do_crossattention_dur_loss_ce  \
            --aug_rnd_drop \
            --exp_name 'breakfast_stage2' \
            --experiment_path save_models \ 
            --data_root data \ 
            --seed 1543417815 \
            --pretrained_model pretrained_models/breakfast/split3/breakfast_split3_stage1.model

# split 4
python run.py --use_cuda --save_args \
            --dataset 'breakfast' --split 4 \
            --split_segments \
            --use_pe_tgt \
            --use_alignment_dec --do_crossattention_dur_loss_ce  \
            --aug_rnd_drop \
            --exp_name 'breakfast_stage2' \
            --experiment_path save_models \ 
            --data_root data \ 
            --seed 1543417815 \
            --pretrained_model pretrained_models/breakfast/split4/breakfast_split4_stage1.model
