# split 1
python run.py --use_cuda --save_args \
            --dataset '50salads' --split 1 \
            --split_segments \
            --use_pe_tgt \
            --AttentionPoolType_dec 'avg' \
            --use_alignment_dec --do_crossattention_dur_loss_ce  \
            --aug_rnd_drop \
            --exp_name 'salad_stage2' \
            --experiment_path save_models \ 
            --data_root data \ 
	    --seed 3920756584 \
	    --pretrained_model pretrained_models/50salads/split1/50salads_split1_stage1.model

# split 2
python run.py --use_cuda --save_args \
            --dataset '50salads' --split 2 \
            --split_segments \
            --use_pe_tgt \
            --AttentionPoolType_dec 'avg' \
            --use_alignment_dec --do_crossattention_dur_loss_ce  \
            --aug_rnd_drop \
            --exp_name 'salad_stage2' \
            --experiment_path save_models \ 
            --data_root data \ 
            --seed 3920756584 \
            --pretrained_model pretrained_models/50salads/split2/50salads_split2_stage1.model

# split 3
python run.py --use_cuda --save_args \
            --dataset '50salads' --split 3 \
            --split_segments \
            --use_pe_tgt \
            --AttentionPoolType_dec 'avg' \
            --use_alignment_dec --do_crossattention_dur_loss_ce  \
            --aug_rnd_drop \
            --exp_name 'salad_stage2' \
            --experiment_path save_models \ 
            --data_root data \ 
            --seed 3920756584 \
            --pretrained_model pretrained_models/50salads/split3/50salads_split3_stage1.model

# split 4
python run.py --use_cuda --save_args \
            --dataset '50salads' --split 4 \
            --split_segments \
            --use_pe_tgt \
            --AttentionPoolType_dec 'avg' \
            --use_alignment_dec --do_crossattention_dur_loss_ce  \
            --aug_rnd_drop \
            --exp_name 'salad_stage2' \
            --experiment_path save_models \ 
            --data_root data \ 
            --seed 3920756584 \
            --pretrained_model pretrained_models/50salads/split4/50salads_split4_stage1.model

# split 5
python run.py --use_cuda --save_args \
            --dataset '50salads' --split 5 \
            --split_segments \
            --use_pe_tgt \
            --AttentionPoolType_dec 'avg' \
            --use_alignment_dec --do_crossattention_dur_loss_ce  \
            --aug_rnd_drop \
            --exp_name 'salad_stage2' \
            --experiment_path save_models \ 
            --data_root data \ 
            --seed 3920756584 \
            --pretrained_model pretrained_models/50salads/split5/50salads_split5_stage1.model
