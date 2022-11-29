# split 1
python run.py --use_cuda --save_args \
            --dataset '50salads' --split 1 \
            --split_segments \
	    --use_pe_tgt \
            --AttentionPoolType_dec 'avg' \
            --do_framewise_loss --do_framewise_loss_g --framewise_loss_g_apply_nothing  \
            --do_segwise_loss --do_segwise_loss_g --segwise_loss_g_apply_logsoftmax \
            --do_crossattention_action_loss_nll \
            --exp_name 'salad_stage1' \
            --experiment_path save_models \ 
            --data_root data \ 
	    --seed 3920756584

# split 2
python run.py --use_cuda --save_args \
            --dataset '50salads' --split 2 \
            --split_segments \
            --use_pe_tgt \
            --AttentionPoolType_dec 'avg' \
            --do_framewise_loss --do_framewise_loss_g --framewise_loss_g_apply_nothing  \
            --do_segwise_loss --do_segwise_loss_g --segwise_loss_g_apply_logsoftmax \
            --do_crossattention_action_loss_nll \
            --exp_name 'salad_stage1' \
            --experiment_path save_models \ 
            --data_root data \ 
            --seed 3920756584

# split 3
python run.py --use_cuda --save_args \
            --dataset '50salads' --split 3 \
            --split_segments \
            --use_pe_tgt \
            --AttentionPoolType_dec 'avg' \
            --do_framewise_loss --do_framewise_loss_g --framewise_loss_g_apply_nothing  \
            --do_segwise_loss --do_segwise_loss_g --segwise_loss_g_apply_logsoftmax \
            --do_crossattention_action_loss_nll \
            --exp_name 'salad_stage1' \
            --experiment_path save_models \ 
            --data_root data \ 
            --seed 3920756584

# split 4
python run.py --use_cuda --save_args \
            --dataset '50salads' --split 4 \
            --split_segments \
            --use_pe_tgt \
            --AttentionPoolType_dec 'avg' \
            --do_framewise_loss --do_framewise_loss_g --framewise_loss_g_apply_nothing  \
            --do_segwise_loss --do_segwise_loss_g --segwise_loss_g_apply_logsoftmax \
            --do_crossattention_action_loss_nll \
            --exp_name 'salad_stage1' \
            --experiment_path save_models \ 
            --data_root data \ 
            --seed 3920756584

# split 5
python run.py --use_cuda --save_args \
            --dataset '50salads' --split 5 \
            --split_segments \
            --use_pe_tgt \
            --AttentionPoolType_dec 'avg' \
            --do_framewise_loss --do_framewise_loss_g --framewise_loss_g_apply_nothing  \
            --do_segwise_loss --do_segwise_loss_g --segwise_loss_g_apply_logsoftmax \
            --do_crossattention_action_loss_nll \
            --exp_name 'salad_stage1' \
            --experiment_path save_models \ 
            --data_root data \ 
            --seed 3920756584
