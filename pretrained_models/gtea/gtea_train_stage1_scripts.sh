# split 1
python run.py --use_cuda --save_args --step_size 800 \
            --dataset 'gtea' --split 1 \
            --split_segments \
            --use_pe_tgt \
            --do_framewise_loss --do_framewise_loss_g --framewise_loss_g_apply_nothing \
            --do_segwise_loss --do_segwise_loss_g --segwise_loss_g_apply_logsoftmax \
            --do_crossattention_action_loss_nll \
            --exp_name 'gtea_stage1' \
            --experiment_path save_models \ 
            --data_root data \ 
	    --seed 3672463022

# split 2
python run.py --use_cuda --save_args --step_size 800 \
            --dataset 'gtea' --split 2 \
            --split_segments \
            --use_pe_tgt \
            --do_framewise_loss --do_framewise_loss_g --framewise_loss_g_apply_nothing \
            --do_segwise_loss --do_segwise_loss_g --segwise_loss_g_apply_logsoftmax \
            --do_crossattention_action_loss_nll \
            --exp_name 'gtea_stage1' \
            --experiment_path save_models \ 
            --data_root data \ 
            --seed 3672463022

# split 3
python run.py --use_cuda --save_args --step_size 800 \
            --dataset 'gtea' --split 3 \
            --split_segments \
            --use_pe_tgt \
            --do_framewise_loss --do_framewise_loss_g --framewise_loss_g_apply_nothing \
            --do_segwise_loss --do_segwise_loss_g --segwise_loss_g_apply_logsoftmax \
            --do_crossattention_action_loss_nll \
            --exp_name 'gtea_stage1' \
            --experiment_path save_models \ 
            --data_root data \ 
            --seed 3672463022

# split 4
python run.py --use_cuda --save_args --step_size 800 \
            --dataset 'gtea' --split 4 \
            --split_segments \
            --use_pe_tgt \
            --do_framewise_loss --do_framewise_loss_g --framewise_loss_g_apply_nothing \
            --do_segwise_loss --do_segwise_loss_g --segwise_loss_g_apply_logsoftmax \
            --do_crossattention_action_loss_nll \
            --exp_name 'gtea_stage1' \
            --experiment_path save_models \ 
            --data_root data \ 
            --seed 3672463022
