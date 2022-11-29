# split 1
python run.py \
        --use_cuda \
        --dataset 50salads \
        --split 1 \
        --path_inference_model pretrained_models/50salads/split1/50salads_split1_stage2.model \
        --inference_only \
        --split_segments \
        --use_pe_tgt \
	--use_alignment_dec \
	--AttentionPoolType_dec 'avg' \
        --data_root data \ 
        --data_root_mean_duration mean_dur 

# split 2
python run.py \
        --use_cuda \
        --dataset 50salads \
        --split 2 \
        --path_inference_model pretrained_models/50salads/split2/50salads_split2_stage2.model \
        --inference_only \
        --split_segments \
        --use_pe_tgt \
	--use_alignment_dec \
	--AttentionPoolType_dec 'avg' \
        --data_root data \ 
        --data_root_mean_duration mean_dur 

# split 3
python run.py \
        --use_cuda \
        --dataset 50salads \
        --split 3 \
        --path_inference_model pretrained_models/50salads/split3/50salads_split3_stage2.model \
        --inference_only \
        --split_segments \
        --use_pe_tgt \
	--use_alignment_dec \
	--AttentionPoolType_dec 'avg' \
        --data_root data \ 
        --data_root_mean_duration mean_dur 

# split 4
python run.py \
        --use_cuda \
        --dataset 50salads \
        --split 4 \
        --path_inference_model pretrained_models/50salads/split4/50salads_split4_stage2.model \
        --inference_only \
        --split_segments \
        --use_pe_tgt \
	--use_alignment_dec \
	--AttentionPoolType_dec 'avg' \
        --data_root data \ 
        --data_root_mean_duration mean_dur 

# split 5
python run.py \
        --use_cuda \
        --dataset 50salads \
        --split 5 \
        --path_inference_model pretrained_models/50salads/split5/50salads_split5_stage2.model \
        --inference_only \
        --split_segments \
        --use_pe_tgt \
	--use_alignment_dec \
	--AttentionPoolType_dec 'avg' \
        --data_root data \ 
        --data_root_mean_duration mean_dur 
