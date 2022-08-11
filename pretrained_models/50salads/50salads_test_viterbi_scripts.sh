# split 1
python run.py \
        --use_cuda \
        --dataset 50salads \
        --split 1 \
	--use_viterbi \
        --viterbi_sample_rate 2 \
        --path_inference_model pretrained_models/50salads/split1/50salads_split1_stage1.model \
        --inference_only \
        --split_segments \
        --use_pe_tgt \
	--AttentionPoolType_dec 'avg' \
        --data_root data \ 
        --data_root_mean_duration mean_dur 

# split 2
python run.py \
        --use_cuda \
        --dataset 50salads \
        --split 2 \
	--use_viterbi \
        --viterbi_sample_rate 2 \
        --path_inference_model pretrained_models/50salads/split2/50salads_split2_stage1.model \
        --inference_only \
        --split_segments \
        --use_pe_tgt \
	--AttentionPoolType_dec 'avg' \
        --data_root data \ 
        --data_root_mean_duration mean_dur 

# split 3
python run.py \
        --use_cuda \
        --dataset 50salads \
        --split 3 \
	--use_viterbi \
        --viterbi_sample_rate 2 \
        --path_inference_model pretrained_models/50salads/split3/50salads_split3_stage1.model \
        --inference_only \
        --split_segments \
        --use_pe_tgt \
	--AttentionPoolType_dec 'avg' \
        --data_root data \ 
        --data_root_mean_duration mean_dur 

# split 4
python run.py \
        --use_cuda \
        --dataset 50salads \
        --split 4 \
	--use_viterbi \
        --viterbi_sample_rate 2 \
        --path_inference_model pretrained_models/50salads/split4/50salads_split4_stage1.model \
        --inference_only \
        --split_segments \
        --use_pe_tgt \
	--AttentionPoolType_dec 'avg' \
        --data_root data \ 
        --data_root_mean_duration mean_dur 

# split 5
python run.py \
        --use_cuda \
        --dataset 50salads \
        --split 5 \
	--use_viterbi \
        --viterbi_sample_rate 2 \
        --path_inference_model pretrained_models/50salads/split5/50salads_split5_stage1.model \
        --inference_only \
        --split_segments \
        --use_pe_tgt \
	--AttentionPoolType_dec 'avg' \
        --data_root data \ 
        --data_root_mean_duration mean_dur 
