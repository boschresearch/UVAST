# split 1
python run.py \
        --use_cuda \
	--dataset breakfast \
        --split 1 \
	--use_viterbi \
        --viterbi_sample_rate 5 \
        --path_inference_model pretrained_models/breakfast/split1/breakfast_split1_stage1.model \
        --inference_only \
        --split_segments \
        --use_pe_tgt \
        --data_root data \ 
        --data_root_mean_duration mean_dur  

# split 2
python run.py \
        --use_cuda \
        --dataset breakfast \
        --split 2 \
	--use_viterbi \
        --viterbi_sample_rate 5 \
        --path_inference_model pretrained_models/breakfast/split2/breakfast_split2_stage1.model \
        --inference_only \
        --split_segments \
        --use_pe_tgt \
        --data_root data \ 
        --data_root_mean_duration mean_dur  

# split 3
python run.py \
        --use_cuda \
        --dataset breakfast \
        --split 3 \
	--use_viterbi \
        --viterbi_sample_rate 5 \
        --path_inference_model pretrained_models/breakfast/split3/breakfast_split3_stage1.model \
        --inference_only \
        --split_segments \
        --use_pe_tgt \
        --data_root data \ 
        --data_root_mean_duration mean_dur  

# split 4
python run.py \
        --use_cuda \
        --dataset breakfast \
        --split 4 \
	--use_viterbi \
        --viterbi_sample_rate 5 \
        --path_inference_model pretrained_models/breakfast/split4/breakfast_split4_stage1.model \
        --inference_only \
        --split_segments \
        --use_pe_tgt \
        --data_root data \ 
        --data_root_mean_duration mean_dur  
