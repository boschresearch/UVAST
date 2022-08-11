# split 1
python run.py \
	--use_cuda \
	--dataset gtea \
	--split 1 \
	--path_inference_model pretrained_models/gtea/split1/gtea_split1_stage2.model \
	--inference_only \
	--split_segments \
	--use_pe_tgt \
	--use_alignment_dec \
	--data_root data \ 
	--data_root_mean_duration mean_dur 

# split 2
python run.py \
        --use_cuda \
        --dataset gtea \
        --split 2 \
        --path_inference_model pretrained_models/gtea/split2/gtea_split2_stage2.model \
        --inference_only \
        --split_segments \
        --use_pe_tgt \
        --use_alignment_dec \
        --data_root data \ 
        --data_root_mean_duration mean_dur 

# split 3
python run.py \
        --use_cuda \
        --dataset gtea \
        --split 3 \
        --path_inference_model pretrained_models/gtea/split3/gtea_split3_stage2.model \
        --inference_only \
        --split_segments \
        --use_pe_tgt \
        --use_alignment_dec \
        --data_root data \ 
        --data_root_mean_duration mean_dur 

# split 4
python run.py \
        --use_cuda \
        --dataset gtea \
        --split 4 \
        --path_inference_model pretrained_models/gtea/split4/gtea_split4_stage2.model \
        --inference_only \
        --split_segments \
        --use_pe_tgt \
        --use_alignment_dec \
        --data_root data \ 
        --data_root_mean_duration mean_dur 
