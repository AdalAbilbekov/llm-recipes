export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_GID_INDEX=1
export NCCL_SOCKET_IFNAME=eth0 
export NCCL_IB_HCA=mlx5_0,mlx5_10,mlx5_11,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9
export NCCL_IB_TIMEOUT=22
export NCCL_IB_DISABLE=0
export NCCL_IB_RETRY_CNT=7
export NCCL_NET=IB
# export NCCL_LAUNCH_TIMEOUT=3000
# export NCCL_ASYNC_ERROR_HANDLING
# export WANDB_API_KEY=1a46c822eeac94d3222e7e28ffd6ade75aaffbe4
# export PYTHONPATH=.

# CUDA_VISIBLE_DEVICES=0 python finetuning.py --model_name /data/nvme3n1p1/adal_workspace/v2_distillation/checkpoints/db10_issai_batyr_tokenizer_base_37800 --dataset.file datasets/loader/squad.py --lr 1e-6 --num_epochs 100 --batch_size_training 1 --val_batch_size 1 --output_dir /data/nvme3n1p1/adal_workspace/v2_distilation_process/models/smolm2_scp --distillation_config.model_name /data/nvme3n1p1/adal_workspace/v2_distillation/checkpoints/Meta-Llama-3.1-8B-Instruct --distillation --distillation_config.enable_fsdp False --distillation_config.pure_bf16 --distillation_config.distil_factor 1.5 --distillation_config.checkpoint_type StateDictType.FULL_STATE_DICT --save_step 10 --context_length 1024

# accelerate launch finetuning.py --model_name HuggingFaceTB/SmolLM2-1.7B-Instruct --dataset.file datasets/loader/squad.py --lr 1e-6 --num_epochs 100 --batch_size_training 12 --val_batch_size 10 --output_dir /data/nvme3n1p1/adal_workspace/v2_distilation_process/models/smolm2_scp --distillation_config.model_name /data/nvme3n1p1/adal_workspace/distilation_process/llm-recipes/checkpoints/Meta-Llama-3.1-8B-Instruct --distillation --distillation_config.enable_fsdp --distillation_config.pure_bf16 --distillation_config.distil_factor 1.5 --save_step 10 --context_length 8192

torchrun --nproc_per_node=8 --nnodes=2 --master_addr=10.12.190.151 --master_port=1234 --node_rank=1 finetuning_by_data_chunks.py --model_name /data/nvme3n1p1/adal_workspace/v2_distillation/checkpoints/db10_issai_batyr_tokenizer_base_37800 --subset_length 10000 --val_set_length 5000 --dataset.file datasets/loader/squad.py --dataset.dataset_path "/data/nvme3n1p1/critical_all/critical_*/*.json" --lr 1e-6  --num_epochs 100 --batch_size_training 8 --val_batch_size 1 --output_dir /data/nvme1n1p1/cacheproblem_adal_checkpoints_knowledge_distillation_8b28b --distillation_config.model_name /data/nvme3n1p1/adal_workspace/v2_distillation/checkpoints/stable_base_sft_ci4_19200 --distillation --distillation_config.enable_fsdp True --distillation_config.pure_bf16 --distillation_config.distil_factor 1.5 --save_step 500 --context_length 2048

# torchrun --nproc_per_node=8 --nnodes=4 --master_addr=10.12.190.148 --master_port=1234 --node_rank=0 finetuning.py --model_name /data/nvme3n1p1/adal_workspace/v2_distillation/checkpoints/SmolLM2-1.7B-Instruct --dataset.file datasets/loader/squad_orig.py --lr 1e-6 --num_epochs 100 --batch_size_training 2 --val_batch_size 2 --output_dir /data/nvme5n1p1/distilled_llama8B_custom_tokenizer_091124 --distillation_config.model_name /data/nvme3n1p1/adal_workspace/v2_distillation/checkpoints/stable_base_sft_ci4_19200 --distillation --distillation_config.enable_fsdp True --distillation_config.pure_bf16 --distillation_config.distil_factor 1.5 --distillation_config.checkpoint_type StateDictType.FULL_STATE_DICT --save_step 500 --context_length 1024

# torchrun --nproc-per-node=8 --nnodes=4 --node_rank=0 --rdzv-id=456 --rdzv-backend=c10d --rdzv-endpoint=10.12.190.148:1234 finetuning.py --model_name /data/nvme3n1p1/adal_workspace/v2_distillation/checkpoints/db10_issai_batyr_tokenizer_base_37800 --dataset.file datasets/loader/squad.py --lr 1e-6 --num_epochs 100 --batch_size_training 2 --val_batch_size 2 --output_dir /data/nvme3n1p1/adal_workspace/v2_distillation/checkpoints/try_1_multinode --distillation_config.model_name /data/nvme3n1p1/adal_workspace/v2_distillation/checkpoints/Meta-Llama-3.1-8B-Instruct --distillation --distillation_config.enable_fsdp --distillation_config.pure_bf16 --distillation_config.distil_factor 1.5 --save_step 10 --context_length 8192
