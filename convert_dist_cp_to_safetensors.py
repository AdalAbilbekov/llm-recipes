from transformers import AutoModelForCausalLM
import torch.distributed._shard.checkpoint as dist_cp
import torch

# source: https://discuss.huggingface.co/t/transformers-trainer-accelerate-fsdp-how-do-i-load-my-model-from-a-checkpoint/61585

model = AutoModelForCausalLM.from_pretrained(
    "/data/nvme3n1p1/adal_workspace/v2_distillation/checkpoints/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)


state_dict = {
        "model": model.state_dict()
    }

dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader= dist_cp.FileSystemReader("/data/nvme3n1p1/adal_workspace/v2_distillation/checkpoints/knowledge_distilled_60660"),
                no_dist=True,
            )

model.load_state_dict(state_dict["model"])

new_state_dict = model.state_dict()