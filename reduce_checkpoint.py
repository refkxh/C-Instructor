import os

import torch


input_dir = "results_lm_vis_final_rxr"
input_file = os.path.join(input_dir, "checkpoint-7B.pth")
output_file = os.path.join(input_dir, "checkpoint-7B-reduced.pth")

checkpoint = torch.load(input_file, map_location="cpu")
reduced_checkpoint = {}

train_param_name = [
    "gate",
    "clip_proj",
    "clip_proj_norm",
    "clip_ob_proj",
    "clip_ob_proj_norm",
    "ob_ang_linear",
    "ob_ang_layer_norm",
    "visual_query",
    "visual_blocks",
    "visual_proj",
    "visual_proj_norm",
    "adapter_query",
    "ob_query",
    "action_query",
    "history_embeddings",
    "logits_temp",
]

for key, value in checkpoint["model"].items():
    if key.startswith("llama.layers"):
        layer_num = int(key.split(".")[2])
        if layer_num >= 30:
            reduced_checkpoint[key] = value
    elif key.startswith("llama.norm"):
        reduced_checkpoint[key] = value
    else:
        for train_name in train_param_name:
            if train_name in key:
                reduced_checkpoint[key] = value

print(f"Saved keys: {reduced_checkpoint.keys()}")
checkpoint["model"] = reduced_checkpoint
torch.save(checkpoint, output_file)
