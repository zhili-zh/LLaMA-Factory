import os
import subprocess
import yaml

# 基础yaml文件路径
base_yaml = "examples/train_lora/llama3_lora_sft_hevi.yaml"

# 要跑的模型列表 (模型名, 模板名, 模型规模)
models = [
    ("google/gemma-3-4b-it", "gemma3", "4B"),
    ("google/gemma-3-1b-it", "gemma3", "1B"),
    ("meta-llama/Llama-3.2-1B-Instruct", "llama3", "1B"),
    ("meta-llama/Llama-3.2-3B-Instruct", "llama3", "3B"),
    ("microsoft/Phi-3.5-mini-instruct", "phi", "3.5B"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral", "7B"),
    ("Qwen/Qwen3-1.7B", "qwen3_nothink", "1.7B"),
    ("Qwen/Qwen3-4B-Instruct-2507", "qwen3_nothink", "4B"),
    ("Qwen/Qwen3-8B", "qwen3_nothink", "8B"),
    ("Qwen/Qwen2.5-7B-Instruct-1M", "qwen3_nothink", "7B"),
]

def adjust_batch_params(config, scale):
    """根据模型规模调整 batch size 和 grad_accum"""
    if "1B" in scale or "3B" in scale or "3.5B" in scale or "4B" in scale:
        config["per_device_train_batch_size"] = 2
        config["gradient_accumulation_steps"] = 8
    elif "7B" in scale:
        config["per_device_train_batch_size"] = 1
        config["gradient_accumulation_steps"] = 8
    elif "8B" in scale:
        config["per_device_train_batch_size"] = 1
        config["gradient_accumulation_steps"] = 16
    return config

# 逐个模型运行
for model, template, scale in models:
    run_name = model.split("/")[-1]
    output_dir = f"saves/{run_name}/lora/sft"

    # 读取基础yaml
    with open(base_yaml, "r") as f:
        config = yaml.safe_load(f)

    # 修改关键字段
    config["run_name"] = run_name
    config["model_name_or_path"] = model
    config["template"] = template
    config["output_dir"] = output_dir

    # 调整 batch 参数
    config = adjust_batch_params(config, scale)

    # 保存临时yaml
    temp_yaml = f"examples/train_lora/{run_name}_sft.yaml"
    with open(temp_yaml, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    # 调用训练命令
    print(f"🚀 Running LoRA finetune for {model} ({scale})")
    subprocess.run(["llamafactory-cli", "train", temp_yaml], check=True)




