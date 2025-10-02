import os
import subprocess
import yaml

# 基础yaml文件路径
base_yaml = "examples/train_lora/llama3_lora_sft_hevi.yaml"

# 指定 wandb 项目（你说的 hevi_finetune）
WANDB_PROJECT = "hevi_finetune_1002"
# 如果有团队空间可填，否则留空即可（默认个人空间）
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")

# 要跑的模型列表 (模型名, 模板名, 模型规模)
MODELS = [
    # ("google/gemma-3-4b-it", "gemma3", "4B"),
    # ("google/gemma-3-1b-it", "gemma3", "1B"),
    # ("meta-llama/Llama-3.2-1B-Instruct", "llama3", "1B"),
    # ("meta-llama/Llama-3.2-3B-Instruct", "llama3", "3B"),
    ("microsoft/Phi-3.5-mini-instruct", "phi", "3.5B"),
#     ("mistralai/Mistral-7B-Instruct-v0.3", "mistral", "7B"),
#     ("Qwen/Qwen3-1.7B", "qwen3_nothink", "1.7B"),
#     ("Qwen/Qwen3-4B-Instruct-2507", "qwen3_nothink", "4B"),
#     ("Qwen/Qwen3-8B", "qwen3_nothink", "8B"),
#     ("Qwen/Qwen2.5-7B-Instruct-1M", "qwen3_nothink", "7B"),
]

def adjust_batch(config, scale: str):
    """根据模型规模粗调 batch 和累积步数（按你的 48GB 显存假设）"""
    if any(s in scale for s in ["1B", "3B", "3.5B", "4B"]):
        config["per_device_train_batch_size"] = 2
        config["gradient_accumulation_steps"] = 8
    elif "7B" in scale:
        config["per_device_train_batch_size"] = 1
        config["gradient_accumulation_steps"] = 8
    elif "8B" in scale:
        config["per_device_train_batch_size"] = 1
        config["gradient_accumulation_steps"] = 16
    return config

def need_torch26_for_gemma(model_id: str) -> bool:
    """Gemma 3 在 transformers 新实现里需要 torch>=2.6"""
    if "gemma-3" in model_id.lower():
        return version.parse(torch.__version__) < version.parse("2.6")
    return False

for model_id, template, scale in MODELS:
    # torch 版本不足时跳过 Gemma 3
    if need_torch26_for_gemma(model_id):
        print(f"⏭️  Skip {model_id} because torch<{2.6} (current {torch.__version__}).")
        continue

    run_name   = model_id.split("/")[-1]
    output_dir = f"saves/{run_name}/lora/sft"

    # 读取基线 yaml
    with open(base_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    # 覆盖关键字段
    cfg["run_name"] = run_name
    cfg["model_name_or_path"] = model_id
    cfg["template"] = template
    cfg["output_dir"] = output_dir
    cfg["report_to"] = "wandb"  # 确保启用 wandb

    # （可选）不从旧 checkpoint 恢复，避免 torch<2.6 的 resume 限制
    # 需要 resume 就把下一行注释掉，并确保你是 torch>=2.6
    cfg["resume_from_checkpoint"] = None
    cfg["save_safetensors"] = True  # 用更安全的权重格式

    # 自动调整 batch 参数
    cfg = adjust_batch(cfg, scale)

    # 写出临时 yaml
    temp_yaml = f"examples/train_lora/{run_name}_sft.yaml"
    with open(temp_yaml, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    # 为子进程注入 wandb 环境变量（关键点！）
    env = os.environ.copy()
    env["WANDB_PROJECT"] = WANDB_PROJECT          # ✅ 固定到 hevi_finetune
    if WANDB_ENTITY:
        env["WANDB_ENTITY"] = WANDB_ENTITY       # 可选：团队/组织名
    env["WANDB_NAME"] = run_name                  # 让 run 的名字和模型一致，便于区分

    print(f"\n🚀 Running LoRA finetune for {model_id} ({scale}) "
          f"→ project={env['WANDB_PROJECT']} run_name={env['WANDB_NAME']}")
    try:
        subprocess.run(["llamafactory-cli", "train", temp_yaml], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"❌ {model_id} failed with exit code {e.returncode}")

"""
llamafactory-cli export \
  --model_name_or_path mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter_name_or_path saves/Mistral-7B-Instruct-v0.3-lora/ \
  --export_dir merged_models/Mistral-7B-Instruct-v0.3-lora \
  --export_size 2 \
  --export_device cpu \
  --export_legacy_format False \
  --template mistral

llamafactory-cli export \
  --model_name_or_path microsoft/Phi-3.5-mini-instruct \
  --adapter_name_or_path saves/phi3.5-mini-lora/ \
  --export_dir merged_models/Phi-3.5-mini-instruct-lora \
  --export_size 2 \
  --export_device auto \
  --export_legacy_format True \
  --template phi

llamafactory-cli export \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct-1M \
  --adapter_name_or_path saves/Qwen2.5-7B-Instruct-1M-lora/ \
  --export_dir merged_models/Qwen2.5-7B-Instruct-1M-lora \
  --export_size 2 \
  --export_device cpu \
  --export_legacy_format False \
  --template qwen3_nothink

"""