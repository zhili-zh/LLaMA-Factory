from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "microsoft/Phi-3.5-mini-instruct"
adapter_path = "saves/phi3.5-mini-lora/"
export_dir = "merged_models/Phi-3.5-mini-instruct-lora"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model = PeftModel.from_pretrained(base, adapter_path)

# merge & unload
model = model.merge_and_unload()

# save
model.save_pretrained(export_dir, safe_serialization=True)
tokenizer.save_pretrained(export_dir)
