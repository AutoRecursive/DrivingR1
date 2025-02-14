from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
import torch

def init_model(
    model_name: str = "meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length: int = 512,
    lora_rank: int = 32,
    load_in_4bit: bool = True,
    gpu_memory_utilization: float = 0.6,
) -> tuple[FastLanguageModel, any]:
    """
    初始化和配置模型
    
    Args:
        model_name: 模型名称
        max_seq_length: 最大序列长度
        lora_rank: LoRA秩
        load_in_4bit: 是否使用4bit量化
        gpu_memory_utilization: GPU内存使用率
        
    Returns:
        tuple: (model, tokenizer)
    """
    # 在所有函数之前使用PatchFastRL来patch GRPO和其他RL算法
    PatchFastRL("GRPO", FastLanguageModel)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        fast_inference = True,  # 启用vLLM快速推理
        max_lora_rank = lora_rank,
        gpu_memory_utilization = gpu_memory_utilization,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth",  # 启用长上下文微调
        random_state = 3407,
    )
    
    return model, tokenizer 