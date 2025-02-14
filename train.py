from src.model import init_model
from src.data_processing import process_driving_data
from src.reward_functions import control_reward_func, xmlcount_reward_func, soft_format_reward_func, strict_format_reward_func
import torch
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported
import os
from datetime import datetime

def main():
    # 初始化模型和tokenizer
    model, tokenizer = init_model(
        model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
        max_seq_length=512,
        lora_rank=32,
        load_in_4bit=True,
        gpu_memory_utilization=0.6
    )
    
    # 加载和处理数据
    dataset = process_driving_data("./vqa_test_1k.pkl", tokenizer)
    
    # 设置训练参数
    training_args = GRPOConfig(
        use_vllm = True,  # 使用vLLM进行快速推理
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,  # 可以增加到4以获得更平滑的训练
        num_generations = 6,  # 如果内存不足可以减少
        max_prompt_length = 1024,
        max_completion_length = 200,
        max_steps = 250,
        save_steps = 250,
        max_grad_norm = 0.1,
        report_to = "none",  # 可以使用Weights & Biases
        output_dir = "outputs",
    )
    
    # 创建Trainer
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            control_reward_func
        ],
        args = training_args,
        train_dataset = dataset,
    )
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存最终模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = f"./model_checkpoints_{timestamp}"
    os.makedirs(final_model_path, exist_ok=True)
    
    # 保存LoRA权重
    model.save_lora(os.path.join(final_model_path, "lora_weights"))
    print(f"Training completed. Model saved to {final_model_path}")

if __name__ == "__main__":
    main() 