# Driving Agent Training

This is an autonomous driving agent training project based on the Llama 3.1 8B model. The project uses GRPO (Generative Reinforcement Policy Optimization) method to train the model to generate appropriate control instructions based on scene descriptions.

## Motivation & Background

This project attempts to reproduce the DeepSeek-R1 "Aha Moment" in the context of LLM-based autonomous driving. The implementation is migrated from Colab and based on Unsloth's framework. The pipeline (on Colab) has been tested to work effectively with both LLAMA and Qwen (by modifying chat template and formats) models at the 8B parameter level (but this repo is still under development).

Thanks to Unsloth's optimizations, these large models can be efficiently trained with PEFT (Parameter Efficient Fine-Tuning) using just 14GB of GPU memory, making it accessible for research and experimentation on consumer-grade hardware.

### Limitations & Considerations

While this serves as a proof-of-concept implementation of R1 for autonomous driving, it's important to note that the current dataset has significant limitations. The training data lacks crucial information such as velocity and other important geometric parameters, which would be essential for real-world autonomous driving applications. This implementation should be considered as a minimal reproduction for educational and research purposes.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── train.py
└── src/
    ├── data_processing.py
    ├── model.py
    └── reward_functions.py
```

## Key Features

- Data Processing: Convert raw driving scenario data into the required training format
- Model Training: Train the Llama model using GRPO method
- Reward Functions: Include control value rewards and XML format rewards

## Dataset

The training dataset (`vqa_test_1k.pkl`) is sourced from [Wayve's Driving-with-LLMs repository](https://github.com/wayveai/Driving-with-LLMs/blob/main/data/vqa_test_1k.tar.gz). This dataset contains driving scenarios with corresponding control instructions.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Data Preparation:
   - Ensure you have the `vqa_test_1k.pkl` data file
   - Data format should include scene descriptions and corresponding control instructions

2. Training:
   ```bash
   python train.py
   ```

## Model Input/Output Format

Input Format:
```
Scene description text
```

Output Format:
```xml
<reasoning>
Reasoning process
</reasoning>
<answer>
longitudinal: {acceleration value between 0-1}, lateral: {steering value between -1 to 1}
</answer>
```

## Notes

- Model uses 4-bit quantization to reduce memory usage
- Uses LoRA for efficient fine-tuning
- Requires sufficient GPU memory to run the 8B parameter model 