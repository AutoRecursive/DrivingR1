import re
import pickle
from datasets import Dataset
from transformers import PreTrainedTokenizer

def get_control_longitudinal(text: str) -> float | None:
    """从文本中提取纵向控制值"""
    accelerator = re.search(r"Accelerator pedal (\d+)%", text)
    brake = re.search(r"Brake pedal (\d+)%", text)
    accelerator_value = int(accelerator.group(1)) / 100 if accelerator else None
    brake_value = int(brake.group(1)) / 100 if brake else None
    if accelerator_value is None or brake_value is None:
        return None
    x = accelerator_value - brake_value
    control_longitudinal = (x + 1.0) / 2.0
    return control_longitudinal

def get_control_lateral(text: str) -> float | None:
    """从文本中提取横向控制值"""
    match = re.search(r"(\d+)% to the (right|left)\.", text, re.IGNORECASE)
    if match:
        percentage, direction = match.groups()
        value = int(percentage) / 100.0
        if direction.lower() == "right":
            value *= -1
        return value
    return None

def clean_input_prompt(text: str) -> str:
    """清理输入提示，只保留'Here are my actions'之前的部分"""
    parts = text.split("Here are my actions")
    return parts[0].strip()

def truncate_text(text: str, tokenizer: PreTrainedTokenizer, max_length: int = 400) -> str:
    """将文本截断到最大长度"""
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length:
        truncated_tokens = tokens[:max_length]
        truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
        return truncated_text
    return text

def process_driving_data(data_path: str, tokenizer: PreTrainedTokenizer) -> Dataset:
    """处理驾驶数据并返回Dataset格式"""
    with open(data_path, 'rb') as file:
        data = pickle.load(file)

    processed_data = []
    for item in data:
        cleaned_input_prompt = clean_input_prompt(item['input_prompt'])
        cleaned_prompt = truncate_text(cleaned_input_prompt, tokenizer)

        control_long = get_control_longitudinal(item['response_content'])
        control_lat = get_control_lateral(item['response_content'])

        if control_long is not None and control_lat is not None:
            processed_item = {
                'prompt': [
                    {'role': 'system', 'content': 'You are a fully autonomous AI driver. Try to provide control values in the following format: <reasoning>Your thoughts</reasoning><answer>longitudinal: {from 0(full brake) to 1(full pedal) as percentage}, lateral: { steer angle from 0 to 1 as percentage}</answer>'},
                    {'role': 'user', 'content': cleaned_prompt}
                ],
                'answer': f"<reasoning></reasoning><answer>longitudinal: {control_long}, lateral: {control_lat}</answer>"
            }
            processed_data.append(processed_item)

    return Dataset.from_list(processed_data)

def extract_control_values(text: str) -> tuple[float, float] | None:
    """从回答中提取控制值"""
    pattern = r'<answer>longitudinal: ([-\d.]+), lateral: ([-\d.]+)</answer>'
    match = re.search(pattern, text)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None 