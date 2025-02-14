import numpy as np
import re
from typing import List, Dict, Any
from .data_processing import extract_control_values

def control_reward_func(prompts: List[Dict[str, Any]], completions: List[List[Dict[str, str]]], answer: List[str], **kwargs) -> List[float]:
    """
    计算控制值的奖励函数
    
    Args:
        prompts: 提示列表
        completions: 模型完成的回答列表
        answer: 目标答案列表
        
    Returns:
        List[float]: 奖励值列表
    """
    responses = [completion[0]['content'] for completion in completions]
    rewards = []

    for response, target_answer in zip(responses, answer):
        # 提取目标值和预测值
        target_values = extract_control_values(target_answer)
        predicted_values = extract_control_values(response)

        if target_values is None or predicted_values is None:
            rewards.append(0.0)
            continue

        # 解包值
        target_long, target_lat = target_values
        pred_long, pred_lat = predicted_values

        # 计算欧氏距离
        distance = np.sqrt((target_long - pred_long)**2 + (target_lat - pred_lat)**2)

        # 将距离转换为reward
        # 距离为0时得到最高分2.0，距离越大分数越低
        reward = 2.0 * np.exp(-2 * distance)

        rewards.append(reward)

    return rewards

def extract_xml_answer(text: str) -> str:
    """提取XML格式的答案"""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def count_xml(text: str) -> float:
    """计算XML标签的完整性"""
    score = 0.0
    
    # 检查reasoning标签
    if "<reasoning>" in text and "</reasoning>" in text:
        score += 0.5
        
    # 检查answer标签
    if "<answer>" in text and "</answer>" in text:
        score += 0.5
        
    return score

def xmlcount_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
    """
    计算XML格式完整性的奖励函数
    
    Args:
        completions: 模型完成的回答列表
        
    Returns:
        List[float]: 奖励值列表
    """
    responses = [completion[0]['content'] for completion in completions]
    return [count_xml(response) for response in responses] 