import re
import requests
from xpinyin import Pinyin


def cosine_similarity(str1, str2):    
    url = "http://127.0.0.1:8000/similarity"
    
    payload = {
        "str1": str1,
        "str2": str2
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() 
        result = response.json()
        return result["similarity"]
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return 0.


def get_diversity_score(predict_response_list):
    score = 0
    for i in range(len(predict_response_list)):
        for j in range(i + 1, len(predict_response_list)):
            temp_score = cosine_similarity(predict_response_list[i], predict_response_list[j])
            score += temp_score
    try:
        return -score / (len(predict_response_list) * (len(predict_response_list) - 1) / 2)
    except:
        return 0.
    

def sug_score(
    predict_response_list, 
    prompt: str
):
    user_input = re.findall(
        r'\<当前输入部分\>(.*?)\-',
        prompt,
        re.DOTALL
    )
    
    if not user_input:
        return 0.
    
    p = Pinyin()
    user_input = user_input[0].strip()
    user_input_pinyin = p.get_pinyin(user_input, splitter="")
    # print(f"{user_input_pinyin=}")
    
    penalty = 0.
    for predict_response in predict_response_list:
        if any(c.isascii() and c.isalpha() for c in predict_response):
            penalty += -1.
        else:
            predict_respponse_pinyin = p.get_pinyin(predict_response, splitter="")
            # print(f"{predict_respponse_pinyin=}")
            if user_input_pinyin not in predict_respponse_pinyin:
                penalty += -1.
    return penalty


def compute_score(
    solution_str, 
    ground_truth,
    *args,
    **kwargs
):
    split_pattern = "<|im_start|>assistant"
    elements = solution_str.split(split_pattern)                  # split the response from total sequence
        
    if len(elements) < 2:
        return -5.
    
    prompt = split_pattern.join(elements[:-1]) + split_pattern
    sequence = elements[-1].strip()
    
    predict_response = re.findall(
        r'<answer>(.*?)</answer>', 
        sequence
    )

    if len(predict_response) != 1:
        return -5.
    
    response = predict_response[0]
    required_tags = ['<think>', '</think>', '<answer>', '</answer>']

    for tag in required_tags:
        count = sequence.count(tag)
        if count != 1:
            return -5.
    
    try:
        response_list = eval(response)
        assert isinstance(response_list, list)
        assert len(response_list) == 3
    except:
        return -5.
    
    all_rewards = []
    for r in response_list:
        similarity = cosine_similarity(r, ground_truth)
        all_rewards.append(similarity)
    
    bon_reward = max(all_rewards)
    diversity_score = get_diversity_score(response_list)
    sug_format_penalty = sug_score(response_list, prompt)
    
    return bon_reward * 2 + diversity_score + sug_format_penalty