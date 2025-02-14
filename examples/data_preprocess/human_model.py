# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
from tqdm import tqdm

import pandas as pd


def process_dataframe_rows(df, processing_function):
    """
    逐行处理 Pandas DataFrame 并返回一个新的 DataFrame。
    """
    processed_rows = []

    for index, row in tqdm(df.iterrows()):
        processed_data = processing_function(row)
        if not isinstance(processed_data, dict):
            raise ValueError("处理函数必须返回一个字典。")
        processed_rows.append(processed_data)

    new_df = pd.DataFrame(processed_rows)
    return new_df


def processing_function(row):
    data = {
        "data_source": "human_model",
        "prompt": row["context_messages"],
        "ability": "nlp",
        "reward_model": {
            "style": "rule",
            "ground_truth": row["answer"],
        },
        "extra_info": {
            'split': "",
            'answer': row["answer"],
            'question': row["input"],
            'sys_prompt': row["sys_prompt"],
        }
    }
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_path', default='/cpfs/user/bupo/data/intent_rm_dataset/human_model_add_sug_task.jsonl')
    # parser.add_argument('--local_path', default='/cpfs/user/linke/human_model/test_data/hm_24-10_test_500.jsonl')
    parser.add_argument('--save_path', default='/cpfs/user/bupo/data/intent_rm_dataset/human_model/train.parquet')
    args = parser.parse_args()
    
    origin_dataset = pd.read_json(args.local_path, lines=True)
    formated_dataset = process_dataframe_rows(origin_dataset, processing_function)
    
    # print(df.head())
    # print('--------------------')
    # print(new_df.head())

    formated_dataset.to_parquet(os.path.join(args.save_path))
    print(f'[Done] Dataset is saved to {args.save_path}.')