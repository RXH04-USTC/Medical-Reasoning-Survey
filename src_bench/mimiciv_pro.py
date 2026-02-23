import pandas as pd
import random
import json

# 数据加载
data1 = pd.read_csv('PATH_TO_DATA/data4LLM_CONCISE_TITLE_1_final.csv')
data2 = pd.read_csv('PATH_TO_DATA/data4LLM_CONCISE_TITLE_2_final.csv')
data3 = pd.read_csv('PATH_TO_DATA/data4LLM_CONCISE_TITLE_3_final.csv')
data4 = pd.read_csv('PATH_TO_DATA/data4LLM_CONCISE_TITLE_4_final.csv')

data1_sample = data1.sample(n=250, random_state=42)
data2_sample = data2.sample(n=250, random_state=42)
data3_sample = data3.sample(n=250, random_state=42)
data4_sample = data4.sample(n=250, random_state=42)

data_combined = pd.concat([data1_sample, data2_sample, data3_sample, data4_sample], ignore_index=True)

prompt_template = """You are taking a medical competency test.
I will provide you with a patient's diagnosis list, clinical note describing the patient's health condition, and a set of procedure options.
Your task is to identify choose the most appropriate procedure list from the given options.

Diagnosis:
{diagnosis}

Clinical Note:
{note}

"""

sample_neg_choice_num = 7
output_path = "PATH_TO_OUTPUT/mimic_iv_procedure.jsonl"

test_data = []

for index, row in data_combined.iterrows():
    diagnosis = row['diagnose']
    note = row['NOTE_CONTENT']
    procedures = row['procedure']

    distractors = data_combined[data_combined.index != index].sample(sample_neg_choice_num)['procedure'].tolist()

    correct_pos = random.randint(0, sample_neg_choice_num)
    options = distractors.copy()
    options.insert(correct_pos, procedures)
    
    # 构建选项字典
    options_dict = {
        "A": options[0],
        "B": options[1],
        "C": options[2],
        "D": options[3],
        "E": options[4],
        "F": options[5],
        "G": options[6],
        "H": options[7]
    }
    
    # 生成问题文本（不包含选项）
    question = prompt_template.format(
        diagnosis=diagnosis,
        note=note
    )
    
    # 确定正确答案的索引和文本
    answer_idx = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][correct_pos]
    answer = procedures  # 正确答案是原过程列表

    test_data.append({
        "question": question,
        "options": options_dict,
        "answer_idx": answer_idx,
        "answer": answer
    })

with open(output_path, 'w') as f:
    for item in test_data:
        f.write(json.dumps(item) + '\n')