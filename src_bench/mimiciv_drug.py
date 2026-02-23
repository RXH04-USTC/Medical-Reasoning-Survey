import pandas as pd
import json
import random
import pickle

# 数据加载
data1 = pd.read_csv('PATH_TO_DATA/data4LLM_CONCISE_TITLE_1_final.csv')  # 替换为实际路径
data2 = pd.read_csv('PATH_TO_DATA/data4LLM_CONCISE_TITLE_2_final.csv')  # 替换为实际路径
data3 = pd.read_csv('PATH_TO_DATA/data4LLM_CONCISE_TITLE_3_final.csv')  # 替换为实际路径
data4 = pd.read_csv('PATH_TO_DATA/data4LLM_CONCISE_TITLE_4_final.csv')  # 替换为实际路径

data1_sample = data1.sample(n=250, random_state=42)
data2_sample = data2.sample(n=250, random_state=42)
data3_sample = data3.sample(n=250, random_state=42)
data4_sample = data4.sample(n=250, random_state=42)

data_combined = pd.concat([data1_sample, data2_sample, data3_sample, data4_sample], ignore_index=True)

prompt_template = """You are taking a medical competency test.
I will provide you with a patient's diagnosis list, procedure list, clinical note describing the patient's health condition, and a partial medication list.
One medication that should be prescribed for this patient is missing from the list.
Your task is to identify the most appropriate missing medication from the given options, based on the patient's health condition.

Diagnosis:
{diagnosis}

Procedures:
{procedures}

Clinical Note:
{note}

Partial Medication List:
{partial_med_list}

Which medication is missing from the partial medication list?"""

test_data = []

sample_med_num = 1
sample_neg_choice_num = 7
output_path = "PATH_TO_OUTPUT/mimic_iv_med.jsonl"

with open('PATH_TO_DATA/med_name2idx.json', 'r') as f:
    med_name2idx = json.load(f)
med_idx2name = {v: k for k, v in med_name2idx.items()}
all_med_names = list(med_name2idx.keys())
ddi = pickle.load(open('//ddi_A_final.pkl', 'rb'))

for index, row in data_combined.iterrows():
    diagnosis = row['diagnose']
    procedures = row['procedure']
    note = row['NOTE_CONTENT']
    gt_med = eval(row['drug_name'])

    med_to_remove = random.sample(gt_med, min(sample_med_num, len(gt_med)))
    med_not_in_gt = list(set(all_med_names) - set(gt_med))

    for med in med_to_remove:
        partial_med_list = gt_med.copy()
        partial_med_list.remove(med)
        # 根据ddi选择错误选项
        med_idx = med_name2idx[med]
        # 找出ddi[med_idx]这个list当中为1的那些ddi_idx
        ddi_indices = [i for i, val in enumerate(ddi[med_idx]) if val == 1]
        # 将这些ddi_idx转换为药物名称
        ddi_med_names = [med_idx2name[i] for i in ddi_indices if med_idx2name[i] in med_not_in_gt]
        ddi_med_names_not_in_gt = list(set(ddi_med_names) - set(partial_med_list))
        ### 如果len(ddi_med_names_not_in_gt) >= sample_neg_choice_num,则从中随机选择sample_neg_choice_num个作为错误选项
        ### 如果len(ddi_med_names_not_in_gt) < sample_neg_choice_num,则先将ddi_med_names_not_in_gt全部加入错误选项中,然后再从med_not_in_gt中随机选择剩余数量的不在ddi_med_names_not_in_gt的药物名称加入错误选项中
        if len(ddi_med_names_not_in_gt) >= sample_neg_choice_num:
            distractors = random.sample(ddi_med_names_not_in_gt, sample_neg_choice_num)
        else:
            distractors = ddi_med_names_not_in_gt.copy()
            remaining_num = sample_neg_choice_num - len(distractors)
            additional_distractors = random.sample(list(set(med_not_in_gt) - set(ddi_med_names_not_in_gt)), remaining_num)
            distractors.extend(additional_distractors)
        random.shuffle(distractors)
        correct_pos = random.randint(0, sample_neg_choice_num)
        options = distractors.copy()
        options.insert(correct_pos, med)
        
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
            procedures=procedures,
            note=note,
            partial_med_list=partial_med_list
        )
        
        # 确定正确答案的索引和文本
        answer_idx = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][correct_pos]
        answer = med  # 正确答案是从原列表中移除的那个药物

        test_data.append({
            "question": question,
            "options": options_dict,
            "answer_idx": answer_idx,
            "answer": answer
        })

with open(output_path, 'w') as f:
    for item in test_data:
        f.write(json.dumps(item) + '\n')