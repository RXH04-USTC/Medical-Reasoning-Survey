#%%
#这是针对通用模型的评分逻辑，对于给出自定义评分逻辑的模型，请替换对应的match_choice函数和score函数
from collections import defaultdict
import re
import json
import difflib
import os

def process_result_file(result_file, verbose=False):
    """
    Process a single result_with_score.json file:
    - Computes accuracy from 'score'
    - Compares it to accuracy in accuracy.json
    - If mismatch, tries to resolve by checking if first or second answer is correct more often
    - Updates scores and returns accuracy mismatch if still present

    Args:
        result_file (str or Path): Path to result_with_score.json
        verbose (bool): Print mismatches and errors if True

    Returns:
        tuple or None: (result_file, recalculated_accuracy, logged_accuracy) if mismatch persists
    """
    result_file = str(result_file)
    
    if "openmedcases" in result_file:
        return None

    try:
        with open(result_file, "r") as f:
            results = json.load(f)
    except Exception as e:
        if verbose:
            print(f"Failed to load {result_file}: {e}")
        return None

    try:
        scores = [x["score"] for x in results]
        accuracy = sum(scores) / len(scores)
    except:
        if verbose:
            print(f"Invalid scores in {result_file}")
        return None

    accuracy_path = os.path.join(os.path.dirname(result_file), "accuracy.json")
    try:
        with open(accuracy_path, "r") as f:
            accuracy_dict = json.load(f)
    except:
        if verbose:
            print(f"Missing accuracy.json for {result_file}")
        return None

    if "normal" in accuracy_dict:
        key = "normal"
    elif "unknown" in accuracy_dict:
        key = "unknown"
    else:
        raise ValueError(f"Unknown accuracy key for {result_file}")

    logged_accuracy = accuracy_dict[key][0]

    # Accept if match
    if abs(accuracy - logged_accuracy) < 0.01:
        return None

    # Count front vs back correct answers
    count_front, count_back = 0, 0
    for result in results:
        ans = result["ans"]
        answer_idx = result["answer_idx"]
        if len(ans) != 2:
            continue  # malformed answer

        if ans[0].lower() == answer_idx.lower():
            count_front += 1
        if ans[1].lower() == answer_idx.lower():
            count_back += 1

    front = count_front > count_back

    # Update scores
    for result in results:
        chosen_ans = result["ans"][0] if front else result["ans"][1]
        result["score"] = int(chosen_ans.lower() == result["answer_idx"].lower())

    # Recalculate accuracy
    scores = [x["score"] for x in results]
    try:
        accuracy = sum(scores) / len(scores)
    except:
        if verbose:
            print(f"Error recomputing accuracy in {result_file}")
        return None

    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)

    if abs(accuracy - logged_accuracy) > 0.01:
        return (result_file, accuracy, logged_accuracy)

    return None

def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()

def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0
    
    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)
        
        # If the current string is more similar than the previous most similar string, update the variables
        if similarity >= highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity

    return most_similar_index


# Updated function
def match_choice(text, options):
    """
    Extracts the answer choice from the model's output text.
    It now prioritizes finding the answer within a \boxed{} expression.
    """
    # Create a string of valid option letters (e.g., "ABCD")
    match_options = 'ABCDEFGHIJKLMN'[:len(options)]
    
    # --- NEW, MORE ROBUST LOGIC: PRIORITY 1 ---
    # Step 1: Broadly find any content within \boxed{...}.
    # The (.*?) is a non-greedy capture of any character.
    # re.DOTALL allows '.' to match newline characters, in case the content spans multiple lines.
    boxed_content_pattern = r"\\boxed{(.*?)}"
    boxed_content_match = re.search(boxed_content_pattern, text, re.DOTALL)
    
    if boxed_content_match:
        # Step 2: Extract the content found inside the braces.
        # e.g., inner_text could be "B", "\text{B}", or "The correct answer is B".
        inner_text = boxed_content_match.group(1)
        
        # Step 3: Find all occurrences of valid option letters within that inner text.
        option_pattern = r"[" + match_options + r"]"
        possible_ans = re.findall(option_pattern, inner_text, re.IGNORECASE)
        
        if possible_ans:
            # If we found one or more option letters (e.g., in a sentence),
            # we assume the LAST one is the intended final answer.
            final_ans = possible_ans[-1].upper().strip()
            # Return it as a high-confidence match (ans_type = 0)
            return [final_ans, final_ans], 0

    # --- ORIGINAL FALLBACK LOGIC ---
    # If \boxed{} is not found, proceed with the old methods.
    
    # For HuatuoGPT-o1
    if '## Final Response' in text:
        text = text.split('## Final Response')[-1].strip()
    # for our model
    if '## Final Answer' in text:
        text = text.split('## Final Answer')[-1].strip()

    if "<answer>" in text:
        text = text.split("<answer>")[-1].split("</answer>")[0].strip()

    if "answer is" in text:
        text = text.split("answer is")[-1].strip()
    
    if "Answer is" in text:
        text = text.split("Answer is")[-1].strip()

    if "answer:" in text:
        text = text.split("answer:")[-1].strip()

    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    
    # for strict prompt 
    matches = list(re.finditer(r"(answer is\s*?)([A-N])", text, re.S))
    if matches:
        ans_first = matches[0].group(2)
        ans_last = matches[-1].group(2)
        return [ans_first, ans_last], 1

    # non strict
    matches = list(re.finditer(r"([\u4e00-\u9fff]|is |是|项|\*|\W|\ |\(|为|^|'|\"|#)(?![aA] )([" + match_options + r"])(\W|[\u4e00-\u9fff]|$)", text, re.S))
    if matches:
        ans_first = matches[0].group(2)
        ans_last = matches[-1].group(2)
        return [ans_first, ans_last], 1

    text = text.lower()
    opsindex = [(opt, text.rindex(options[opt].lower())) for opt in options if options[opt].lower() in text]
    if len(opsindex) > 0:
        ans_last = sorted(opsindex, key=lambda x: x[1], reverse=True)[0][0]
        opsindex = [(opt, text.index(options[opt].lower())) for opt in options if options[opt].lower() in text]
        ans_first = sorted(opsindex, key=lambda x: x[1], reverse=True)[0][0]
        return [ans_first, ans_last], 2
    else:
        oplabels = [x for x in options]
        opans = [options[x].lower() for x in options]
        ansindex = find_most_similar_index(opans, text.lower())
        return [oplabels[ansindex], oplabels[ansindex]], 3


# Updated function
def score(data, ignore_miss=False):
    res = {}
    wrong_data = []
    cor_data = []
    for da in data:
        if 'source' not in da:
            da['source'] = 'unknown'
        if da['source'] not in res:
            res[da['source']] = [0, 0, 0, 0]

        output = da['output']
        ans, ans_type = match_choice(output, da['options'])
        
        # --- MODIFIED LINE ---
        # Allow ans_type 0 (from \boxed{}) to be counted as a valid match
        if ignore_miss and ans_type not in [0, 1]:
            da['score'] = 0
            continue

        da['ans'] = ans
        da['ans_type'] = ans_type

        if ans[0].lower() == da['answer_idx'].lower():
            da['score'] = 1
            res[da['source']][1] += 1
            cor_data.append(da)
        else:
            da['score'] = 0
            wrong_data.append(da)
        
        if ans[1].lower() == da['answer_idx'].lower():
            # This logic seems to be for comparing first vs. last mention.
            # For a boxed answer, ans[0] and ans[1] are the same, so this logic is fine.
            da['score'] = 1 # This might overwrite the score, but it will be the same value
            res[da['source']][3] += 1
        else:
            # If the first answer was correct but the last wasn't (shouldn't happen for boxed),
            # this would set the score back to 0. The subsequent logic handles this.
            da['score'] = 0

        res[da['source']][2] += 1

    for k in res:
        head_match_score = res[k][1] / res[k][2]
        tail_match_score = res[k][3] / res[k][2]
        if head_match_score > tail_match_score:
            res[k][0] = head_match_score
        else:
            res[k][0] = tail_match_score

    return res, wrong_data, cor_data



def get_results(res_path):

    if res_path.endswith(".jsonl"):
        data = []
        with open(res_path) as f:
            for line in f:
                data.append(json.loads(line))
    else:
        with open(res_path) as f:
            data = json.load(f) 

    res,wrong_data,cor_data =  score(data)  

    res_file_name = os.path.basename(res_path)
    print(f"*Logging_file: {res_file_name}*")
    print(json.dumps(res,indent=4))

    accuracy_file = os.path.join(os.path.dirname(res_path), 'accuracy.json')
    results_with_score_file = os.path.join(os.path.dirname(res_path), 'result_with_score.json')

    with open(accuracy_file, 'w') as fw:
        json.dump(res, fw, ensure_ascii=False, indent=2)

    with open(results_with_score_file, 'w') as fw:
        json.dump(data, fw, ensure_ascii=False, indent=2)

    process_result_file(results_with_score_file)

# if __name__ == "__main__":
#     get_results('output_file_path')