import argparse
from tqdm import tqdm
import openai
from jinja2 import Template
import os
import json
from transformers import AutoTokenizer
from scorer import get_results
from datasets import load_dataset


def postprocess_output(pred):
    pred = pred.replace("</s>", "")
    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred

def load_file(input_fp, eval_benchmark=None):
    # 如果 input_fp 是一个本地文件 (例如, ../benchs/xxx.jsonl), 直接读取 JSON Lines.
    if os.path.isfile(input_fp):
        print(f"Loading local file: {input_fp}")
        input_data = []
        with open(input_fp, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # 如果 options 字段是 JSON 字符串，尝试解析；否则保持原样。
                if "options" in item and isinstance(item["options"], str):
                    try:
                        item["options"] = json.loads(item["options"])
                    except Exception:
                        # leave it as string if parsing fails
                        pass
                input_data.append(item)
        return input_data

    # 回退到 Hugging Face Hub 加载器（原始行为）
    print(f"Loading from Hugging Face Hub: {input_fp}, subset: {eval_benchmark}")
    dataset = load_dataset(input_fp, eval_benchmark)['test']
    input_data = [item for item in dataset]

    for item in input_data:
        # original code assumes options is a JSON string
        item["options"] = json.loads(item["options"])
    return input_data

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dataset', type=str, required=True)
    parser.add_argument('--eval_benchmark', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=6000)
    parser.add_argument('--max_tokens', type=int, default=-1)
    parser.add_argument('--use_chat_template', action="store_true")
    parser.add_argument('--strict_prompt', action="store_true")
    parser.add_argument('--task', type=str,default='api')
    parser.add_argument('--port', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=1024)    
    parser.add_argument('--reasoning', action="store_true")
    parser.add_argument('--temperature', type=float, default=0.2)
    args = parser.parse_args()

    print(f"Using local API server at port {args.port}")
    client = openai.Client(
        base_url=f"http://127.0.0.1:{args.port}/v1", 
        # base_url=f"http://localhost:{args.port}/v1",
        api_key="EMPTY"
    )
    model = client.models.list().data[0].id
    print(f"Using model {model}")
    if args.use_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, padding_side='left')
        template = Template(tokenizer.chat_template)

    def call_model(prompts, model, max_new_tokens=50, temperature=0.5, print_example =False):
        if print_example:
            print("Example:")
            print(prompts[1])
        preds = []
        if args.use_chat_template: 
            prompts = [template.render(messages=[{"role": "user", "content": prom}],bos_token= tokenizer.bos_token,add_generation_prompt=True) for prom in prompts]
        
        if args.max_tokens > 0:
            new_prompts = []
            for prompt in prompts:
                input_ids = tokenizer.encode(prompt,add_special_tokens= False)
                if len(input_ids) > args.max_tokens:
                    input_ids = input_ids[:args.max_tokens]
                    new_prompts.append(tokenizer.decode(input_ids))
                else:
                    new_prompts.append(prompt[-args.max_tokens:])
            prompts = new_prompts

        response = client.completions.create(
            model=model,
            prompt=prompts,
            temperature=temperature, top_p=0.9, max_tokens=max_new_tokens,
        )
        preds = [x.text for x in response.choices]
        postprocessed_preds = [postprocess_output(pred) for pred in preds]
        return postprocessed_preds, preds

    # args.eval_dataset 现在将是完整的文件路径, e.g., ../benchs/GPQA_Medical_test.jsonl
    # args.eval_benchmark 仅用于元数据和文件夹命名, e.g., GPQA_Medical_test
    input_data = load_file(args.eval_dataset, args.eval_benchmark)
    ###这里的prompt是通用模型的设置，对于有自定义prompt的模型，优先采用原作者提供的prompt
    final_results = []
    if 'medxpert' in args.eval_benchmark.lower():
        # 对于 medxpert 数据集，选项已经包含在 question 中，不需要 option_str
        print("检测到 'medxpert' benchmark，使用不含 options 的 prompt 模板。")
        query_prompt = "Please answer the following medical multiple-choice question by analyzing the problem. Enclose the letter of the correct option in \\boxed{{}}. Ensure the box contains ONLY the single letter, without the option text.\n{question}"
    else:
        # 对于其他数据集，使用标准的 prompt 模板
        print("使用标准 prompt 模板（包含 question 和 options）。")
        query_prompt = "Please answer the following medical multiple-choice question by analyzing the problem. Enclose the letter of the correct option in \\boxed{{}}. Ensure the box contains ONLY the single letter, without the option text.\n{question}\n{option_str}"
    for idx in tqdm(range(0, len(input_data), args.batch_size)):
        batch = input_data[idx : min(idx + args.batch_size, len(input_data))]
        if len(batch) == 0:
            break

        for item in batch:
            if "options" in item:
                item['option_str'] = '\n'.join([ f'{op}. {ans}' for op,ans in item['options'].items()])
            else:
                item['option_str'] = ""
            item["input_str"] = query_prompt.format_map(item)

        processed_batch = [ item["input_str"] for item in batch]
    
        if idx == 0:
            print_example = True
        else:
            print_example = False
        
        preds, _ = call_model(
            processed_batch, model=model, max_new_tokens=args.max_new_tokens, temperature=args.temperature, print_example=print_example)

        for j, item in enumerate(batch):
            pred = preds[j]
            if len(pred) == 0:
                continue
            item["output"] = pred
            final_results.append(item)

    model_name = model.split('/')[-1]

    # 【修改点】: 移除了 reasoning/non-reasoning 子文件夹
    task_floder = f'./results/{model_name}_port{args.port}/{args.eval_benchmark}'

    os.makedirs(task_floder, exist_ok=True)

    result_file = os.path.join(task_floder, 'result.json')
    # final_results = [item for item in final_results if item['answer_idx'] != None]
    with open(result_file, 'w') as fw:
        json.dump(final_results, fw, ensure_ascii=False, indent=2)
    # get results
    get_results(result_file)

if __name__ == "__main__":
    main()