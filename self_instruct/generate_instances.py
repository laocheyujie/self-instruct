import os
import json
import random
import tqdm
import re
import argparse
import pandas as pd
from collections import OrderedDict
from gpt3_api import make_requests as make_gpt3_requests
from templates.instance_gen_template import output_first_template_for_clf, input_first_template_for_gen


random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="machine_generated_instructions.jsonl"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="machine_generated_instances.jsonl",
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        help="if specified, only generate instance input for this many instructions",
    )
    parser.add_argument(
        "--max_instances_to_generate",
        type=int,
        default=5,
        help="The max number of instances to generate for each instruction.",
    )
    parser.add_argument(
        "--generation_tasks_only",
        action="store_true",
        help="If specified, only do for generation tasks.",
    )
    parser.add_argument(
        "--classification_tasks_only",
        action="store_true",
        help="If specified, only do for classification tasks.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci",
        help="The engine to use."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send in a batch."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="The API key to use. If not specified, the key will be read from the environment variable OPENAI_API_KEY."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(os.path.join(args.batch_dir, args.input_file), encoding='utf-8') as fin:
        lines = fin.readlines()
        if args.num_instructions is not None:
            lines = lines[:args.num_instructions]
        tasks = []
        for line in lines:
            data = json.loads(line)
            if "metadata" in data:
                data["instruction_metadata"] = data["metadata"]
                del data["metadata"]
            tasks.append(data)

    task_clf_types = {}
    with open(os.path.join(args.batch_dir, "is_clf_or_not_davinci_template_1.jsonl"), encoding='utf-8') as fin:
        for line in fin:
            data = json.loads(line)
            task_clf_types[data["instruction"]] = data["is_classification"].strip() in ["Yes", "yes", "YES"]

    if args.classification_tasks_only:
        tasks = [task for task in tasks if task_clf_types[task["instruction"]]]
    
    if args.generation_tasks_only:
        tasks = [task for task in tasks if not task_clf_types[task["instruction"]]]

    output_path = os.path.join(args.batch_dir, args.output_file)
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path, encoding='utf-8') as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")

    progress_bar = tqdm.tqdm(total=len(tasks))
    # 使用文件操作打开一个输出文件，以utf-8的编码格式，然后把文件对象赋值给fout
    with open(output_path, "w", encoding='utf-8') as fout:
        # 迭代任务数据，步长为request_batch_size
        for batch_idx in range(0, len(tasks), args.request_batch_size):
            # 获取当前批次的任务
            batch = tasks[batch_idx: batch_idx + args.request_batch_size]
            # 检查批次中的所有指令是否都在已存在的请求中
            if all(d["instruction"] in existing_requests for d in batch):
                # 如果都在，则直接从已存在的请求中获取数据，并写入到输出文件中
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    # 只选择关键字段创建有序字典
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "raw_instances", "instance_metadata", "instruction_metadata", 
                            "most_similar", "avg_similarity_score"]
                        )
                    # 写入数据到输出文件
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                # 如果不都在，那么需要构建请求的prompts
                prompts = []
                for task in batch:
                    # 根据任务的类型，使用不同的模板构建prompt
                    if task_clf_types[task["instruction"]]:
                        prompt = output_first_template_for_clf + " " + task["instruction"].strip() + "\n"
                        prompts.append(prompt)
                    else:
                        prompt = input_first_template_for_gen + " " + task["instruction"].strip() + "\n"
                        prompts.append(prompt)
                # 使用GPT-3引擎发送请求
                results = make_gpt3_requests(
                    engine=args.engine,
                    prompts=prompts,
                    # because the clf template is longer, we need to decrease the max_tokens
                    # 根据任务类型调整最大token数
                    max_tokens=300 if any(task_clf_types[task["instruction"]] for task in batch) else 350,
                    temperature=0,
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=1.5,
                    stop_sequences=[f"Example {args.max_instances_to_generate + 1}", "Task:"],
                    logprobs=1,
                    n=1,
                    best_of=1,
                    api_key=args.api_key,
                    organization=args.organization)
                # 将结果写入到输出文件中
                for i in range(len(batch)):
                    data = batch[i]
                    # 保存请求的元数据
                    data["instance_metadata"] = results[i]
                    # 如果结果存在，则保存生成的实例
                    if results[i]["response"] is not None:
                        data["raw_instances"] = results[i]["response"]["choices"][0]["text"]
                    else:
                        # 如果结果不存在，则设置为空
                        data["raw_instances"] = ""
                    # 构建有序字典
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "raw_instances", "instance_metadata", "instruction_metadata", 
                            "most_similar", "avg_similarity_score"]
                        )
                    # 写入数据到输出文件
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            # 更新进度条
            progress_bar.update(len(batch))
