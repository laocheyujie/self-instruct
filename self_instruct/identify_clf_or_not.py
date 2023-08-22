import os
import json
import random
import tqdm
import re
import argparse
import pandas as pd
from collections import OrderedDict
from gpt3_api import make_requests as make_gpt3_requests
from templates.clf_task_template import template_1


random.seed(42)


templates = {
    "template_1": template_1
}

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--template", type=str, default="template_1", help="Which template to use.")
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        help="if specified, only generate instance input for this many instructions",
    )
    parser.add_argument(
        "--template", 
        type=str, 
        default="template_1", 
        help="Which template to use. Currently only `template_1` is supported.",
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
        help="The API key to use. If not specified, the key will be read from the environment variable `OPENAI_API_KEY`."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")) as fin:
        lines = fin.readlines()
        if args.num_instructions is not None:
            lines = lines[:args.num_instructions]

    output_path = os.path.join(args.batch_dir, f"is_clf_or_not_{args.engine}_{args.template}.jsonl")
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")

    progress_bar = tqdm.tqdm(total=len(lines))
    # 执行输出过程
    # 使用文件操作打开一个输出文件，然后把文件对象赋值给fout
    with open(output_path, "w") as fout:
        # 迭代输入的数据行，步长为request_batch_size
        for batch_idx in range(0, len(lines), args.request_batch_size):
            # 对每个批次，将批次中的数据行转换为JSON对象
            batch = [json.loads(line) for line in lines[batch_idx: batch_idx + args.request_batch_size]]
            # 检查批次中的所有指令是否都在已存在的请求中
            if all(d["instruction"] in existing_requests for d in batch):
                # 如果都在，则直接从已存在的请求中获取数据，并写入到输出文件中
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "is_classification"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                # 如果不都在，那么需要使用GPT-3引擎生成数据
                # 首先构造一个提示，这个提示包含前缀和指令
                # prefix = compose_prompt_prefix(human_written_tasks, batch[0]["instruction"], 8, 2)
                prefix = templates[args.template]
                prompts = [prefix + " " + d["instruction"].strip() + "\n" + "Is it classification?" for d in batch]
                # 调用函数使用GPT-3引擎对批处理的输入数据进行处理
                # 处理的参数包括最大的输出词汇数量、输出的随机性、输出结果的顶部概率等
                results = make_gpt3_requests(
                    engine=args.engine,
                    prompts=prompts,
                    max_tokens=3,
                    temperature=0,
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop_sequences=["\n", "Task"],
                    logprobs=1,
                    n=1,
                    best_of=1,
                    api_key=args.api_key,
                    organization=args.organization)
                # 将结果写入到输出文件中
                for i in range(len(batch)):
                    data = batch[i]
                    # 如果结果存在，则将结果中的"is_classification"字段保存到数据中
                    if results[i]["response"] is not None:
                        data["is_classification"] = results[i]["response"]["choices"][0]["text"]
                    else:
                        # 如果结果不存在，则将"is_classification"字段设置为空
                        data["is_classification"] = ""
                    # 构造一个字典，包含指令和"is_classification"字段
                    data = {
                        "instruction": data["instruction"],
                        "is_classification": data["is_classification"]
                    }
                    # 对字典进行排序，然后将字典转换为JSON字符串，并写入到输出文件中
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "is_classification"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            progress_bar.update(len(batch))
