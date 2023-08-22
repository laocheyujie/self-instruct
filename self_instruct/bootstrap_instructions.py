import os
import json
import random
import re
import string
import tqdm
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer
from gpt3_api import make_requests as make_gpt3_requests


random.seed(42)

# 构建prompt数据，针对是否分类分别构建不同的prompt数据
# 定义一个函数，该函数用于将多个提示指令编码成一个字符串
# 该函数接受两个参数，第一个参数是提示指令列表，第二个参数表示是否是分类任务，是=>输出优先，否=>输入优先，对应的 prompt_instructions/prompt_instances 不一样
def encode_prompt(prompt_instructions, classification=False):
    """Encode multiple prompt instructions into a single string."""
    # 如果当前任务是分类任务，那么设置提示信息为一个固定的字符串
    if classification:
        # 这个提示信息是引导用户生成一系列的分类任务，如果可能的话，要求用户明确指定可能的输出标签
        # prompt = "Come up with a series of classification tasks. Try to specify the possible output labels when possible.\n"
        prompt = "Referring to a series of classification tasks, generate 8 more new tasks. Try to specify the possible output labels when possible.\n"
    # 如果当前任务不是分类任务，那么设置提示信息为另一个固定的字符串
    else:
        # 这个提示信息是引导用户生成一系列的任务
        # prompt = "Come up with a series of tasks:\n"
        prompt = "Referring to these eight tasks, generate 8 more new tasks:\n"
    # 循环处理每一条提示指令
    for idx, instruction in enumerate(prompt_instructions):
        # 使用正则表达式将指令中的多余空格替换为单个空格，并去掉前后的空格以及末尾的冒号
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        # 将处理后的指令添加到提示信息中，注意指令前面需要添加序号
        prompt += f"{idx+1}. {instruction}\n"
    # 在所有指令之后添加一个空白的序号，这个序号是接下来用户需要填写的新任务的序号
    prompt += f"{len(prompt_instructions) + 1}."
    return prompt


def sample_machine_instructions(machine_instructions, similarities, n):
    """Sample n machine instructions from a list of machine instructions."""
    return random.sample(machine_instructions, min(n, len(machine_instructions)))


def find_word_in_string(w, s):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)


def post_process_gpt3_response(response):
    if response is None or response["choices"][0]["finish_reason"] == "length":
        return []
    raw_instructions = re.split(r"\n\d+\s?\. ", response["choices"][0]["text"])
    instructions = []
    for inst in raw_instructions:
        inst = re.sub(r"\s+", " ", inst).strip()
        inst = inst.strip().capitalize()
        if inst == "":
            continue
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        if any(find_word_in_string(word, inst) for word in ["image", "images", "graph", "graphs", "picture", "pictures", "file", "files", "map", "maps", "draw", "plot", "go to"]):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result. 
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append(inst)
    return instructions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        default="data/gpt3_generations/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--num_instructions_to_generate",
        type=int,
        default=100,
        help="th",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci",
        help="The engine to use."
    )
    parser.add_argument(
        "--num_prompt_instructions",
        type=int,
        default=8,
        help="The number of instructions to use in the prompt."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send to GPT3 at a time."
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


if __name__ == "__main__":
    args = parse_args()
    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    if args.use_clf_seed_tasks_only:
        seed_tasks = [t for t in seed_tasks if t["is_classification"]]
    seed_instructions = [t["instruction"] for t in seed_tasks]
    print(f"Loaded {len(seed_instructions)} human-written seed instructions")
    
    os.makedirs(args.batch_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions，使用生成模型得到新的100条 instruction 提示
    machine_instructions = []
    # 开始生成 100 条 instruction 提示数据
    if os.path.exists(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")):
        with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info["instruction"])
                request_idx = instruction_info["request_idx"] + 1
        print(f"Loaded {len(machine_instructions)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    
    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    if machine_instructions:
        progress_bar.update(len(machine_instructions))

    # 开始生成100条instruction提示数据
    # 使用文件操作打开一个文件，该文件位于指定的批处理目录中
    # 文件名为"machine_generated_instructions.jsonl"，以追加模式打开，然后把文件对象赋值给fout
    with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "a") as fout:
        # 进入循环，当生成模型产生的指令数量未达到用户指定的数量时，继续产生新的指令
        while len(machine_instructions) < args.num_instructions_to_generate:
        # 初始化一个列表，用于保存批处理的输入数据
            batch_inputs = []
            # args.request_batch_size 为 5
            # 循环指定的批处理大小的次数，每次循环都会产生一条新的指令
            for _ in range(args.request_batch_size):
                # sample machine instructions from the pool（从生成模型中选，n表示最少的条数，这里为2）
                # 调用函数从生成模型中抽样生成指令，这里选择的指令数量为2，然后将生成的指令保存到变量prompt_instructions
                prompt_instructions = sample_machine_instructions(
                    machine_instructions, 
                    similarities=None,
                    n=2)
                # sample human instructions from the pool
                # 从默认的175条中选再选几条，相当于一共选了8条，其中从175条中选6条，使用LLM生成2条
                # 最开始的时候，machine_instructions 为空，因此会直接从175条中直接选8条
                prompt_instructions += random.sample(seed_instructions, args.num_prompt_instructions - len(prompt_instructions))
                random.shuffle(prompt_instructions)
                # 将这8条指令编码成模型可以接收的输入格式，然后保存到变量prompt
                prompt = encode_prompt(prompt_instructions, classification=args.use_clf_seed_tasks_only)
                # 将编码后的输入添加到批处理的输入数据列表中
                batch_inputs.append(prompt)
                
            # 调用函数使用GPT-3引擎对批处理的输入数据进行处理，处理的参数包括最大的输出词汇数量、输出的随机性、输出结果的顶部概率等
            results = make_gpt3_requests(
                engine=args.engine,
                prompts=batch_inputs,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=2,
                stop_sequences=["\n\n", "\n16", "16.", "16 ."],
                logprobs=1,
                n=1,
                best_of=1,
                api_key=args.api_key,
                organization=args.organization,
            )
            instructions = []
            all_metadata = []
            for result in results:
                new_instructions = post_process_gpt3_response(result["response"])
                instructions += new_instructions
                all_metadata += [result] * len(new_instructions)

            for inst, metadata in zip(instructions, all_metadata):
                with Pool(4) as p:
                    rouge_scores = p.map(partial(scorer.score, inst), seed_instructions + machine_instructions)
                rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]
                # rouge_scores = [scorer.score(inst, e_inst)["rougeL"].fmeasure for e_inst in human_instructions + machine_instructions]
                if max(rouge_scores) > 0.7:
                    continue
                all_instructions = seed_instructions + machine_instructions
                most_similar_instructions = {
                        all_instructions[i] : rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                    }
                machine_instructions.append(inst)
                fout.write(json.dumps({
                    "instruction": inst,
                    "most_similar": most_similar_instructions,
                    "avg_similarity_score": float(np.mean(rouge_scores)),
                    "metadata": metadata,
                    "request_idx": request_idx
                }) + "\n")
                progress_bar.update(1)
            request_idx += 1
