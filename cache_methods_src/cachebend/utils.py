# Adapted from https://github.com/YaoJiayi/CacheBlend/blob/main/example/utils.py
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoConfig, AutoModelForCausalLM
import json
import torch
import collections
import string
import re
from typing import Any
from transformers import AutoTokenizer
import gc
import subprocess
from typing import Dict
# rouge_score and vllm are imported lazily inside the functions that need them
# (compute_rl, load_model_vllm) — they are heavy optional deps.
import time
class Timer:
    def __init__(self,prefix,stats:Dict=None,verbose=False):
        self.prefix = prefix
        self.verbose = verbose
        self.stats = stats if stats is not None else {}
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self,exc_type, exc_value, traceback):
        end = time.perf_counter()
        #if self.print: logger.info(f"{self.prefix} Time taken: {end - self.start:.6f} seconds")
        #else: torch.npu.synchronize()
        self.stats[self.prefix] = self.stats.get(self.prefix,0.0)+ (end-self.start)
        if self.verbose:print(f"{self.prefix} -> {end-self.start}")


def parse_npu_process_info(npu_id, chip_id):
    try:
        output = subprocess.check_output(['npu-smi', 'info'], text=True)
        pattern = re.compile( r'\|\s*(\d+)\s+(\d+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|')

        # Loop through each line and apply regex
        for line in output.splitlines():
            match = pattern.match(line)
            if match:
                npu, chip = int(match.group(1)), int(match.group(2))
                if npu == npu_id and chip == chip_id:
                    process_id = match.group(3).strip()
                    process_name = match.group(4).strip()
                    mem_mb = match.group(5).strip()
                    return {
                            "npu": npu,
                            "chip": chip,
                            "pid": process_id,
                            "name": process_name,
                            "memory_mb": int(mem_mb)
                            }

        return None  # Not found
    except subprocess.CalledProcessError as e:
        print("Error running npu-smi:", e)
        return None

def get_mem(npu_id, chip_id=0):
    try:
        return parse_npu_process_info(npu_id,chip_id)['memory_mb']
    except:
        return -1

def report_npu_tensors2():
    import gc
    import torch

    counter = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == 'npu':
                key = (tuple(obj.shape), str(obj.dtype))
                counter[key] = counter.get(key, 0) + 1
        except Exception:
            pass

    print("NPU tensor summary:")
    total_bytes = 0
    for (shape, dtype), count in counter.items():
        try:
            # Get element size
            element_size = torch.empty((), dtype=getattr(torch, dtype.split('.')[1])).element_size()
            # Total number of elements in one tensor
            numel = torch.tensor(shape).prod().item()
            # Total size in bytes
            size_bytes = count * numel * element_size
            size_mb = size_bytes / 1e6
            total_bytes += size_bytes
            print(f"{count:5d} tensor(s) of shape={shape}, dtype={dtype} → {size_mb:.2f} MB total")
        except Exception:
            print(f"{count:5d} tensor(s) of shape={shape}, dtype={dtype} → [Size calc failed]")

    print(f"~{total_bytes / 1e6:.2f} MB total on NPU")

def report_npu_tensors():
    counter = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == 'npu':
                key = (tuple(obj.shape), str(obj.dtype))
                counter[key] = counter.get(key, 0) + 1
        except Exception:
            pass
    print("NPU tensor summary:")
    for (shape, dtype), count in counter.items():
        print(f"{count:5d} tensor(s) of shape={shape}, dtype={dtype}")
    total_mb = sum(torch.empty(shape, dtype=getattr(torch, dtype.split('.')[1])).npu().element_size() *
                   torch.tensor(shape).prod().item() * count
                   for (shape, dtype), count in counter.items()) / 1e6
    print(f"~{total_mb:.2f} MB total on NPU")

def docs_to_ids(docs: list[str], tokenizer: AutoTokenizer) -> list[int]:
    res = []
    if tokenizer.bos_token_id is not None:
        res.append(tokenizer.bos_token_id)

    for doc in docs:
        # Tokenize and convert to input IDs
        input_ids = tokenizer.encode(doc, add_special_tokens=False)
        res.extend(input_ids)

    return res

def normalize_question(question: str):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]

def parse_generation(s: str):
    s = s.lstrip('\n').split('\n')[0]
    if s.startswith("Yes") or s.startswith("yes"):
        s = "Yes"
    elif (s.split()[0]).startswith("No") or (s.split()[0]).startswith("no"):
        s = "No"
    return s

def normalize_answer(s: str):
    def remove_articles(text: str):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str):
        return " ".join(text.split())

    def remove_punc(text: str):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def build_qa_prompt(example: dict[str, Any], query_prompt: str):
    """
    example['question']: str
    example['ctxs']: list[str] the documents
    """
    q = normalize_question(example["question"])
    doc_prompts = [f"{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"]]
    q_prompt = f"{query_prompt}{q}\nAnswer:"
    return doc_prompts, q_prompt

def build_fewshot_prompt(example):
    q = "\n\n"+example["question"]
    doc_prompts = [f"{ctx['text']}" for ctx in example["ctxs"]]
    q_prompt = f"{q}"
    return doc_prompts, q_prompt

def compute_f1(a_pred: str, a_gold: str|list[str]|list[list[str]], tokenizer: PreTrainedTokenizerBase):
    if not isinstance(a_gold, str):
        res = 0.0
        for a in a_gold:
            res = max(res, compute_f1(a_pred, a, tokenizer))
        return res
    a_pred = parse_generation(a_pred)
    gold_toks = tokenizer.encode(normalize_answer(a_gold), add_special_tokens=False)
    pred_toks = tokenizer.encode(normalize_answer(a_pred), add_special_tokens=False)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_rl(pred, gold):
    from rouge_score import rouge_scorer
    # Initialize scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # CASE 1: Gold is a list of answers (TriviaQA style)
    if isinstance(gold, list):
        best_score = 0.0
        for answer in gold:
            score = scorer.score(answer, pred)['rougeL'].fmeasure
            if score > best_score:
                best_score = score
        return best_score

    # CASE 2: Gold is a single string
    return scorer.score(gold, pred)['rougeL'].fmeasure

def compute_llm_judge_binary(pred: str,
    gold: str,
    question: str,
    llm=None,
    tokenizer=None,
    system_prompt: str = "",
    device: str = "npu:0",
    model_path: str = "/data/weights/llama3.1-8B"
    ) -> int : 
    """
    A simple implementation of llm as a judge to evaluate the prediction of an llm against
    the ground truth answer,  currently designed for qa tasks.
    It assumes there is already an llm object and the tokenizer you can pass to it
    if its none it creates it anew using our llama 8 B at /data/weights/llama3.1-8BI
    The judge will be provided with a system prompt explainin its role and that it will be provided with,
    in order, the question of the qua task, the ground truth answer and the anser provided by the llm, it has to output a single token,
    either 0 or 1, 1 if the answer of the llm is conidered correct, in the sense that he answer correctly and didn't just guessed by repeating
    part of the answers. It willa also be provided with a few in context examples, like from the following:
    
        Question: What was the name of the Football Association Challenge Cup in 1894-95?
    Answer:   FA Cup
    Methods       | F1     | RL     | Prediction
    ----------------------------------------------------------------------
    baseline     | 0.000  | 0.000  | There is no mention.
    cacheblend   | 0.330  | 0.290  | The Football Association Challenge Cup.
    zcf          | 0.330  | 0.290  | The Football Association Challenge Cup.
    cachecraft   | 0.000  | 0.000  | There is no mention.
    
    here some methods are just getting better scores by repeating part of the question
    
    
        --- Example 164 ---
    Question: What are the official languages of East Timor?
    Answer:   Portuguese and Tetum
    Method       | F1     | RL     | Prediction
    ----------------------------------------------------------------------
    baseline     | 0.000  | 0.000  | No information about East Timor.
    cacheblend   | 0.920  | 0.860  | Portuguese and Tetum languages.
    zcf          | 0.670  | 0.330  | Tetum and Portuguese.
    cachecraft   | 0.000  | 0.000  | No information about East Timor.
    
    This is a case of genuine good answer, not just repetition
    
    
    :param pred: the string outputted by the llm being evaluated
    :param gold: the string ground truth answer from the dataset
    :param question: a string question
    :param llm: the llm object to act as a judge
    :param tokenizer: the tokenizer object
    :return: 1 if the judge cnosider the preciction semantically correct given the gold, 0 otherwise
    """
    
    if (llm is None) or (tokenizer is None):
        
        llm = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
        llm.eval()
        llm.to(device)

        # Reconstruct your build_tokenizer logic inline
        special_tokens_dict = {"sep_token": '<DSEP>'}
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        num_added = tokenizer.add_special_tokens(special_tokens_dict)
        assert num_added, "Failed to add <DSEP>"

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
            # No resize → safe per your comment

        if tokenizer.bos_token is None:
            tokenizer.bos_token = ' '
            tokenizer.bos_token_id = tokenizer.encode(tokenizer.bos_token)[0]

    
    if not system_prompt or system_prompt.strip() == "":
        system_prompt = (
            "You are an expert judge evaluating answers in a question-answering task. "
            "You will be given a QUESTION, a GROUND TRUTH ANSWER, and a PREDICTED ANSWER. "
            "Output '1' if the predicted answer is factually correct and demonstrates understanding. "
            "Output '0' if it is incorrect, vague, uninformative, or merely repeats phrases from the question. "
            "Respond with only a single token: '0' or '1'."
        )

    # In-context examples 
    examples = """
    Example 1:
    Question: What was the name of the Football Association Challenge Cup in 1894-95?
    Ground Truth Answer: FA Cup
    Predicted Answer: The Football Association Challenge Cup.
    Judgment: 0

    Example 2:
    Question: What are the official languages of East Timor?
    Ground Truth Answer: Portuguese and Tetum
    Predicted Answer: Tetum and Portuguese.
    Judgment: 1

    Example 3:
    Question: Who discovered penicillin?
    Ground Truth Answer: Alexander Fleming
    Predicted Answer: Alexander Fleming.
    Judgment: 1

    Example 4:
    Question: What is the capital of Canada?
    Ground Truth Answer: Ottawa
    Predicted Answer: The capital of Canada is Ottawa.
    Judgment: 1

    Example 5:
    Question: What is the capital of Canada?
    Ground Truth Answer: Ottawa
    Predicted Answer: The capital city of Canada.
    Judgment: 0

    Example 6:
    Question: When did the Berlin Wall fall?
    Ground Truth Answer: 1989
    Predicted Answer: 1989.
    Judgment: 1

    Example 7:
    Question: How many planets are in the Solar System?
    Ground Truth Answer: 8
    Predicted Answer: There are many planets.
    Judgment: 0

    Example 8:
    Question: What is the main ingredient in guacamole?
    Ground Truth Answer: Avocado
    Predicted Answer: Avocados.
    Judgment: 1
    """

    # Build prompt
    prompt = (
        f"{system_prompt}\n\n"
        f"{examples}\n"
        f"Question: {question}\n"
        f"Ground Truth Answer: {gold}\n"
        f"Predicted Answer: {pred}\n"
        f"Judgment:"
    )

    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate a single token 
    #deterministic
    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    
    gen_token_id = outputs[0, -1].item()
    gen_text = tokenizer.decode([gen_token_id], skip_special_tokens=True).strip()

    
    if gen_text == "1":
        return 1
    elif gen_text == "0":
        return 0
    else:
        #for now conservative, everything that is not good is 
        #considered bad
        return 0  
    
def compute_llm_judge_binary1(
    pred: str,
    gold: str,
    question: str,
    llm=None,
    tokenizer=None,
    system_prompt: str = "",
    device: str = "npu:0",
    model_path: str = "/data/weights/llama3.1-8B"
) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if llm is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Add your special tokens (though likely unused in judge prompt)
        tokenizer.add_special_tokens({"sep_token": "<DSEP>"})
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        llm = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(device).eval()

    if not system_prompt.strip():
        system_prompt = (
            "You are an expert judge for QA tasks. Given a question, ground truth answer, and predicted answer, "
            "output ONLY '1' if the prediction is factually correct and shows understanding, or '0' if it is wrong, "
            "vague, or just repeats the question. Output exactly one character: 0 or 1."
        )

    examples = """  Example 1:
                    Question: What was the name of the Football Association Challenge Cup in 1894-95?
                    Ground Truth Answer: FA Cup
                    Predicted Answer: The Football Association Challenge Cup.
                    Judgment: 0

                    Example 2:
                    Question: What are the official languages of East Timor?
                    Ground Truth Answer: Portuguese and Tetum
                    Predicted Answer: Tetum and Portuguese.
                    Judgment: 1"""

    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{examples}\n\nQuestion: {question}\nGround Truth Answer: {gold}\nPredicted Answer: {pred}\nJudgment:"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    gen_token_id = outputs[0, -1].item()
    gen_text = tokenizer.decode([gen_token_id], skip_special_tokens=True).strip()

    

    return 1 if gen_text == "1" else 0
        


metric_name2f = {
    'f1': compute_f1,
    'rl': compute_rl
}

def load_model(model_path: str, dtype=torch.bfloat16, device='npu'):
    config = AutoConfig.from_pretrained(model_path)
    config.sliding_window = None
    llm: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, config=config, attn_implementation="eager")
    llm.to(dtype)
    llm.to(device)
    llm.eval()
    return llm

def load_model_vllm(model_path: str, dtype="bfloat16", device="npu"):
    from vllm import LLM
    llm = LLM(
        model=model_path,
        dtype=dtype,
        device=device,
        trust_remote_code=True,
        tensor_parallel_size=2,
        max_model_len=10000,
        max_num_seqs=1
    )
    return llm