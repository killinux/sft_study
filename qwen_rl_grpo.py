"""
例子二 - B：用真实的 Qwen 模型做 RL（强化学习，GRPO 算法）
=============================================================
目标：让小白亲眼看到 "RL = 自己答 + 按奖励改进，能超越示范"。

和 SFT 不同，这里【不给标准答案】。我们只给一个"打分器"(reward)：
  - 答案数值正确         -> +1.0
  - 输出里有 \\boxed{} 格式 -> +0.2
模型自己生成多个答案，谁得分高就强化谁 —— 这正是 DeepSeek-R1 / o1 用的
"可验证奖励强化学习(RLVR)"的迷你版。用的算法是 GRPO（PPO 的轻量替代）。

模型：Qwen/Qwen2.5-0.5B-Instruct    技术：LoRA    设备：cuda/mps/cpu 自动

运行：
    pip install -r requirements.txt
    python qwen_rl_grpo.py
注意：RL 比 SFT 慢很多（要不断"生成→打分→更新"）。Mac 上建议先把
max_steps 调小（如 30）感受趋势；想看明显提升就多跑一些步或用 GPU。
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import random
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = "你是一个数学小助手。先简要推理，最后用 \\boxed{} 给出最终答案。"

# 从文本里抽取最后一个 \boxed{数字}
BOXED = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_answer(text):
    matches = list(BOXED.finditer(text))
    return int(matches[-1].group(1)) if matches else None


def _content(completion):
    # 对话格式下，completion 形如 [{"role":"assistant","content":"..."}]
    if isinstance(completion, list):
        return completion[-1]["content"]
    return completion


# ---------- 奖励函数（RL 的灵魂：不给答案，只给"分数"）----------
def reward_format(completions, **kwargs):
    """有 \\boxed{} 格式就加 0.2 分（鼓励规范输出）"""
    return [0.2 if BOXED.search(_content(c)) else 0.0 for c in completions]


def reward_correct(completions, answer, **kwargs):
    """算对了加 1.0 分（这才是真正想优化的目标）"""
    scores = []
    for c, gold in zip(completions, answer):
        pred = extract_answer(_content(c))
        scores.append(1.0 if (pred is not None and pred == gold) else 0.0)
    return scores


# ---------- 训练数据：只有题目(prompt) + 正确答案(answer)，没有解题过程 ----------
def build_dataset(n=256, seed=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        op = rng.choice(["+", "-"])
        ans = a + b if op == "+" else a - b
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"计算 {a} {op} {b}"},
            ],
            "answer": ans,   # 只给打分器用，不会喂给模型
        })
    return Dataset.from_list(rows)


# ---------- 评测：跑若干新题，看"做对率"和"格式合规率"----------
@torch.no_grad()
def evaluate(model, tok, n=20, seed=999):
    rng = random.Random(seed)
    correct = fmt = 0
    samples = []
    model.eval()
    dev = next(model.parameters()).device
    for i in range(n):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        op = rng.choice(["+", "-"])
        gold = a + b if op == "+" else a - b
        q = f"计算 {a} {op} {b}"
        text = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(dev)
        out = model.generate(**ids, max_new_tokens=160, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        resp = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        fmt += int(BOXED.search(resp) is not None)
        correct += int(extract_answer(resp) == gold)
        if i < 3:
            samples.append((q, gold, resp.replace("\n", " ")[:110]))
    print(f"  做对率 {correct}/{n} = {correct/n:.0%} | 含\\boxed格式 {fmt}/{n} = {fmt/n:.0%}")
    for q, gold, resp in samples:
        print(f"    Q:{q}(正确{gold}) -> {resp!r}")


def main():
    device = pick_device()
    print(f"设备 = {device}\n加载模型 {MODEL_NAME} ...")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.to(device)

    print("\n========== RL 训练【前】==========")
    evaluate(model, tok)

    lora = LoraConfig(
        r=8, lora_alpha=8, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    args = GRPOConfig(
        output_dir="out_qwen_grpo",
        learning_rate=1e-5,
        per_device_train_batch_size=16,   # 2 道题 × 每题 8 个候选 = 16
        gradient_accumulation_steps=1,
        num_generations=8,                # 每道题自己生成 8 个答案互相比较
        max_prompt_length=128,
        max_completion_length=160,
        temperature=0.9,                  # 采样温度，制造多样性以便"探索"
        max_steps=80,                     # Mac 上偏慢，想快可改 30；想效果好可改大
        beta=0.0,                         # KL 惩罚系数，0=不需要参考模型，最省内存
        gradient_checkpointing=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_format, reward_correct],  # 两个奖励相加
        args=args,
        train_dataset=build_dataset(),
        peft_config=lora,
    )
    print("\n开始 GRPO 强化学习（自己生成→按奖励改进）...")
    trainer.train()

    print("\n========== RL 训练【后】==========")
    evaluate(trainer.model, tok)

    print("\n要点：RL 不靠抄标准答案，而是靠'奖励'自己摸索更好的解法 ——")
    print("做对率/格式合规率应当上升。这就是 RLHF / RLVR(如 DeepSeek-R1) 的核心思想。")


if __name__ == "__main__":
    main()
