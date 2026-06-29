"""
例子二 - A：用真实的 Qwen 模型做 SFT（监督微调 / 模仿学习）
=============================================================
目标：让小白亲眼看到 "SFT = 模仿示范"。

我们给模型看一批"算术题 -> 标准解法"的示范（老师写好的答案），
用监督学习（next-token 预测 + 交叉熵）让模型去"模仿"这种解题风格。
训练完后，模型回答的【风格】会从原始的简短作答，变成示范里那种
"我们来计算 … 所以最终答案是 \boxed{…}" 的样子。

模型：Qwen/Qwen2.5-0.5B-Instruct（0.5B 很小，Mac 也能跑）
技术：LoRA 轻量微调（只训练极少量参数，省显存/内存）
设备：自动选择 cuda / mps(苹果GPU) / cpu

>>> 重要经验（本仓库实测得到的关键知识）<<<
小模型 + 高度重复的数据，如果"训练过头"(loss 压到很低)，会发生
"模型塌缩"——它把训练样本背下来了，但自由生成时输出乱码。
所以 SFT 不是练得越久越好：要练到刚好学会风格(本例 loss≈0.5~0.6)即可。
这本身就是 SFT 的一个真实坑（见 README 的"常见坑"）。

运行：
    pip install -r requirements.txt
    python qwen_sft.py
首次运行会自动从 HuggingFace 下载模型（约 1GB）。
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # MPS 不支持的算子自动回退 CPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import random
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = "你是一个数学小助手。"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# 老师示范的"标准解法"风格 —— SFT 就是要让模型模仿它
def make_solution(a, b, op, ans):
    return f"我们来计算 {a} {op} {b}。{a} {op} {b} = {ans}，所以最终答案是 \\boxed{{{ans}}}。"


# ---------- 1) 造一份迷你"教材"：题目(prompt) -> 标准解法(completion) ----------
# 用 prompt-completion 格式：TRL 默认只对 completion(答案) 部分计算 loss，
# 不会训练到 system/user 等模板 token（这样更稳，不易把模型带崩）。
def build_dataset(n=200, seed=0):
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
            "completion": [
                {"role": "assistant", "content": make_solution(a, b, op, ans)},
            ],
        })
    return Dataset.from_list(rows)


# ---------- 工具：让当前模型回答一道题（对比训练前 vs 训练后）----------
@torch.no_grad()
def ask(model, tok, question):
    model.eval()
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(next(model.parameters()).device)
    out = model.generate(**ids, max_new_tokens=80, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    device = pick_device()
    print(f"设备 = {device}\n加载模型 {MODEL_NAME} ...")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.to(device)

    test_q, new_q = "计算 37 + 48", "计算 88 - 19"
    print("\n========== 训练【前】（原始 Qwen：简短作答，没有统一风格）==========")
    print(f"问：{test_q}\n答：{ask(model, tok, test_q)}")

    # ---------- 2) 配置 LoRA + SFT 训练 ----------
    # 经验配置：attention-only + alpha=r(缩放1.0) + 适中学习率/轮数，
    # 让 loss 落在 ~0.5~0.6（学会风格但不塌缩）。
    lora = LoraConfig(
        r=8, lora_alpha=8, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    args = SFTConfig(
        output_dir="out_qwen_sft",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=8e-5,
        logging_steps=10,
        max_length=128,
        gradient_checkpointing=False,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=build_dataset(),
        processing_class=tok,
        peft_config=lora,
    )
    print("\n开始 SFT（监督微调 / 模仿示范）...")
    trainer.train()

    print("\n========== 训练【后】（已模仿出示范的解题风格）==========")
    print(f"问：{test_q}\n答：{ask(trainer.model, tok, test_q)}")
    print(f"\n问：{new_q}（训练时没出现的新题，看风格是否迁移过去）\n答：{ask(trainer.model, tok, new_q)}")

    print("\n要点：SFT 让模型【模仿】了示范的解题风格 —— 这就是'监督微调'。")
    print("局限：水平上限被示范锁死；且练过头会塌缩。想'超越示范'见 RL 例子。")


if __name__ == "__main__":
    main()
