"""
SFT vs RL 极简对比实验（纯标准库，无需安装任何依赖）
====================================================
任务设定：一个 prompt 有 4 个候选回答 A/B/C/D。
- "真实好坏"(reward)：D 最好，C 次之，B/A 较差。
- 但"人类老师"的示范里几乎只示范稳妥的 C，很少给出最优的 D。

我们用同一个超简单的"策略"(4 个 logits + softmax) 分别用两种方式训练：
  1) SFT  ：最大似然，模仿老师的示范分布（行为克隆 / 模仿学习）
  2) RL   ：自己采样回答 -> 按 reward 调整（REINFORCE 策略梯度）
  3) SFT->RL：先 SFT 打底，再 RL 精修（这正是真实大模型的训练流水线）

看点：SFT 的天花板 = 老师示范的质量；RL 能靠"试错探索"超越老师。
"""
import random, math

random.seed(0)  # 固定随机种子，结果可复现

ACTIONS = ["A", "B", "C", "D"]

# 真实回报：D 最优（=1.0），但老师很少示范它
TRUE_REWARD = {"A": 0.1, "B": 0.3, "C": 0.6, "D": 1.0}

# 老师的示范分布（SFT 的训练数据）：偏爱"稳妥"的 C，几乎不示范最优的 D
TEACHER_DEMO = {"A": 0.05, "B": 0.15, "C": 0.75, "D": 0.05}


def softmax(logits):
    m = max(logits.values())
    exps = {a: math.exp(logits[a] - m) for a in ACTIONS}
    z = sum(exps.values())
    return {a: exps[a] / z for a in ACTIONS}


def avg_reward(probs):
    """该策略下的期望真实回报（衡量'效果'）"""
    return sum(probs[a] * TRUE_REWARD[a] for a in ACTIONS)


def show(tag, probs):
    dist = "  ".join(f"{a}:{probs[a]:.2f}" for a in ACTIONS)
    print(f"{tag:9s} 策略分布[ {dist} ]  ->  平均回报 = {avg_reward(probs):.3f}")


# ============ SFT：最大似然，把策略分布拉向老师示范分布 ============
def train_sft(steps=3000, lr=0.5):
    logits = {a: 0.0 for a in ACTIONS}
    for _ in range(steps):
        p = softmax(logits)
        # 交叉熵损失对 logit 的梯度 = p_a - q_a (q 是老师示范分布)
        for a in ACTIONS:
            logits[a] -= lr * (p[a] - TEACHER_DEMO[a])
    return logits


# ============ RL：自己生成回答，按 reward 高低调整概率 (REINFORCE) ============
def train_rl(init_logits=None, steps=5000, lr=0.2):
    logits = dict(init_logits) if init_logits else {a: 0.0 for a in ACTIONS}
    baseline = 0.0  # 滑动平均基线，降低梯度方差
    for _ in range(steps):
        p = softmax(logits)
        # 1) 模型"自己"采样一个回答（探索）
        r, cum, choice = random.random(), 0.0, ACTIONS[-1]
        for a in ACTIONS:
            cum += p[a]
            if r <= cum:
                choice = a
                break
        # 2) 环境给出 reward
        reward = TRUE_REWARD[choice]
        baseline += 0.01 * (reward - baseline)
        adv = reward - baseline  # 优势：比平均好就加强，比平均差就削弱
        # 3) 策略梯度更新：好回答提高概率，差回答降低概率
        for a in ACTIONS:
            indicator = 1.0 if a == choice else 0.0
            logits[a] += lr * adv * (indicator - p[a])
    return logits


if __name__ == "__main__":
    print("真实回报 :", TRUE_REWARD, "   <- D 最优")
    print("老师示范 :", TEACHER_DEMO, "   <- 几乎不示范 D")
    print(f"(老师本人的平均水平 = {avg_reward(TEACHER_DEMO):.3f})\n")

    sft = train_sft()
    show("仅 SFT", softmax(sft))           # 学成了"老师的复制品"

    rl = train_rl()
    show("仅 RL", softmax(rl))             # 靠试错发现了最优 D

    rl_after_sft = train_rl(init_logits=sft)
    show("SFT->RL", softmax(rl_after_sft)) # 先模仿打底，再优化超越

    print("\n结论：SFT 的上限被老师示范锁死(~0.55)；RL 通过探索发现了老师没怎么教的最优解 D。")
