"""固定窗口贪心评估：衡量 RL 真实学习效果。

训练过程中的 mean_reward 不是有效的学习曲线——每个 RL step 用不同的
start_idx（不同市场窗口），奖励被大盘 beta 主导，跨 step 不可比。
本模块改为在一组【固定】的 held-out（VAL）窗口上以【贪心】（argmax，
确定性）方式跑策略，因此得到的指标跨训练步直接可比，构成真实学习曲线。

复用 backtest_rl 的 run_one_episode / compute_episode_metrics，保证评估口径
与最终回测完全一致。
"""
import numpy as np
import torch

import config
from backtest_rl import (run_one_episode, compute_episode_metrics,
                         load_benchmark_close_map)


def build_eval_windows(env, start_date, end_date, seq_len, episode_len,
                       max_windows):
    """在 [start_date, end_date] 内取非重叠的 EPISODE_LEN 窗口，
    均匀下采样到至多 max_windows 个。返回固定的 start_idx 列表。"""
    cand = [i for i, d in enumerate(env.dates)
            if start_date <= d <= end_date and i >= seq_len
            and i + episode_len - 1 < len(env.dates)]
    strided = cand[::episode_len]  # 非重叠步进
    if max_windows > 0 and len(strided) > max_windows:
        sel = np.linspace(0, len(strided) - 1, max_windows).astype(int)
        strided = [strided[k] for k in sel]
    return strided


@torch.no_grad()
def evaluate_greedy(encoder, policy, env, obs_cache, device, eval_starts,
                    close_map=None):
    """在固定窗口上以贪心(argmax)方式评估策略，返回可跨训练步比较的指标。

    切到 eval 模式（denoiser 确定性零噪声、dropout 关闭），跑完后恢复原模式。
    复用 backtest_rl.run_one_episode / compute_episode_metrics 保证口径一致。
    """
    if not eval_starts:
        return None

    enc_was_training = encoder.training
    pol_was_training = policy.training
    encoder.eval()
    policy.eval()

    bins = torch.tensor(config.BINS, device=device, dtype=torch.float32)
    saved_state = env.clone()  # 评估会改写 env，先存档稍后还原

    episodes = []
    try:
        for start_idx in eval_starts:
            ep = run_one_episode(encoder, policy, env, obs_cache, device,
                                 start_idx, bins)
            episodes.append(ep)
    finally:
        # 还原 env 训练态，避免污染后续 RL 采样
        env.__dict__.update(saved_state.__dict__)
        if enc_was_training:
            encoder.train()
        if pol_was_training:
            policy.train()

    metrics, r, b, x, mdd = compute_episode_metrics(episodes, close_map)
    return metrics

