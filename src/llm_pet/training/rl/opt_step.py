import torch


import copy
from llm_pet.data_set.vocab import encode
from llm_pet.training.rl.answer_rollout import rollout
from llm_pet.training.rl.reward import compute_reward, reference_logps, KL_COEF


# Algorithm (the GRPO-style recipe discussed): for each question we sample a
# GROUP of answers, score each with compute_reward(), and use the GROUP MEAN as
# the baseline. The advantage of an answer is (its reward - group mean), z-scored
# by the group std so easy and hard questions contribute comparably. The policy
# loss is the standard REINFORCE term  -(advantage * sum logp(tokens)), plus a
# per-token KL penalty pulling the policy toward a FROZEN copy of the start
# model so we sharpen accuracy without wrecking fluency/format.
#
# This is deliberately a plain torch loop (not a pl.Trainer fit) because the RL
# objective isn't a per-batch supervised loss -- it needs sampling, scoring and
# a custom gradient. It reuses the model, tokenizer, reward and data helpers
# already defined above.


RL_GROUP_SIZE = 8  # answers sampled per question (group baseline)
RL_NUM_QUESTIONS = 400  # GSM8k questions to train on
RL_MAX_NEW_TOKENS = 128  # decode budget per answer (<= 128 position cap)
RL_LR = 1e-5  # small LR: we are refining, not retraining
RL_QUESTIONS_PER_LOG = 10


def rl_finetune(
    policy,
    qa_pairs,
    group_size=RL_GROUP_SIZE,
    lr=RL_LR,
    max_new_tokens=RL_MAX_NEW_TOKENS,
    kl_coef=KL_COEF,
):
    """Run REINFORCE-with-group-baseline RL fine-tuning over (question, answer)
    pairs. `policy` is updated in place."""
    device = policy.device
    policy.train()

    # frozen reference = a deep copy of the start model, used only for KL.
    ref_policy = copy.deepcopy(policy).to(device)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=lr)

    running_reward = []
    for qi, (question, answer) in enumerate(qa_pairs):
        ctx_ids = encode(question)[-policy.hparams.max_context :]
        if not ctx_ids:
            continue

        # ---- sample a group of answers, score each ----
        group = []
        rewards = []
        for _ in range(group_size):
            roll = rollout(policy, ctx_ids, max_new_tokens, sample=True)
            if roll["token_ids"].numel() == 0:
                continue
            r = compute_reward(roll["text"], answer)
            roll["reward"] = r
            group.append(roll)
            rewards.append(r)
        if len(group) < 2:
            continue  # need >=2 to form a baseline

        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        baseline = rewards_t.mean()
        std = rewards_t.std().clamp_min(1e-6)  # z-score within the group
        running_reward.append(float(baseline))

        # ---- accumulate policy-gradient + KL loss over the group ----
        opt.zero_grad()
        total_loss = 0.0
        for roll, r in zip(group, rewards):
            advantage = (r - baseline) / std  # scalar tensor

            logps = roll["logps"]  # (L,) differentiable
            mask = roll["mask"]

            # REINFORCE: push up logp of tokens in better-than-average answers.
            # advantage is detached (it is a return, not a function of params).
            pg = -(advantage.detach() * (logps * mask).sum())

            # KL(policy || reference) per token, estimated on the sampled tokens:
            #   E[logp_policy - logp_ref]; keeps us near the fluent start model.
            ref_lp = reference_logps(ref_policy, ctx_ids, roll["token_ids"]).to(device)
            L = min(logps.size(0), ref_lp.size(0))
            kl = ((logps[:L] - ref_lp[:L]) * mask[:L]).sum()

            total_loss = total_loss + pg + kl_coef * kl

        total_loss = total_loss / len(group)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if (qi + 1) % RL_QUESTIONS_PER_LOG == 0:
            window = running_reward[-RL_QUESTIONS_PER_LOG:]
            avg = sum(window) / len(window)
            print(
                f"[rl] question {qi + 1}/{len(qa_pairs)} "
                f"group_mean_reward(last {len(window)})={avg:.3f} "
                f"loss={float(total_loss):.3f}"
            )

    return policy
