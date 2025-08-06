import torch


def ppo_loss(agent, mb_obs, mb_logprobs, mb_actions, mb_values, mb_returns, mb_advantages, mb_advantages2, norm_adv,
             clip_coef, ent_coef, vf_coef, clip_vloss):
    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(mb_obs, mb_actions.long())
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    # if norm_adv:
    # if norm_adv and mb_advantages.shape[0] > 1:
    mb_advantages = (mb_advantages - mb_advantages2.mean()) / (mb_advantages2.std() + 1e-8)

    # Policy loss
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1)
    # if clip_vloss:
    #     v_loss_unclipped = (newvalue - mb_returns) ** 2
    #     v_clipped = mb_values + torch.clamp(newvalue - mb_values, -clip_coef, clip_coef, )
    #     v_loss_clipped = (v_clipped - mb_returns) ** 2
    #     v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
    #     v_loss = 0.5 * v_loss_max.mean()
    # else:
    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

    return loss, pg_loss, v_loss, entropy_loss, logratio, ratio


def ppo_gae(agent, next_done, next_obs, rewards, dones, values, gamma, gae_lambda, device, num_steps):
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards, device=device)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = ~next_done
                nextvalues = next_value
            else:
                nextnonterminal = ~dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
    return advantages, returns
