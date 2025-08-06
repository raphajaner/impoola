import matplotlib.pyplot as plt
import numpy as np


def polynomial_scheduler(pruning_ratio_dict, steps, start_step=0, end_step=1, power=3, prune_steps=None):
    """
    Polynomial scheduler for pruning ratio.

    Args:
    - pruning_ratio_dict (Dict[nn.Module, float]): layer-specific pruning ratio.
    - steps (int): number of overall steps.
    - start_step (int): step to start pruning.
    - end_step (int): step when no pruning occurs anymore.
    - power (int): power of polynomial. Default: 3.
    - prune_steps (int): step to prune the model
    """
    # prefill the list with the start_step
    assert 0 <= start_step <= 1 and 0 <= end_step <= 1 and start_step < end_step, " Set correct start_step and end_step"
    assert steps > 0, "Steps should be greater than 0"

    start_step = int(start_step * steps)
    end_step = int(end_step * steps)
    schedule = [0] * start_step
    # calculate the polynomial
    prune_schedule = []
    for i in range(start_step, end_step + 1):
        prune_schedule.append(pruning_ratio_dict * (1 - (1 - ((i - start_step) / (end_step - start_step))) ** power))

    if prune_steps is not None:
        # min_val, max_val = min(prune_schedule[1:]), max(prune_schedule[:-1])
        # discrete_steps = np.linspace(min_val, max_val, prune_steps)
        # get all the discrete steps along the step axis
        discrete_steps = [prune_schedule[i] for i in range(0, len(prune_schedule)) if i % (len(prune_schedule) // prune_steps) == 0][:-1] + [prune_schedule[-1] + 1e-6]
        indices = np.digitize(prune_schedule, discrete_steps) - 1
        prune_schedule = [discrete_steps[i] for i in indices]

    schedule += prune_schedule
    schedule += [pruning_ratio_dict] * (steps - end_step)
    # Pruning in pruner starts at step = idx = 1, i.e., the idx=0 is a dummy step

    if len(schedule) != steps + 1:
        import pdb;
        pdb.set_trace()

    # plt schedule
    # plt.plot(schedule)
    # plt.axvline(x=start_step, color='r', linestyle='--')
    # plt.axvline(x=end_step, color='r', linestyle='--')
    # plt.show()
    # vert line for pruning starts and ends

    assert len(schedule) == steps + 1
    return schedule


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    if t >= duration:
        return end_e
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
