import os
from dataclasses import dataclass
from functools import partial
from typing import Optional
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmaps
import matplotlib as mpl
from matplotlib.pyplot import tight_layout

# Use LaTeX-style fonts
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Latin Modern Roman']

import wandb
import tyro
import torch

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from impoola.utils.utils import load_agent_from_wandb
from impoola.train.nn import layer_init_orthogonal
from impoola.utils.utils import calc_translation_sensitivity


@dataclass
class Args:
    # General Settings
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: Optional[int] = None
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "impoola"  # "pruning4drl_results"  # "drl_pruning"
    """the wandb's project name"""
    wandb_entity: str = 'tumwcps'
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    env_track_setting: str = "generalization"
    """the track setting of the environment"""
    n_episodes_rollout: int = int(5e3)
    """the number of episodes to rollout for evaluation"""
    env_id: str = "bigfish"
    """the id of the environment"""
    num_envs: int = 64
    """the number of parallel game environments"""
    wandb_run_name: str = "dzug2qkm"  # gkd5geoy=impala  or dzug2qkm=impoola
    """the wandb run name of the agent to load"""
    render: bool = False
    """if toggled, the environment will be rendered"""
    t_pixels: int = 8
    """the number of pixels to translate the image by"""


if __name__ == "__main__":
    args = tyro.cli(Args)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    run_name = f"inference_{args.wandb_run_name}_{device_name}"
    print(f"Running inference on {device_name} device")

    if args.cuda:
        # torch.backends.cudnn.allow_tf32 = True
        # torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    agent_wandb_tags = {
        # "ppo_impala_s2_random_no_anneal_lower_mid_lr": "Impala",
        # "ppo_impoola_s2_pool_2": "Impala w/ AvgPool(2,2)",
        # "ppo_impoola_s2_depth_conv": "Impala w/ DepthwiseConv2D",
        # "ppo_impoola_s2_max_pool": "Impoola w/ MaxPool(1,1)",
        # "ppo_impoola_s2_stacked_pool": "Impoola w/ StackedAvgPool",
        # "ppo_impoola_s2_deeper": "Impoola w/ 4 Blocks))",
        # "ppo_impala_s2_moe_random": "Impala w/ MoE",
        # "ppo_impoola_s2_pos_encoding": "Impoola w/ Pos. Enc.",
        # "ppo_impoola_s2_random_no_anneal_lower_mid_lr": "Impoola",
        'ppo_nature_s2': "Nature",
        'ppo_nature_s2_pool_new': "Nature w/ GAP",

    }
    agent_wandb_runs = {}

    api = wandb.Api()
    project_name = args.wandb_entity + "/" + args.wandb_project_name

    for tag, agent_name in agent_wandb_tags.items():
        # Fetch all runs in the project
        # runs = api.runs(project_name, filters={"tags": tag, "state": "finished"})
        runs = api.runs(project_name, filters={"tags": tag, "state": "finished", "config.env_id": "bigfish"})
        if runs:
            for artifact in runs[0].logged_artifacts():
                if artifact.type == "model":
                    artifact_name = artifact.name
                    # strip off the model/ prefix and postfix
                    artifact_name = artifact_name[6:].split(":")[0]
                    # remove the postfix which has a : at the beginning using .split
                    agent_wandb_runs[agent_name] = artifact_name
                    break

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        config=vars(args),
        name=run_name,
        mode="disabled" if not args.track else "online",
    )

    # Fishes with background
    # fishes = []
    # for name in ['fish1', 'fish2', 'fish3']:
    #     data = np.load(f'bigfish_assets/{name}.npy').transpose(2, 0, 1)
    #     # data = data.reshape(1, *data.shape)
    #     data[data == 255] = 0
    #     fishes.append(data)
    # fishes = np.stack(fishes)
    # more_fishes = np.load('bigfish_assets/fish_array_1000.npy').transpose(0, 3, 1, 2).astype(np.uint8)
    # fishes = np.vstack([fishes, more_fishes])

    fishes = np.load('bigfish_assets/fish_array_new_512.npy').transpose(0, 3, 1, 2)
    fishes = (fishes * 255).astype(np.uint8)  # [:128]
    # Backgrounds
    # background0 = np.load('bigfish_assets/water1.npy').transpose(2, 0, 1).reshape(1, 3, 64, 64)
    # background1 = np.load('bigfish_assets/underwater1.npy').transpose(2, 0, 1).reshape(1, 3, 64, 64)
    # background = np.vstack([background0, background1])
    backgrounds = []
    for name in [
        'water1',
        'water2', 'water3', 'water4',
        'underwater1',
        'underwater2', 'underwater3'
    ]:
        data = np.load(f'bigfish_assets/{name}.npy').transpose(2, 0, 1).astype(np.uint8)
        backgrounds.append(data)
    backgrounds = np.stack(backgrounds)


    def overlay_background_with_fish(backgrounds, fishes, offset_y, offset_x):
        all_images = []

        for i in range(len(backgrounds)):
            for j in range(len(fishes)):
                fish = fishes[j]
                # move the fish by the offset using np.roll
                fish = np.roll(fish, offset_y, axis=1)
                fish = np.roll(fish, offset_x, axis=2)
                # overlay the fish on the background
                out1 = backgrounds[i].copy()
                out1[fish > 0] = fish[fish > 0]
                all_images.append(out1)

                # # Augment data with flipped fish
                # fish_flipped = np.flip(fish, axis=2)
                # fish_flipped = np.roll(fish_flipped, offset_x, axis=1)
                # fish_flipped = np.roll(fish_flipped, offset_y, axis=2)
                # out2 = backgrounds[i].copy()
                # out2[fish_flipped > 0] = fish_flipped[fish_flipped > 0]
                # all_images.append(out2)
        return np.stack(all_images).astype(np.uint8)


    # overlay the fish on the background
    # obs = overlay_background_with_fish(backgrounds[[0]], fishes[[0]], 0, 20)
    # plt.imshow(obs[0].transpose(1, 2, 0))
    # plt.show()

    obs_generator = partial(overlay_background_with_fish, backgrounds, fishes)

    # save all the images for the centered fish and background in the same folder as pfs
    for i, obs in enumerate(obs_generator(0, 0)):
        os.makedirs('bigfish_assets/overlayed', exist_ok=True)
        plt.imsave(f'bigfish_assets/overlayed/obs_{i}.png', obs.transpose(1, 2, 0))

    n_agents = len(agent_wandb_runs)

    fig, ax = plt.subplots(
        nrows=1, ncols=1 * n_agents, figsize=(6 * (1 * n_agents), 5), sharey=True)

    for i_row, (agent_name, wandb_run_name) in enumerate(agent_wandb_runs.items()):

        # if 'impala' in agent_name:
        #     agent_type = 'Impala'
        # elif 'impoola' in agent_name:
        #     agent_type = 'Impoola'
        # else:
        #     agent_type = 'Unknown!'

        agent, _, _, config, _ = load_agent_from_wandb(None, device, wandb_run_name)
        args.env_id = config["env_id"]
        args.env_track_setting = config["env_track_setting"]

        # Restore training settings
        if args.seed is None:
            args.seed = config["seed"]
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        with torch.no_grad():
            example_input = 127 * np.ones((1,) + (3, 64, 64)).astype(np.float32)
            example_input = torch.tensor(example_input).to(device)
            agent(example_input)

        # visualize the translation map
        # ax.imshow(translation_map.cpu().numpy(), cmap='hot')
        global_min = 0
        global_max = 1.0

        flipped_hot = cmaps.get_cmap('plasma')  # .reversed()

        translation_map_actor, translation_map_critic = \
            calc_translation_sensitivity(agent, obs_generator, device, t_pixels=args.t_pixels)

        # # Reinitialize the agent
        # for name, param in agent.named_modules():
        #     if hasattr(param, "reset_parameters"):
        #         param.reset_parameters()
        #         if 'actor' in name:
        #             layer_init_orthogonal(param, std=0.01)
        #         elif 'critic' in name:
        #             layer_init_orthogonal(param, std=1.0)
        #
        # translation_map_actor_init, translation_map_critic_init = \
        #     calc_translation_sensitivity(agent, obs_generator, device, t_pixels=args.t_pixels)

        # Actor
        # ax[i_row, 0].imshow(
        #     translation_map_actor_init, cmap=flipped_hot, interpolation='nearest', vmin=global_min, vmax=global_max,
        #     aspect='equal')
        # ax[i_row, 0].set_title(f"Untrained Actor Sensitivity {agent_type}")
        # ax[i_row, 0].set_xlabel("Translation in X")
        # ax[i_row, 0].set_ylabel("Translation in Y")

        ax[i_row].imshow(
            translation_map_actor, cmap=flipped_hot, interpolation='nearest', vmin=global_min, vmax=global_max,
            aspect='equal')
        # set bold title using latex styoe with \textbf
        title_name = r'\textbf{' + agent_name + r'}'
        ax[i_row].set_title(f"{title_name}", fontsize="xx-large")

        ax[i_row].set_xlabel("Translation in x", fontsize="xx-large")
        if i_row == 0:
            ax[i_row].set_ylabel("Translation in y", fontsize="xx-large")

        # Critic
        # ax[i_row, 2].imshow(
        #     translation_map_critic_init, cmap=flipped_hot, interpolation='nearest', vmin=global_min, vmax=global_max,
        #     aspect='equal')
        # ax[i_row, 2].set_title(f"Critic Sensitivity {agent_type}")
        # ax[i_row, 2].set_xlabel("Translation in X")

        # ax[i_row + n_agents].imshow(
        #     translation_map_critic, cmap=flipped_hot, interpolation='nearest', vmin=global_min, vmax=global_max,
        #     aspect='equal')
        # ax[i_row + n_agents].set_title(f"{agent_name}")
        # ax[i_row + n_agents].set_xlabel("Translation in X")

        for i_column in range(1 * n_agents):
            ax[i_column].spines['top'].set_visible(False)
            ax[i_column].spines['right'].set_visible(False)
            ax[i_column].set_yticks(np.linspace(0, 2 * args.t_pixels, 5, dtype=int))
            ax[i_column].set_yticklabels(np.linspace(args.t_pixels, -args.t_pixels, 5, dtype=int))
            ax[i_column].set_xticks(np.linspace(0, 2 * args.t_pixels, 5, dtype=int))
            ax[i_column].set_xticklabels(np.linspace(-args.t_pixels, args.t_pixels, 5, dtype=int))

        # cleanup agent etc
        del agent
        torch.cuda.empty_cache()

        # fig.text(-0.05, 1 - 0.1 - 0.5 * i_row, agent_type, ha='center', va='center', fontsize=16, rotation=90)
    # place colorbar manually to the right of the last column
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=flipped_hot), ax=ax, orientation='vertical', fraction=0.05, pad=0.02) #, shrink=0.5)


    cbar.set_label('Sensitivity', rotation=270, labelpad=20, fontsize="xx-large")
    # fig.tight_layout()

    # fig.text(0.25, 1 - 0.02, f'Actor Sensitivity', ha='left', va='center', fontsize=16, fontweight='bold')
    # fig.text(0.7, 1 - 0.02, f'Critic Sensitivity', ha='center', va='center', fontsize=16, fontweight='bold')

    # set max tick of cbar to global_max
    # cbar.set_ticks([global_min, 1])
    # cbar.set_ticklabels([f'{0}', f'{global_max}'])

    # save as high quality pdf
    fig.savefig(f"translation_sensitivity_bigfish_nature.pdf", dpi=300)
    # plt.show()
    # run bash command pdfcrop translation_sensitivity_bigfish.pdf translation_sensitivity_bigfish.pdf
    # to crop the pdf
    os.system("pdfcrop translation_sensitivity_bigfish_nature.pdf translation_sensitivity_bigfish_nature.pdf")
