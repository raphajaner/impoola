from functools import partial

from torch import nn
import torch
from torch.nn.utils.parametrizations import spectral_norm
from torch.nn.utils.parametrize import remove_parametrizations


def add_parametrizations_to_agent(agent, orig_modules):
    for module in orig_modules:
        spectral_norm(module, name="weight")
    return agent


def remove_parametrizations_from_agent(agent):
    orig_modules = list()
    for module in agent.modules():
        if hasattr(module, 'parametrizations'):
            remove_parametrizations(module, "weight", leave_parametrized=True)
            orig_modules.append(module)
            assert not hasattr(module, 'parametrizations')
    return orig_modules


def register_activation_hook(model):
    @torch.inference_mode()
    def forward_hook(module, input, output, name):
        # setattr(module, f'activation_', output)
        if hasattr(module, f'activation_'):
            if getattr(module, f'activation_').shape[0] != output.shape[0]:
                return
            elif getattr(module, f'activation_').shape[1] != output.shape[1]:
                setattr(module, f'activation_', output)
            else:
                # import pdb; pdb.set_trace()
                # # concatenate the output
                activation_ = torch.cat([getattr(module, f'activation_'), output], dim=0)
                setattr(module, f'activation_', activation_)
                # setattr(module, f'activation_', output)
        else:
            setattr(module, f'activation_', output)

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            handle = module.register_forward_hook(partial(forward_hook, name=name))
            handles.append(handle)
    return handles


def remove_activation_hooks(model, handles):
    for name, module in model.named_modules():
        if hasattr(module, f'activation_'):
            del module.activation_

    for handle in handles:
        handle.remove()


def get_activations_from_hooks(model):
    activations = {}
    for name, module in model.named_modules():
        if hasattr(module, f'activation_'):
            activations[name] = getattr(module, f'activation_')
    return activations


def remove_relu_activation_hooks(handles):
    for handle in handles:
        handle.remove()


def get_relu_activations_from_hooks(model):
    activations = {}
    for name, module in model.named_modules():
        if hasattr(module, f'activation_'):
            activations[name] = nn.functional.relu(getattr(module, f'activation_'))
    return activations


def register_gradient_accumulation_hooks(model):
    @torch.inference_mode()
    def backward_hook(module, grad_input, grad_output, name):
        setattr(module, name, grad_output)

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            handle = module.register_backward_hook(partial(backward_hook, name=name))
            handles.append(handle)
    return handles


def remove_gradient_accumulation_hooks(handles):
    for handle in handles:
        handle.remove()


def get_gradient_accumulation_from_hooks(model):
    gradient_accumulation = {}
    for name, module in model.named_modules():
        if hasattr(module, f'{name}_output'):
            gradient_accumulation[name] = getattr(module, f'{name}_output')
    return gradient_accumulation
