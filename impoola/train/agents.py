import torch.nn as nn
from torch.distributions.categorical import Categorical

from impoola.train.nn import layer_init_orthogonal, layer_init_normed, encoder_factory


class DQNAgent(nn.Module):
    def __init__(
            self,
            encoder_type,
            envs,
            width_scale=1, out_features=256, cnn_filters=(16, 32, 32), activation='relu',
            use_layer_init_normed=False,
            use_pooling_layer=False, pooling_layer_kernel_size=1,
            use_dropout=False,
            use_1d_conv=False,
            use_depthwise_conv=False,
            use_moe=False,
            use_simba=False,
            positional_encoding=False
    ):
        super().__init__()

        # Encode input images (input as int8, conversion to float32 is done in the encoder forward pass)
        encoder, out_features = encoder_factory(
            encoder_type=encoder_type,
            envs=envs,
            width_scale=width_scale, out_features=out_features, cnn_filters=cnn_filters, activation=activation,
            use_layer_init_normed=use_layer_init_normed,
            use_pooling_layer=use_pooling_layer, pooling_layer_kernel_size=pooling_layer_kernel_size,
            use_dropout=use_dropout,
            use_1d_conv=use_1d_conv,
            use_depthwise_conv=use_depthwise_conv,
            use_moe=use_moe,
            use_simba=use_simba,
            positional_encoding=positional_encoding
        )
        self.encoder = encoder
        self.out_features = out_features

        self.value = layer_init_orthogonal(nn.Linear(out_features, envs.single_action_space.n), std=0.01)

    def forward(self, x):
        return self.value(self.encoder(x))

    def get_action(self, x, deterministic=False):
        q_values = self.forward(x)
        if deterministic:
            return q_values.argmax(dim=1)  # same as dist.mode
        else:
            return Categorical(logits=q_values).sample()


class ActorCriticAgent(nn.Module):
    def __init__(
            self,
            encoder_type,
            envs,
            width_scale=1, out_features=256, cnn_filters=(16, 32, 32), activation='relu',
            use_layer_init_normed=False,
            use_pooling_layer=False, pooling_layer_kernel_size=1,
            use_dropout=False,
            use_1d_conv=False,
            use_depthwise_conv=False,
            use_moe=False,
            use_simba=False,
            positional_encoding=False
    ):
        super().__init__()

        # Encode input images (input as int8, conversion to float32 is done in the encoder forward pass)
        encoder, out_features = encoder_factory(
            encoder_type=encoder_type,
            envs=envs,
            width_scale=width_scale, out_features=out_features, cnn_filters=cnn_filters, activation=activation,
            use_layer_init_normed=use_layer_init_normed,
            use_pooling_layer=use_pooling_layer, pooling_layer_kernel_size=pooling_layer_kernel_size,
            use_dropout=use_dropout,
            use_1d_conv=use_1d_conv,
            use_depthwise_conv=use_depthwise_conv,
            use_moe=use_moe,
            use_simba=use_simba,
            positional_encoding=positional_encoding
        )
        self.encoder = encoder
        self.out_features = out_features

        # Actor head
        actor = nn.Linear(out_features if not use_simba else cnn_filters[-1] * width_scale, envs.single_action_space.n)
        # self.actor = layer_init_normed(actor, norm_dim=1, scale=0.1) if use_layer_init_normed else actor
        self.actor = layer_init_orthogonal(actor, std=0.01)

        # Critic head
        critic = nn.Linear(out_features if not use_simba else cnn_filters[-1] * width_scale, 1)
        # self.critic = layer_init_normed(critic, norm_dim=1, scale=0.1) if use_layer_init_normed else critic
        self.critic = layer_init_orthogonal(critic, std=1.0)

    def forward(self, x):
        hidden = self.encoder(x)
        return self.actor(hidden), self.critic(hidden)

    def get_value(self, x):
        return self.forward(x)[1]

    def get_pi(self, x):
        return Categorical(logits=self.forward(x)[0])

    def get_action(self, x, deterministic=False):
        pi = self.get_pi(x)
        return pi.sample() if not deterministic else pi.mode

    def get_action_and_value(self, x, action=None):
        raise NotImplementedError


class PPOAgent(ActorCriticAgent):
    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        pi = Categorical(logits=logits)
        if action is None:
            action = pi.sample()
        return action, pi.log_prob(action), pi.entropy(), value, pi.logits

    def get_pi_and_value(self, x):
        logits, value = self.forward(x)
        return Categorical(logits=logits), value


class PPGAgent(ActorCriticAgent):

    def __init__(self, envs, width_scale=1, out_features=256, chans=(16, 32, 32), activation='relu',
                 use_layer_init_normed=False, use_spectral_norm=False,
                 use_pooling_layer=False, pooling_layer_kernel_size=1,
                 use_dropout=False,
                 use_1d_conv=False,
                 use_depthwise_conv=False):
        super().__init__(envs, width_scale, out_features, chans, activation, use_layer_init_normed, use_spectral_norm,
                         use_pooling_layer, pooling_layer_kernel_size, use_dropout, use_1d_conv, use_depthwise_conv)

        # Aux critic head
        aux_critic = nn.Linear(out_features, 1)
        self.aux_critic = layer_init_normed(aux_critic, norm_dim=1, scale=0.1) if use_layer_init_normed else aux_critic

    def forward(self, x, ):
        hidden = self.encoder(x)
        return self.actor(hidden), self.critic(hidden), self.aux_critic(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.encoder(x)
        logits = self.actor(hidden)
        pi = Categorical(logits=logits)
        if action is None:
            action = pi.sample()
        return action, pi.log_prob(action), pi.entropy(), self.critic(hidden.detach()), pi.logits

    def get_pi_and_value(self, x):
        hidden = self.encoder(x)
        return Categorical(logits=self.actor(hidden)), self.critic(hidden.detach())

    # PPG logic:
    def get_pi_value_and_aux_value(self, x):
        hidden = self.encoder(x)
        return Categorical(logits=self.actor(hidden)), self.critic(hidden.detach()), self.aux_critic(hidden)
