"""Policy models"""

from torch import nn

from modeling.common_models import Encoder


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        n_cond=20,
        n_feature=256,
        n_actions=2,
        h_height=14,
        h_width=3,
        n_hidden=256,
    ):
        super().__init__()
        self.n_channels = 4
        self.n_cond = n_cond
        self.n_feature = n_feature
        self.n_actions = n_actions
        self.h_height = h_height
        self.h_width = h_width
        self.n_hidden = n_hidden
        self.encoder = Encoder(
            Encoder.Config(
                a_size=0, n_inputs=self.n_cond, n_channels=self.n_channels
            )
        )
        self.n_outputs = self.n_actions
        self.hsize = self.n_feature * self.h_height * self.h_width
        self.proj = nn.Linear(self.hsize, self.n_hidden)

        self.fc = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_outputs),
        )

    def forward(
        self,
        state_images,
        states,
        context=None,
        sample=True,
        normalize_inputs=False,
        normalize_outputs=False,
        n_samples=1,
    ):
        if normalize_inputs:
            state_images = state_images.clone().float().div_(255.0)
            states -= self.stats["s_mean"].cuda().view(1, 4).expand(states.size())
            states /= self.stats["s_std"].cuda().view(1, 4).expand(states.size())
            if state_images.dim() == 4:  # if processing single vehicle
                state_images = state_images.cuda().unsqueeze(0)
                states = states.cuda().unsqueeze(0)

        bsize = state_images.size(0)
        h = self.encoder(state_images, states).view(bsize, self.hsize)
        h = self.proj(h)  # from hidden_size to n_hidden
        a = self.fc(h).view(bsize, self.n_outputs)

        if normalize_outputs:
            a = a.data
            a.clamp_(-3, 3)
            a *= self.stats["a_std"].view(1, 2).expand(a.size()).cuda()
            a += self.stats["a_mean"].view(1, 2).expand(a.size()).cuda()
        return a
