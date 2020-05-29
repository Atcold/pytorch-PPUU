from dataclasses import dataclass

from torch import nn

import configs


class Encoder(nn.Module):
    @dataclass
    class Config(configs.ConfigBase):
        n_feature: int = 256
        layers: int = 3
        n_channels: int = 3
        n_cond: int = 20
        n_inputs: int = 4
        state_input_size: int = 4
        states: bool = True
        a_size: int = 3
        hidden_size: int = 14 * 3 * 256
        dropout: float = 0.0
        height: int = 117
        width: int = 24

    @staticmethod
    def build_frame_encoder(input_size, output_size, layers, dropout):
        if layers == 3:
            assert output_size % 4 == 0
            feature_maps = (
                output_size // 4,
                output_size // 2,
            )
        elif layers == 4:
            assert output_size % 8 == 0
            feature_maps = (
                output_size // 8,
                output_size // 4,
                output_size // 2,
            )
        encoder_layers = []
        current_size = input_size
        for i, next_size in enumerate(feature_maps):
            encoder_layers += [
                nn.Conv2d(current_size, next_size, 4, 2, 1),
                nn.Dropout2d(p=dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            current_size = next_size
        encoder_layers.append(nn.Conv2d(current_size, output_size, 4, 2, 1))
        return nn.Sequential(*encoder_layers)

    @staticmethod
    def build_values_encoder(input_size, hidden_size, output_size, dropout):
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config.n_inputs = (
            self.config.n_cond
            if self.config.n_inputs is None
            else self.config.n_inputs
        )
        self.f_encoder = Encoder.build_frame_encoder(
            self.config.n_channels * self.config.n_inputs,
            self.config.n_feature,
            self.config.layers,
            self.config.dropout,
        )
        if self.config.states:
            self.s_encoder = Encoder.build_values_encoder(
                self.config.state_input_size * self.config.n_inputs,
                self.config.n_feature,
                self.config.hidden_size,
                self.config.dropout,
            )
        if self.config.a_size > 0:
            # action or cost encoder
            self.a_encoder = Encoder.build_values_encoder(
                self.config.a_size,
                self.config.n_feature,
                self.config.hidden_size,
                self.config.dropout,
            )

    def forward(self, images, states=None, actions=None):
        bsize = images.size(0)
        h = self.f_encoder(
            images.view(
                bsize,
                self.config.n_inputs * self.config.n_channels,
                self.config.height,
                self.config.width,
            )
        )
        if states is not None:
            h = h + self.s_encoder(states.contiguous().view(bsize, -1)).view(
                h.size()
            )
        if actions is not None:
            a = self.a_encoder(
                actions.contiguous().view(bsize, self.config.a_size)
            )
            h = h + a.view(h.size())
        return h


class UNetwork(nn.Module):
    """U-Net that encodes and decodes a frame"""

    def __init__(self, n_feature=256, layers=3, dropout=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_feature, n_feature, 4, 2, 1),
            nn.Dropout2d(p=dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feature, n_feature, (4, 1), 2, 1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_feature, n_feature, (4, 1), 2, 1),
            nn.Dropout2d(p=dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(n_feature, n_feature, (4, 3), 2, 0),
        )

        assert layers == 3  # hardcoded sizes
        self.hidden_size = n_feature * 3 * 2
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, n_feature),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n_feature, self.hidden_size),
        )

    def forward(self, h):
        h1 = self.encoder(h)
        h2 = self.fc(h1.view(-1, self.hidden_size))
        h2 = h2.view(h1.size())
        h3 = self.decoder(h2)
        return h3


class Decoder(nn.Module):
    """Decodes a hidden state into a predicted frame,
    a predicted state and a predicted cost vector.
    """

    def __init__(
        self,
        layers=3,
        n_feature=256,
        dropout=0.0,
        h_height=14,
        h_width=3,
        height=117,
        width=24,
    ):
        """Dependencies:
            opt.layers
            opt.feature
            opt.dropout
            opt.h_height
            opt.h_width
            opt.height
            opt.width
        """
        super(decoder, self).__init__()
        self.layers = layers
        self.n_feature = n_feature
        self.dropout = dropout
        self.h_height = h_heght
        self.h_width = h_width
        self.height = height
        self.width = width

        assert self.layers == 3
        assert self.n_feature % 4 == 0
        self.feature_maps = [
            self.n_feature,
            int(self.n_feature / 2),
            int(self.n_feature / 4),
        ]
        self.f_decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.feature_maps[0], self.feature_maps[1], (4, 4), 2, 1
            ),
            nn.Dropout2d(p=self.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                self.feature_maps[1], self.feature_maps[2], (5, 5), 2, (0, 1),
            ),
            nn.Dropout2d(p=self.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.feature_maps[2], 3, (2, 2), 2, (0, 1)),
        )

        self.h_reducer = nn.Sequential(
            nn.Conv2d(self.n_feature, self.n_feature, 4, 2, 1),
            nn.Dropout2d(p=self.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.n_feature, self.n_feature, (4, 1), (2, 1), 0,),
            nn.Dropout2d(p=self.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.s_predictor = nn.Sequential(
            nn.Linear(2 * self.n_feature, self.n_feature),
            nn.Dropout(p=self.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.n_feature, self.n_feature),
            nn.Dropout(p=self.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.n_feature, 4),
        )

    def forward(self, h):
        bsize = h.size(0)
        h = h.view(bsize, self.n_feature, self.h_height, self.h_width)
        h_reduced = self.h_reducer(h).view(bsize, -1)
        pred_state = self.s_predictor(h_reduced)
        pred_image = self.f_decoder(h)
        pred_image = pred_image[:, :, : self.height, : self.width].clone()
        pred_image = pred_image.view(bsize, 1, 3, self.height, self.width)
        return pred_image, pred_state


if __name__ == "__main__":
    config = Encoder.Config.parse_from_command_line()
    print(config)
    encoder = Encoder(config)
    print("ok")
