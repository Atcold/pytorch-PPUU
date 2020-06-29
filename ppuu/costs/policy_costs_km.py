"""Cost model that uses the kinematic model.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field

from ppuu.costs.policy_costs_continuous import PolicyCostContinuous


class PolicyCostKM(PolicyCostContinuous):
    @dataclass
    class Config(PolicyCostContinuous.Config):
        masks_power: int = 2
        agg_func_str: str = "logsumexp-31"

        u_reg: float = field(default=3.09)
        lambda_a: float = field(default=2.29)
        lambda_l: float = field(default=6.44)
        lambda_o: float = field(default=5.0)
        lambda_p: float = field(default=10.23)

        def __post_init__(self):
            agg_func = torch.logsumexp
            if self.agg_func_str == "sum":
                agg_func = torch.sum
            elif self.agg_func_str.startswith("logsumexp"):
                beta = float(self.agg_func_str.split("-")[1])

                def foo(*args):
                    return (
                        1 / beta * torch.logsumexp(args[0] * beta, *args[1:])
                    )

                agg_func = foo
            self.agg_func = agg_func

    def get_masks(self, images, states, car_size, unnormalize):
        bsize, npred, nchannels, crop_h, crop_w = images.shape

        states = states.view(bsize * npred, 4).clone()

        if unnormalize:
            states = (
                states
                * (
                    1e-8
                    + self.data_stats["s_std"].view(1, 4).expand(states.size())
                ).cuda()
            )
            states = (
                states
                + self.data_stats["s_mean"]
                .view(1, 4)
                .expand(states.size())
                .cuda()
            )

        states = states.view(bsize, npred, 4)

        LANE_WIDTH_METRES = 3.7
        LANE_WIDTH_PIXELS = 24  # pixels / 3.7 m, lane width
        SCALE = 1 / 4
        PIXELS_IN_METRE = SCALE * LANE_WIDTH_PIXELS / LANE_WIDTH_METRES
        MAX_SPEED_MS = 130 / 3.6  # m/s
        LOOK_AHEAD_M = MAX_SPEED_MS  # meters
        LOOK_SIDEWAYS_M = 2 * LANE_WIDTH_METRES  # meters
        METRES_IN_FOOT = 0.3048

        car_size = car_size.cuda()
        positions = states[:, :, :2]
        speeds = states[:, :, 2:]
        speeds_norm = speeds.norm(1, 2) / PIXELS_IN_METRE
        # speeds_o = torch.ones_like(speeds).cuda()
        # directions = torch.atan2(speeds_o[:, :, 1], speeds_o[:, :, 0])
        directions = torch.atan2(speeds[:, :, 1], speeds[:, :, 0])

        positions_adjusted = positions - positions.detach()
        # here we flip directions because they're flipped otherwise
        directions_adjusted = -(directions - directions.detach())

        width, length = car_size[:, 0], car_size[:, 1]  # feet
        width = width * METRES_IN_FOOT
        width = width.view(bsize, 1)

        length = length * METRES_IN_FOOT
        length = length.view(bsize, 1)

        REPEAT_SHAPE = (bsize, npred, 1, 1)

        y_d = width / 2 + LANE_WIDTH_METRES
        x_s = (
            1.5 * torch.clamp(speeds_norm.detach(), min=10) + length * 1.5 + 1
        )

        x_s = x_s.view(REPEAT_SHAPE)
        x_s_rotation = torch.ones(REPEAT_SHAPE).cuda() * 1

        y = torch.linspace(-LOOK_SIDEWAYS_M, LOOK_SIDEWAYS_M, crop_w).cuda()
        # x should be from positive to negative, as when we draw the image,
        # cars with positive distance are ahead of us.
        # also, if it's reversed, the -x_pos in x_prime calculation becomes
        # +x_pos.
        x = torch.linspace(LOOK_AHEAD_M, -LOOK_AHEAD_M, crop_h).cuda()
        xx, yy = torch.meshgrid(x, y)
        xx = xx.repeat(bsize, npred, 1, 1)
        yy = yy.repeat(bsize, npred, 1, 1)

        c, s = torch.cos(directions_adjusted), torch.sin(directions_adjusted)
        c = c.view(REPEAT_SHAPE)
        s = s.view(REPEAT_SHAPE)

        x_pos = positions_adjusted[:, :, 0]
        x_pos = x_pos.view(*x_pos.shape, 1, 1)
        y_pos = positions_adjusted[:, :, 1]
        y_pos = y_pos.view(*y_pos.shape, 1, 1)

        x_prime = c * xx - s * yy - x_pos  # <- here
        y_prime = s * xx + c * yy - y_pos  # and here a double - => +

        z_x_prime = torch.clamp(
            (x_s - torch.abs(x_prime))
            / (x_s - length.view(bsize, 1, 1, 1) / 2),
            min=0,
        )
        z_x_prime_rotation = torch.clamp(
            (x_s_rotation - torch.abs(x_prime)) / (x_s_rotation), min=0
        )
        r_y_prime = torch.clamp(
            (y_d.view(bsize, 1, 1, 1) - torch.abs(y_prime))
            / (y_d - width / 2).view(bsize, 1, 1, 1),
            min=0,
        )

        # Acceleration probe
        x_major = z_x_prime ** self.config.masks_power
        # x_major[:, :, (x_major.shape[2] // 2 + 10):, :] = x_major[:, :, (x_major.shape[2] // 2 + 10):, :].clone() ** 2
        y_ramp = torch.clamp(r_y_prime ** self.config.masks_power, max=1)
        result_acceleration = x_major * y_ramp

        # Rotation probe
        # x_ramp = torch.clamp(z_x_prime, max=1).float()
        x_ramp = torch.clamp(z_x_prime ** self.config.masks_power, max=1)
        # x_ramp = (z_x_prime > 0).float()
        # x_ramp[:, :, (x_ramp.shape[2] // 2 + 10):, :] = x_ramp[:, :, (x_ramp.shape[2] // 2 + 10):, :].clone() ** 2
        y_major = r_y_prime ** self.config.masks_power
        result_rotation = x_ramp * y_major

        return result_rotation, result_acceleration

    def compute_proximity_cost_km(self, images, proximity_masks):
        bsize, npred, nchannels, crop_h, crop_w = images.shape
        images = images.view(-1, nchannels, crop_h, crop_w)
        green_contours = self.compute_contours(images)
        green_contours = green_contours.view(bsize, npred, crop_h, crop_w)
        pre_max = proximity_masks * (green_contours ** 2)
        costs = self.config.agg_func(pre_max.view(bsize, npred, -1), 2)

        result = {}
        result["costs"] = costs
        result["masks"] = proximity_masks
        result["pre_max"] = pre_max
        result["contours"] = green_contours
        return result

    def compute_lane_cost_km(self, images, proximity_masks):
        bsize, npred = images.shape[0], images.shape[1]
        lanes = images[:, :, 0].float()
        costs = proximity_masks * (lanes ** 2)
        costs_m = self.config.agg_func(costs.view(bsize, npred, -1), 2)
        return costs_m.view(bsize, npred)

    def compute_offroad_cost_km(self, images, proximity_masks):
        bsize, npred = images.shape[0], images.shape[1]
        offroad = images[:, :, 2]
        costs = proximity_masks * (offroad ** 2)
        costs_m = self.config.agg_func(costs.view(bsize, npred, -1), 2)
        return costs_m.view(bsize, npred)

    def compute_state_costs_for_training(
        self, pred_images, pred_states, car_sizes
    ):
        proximity_masks = self.get_masks(
            pred_images[:, :, :3].contiguous().detach(),
            pred_states["km"],
            car_sizes,
            unnormalize=True,
        )

        npred = pred_images.size(1)
        gamma_mask = (
            torch.tensor([0.99 ** t for t in range(npred + 1)])
            .cuda()
            .unsqueeze(0)
        )
        proximity_cost = self.compute_proximity_cost_km(
            pred_images, proximity_masks[1],
        )["costs"]

        lane_cost = self.compute_lane_cost_km(pred_images, proximity_masks[0])
        offroad_cost = self.compute_offroad_cost_km(
            pred_images, proximity_masks[0]
        )

        lane_loss = torch.mean(lane_cost * gamma_mask[:, :npred])
        offroad_loss = torch.mean(offroad_cost * gamma_mask[:, :npred])
        proximity_loss = torch.mean(proximity_cost * gamma_mask[:, :npred])
        return dict(
            proximity_cost=proximity_cost,
            lane_cost=lane_cost,
            offroad_cost=offroad_cost,
            lane_loss=lane_loss,
            offroad_loss=offroad_loss,
            proximity_loss=proximity_loss,
        )


class PolicyCostKMSplit(PolicyCostKM):
    def compute_state_costs_for_training(
        self, pred_images, pred_states, car_sizes
    ):
        proximity_mask_a = self.get_masks(
            pred_images[:, :, :3].contiguous().detach(),
            pred_states["km_a"],
            car_sizes,
            unnormalize=True,
        )[1]
        proximity_mask_b = self.get_masks(
            pred_images[:, :, :3].contiguous().detach(),
            pred_states["km_b"],
            car_sizes,
            unnormalize=True,
        )[0]

        npred = pred_images.size(1)
        gamma_mask = (
            torch.tensor([0.99 ** t for t in range(npred + 1)])
            .cuda()
            .unsqueeze(0)
        )
        proximity_cost = (
            self.compute_proximity_cost_km(pred_images, proximity_mask_a,)[
                "costs"
            ]
            + self.compute_proximity_cost_km(pred_images, proximity_mask_b,)[
                "costs"
            ]
        ) / 2
        lane_cost = (
            self.compute_lane_cost_km(pred_images, proximity_mask_a)
            + self.compute_lane_cost_km(pred_images, proximity_mask_b)
        ) / 2
        offroad_cost = (
            self.compute_offroad_cost_km(pred_images, proximity_mask_a)
            + self.compute_offroad_cost_km(pred_images, proximity_mask_b)
        ) / 2

        lane_loss = torch.mean(lane_cost * gamma_mask[:, :npred])
        offroad_loss = torch.mean(offroad_cost * gamma_mask[:, :npred])
        proximity_loss = torch.mean(proximity_cost * gamma_mask[:, :npred])
        return dict(
            proximity_cost=proximity_cost,
            lane_cost=lane_cost,
            offroad_cost=offroad_cost,
            lane_loss=lane_loss,
            offroad_loss=offroad_loss,
            proximity_loss=proximity_loss,
        )
