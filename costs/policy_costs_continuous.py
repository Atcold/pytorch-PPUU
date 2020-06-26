"""Improved costs that use edge filter and don't take max of the pixels, but
sum the values instead"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field

from costs.policy_costs import PolicyCost

import configs


class PolicyCostContinuous(PolicyCost):
    @dataclass
    class Config(PolicyCost.Config):
        lambda_p: float = field(default=4.0)

    def compute_lane_cost(self, images, car_size):
        SCALE = 0.25
        bsize, npred, nchannels, crop_h, crop_w = images.size()
        images = images.view(bsize * npred, nchannels, crop_h, crop_w)

        width, length = car_size[:, 0], car_size[:, 1]  # feet
        width = width * SCALE * (0.3048 * 24 / 3.7)  # pixels
        length = length * SCALE * (0.3048 * 24 / 3.7)  # pixels

        # Create separable proximity mask
        width.fill_(24 * SCALE / 2)

        max_x = torch.ceil((crop_h - length) / 2)
        #    max_y = torch.ceil((crop_w - width) / 2)
        max_y = torch.ceil(torch.zeros(width.size()).fill_(crop_w) / 2)
        max_x = (
            max_x.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .cuda()
        )
        max_y = (
            max_y.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .cuda()
        )
        min_y = torch.ceil(
            crop_w / 2 - width
        )  # assumes other._width / 2 = self._width / 2
        min_y = (
            min_y.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .cuda()
        )
        x_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_h))) * crop_h / 2

        x_filter = x_filter.unsqueeze(0).expand(bsize * npred, crop_h).cuda()
        x_filter = torch.min(
            x_filter, max_x.view(bsize * npred, 1).expand(x_filter.size())
        )
        x_filter = (
            x_filter == max_x.unsqueeze(1).expand(x_filter.size())
        ).float()

        y_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_w))) * crop_w / 2
        y_filter = (
            y_filter.view(1, crop_w).expand(bsize * npred, crop_w).cuda()
        )
        #    y_filter = torch.min(y_filter, max_y.view(bsize * npred, 1))
        y_filter = torch.max(y_filter, min_y.view(bsize * npred, 1))
        y_filter = (y_filter - min_y.view(bsize * npred, 1)) / (
            max_y.view(bsize * npred, 1) - min_y.view(bsize * npred, 1)
        )
        x_filter = x_filter.cuda()
        y_filter = y_filter.cuda()
        proximity_mask = torch.bmm(
            x_filter.view(-1, crop_h, 1), y_filter.view(-1, 1, crop_w)
        )
        proximity_mask = proximity_mask.view(bsize, npred, crop_h, crop_w)
        proximity_mask = proximity_mask ** 2
        images = images.view(bsize, npred, nchannels, crop_h, crop_w)
        costs = torch.sum(
            (proximity_mask * images[:, :, 0].float()).view(bsize, npred, -1),
            2,
        )
        return costs.view(bsize, npred), proximity_mask

    def compute_offroad_cost(self, images, proximity_mask):
        bsize, npred, nchannels, crop_h, crop_w = images.size()
        images = images.view(bsize, npred, nchannels, crop_h, crop_w)
        costs = torch.sum(
            (proximity_mask * images[:, :, 2].float()).view(bsize, npred, -1),
            2,
        )
        return costs.view(bsize, npred)

    def compute_contours(self, images):
        """Computes contours of the green channel.
        The idea is to get only edges of the cars so that later
        when we do summing the size of the cars doesn't affect our behavior.
        """
        device = images.device
        horizontal_filter = torch.tensor(
            [[[0.0], [0.0]], [[-1.0], [1.0]], [[0.0], [0.0]]], device=device,
        )
        horizontal_filter = horizontal_filter.expand(1, 3, 2, 1)
        vertical_filter = torch.tensor(
            [[0.0, 0.0], [1.0, -1.0], [0.0, 0.0]], device=device
        ).view(3, 1, 2)
        vertical_filter = vertical_filter.expand(1, 3, 1, 2)

        horizontal = F.conv2d(
            images, horizontal_filter, stride=1, padding=(1, 0)
        )
        horizontal = horizontal[:, :, :-1, :]

        vertical = F.conv2d(images, vertical_filter, stride=1, padding=(0, 1))
        vertical = vertical[:, :, :, :-1]

        _, _, height, width = horizontal.shape

        horizontal_mask = torch.ones((1, 1, height, width), device=device)
        horizontal_mask[:, :, : (height // 2), :] = -1
        horizontal_masked = F.relu(horizontal_mask * horizontal)

        vertical_mask = torch.ones((1, 1, height, width), device=device)
        vertical_mask[:, :, :, (width // 2) :] = -1
        vertical_masked = F.relu(vertical_mask * vertical)

        result = vertical_masked[:][:] + horizontal_masked[:][:]
        return result

    def compute_proximity_cost(
        self,
        images,
        states,
        car_size=(6.4, 14.3),
        green_channel=1,
        unnormalize=False,
        clip=False,
    ):

        SCALE = 0.25
        safe_factor = 1.5
        bsize, npred, nchannels, crop_h, crop_w = images.size()

        images = images.view(bsize * npred, nchannels, crop_h, crop_w)
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

        speed = states[:, 2:].norm(2, 1) * SCALE  # pixel/s
        width, length = car_size[:, 0], car_size[:, 1]  # feet
        width = width * SCALE * (0.3048 * 24 / 3.7)  # pixels
        length = length * SCALE * (0.3048 * 24 / 3.7)  # pixels

        safe_distance = (
            torch.abs(speed) * safe_factor + (1 * 24 / 3.7) * SCALE
        )  # plus one metre (TODO change)

        # Compute x/y minimum distance to other vehicles (pixel version)
        # Account for 1 metre overlap (low data accuracy)
        alpha = 1 * SCALE * (24 / 3.7)  # 1 m overlap collision
        # Create separable proximity mask

        max_x = torch.ceil((crop_h - torch.clamp(length - alpha, min=0)) / 2)
        max_y = torch.ceil((crop_w - torch.clamp(width - alpha, min=0)) / 2)
        max_x = (
            max_x.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .cuda()
        )
        max_y = (
            max_y.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .cuda()
        )

        min_x = torch.clamp(max_x - safe_distance, min=0)
        min_y = torch.ceil(
            crop_w / 2 - width
        )  # assumes other._width / 2 = self._width / 2
        min_y = (
            min_y.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .cuda()
        )

        x_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_h))) * crop_h / 2
        x_filter = x_filter.unsqueeze(0).expand(bsize * npred, crop_h).cuda()
        x_filter = torch.min(
            x_filter, max_x.view(bsize * npred, 1).expand(x_filter.size())
        )
        x_filter = torch.max(x_filter, min_x.view(bsize * npred, 1))

        x_filter = (x_filter - min_x.view(bsize * npred, 1)) / (
            max_x - min_x
        ).view(bsize * npred, 1)
        y_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_w))) * crop_w / 2
        y_filter = (
            y_filter.view(1, crop_w).expand(bsize * npred, crop_w).cuda()
        )
        y_filter = torch.min(y_filter, max_y.view(bsize * npred, 1))
        y_filter = torch.max(y_filter, min_y.view(bsize * npred, 1))
        y_filter = (y_filter - min_y.view(bsize * npred, 1)) / (
            max_y.view(bsize * npred, 1) - min_y.view(bsize * npred, 1)
        )
        x_filter = x_filter.cuda()
        y_filter = y_filter.cuda()
        proximity_mask = torch.bmm(
            x_filter.view(-1, crop_h, 1), y_filter.view(-1, 1, crop_w)
        )
        proximity_mask = proximity_mask.view(bsize, npred, crop_h, crop_w)
        if clip:
            proximity_mask[:, :, 60:] = 0
        images = images.view(-1, nchannels, crop_h, crop_w)
        green_contours = self.compute_contours(images)
        green_contours = green_contours.view(bsize, npred, crop_h, crop_w)
        # pre_max = (proximity_mask * (green_image ** 2))
        # green_contours[green_contours < 0.5] = 0
        proximity_mask = proximity_mask ** 2
        pre_max = proximity_mask * (green_contours ** 2)
        # costs = torch.max(pre_max.view(bsize, npred, -1), 2)[0]
        costs = torch.sum(pre_max.view(bsize, npred, -1), 2)
        # costs = torch.sum((proximity_mask * images[:, :, green_channel].float()).view(bsize, npred, -1), 2)
        # costs = torch.max((proximity_mask * images[:, :, green_channel].float()).view(bsize, npred, -1), 2)[0]

        images = images.view(bsize, npred, nchannels, crop_h, crop_w)
        green_image = images[:, :, green_channel].float()
        pre_max_old = proximity_mask * green_image
        costs_old = torch.max(pre_max_old.view(bsize, npred, -1), 2)[0]

        result = {}
        result["costs"] = costs
        result["costs_old"] = costs_old
        result["masks"] = proximity_mask
        result["pre_max"] = pre_max
        result["contours"] = green_contours

        return result

    def compute_state_costs(self, pred_images, pred_states, car_sizes):
        npred = pred_images.size(1)
        gamma_mask = (
            torch.tensor([0.99 ** t for t in range(npred + 1)])
            .cuda()
            .unsqueeze(0)
        )
        proximity_cost = self.compute_proximity_cost(
            pred_images,
            pred_states.data,
            car_sizes,
            unnormalize=True,
            clip=False,
        )["costs"]
        lane_cost, prox_map_l = self.compute_lane_cost(pred_images, car_sizes)
        offroad_cost = self.compute_offroad_cost(pred_images, prox_map_l)

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
