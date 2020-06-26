from typing import Union

import torch

from modeling.forward_models import ForwardModel


class ForwardModelKM(ForwardModel):
    def predict_states(self, states, actions, stats, timestep=0.1):
        states = states.clone()
        actions = actions.clone()

        ss_std = (
            1e-8 + stats["s_std"][0].view(1, 4).expand(states.size())
        ).cuda()
        ss_mean = stats["s_mean"][0].view(1, 4).expand(states.size()).cuda()
        aa_std = (1e-8 + stats["a_std"][0].view(1, 2)).cuda()
        aa_mean = stats["a_mean"][0].view(1, 2).cuda()

        actions = actions * aa_std + aa_mean
        states = states * ss_std + ss_mean

        a = actions[:, 0]
        b = actions[:, 1]

        positions = states[:, :2]
        speeds = states[:, 2:]
        speeds_norm = speeds.norm(dim=1).view(speeds.shape[0], 1)

        directions_with_negative = speeds / speeds_norm
        directions = torch.stack(
            [
                torch.abs(directions_with_negative[:, 0]),
                directions_with_negative[:, 1],
            ],
            axis=1,
        )

        new_positions = positions + timestep * speeds_norm * directions

        ortho_directions = torch.stack(
            [directions[:, 1], -directions[:, 0]], 1
        )

        new_directions_unnormed = (
            directions
            + ortho_directions * b.unsqueeze(1) * speeds_norm * timestep
        )
        new_directions = new_directions_unnormed / (
            new_directions_unnormed.norm(dim=1).view(speeds.shape[0], 1) + 1e-3
        )

        new_speeds_norm = speeds_norm + a.unsqueeze(1) * timestep
        new_speeds = new_directions * new_speeds_norm

        new_states = torch.cat([new_positions, new_speeds], 1)
        new_states = new_states - ss_mean
        new_states = new_states / ss_std

        return new_states

    def unfold_km(
        self,
        actions_or_policy: Union[torch.nn.Module, torch.Tensor],
        batch,
        Z=None,
    ):
        def cat_inputs(inputs, new_value):
            if len(new_value.shape) < len(inputs.shape):
                new_value = new_value.unsqueeze(1)
            return torch.cat((inputs[:, 1:], new_value), 1)

        input_images = batch["input_images"].clone()
        input_states = batch["input_states"].clone()
        input_states_km = input_states.clone()
        input_states_km_a = input_states.clone()
        input_states_km_b = input_states.clone()

        stats = batch["stats"]
        bsize = batch["input_images"].size(0)

        if torch.is_tensor(actions_or_policy):
            ego_car_required = False
            npred = actions_or_policy.size(1)
        else:
            # If the source of actions is a policy and not a tensor array, we
            # need a version of the state with ego car on the 4th channel.
            ego_car_required = True
            input_ego_car_orig = batch["ego_cars"]
            npred = batch["target_images"].size(1)

            ego_car_new_shape = [*input_images.shape]
            ego_car_new_shape[2] = 1
            input_ego_car = input_ego_car_orig[:, 2][:, None, None].expand(
                ego_car_new_shape
            )
            input_images_with_ego = torch.cat(
                (input_images.clone(), input_ego_car), dim=2
            )

        pred_images, pred_states, pred_actions = [], [], []
        pred_states_km = []
        pred_states_km_a = []
        pred_states_km_b = []

        if Z is None:
            Z = self.sample_z(npred * bsize)
            Z = Z.view(bsize, npred, -1)

        for t in range(npred):
            if torch.is_tensor(actions_or_policy):
                actions = actions_or_policy[:, t]
            else:
                actions = actions_or_policy(
                    input_images_with_ego, input_states
                )

            z_t = Z[:, t]
            pred_image, pred_state = self.forward_single_step(
                input_images, input_states_km, actions, z_t
            )
            pred_state_km = self.predict_states(
                input_states_km[:, -1], actions, stats
            )
            pred_state_km_a = self.predict_states(
                input_states_km_a[:, -1],
                torch.stack([actions[:, 0], actions[:, 1].detach()], axis=1),
                stats,
            )
            pred_state_km_b = self.predict_states(
                input_states_km_b[:, -1],
                torch.stack([actions[:, 0].detach(), actions[:, 1]], axis=1),
                stats,
            )

            input_images = cat_inputs(input_images, pred_image)
            input_states = cat_inputs(input_states, pred_state)
            input_states_km = cat_inputs(input_states_km, pred_state_km)
            input_states_km_a = cat_inputs(input_states_km_a, pred_state_km_a)
            input_states_km_b = cat_inputs(input_states_km_b, pred_state_km_b)

            if ego_car_required:
                pred_image_with_ego = torch.cat(
                    (pred_image, input_ego_car[:, :1]), dim=2
                )
                input_images_with_ego = cat_inputs(
                    input_images_with_ego, pred_image_with_ego
                )

            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_states_km.append(pred_state_km)
            pred_states_km_a.append(pred_state_km_a)
            pred_states_km_b.append(pred_state_km_b)
            pred_actions.append(actions)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_states_km = torch.stack(pred_states_km, 1)
        pred_states_km_a = torch.stack(pred_states_km_a, 1)
        pred_states_km_b = torch.stack(pred_states_km_b, 1)
        pred_actions = torch.stack(pred_actions, 1)

        return dict(
            pred_images=pred_images,
            pred_states=dict(
                vanilla=pred_states,
                km=pred_states_km,
                km_a=pred_states_km_a,
                km_b=pred_states_km_b,
            ),
            pred_actions=pred_actions,
            Z=Z,
        )
