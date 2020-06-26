from typing import Union

import torch


class ForwardModel(torch.nn.Module):
    def __init__(self, file_path):
        super().__init__()
        model = torch.load(file_path)
        if type(model) is dict:
            model = model["model"]
        if not hasattr(model.encoder, "n_channels"):
            model.encoder.n_channels = 3
        setattr(self, "forward_model", model)
        self.forward_model = model

    def __getattr__(self, name):
        """Delegate everything to forward_model"""
        return getattr(self._modules["forward_model"], name)

    def unfold(
        self,
        actions_or_policy: Union[torch.nn.Module, torch.Tensor],
        batch,
        Z=None,
    ):
        input_images = batch["input_images"].clone()
        input_states = batch["input_states"].clone()
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
                input_images, input_states, actions, z_t
            )
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat(
                (input_states[:, 1:], pred_state.unsqueeze(1)), 1
            )

            if ego_car_required:
                pred_image_with_ego = torch.cat(
                    (pred_image, input_ego_car[:, :1]), dim=2
                )
                input_images_with_ego = torch.cat(
                    (input_images_with_ego[:, 1:], pred_image_with_ego), 1
                )

            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_actions.append(actions)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_actions = torch.stack(pred_actions, 1)

        return dict(
            pred_images=pred_images,
            pred_states=pred_states,
            pred_actions=pred_actions,
            Z=Z,
        )
