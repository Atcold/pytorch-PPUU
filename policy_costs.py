"""Functions for planning and training policy networks using the forward
model
"""
from typing import Union

import torch


def compute_lane_cost(images, car_size):
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
    x_filter = (x_filter == max_x.unsqueeze(1).expand(x_filter.size())).float()

    y_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_w))) * crop_w / 2
    y_filter = y_filter.view(1, crop_w).expand(bsize * npred, crop_w).cuda()
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
    images = images.view(bsize, npred, nchannels, crop_h, crop_w)
    costs = torch.max(
        (proximity_mask * images[:, :, 0].float()).view(bsize, npred, -1), 2
    )[0]
    return costs.view(bsize, npred), proximity_mask


def compute_offroad_cost(images, proximity_mask):
    bsize, npred, nchannels, crop_h, crop_w = images.size()
    images = images.view(bsize, npred, nchannels, crop_h, crop_w)
    costs = torch.max(
        (proximity_mask * images[:, :, 2].float()).view(bsize, npred, -1), 2
    )[0]
    return costs.view(bsize, npred)


def compute_proximity_cost(
    images,
    states,
    car_size=(6.4, 14.3),
    green_channel=1,
    unnormalize=False,
    s_mean=None,
    s_std=None,
):
    SCALE = 0.25
    safe_factor = 1.5
    bsize, npred, nchannels, crop_h, crop_w = images.size()
    images = images.view(bsize * npred, nchannels, crop_h, crop_w)
    states = states.view(bsize * npred, 4).clone()

    if unnormalize:
        states = (
            states * (1e-8 + s_std.view(1, 4).expand(states.size())).cuda()
        )
        states = states + s_mean.view(1, 4).expand(states.size()).cuda()

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
    y_filter = y_filter.view(1, crop_w).expand(bsize * npred, crop_w).cuda()
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
    images = images.view(bsize, npred, nchannels, crop_h, crop_w)
    costs = torch.max(
        (proximity_mask * images[:, :, green_channel].float()).view(
            bsize, npred, -1
        ),
        2,
    )[0]
    return costs, proximity_mask


def compute_uncertainty_batch(
    model,
    batch,
    cost_config,
    data_stats,
    npred=200,
    n_models=10,
    Z=None,
    estimation=True,
):
    """Estimates prediction uncertainty using dropout."""
    if estimation:
        torch.set_grad_enabled(False)
    input_images = batch["input_images"]
    input_states = batch["input_states"]
    actions = batch["actions"]
    car_sizes = batch["car_sizes"]

    bsize, ncond, channels, height, width = input_images.shape

    if Z is None:
        Z = model.sample_z(bsize * npred, method="fp")
        if type(Z) is list:
            Z = Z[0]
        Z = Z.view(bsize, npred, -1)
    Z_rep = Z.unsqueeze(0)
    Z_rep = Z_rep.expand(n_models, bsize, npred, -1)

    input_images = input_images.unsqueeze(0)
    input_states = input_states.unsqueeze(0)
    actions = actions.unsqueeze(0)
    input_images = input_images.expand(
        n_models, bsize, ncond, channels, height, width
    )
    input_states = input_states.expand(n_models, bsize, ncond, 4)
    actions = actions.expand(n_models, bsize, npred, 2)
    input_images = input_images.contiguous().view(
        bsize * n_models, ncond, channels, height, width
    )
    input_states = input_states.contiguous().view(bsize * n_models, ncond, 4)
    actions = actions.contiguous().view(bsize * n_models, npred, 2)
    Z_rep = Z_rep.contiguous().view(n_models * bsize, npred, -1)

    model.train()  # turn on dropout, for uncertainty estimation

    predictions = policy_unfold(
        model,
        actions.clone(),
        dict(input_images=input_images, input_states=input_states,),
        Z=Z_rep.clone(),
    )

    car_sizes_temp = (
        car_sizes.unsqueeze(0)
        .expand(n_models, bsize, 2)
        .contiguous()
        .view(n_models * bsize, 2)
    )
    costs = compute_state_costs(
        predictions["pred_images"],
        predictions["pred_states"],
        car_sizes_temp,
        data_stats,
    )

    pred_costs = (
        cost_config.lambda_p * costs["proximity_cost"]
        + cost_config.lambda_l * costs["lane_cost"]
        + cost_config.lambda_o * costs["offroad_cost"]
    )

    pred_images = predictions["pred_images"].view(n_models, bsize, npred, -1)
    pred_states = predictions["pred_states"].view(n_models, bsize, npred, -1)
    pred_costs = pred_costs.view(n_models, bsize, npred, -1)
    # use variance rather than standard deviation, since it is not
    # differentiable at 0 due to sqrt
    pred_images_var = torch.var(pred_images, 0).mean(2)
    pred_states_var = torch.var(pred_states, 0).mean(2)
    pred_costs_var = torch.var(pred_costs, 0).mean(2)
    pred_costs_mean = torch.mean(pred_costs, 0)
    pred_images = pred_images.view(
        n_models * bsize, npred, channels, height, width
    )
    pred_states = pred_states.view(n_models * bsize, npred, 4)

    if not estimation:
        # This is the uncertainty loss of different terms together.
        u_loss_costs = torch.relu(
            (pred_costs_var - model.u_costs_mean) / model.u_costs_std
            - cost_config.u_hinge
        )
        u_loss_states = torch.relu(
            (pred_states_var - model.u_states_mean) / model.u_states_std
            - cost_config.u_hinge
        )
        u_loss_images = torch.relu(
            (pred_images_var - model.u_images_mean) / model.u_images_std
            - cost_config.u_hinge
        )
        total_u_loss = (
            u_loss_costs.mean() + u_loss_states.mean() + u_loss_images.mean()
        )
    else:
        total_u_loss = None
        # We disabled gradients earlier for the estimation case.
        torch.set_grad_enabled(True)

    return dict(
        pred_images_var=pred_images_var,
        pred_states_var=pred_states_var,
        pred_costs_var=pred_costs_var,
        pred_costs_mean=pred_costs_mean,
        total_u_loss=total_u_loss,
    )


def estimate_uncertainty_stats(
    model, dataloader, cost_config, data_stats, n_batches=100, npred=200
):
    """Computes uncertainty estimates for the ground truth actions in the training
    set. This will give us an idea of what normal ranges are using actions the
    forward model was trained on
    """
    u_images, u_states, u_costs = [], [], []
    data_iter = iter(dataloader)
    for i in range(n_batches):
        print(
            f"[estimating normal uncertainty ranges: {i / n_batches:2.1%}]",
            end="\r",
        )
        batch = next(data_iter)
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(model.device)
        result = compute_uncertainty_batch(
            model=model,
            batch=batch,
            cost_config=cost_config,
            data_stats=data_stats,
            npred=npred,
            n_models=10,
            estimation=True,
        )
        u_images.append(result["pred_images_var"])
        u_states.append(result["pred_states_var"])
        u_costs.append(result["pred_costs_var"])

    print("[estimating normal uncertainty ranges: 100.0%]")

    u_images = torch.stack(u_images).view(-1, npred)
    u_states = torch.stack(u_states).view(-1, npred)
    u_costs = torch.stack(u_costs).view(-1, npred)

    model.u_images_mean = u_images.mean(0)
    model.u_states_mean = u_states.mean(0)
    model.u_costs_mean = u_costs.mean(0)

    model.u_images_std = u_images.std(0)
    model.u_states_std = u_states.std(0)
    model.u_costs_std = u_costs.std(0)


def policy_unfold(
    forward_model,
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
        # If the source of actions is a policy and not a tensor array, we need
        # a version of the state with ego car on the 4th channel.
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
        Z = forward_model.sample_z(npred * bsize)
        Z = Z.view(bsize, npred, -1)

    for t in range(npred):
        if torch.is_tensor(actions_or_policy):
            actions = actions_or_policy[:, t]
        else:
            actions = actions_or_policy(input_images_with_ego, input_states)

        z_t = Z[:, t]
        pred_image, pred_state = forward_model.forward_single_step(
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


def compute_state_costs(pred_images, pred_states, car_sizes, stats):
    npred = pred_images.size(1)
    gamma_mask = (
        torch.tensor([0.99 ** t for t in range(npred + 1)]).cuda().unsqueeze(0)
    )
    proximity_cost, _ = compute_proximity_cost(
        pred_images,
        pred_states.data,
        car_sizes,
        unnormalize=True,
        s_mean=stats["s_mean"],
        s_std=stats["s_std"],
    )
    lane_cost, prox_map_l = compute_lane_cost(pred_images, car_sizes)
    offroad_cost = compute_offroad_cost(pred_images, prox_map_l)

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


def compute_combined_loss(
    cost_config,
    proximity_loss=0,
    uncertainty_loss=0,
    lane_loss=0,
    action_loss=0,
    offroad_loss=0,
):
    return cost_config.lambda_p * proximity_loss
    +cost_config.u_reg * uncertainty_loss
    +cost_config.lambda_l * lane_loss
    +cost_config.lambda_a * action_loss
    +cost_config.lambda_o * offroad_loss
