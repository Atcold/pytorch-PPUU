import torch
import torch.optim as optim
import numpy
import utils

##################################################################################
# functions for planning and training policy networks using the forward model
##################################################################################

# estimate prediction uncertainty using dropout
def compute_uncertainty_batch(model, input_images, input_states, actions, targets=None, car_sizes=None, npred=200,
                              n_models=10, Z=None, dirname=None, detach=True, compute_total_loss=False):
    """
    Compute variance over n_models prediction per input + action

    :param model: predictive model
    :param input_images: input context states (traffic + lanes)
    :param input_states: input states (position + velocity)
    :param actions: expert / policy actions (longitudinal + transverse acceleration)
    :param npred: number of future predictions
    :param n_models: number of predictions per given input + action
    :param Z: predictive model latent samples
    :param detach: do not retain computational graph
    :param compute_total_loss: return overall loss
    :return:
    """

    bsize = input_images.size(0)
    if Z is None:
        Z = model.sample_z(bsize * npred, method='fp')
        if type(Z) is list: Z = Z[0]
        Z = Z.view(bsize, npred, -1)

    input_images = input_images.unsqueeze(0)
    input_states = input_states.unsqueeze(0)
    actions      = actions.     unsqueeze(0)
    Z_rep        = Z.           unsqueeze(0)
    input_images = input_images.expand(n_models, bsize, model.opt.ncond, 3, model.opt.height, model.opt.width)
    input_states = input_states.expand(n_models, bsize, model.opt.ncond, 4)
    actions      = actions.     expand(n_models, bsize, npred, 2)
    Z_rep        = Z_rep.       expand(n_models, bsize, npred, -1)
    input_images = input_images.contiguous()
    input_states = input_states.contiguous()
    actions      = actions.     contiguous()
    Z_rep        = Z_rep.       contiguous()
    input_images = input_images.view(bsize * n_models, model.opt.ncond, 3, model.opt.height, model.opt.width)
    input_states = input_states.view(bsize * n_models, model.opt.ncond, 4)
    actions      = actions.     view(bsize * n_models, npred, 2)
    Z_rep        = Z_rep.       view(n_models * bsize, npred, -1)

    model.train()  # turn on dropout, for uncertainty estimation
    pred_images, pred_states = [], []
    for t in range(npred):
        z = Z_rep[:, t]
        pred_image, pred_state = model.forward_single_step(input_images, input_states, actions[:, t], z)
        if detach:
            pred_image.detach_()
            pred_state.detach_()

        input_images = torch.cat((input_images[:, 1:], pred_image), 1)
        input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
        pred_images.append(pred_image)
        pred_states.append(pred_state)

    if npred > 1:
        pred_images = torch.stack(pred_images, 1).squeeze()
        pred_states = torch.stack(pred_states, 1).squeeze()
    else:
        pred_images = torch.stack(pred_images, 1)[:, 0]
        pred_states = torch.stack(pred_states, 1)[:, 0]

    if hasattr(model, 'cost'):
        pred_costs = model.cost(pred_images.view(-1, 3, 117, 24), pred_states.data.view(-1, 4))
        pred_costs = pred_costs.view(n_models, bsize, npred, 2)
        pred_costs = pred_costs[:, :, :, 0] + model.opt.lambda_l * pred_costs[:, :, :, 1]
        if detach:
            pred_costs.detach_()
    else:
        # ipdb.set_trace()
        car_sizes_temp = car_sizes.unsqueeze(0).expand(n_models, bsize, 2).contiguous().view(n_models * bsize, 2)
        pred_costs, _ = utils.proximity_cost(
            pred_images, pred_states.data,
            car_sizes_temp,
            unnormalize=True, s_mean=model.stats['s_mean'], s_std=model.stats['s_std']
        )
        lane_cost, prox_map_l = utils.lane_cost(pred_images, car_sizes_temp)
        offroad_cost = utils.offroad_cost(pred_images, prox_map_l)
        pred_costs += model.opt.lambda_l * lane_cost + model.opt.lambda_o * offroad_cost

    pred_images = pred_images.view(n_models, bsize, npred, -1)
    pred_states = pred_states.view(n_models, bsize, npred, -1)
    pred_costs  = pred_costs. view(n_models, bsize, npred, -1)
    # use variance rather than standard deviation, since it is not differentiable at 0 due to sqrt
    pred_images_var = torch.var(pred_images, 0).mean(2)
    pred_states_var = torch.var(pred_states, 0).mean(2)
    pred_costs_var  = torch.var(pred_costs,  0).mean(2)
    pred_costs_mean = torch.mean(pred_costs, 0)
    pred_images = pred_images.view(n_models * bsize, npred, 3, model.opt.height, model.opt.width)
    pred_states = pred_states.view(n_models * bsize, npred, 4)

    if hasattr(model, 'value_function'):
        pred_v = model.value_function(pred_images[:, -model.value_function.opt.ncond:],
                                      pred_states[:, -model.value_function.opt.ncond:].data)
        if detach:
            pred_v.detach_()
        pred_v = pred_v.view(n_models, bsize)
        pred_v_var = torch.var(pred_v, 0).mean()
        pred_v_mean = torch.mean(pred_v, 0)
    else:
        pred_v_mean = torch.zeros(bsize).cuda()
        pred_v_var = torch.zeros(bsize).cuda()

    if compute_total_loss:
        # this is the uncertainty loss of different terms together. We don't include the uncertainty
        # of the value function, it's normal to have high uncertainty there.
        u_loss_costs  = torch.relu((pred_costs_var  - model.u_costs_mean)  / model.u_costs_std  - model.opt.u_hinge)
        u_loss_states = torch.relu((pred_states_var - model.u_states_mean) / model.u_states_std - model.opt.u_hinge)
        u_loss_images = torch.relu((pred_images_var - model.u_images_mean) / model.u_images_std - model.opt.u_hinge)
        total_u_loss = u_loss_costs.mean() + u_loss_states.mean() + u_loss_images.mean()
    else:
        total_u_loss = None

    return pred_images_var, pred_states_var, pred_costs_var, pred_v_var, pred_costs_mean, pred_v_mean, total_u_loss


# compute uncertainty estimates for the ground truth actions in the training set.
# this will give us an idea of what normal ranges are using actions the forward model
# was trained on
def estimate_uncertainty_stats(model, dataloader, n_batches=100, npred=200):
    u_images, u_states, u_costs, u_values, speeds = [], [], [], [], []
    data_bsize = dataloader.opt.batch_size
    dataloader.opt.batch_size = 8
    for i in range(n_batches):
        print(f'[estimating normal uncertainty ranges: {i / n_batches:2.1%}]', end='\r')
        inputs, actions, targets, ids, car_sizes = dataloader.get_batch_fm('train', npred)
        pred_images_var, pred_states_var, pred_costs_var, pred_v_var, _, _, _ = compute_uncertainty_batch(
            model=model,
            input_images=inputs[0],
            input_states=inputs[1],
            actions=actions,
            npred=npred,
            n_models=10,
            detach=True,
            car_sizes=car_sizes
        )
        u_images.append(pred_images_var)
        u_states.append(pred_states_var)
        u_costs.append(pred_costs_var)
        u_values.append(pred_v_var)
        # speeds.append(inputs[1][:, :, 2:].norm(2, 2))

    print('[estimating normal uncertainty ranges: 100.0%]')  # done :)

    u_images = torch.stack(u_images).view(-1, npred)
    u_states = torch.stack(u_states).view(-1, npred)
    u_costs  = torch.stack(u_costs). view(-1, npred)
    u_values = torch.stack(u_values)
    # speeds = torch.stack(speeds)

    model.u_images_mean = u_images.mean(0)
    model.u_states_mean = u_states.mean(0)
    model.u_costs_mean  = u_costs. mean(0)
    model.u_values_mean = u_values.mean()

    model.u_images_std = u_images.std(0)
    model.u_states_std = u_states.std(0)
    model.u_costs_std  = u_costs. std(0)
    model.u_values_std = u_values.std()
    dataloader.opt.batch_size = data_bsize


def plan_actions_backprop(model, input_images, input_states, car_sizes, npred=50, n_futures=5, normalize=True,
                          bprop_niter=5, bprop_lrt=1.0, u_reg=0.0, actions=None, use_action_buffer=True, n_models=10,
                          save_opt_stats=True, nexec=1, lambda_l=0.0, lambda_o=0.0):
    if use_action_buffer:
        actions = torch.cat((model.actions_buffer[nexec:, :], torch.zeros(nexec, model.opt.n_actions).cuda()), 0).cuda()
    elif actions is None:
        actions = torch.zeros(npred, model.opt.n_actions).cuda()

    model.encoder.n_channels = 3

    if normalize:
        input_images = input_images.clone().float().div_(255.0)
        input_states -= model.stats['s_mean'].view(1, 4).expand(input_states.size())
        input_states /= model.stats['s_std'].view(1, 4).expand(input_states.size())
        input_images = input_images.cuda().unsqueeze(0)
        input_states = input_states.cuda().unsqueeze(0)

    input_images = input_images.expand(n_futures, model.opt.ncond, 3, model.opt.height, model.opt.width)
    input_states = input_states.expand(n_futures, model.opt.ncond, 4)
    input_images = input_images.contiguous().view(n_futures, model.opt.ncond, 3, model.opt.height, model.opt.width)
    input_states = input_states.contiguous().view(n_futures, model.opt.ncond, 4)

    Z = model.sample_z(n_futures * npred, method='fp')
    if type(Z) is list: Z = Z[0]
    Z = Z.view(npred, n_futures, -1)
    Z0 = Z.clone()

    actions.requires_grad = True
    optimizer_a = optim.Adam([actions], bprop_lrt)
    actions_rep = actions.unsqueeze(0).expand(n_futures, npred, model.opt.n_actions)

    if (model.optimizer_a_stats is not None) and save_opt_stats:
        print('loading opt stats')
        optimizer_a.load_state_dict(model.optimizer_a_stats)

    gamma_mask = torch.from_numpy(
        numpy.array([0.99 ** t for t in range(npred + 1)])
    ).float().cuda().unsqueeze(0).expand(n_futures, npred + 1)

    for i in range(bprop_niter):
        optimizer_a.zero_grad()
        model.zero_grad()

        # first calculate proximity cost. Don't use dropout for this, it makes optimization difficult.
        model.eval()
        pred, _ = model.forward([input_images, input_states], actions_rep, None, sampling='fp', z_seq=Z)
        pred_images, pred_states = pred[0], pred[1]
        proximity_cost, _ = utils.proximity_cost(
            pred_images, pred_states.data, car_sizes.expand(n_futures, 2),
            unnormalize=True, s_mean=model.stats['s_mean'], s_std=model.stats['s_std']
        )

        if hasattr(model, 'value_function'):
            v = model.value_function(pred[0][:, -model.value_function.opt.ncond:].contiguous(),
                                     pred[1][:, -model.value_function.opt.ncond:].contiguous().data)
        else:
            v = torch.zeros(n_futures, 1).cuda()
        proximity_loss = torch.mean(torch.cat((proximity_cost, v), 1) * gamma_mask)
        loss = proximity_loss

        if u_reg > 0.0:
            model.train()
            _, _, _, _, _, _, uncertainty_loss = compute_uncertainty_batch(
                model, input_images, input_states, actions_rep, None, car_sizes, npred=npred, n_models=n_models,
                Z=Z.permute(1, 0, 2).clone(), detach=False, compute_total_loss=True
            )
            loss = loss + u_reg * uncertainty_loss
        else:
            uncertainty_loss = torch.zeros(1)

        lane_loss, prox_map_l = utils.lane_cost(pred_images, car_sizes.expand(n_futures, 2))
        lane_loss = torch.mean(lane_loss * gamma_mask[:, :npred])
        offroad_loss = torch.mean(utils.offroad_cost(pred_images, prox_map_l) * gamma_mask[:, :npred])
        # lane_loss = torch.mean(pred[2][:, :, 1] * gamma_mask[:, :npred])
        # lane_loss = torch.mean(pred[2][:, :, 1] * gamma_mask[:, :npred])
        loss = loss + lambda_l * lane_loss + lambda_o * offroad_loss
        loss.backward()
        print('[iter {} | mean pred cost = {:.4f}, uncertainty = {:.4f}, grad = {}'.format(
            i, proximity_loss.item(), uncertainty_loss.item(), actions.grad.data.norm())
        )
        torch.nn.utils.clip_grad_norm([actions], 1)
        optimizer_a.step()

    model.optimizer_a_stats = optimizer_a.state_dict()
    if use_action_buffer:
        model.actions_buffer = actions.data.clone()

    a = actions.data.view(npred, 2)

    if normalize:
        a.clamp_(-3, 3)
        a *= model.stats['a_std'].view(1, 2).expand(a.size()).cuda()
        a += model.stats['a_mean'].view(1, 2).expand(a.size()).cuda()
    return a.cpu().numpy()


def train_policy_net_mpur(model, inputs, targets, car_sizes, n_models=10, sampling_method='fp', lrt_z=0.1,
                          n_updates_z=10, infer_z=False):
    input_images_orig, input_states_orig, input_ego_car_orig = inputs
    target_images, target_states, target_costs = targets
    ego_car_new_shape = [*input_images_orig.shape]
    ego_car_new_shape[2] = 1
    input_ego_car = input_ego_car_orig[:, 2][:, None, None].expand(ego_car_new_shape)

    input_images = torch.cat((input_images_orig, input_ego_car), dim=2)
    input_states = input_states_orig.clone()
    bsize = input_images.size(0)
    npred = target_images.size(1)
    pred_images, pred_states, pred_costs, pred_actions = [], [], [], []

    # total_ploss = torch.zeros(1).cuda()
    # Sample latent variables from a (fixed) prior
    Z = model.sample_z(npred * bsize, method=sampling_method)
    if type(Z) is list: Z = Z[0]
    Z = Z.view(npred, bsize, -1)
    # get initial action sequence, for an episode long npred (= 20) steps
    model.eval()
    for t in range(npred):
        actions, _, _, _ = model.policy_net(input_images, input_states)
        if infer_z:
            h_x = model.encoder(input_images, input_states)
            h_y = model.y_encoder(target_images[:, t].unsqueeze(1).contiguous())
            mu_logvar = model.z_network((h_x + h_y).view(bsize, -1)).view(bsize, 2, model.opt.nz)
            mu = mu_logvar[:, 0]
            logvar = mu_logvar[:, 1]
            z_t = model.reparameterize(mu, logvar, True)
        else:
            z_t = Z[t]
        pred_image, pred_state = model.forward_single_step(input_images[:, :, :3].contiguous(), input_states, actions, z_t)
        # Auto regress: enqueue output as new element of the input
        pred_image = torch.cat((pred_image, input_ego_car[:, :1]), dim=2)
        input_images = torch.cat((input_images[:, 1:], pred_image), 1)
        input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
        pred_images.append(pred_image)
        pred_states.append(pred_state)
        pred_actions.append(actions)

    pred_images = torch.cat(pred_images, 1)
    pred_states = torch.stack(pred_states, 1)
    pred_actions = torch.stack(pred_actions, 1)

    input_images = input_images_orig.clone()
    input_states = input_states_orig.clone()
    if n_updates_z > 0:
        Z_adv = Z.data.clone()
        # optimize z vectors to be more difficult
        # pred_actions = pred_actions.data.clone()
        Z_adv.requires_grad = True
        optimizer_z = optim.Adam([Z_adv], lrt_z)
        for k in range(n_updates_z + 1):
            optimizer_z.zero_grad()
            pred, _ = model.forward([input_images, input_states], pred_actions, None, save_z=False,
                                    z_dropout=0.0, z_seq=Z_adv, sampling='fixed')
            pred_cost_adv, _ = utils.proximity_cost(pred[0], pred[1].data, car_sizes, unnormalize=True,
                                                    s_mean=model.stats['s_mean'], s_std=model.stats['s_std'])

            if k < n_updates_z + 1:
                _, _, _, _, _, _, total_u_loss = compute_uncertainty_batch(
                    model, input_images, input_states, pred_actions, targets, car_sizes, npred=npred, n_models=n_models,
                    detach=False, Z=Z_adv.permute(1, 0, 2), compute_total_loss=True
                )

                loss_z = -pred_cost_adv.mean()  # + total_u_loss.mean()
                loss_z.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_([Z_adv], 1)
                optimizer_z.step()
                # print(f'[z opt | iter: {k} | pred cost: {pred_cost_adv.mean().item()}]')
                print(f'[z opt | iter: {k} | pred cost: {pred_cost_adv.mean().item()} | u_cost: {total_u_loss.mean().item()}]')

    gamma_mask = torch.tensor([0.99 ** t for t in range(npred + 1)]).cuda().unsqueeze(0)
    if not hasattr(model, 'cost'):
        # ipdb.set_trace()
        proximity_cost, _ = utils.proximity_cost(pred_images[:, :, :3].contiguous(), pred_states.data, car_sizes, unnormalize=True,
                                                 s_mean=model.stats['s_mean'], s_std=model.stats['s_std'])
        if n_updates_z > 0:
            proximity_cost = 0.5 * proximity_cost + 0.5 * pred_cost_adv.squeeze()
        lane_cost, prox_map_l = utils.lane_cost(pred_images[:, :, :3].contiguous(), car_sizes)
        offroad_cost = utils.offroad_cost(pred_images[:, :, :3].contiguous(), prox_map_l)
        if hasattr(model, 'value_function'):
            v = model.value_function(pred_images[:, -model.value_function.opt.ncond:, :3].contiguous(),
                                     pred_states[:, -model.value_function.opt.ncond:].contiguous().data)
        else:
            v = torch.zeros(bsize, 1).cuda()
    else:
        pred_costs = model.cost(pred_images[:, :, :3].contiguous().view(-1, 3, 117, 24), pred_states.data.view(-1, 4))
        pred_costs = pred_costs.view(bsize, npred, 2)
        proximity_cost = pred_costs[:, :, 0]
        lane_cost = pred_costs[:, :, 1]

    if hasattr(model, 'value_function'):
        proximity_loss = torch.mean(torch.cat((proximity_cost, v), 1) * gamma_mask)
        lane_loss = torch.mean(lane_cost * gamma_mask[:, :npred])
    else:
        lane_loss = torch.mean(lane_cost * gamma_mask[:, :npred])
        offroad_cost = torch.mean(offroad_cost * gamma_mask[:, :npred])
        proximity_loss = torch.mean(proximity_cost * gamma_mask[:, :npred])

    _, _, _, _, _, _, total_u_loss = compute_uncertainty_batch(
        model, input_images, input_states, pred_actions, targets, car_sizes, npred=npred, n_models=n_models,
        detach=False, Z=Z.permute(1, 0, 2), compute_total_loss=True
    )

    loss_a = pred_actions.norm(2, 2).pow(2).mean()

    pred_images = pred_images[:, :, :3]
    predictions = dict(
        state_img=(pred_images + input_ego_car_orig[:, None].expand_as(pred_images)).clamp(max=1.),
        state_vct=pred_states,
        proximity=proximity_loss,
        lane=lane_loss,
        offroad=offroad_cost,
        uncertainty=total_u_loss,
        action=loss_a,
    )

    return predictions, pred_actions


def get_grad_vid(model, input_images, input_states, car_sizes, device='cuda'):
    input_images, input_states = input_images.clone(), input_states.clone()
    input_images, input_states = utils.normalize_inputs(
        input_images, input_states, model.policy_net.stats, device=device)
    input_images.requires_grad = True
    input_states.requires_grad = True
    input_images.retain_grad()
    input_states.retain_grad()

    proximity_cost, _ = utils.proximity_cost(
        input_images[:, -1:], input_states.data[:, -1:], car_sizes, unnormalize=True,
        s_mean=model.stats['s_mean'], s_std=model.stats['s_std'])
    proximity_loss = torch.mean(proximity_cost)
    lane_cost, _ = utils.lane_cost(input_images[:, -1:], car_sizes)
    lane_loss = torch.mean(lane_cost)

    opt = model.policy_net.options
    loss = proximity_loss + \
           opt.lambda_l * lane_loss
    loss.backward()

    return input_images.grad[:, -1, :3].abs().clamp(max=1.)


def train_policy_net_mper(model, inputs, targets, targetprop=0, dropout=0.0, n_models=10, model_type='vae'):
    input_images, input_states = inputs
    target_images, target_states, target_costs = targets
    bsize = input_images.size(0)
    npred = target_images.size(1)
    pred_images, pred_states, pred_costs, pred_actions = [], [], [], []

    z = None
    total_ploss = torch.zeros(1).cuda()
    z_list = []
    for t in range(npred):
        actions, _, _, _ = model.policy_net(input_images, input_states)
        # encode the inputs
        h_x = model.encoder(input_images, input_states)
        if model_type == 'ten' or model_type == 'vae':
            # encode the targets into z
            h_y = model.y_encoder(target_images[:, t].unsqueeze(1).contiguous())
            if model_type == 'ten':
                z = model.z_network((h_x + h_y).view(bsize, -1))
            elif model_type == 'vae':
                mu_logvar = model.z_network((h_x + h_y).view(bsize, -1)).view(bsize, 2, model.opt.nz)
                mu = mu_logvar[:, 0]
                logvar = mu_logvar[:, 1]
                z = model.reparameterize(mu, logvar, True)
            z_ = z
            z_list.append(z_)
            z_exp = model.z_expander(z_).view(bsize, model.opt.nfeature, model.opt.h_height, model.opt.h_width)
            h_x = h_x.view(bsize, model.opt.nfeature, model.opt.h_height, model.opt.h_width)
            h = h_x + z_exp
        else:
            h = h_x
        a_emb = model.a_encoder(actions).view(h_x.size())
        h = h + a_emb
        h = h + model.u_network(h)
        pred_image, pred_state = model.decoder(h)
        pred_image = torch.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
        # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
        pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
        input_images = torch.cat((input_images[:, 1:], pred_image), 1)
        input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
        pred_images.append(pred_image)
        pred_states.append(pred_state)
        pred_actions.append(actions)

    pred_images = torch.cat(pred_images, 1)
    pred_states = torch.stack(pred_states, 1)
    pred_actions = torch.stack(pred_actions, 1)

    return [pred_images, pred_states, None, total_ploss], pred_actions
