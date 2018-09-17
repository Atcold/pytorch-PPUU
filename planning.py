import torch
import torch.nn as nn
import torch.legacy.nn as lnn
import torch.nn.functional as F
import torch.optim as optim
#import torchvision
from torch.autograd import Variable
import random, pdb, copy, os, math, numpy, copy, time
import utils

##################################################################################
# functions for planning and training policy networks using the forward model
##################################################################################



# estimate prediction uncertainty using dropout
def compute_uncertainty_batch(model, input_images, input_states, actions, targets, car_sizes, npred=200, n_models=10, Z=None, dirname=None, detach=True, compute_total_loss = False):
    bsize = input_images.size(0)
    input_images = input_images.unsqueeze(0).expand(n_models, bsize, model.opt.ncond, 3, model.opt.height, model.opt.width)
    input_states = input_states.unsqueeze(0).expand(n_models, bsize, model.opt.ncond, 4)
    input_images = input_images.contiguous().view(bsize*n_models, model.opt.ncond, 3, model.opt.height, model.opt.width)
    input_states = input_states.contiguous().view(bsize*n_models, model.opt.ncond, 4)
    actions = actions.unsqueeze(0).expand(n_models, bsize, npred, 2).contiguous().view(bsize*n_models, npred, 2)
            
    model.train() 
    if Z is None:
        Z = model.sample_z(bsize * npred, method='fp')
        if type(Z) is list: Z = Z[0]
        Z = Z.view(bsize, npred, -1)
    Z_rep = Z.unsqueeze(0).expand(n_models, bsize, npred, -1).contiguous()
    Z_rep = Z_rep.view(n_models*bsize, npred, -1)

    pred_images, pred_states, pred_costs = [], [], []
    for t in range(npred):
        z = Z_rep[:, t]
        pred_image, pred_state = model.forward_single_step(input_images, input_states, actions[:, t], z)
        if detach:
            pred_image = Variable(pred_image.data)
            pred_state = Variable(pred_state.data)
            
        input_images = torch.cat((input_images[:, 1:], pred_image), 1)
        input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
        pred_images.append(pred_image)
        pred_states.append(pred_state)

    pred_images = torch.stack(pred_images, 1).squeeze()
    pred_states = torch.stack(pred_states, 1).squeeze()
        
    pred_costs, _ = utils.proximity_cost(pred_images, Variable(pred_states.data), car_sizes.unsqueeze(0).expand(n_models, bsize, 2).contiguous().view(n_models*bsize, 2), unnormalize=True, s_mean=model.stats['s_mean'], s_std=model.stats['s_std'])

    pred_images = pred_images.view(n_models, bsize, npred, -1)
    pred_states = pred_states.view(n_models, bsize, npred, -1)
    pred_costs = pred_costs.view(n_models, bsize, npred, -1)
    # use variance rather than standard deviation, since it is not differentiable at 0 due to sqrt
    pred_images_var = torch.var(pred_images, 0).mean(2)
    pred_states_var = torch.var(pred_states, 0).mean(2)
    pred_costs_var = torch.var(pred_costs, 0).mean(2)
    pred_costs_mean = torch.mean(pred_costs, 0)
    pred_images = pred_images.view(n_models*bsize, npred, 3, model.opt.height, model.opt.width)
    pred_states = pred_states.view(n_models*bsize, npred, 4)

    if hasattr(model, 'value_function'):
        pred_v = model.value_function(pred_images[:, -model.value_function.opt.ncond:], Variable(pred_states[:, -model.value_function.opt.ncond:].data))
        if detach:
            pred_v = Variable(pred_v.data)
        pred_v = pred_v.view(n_models, bsize)
        pred_v_var = torch.var(pred_v, 0).mean()
        pred_v_mean = torch.mean(pred_v, 0)
    else:
        pred_v_mean = Variable(torch.zeros(bsize).cuda())
        pred_v_var = Variable(torch.zeros(bsize).cuda())

    if compute_total_loss:
        # this is the uncertainty loss of different terms together. We don't include the uncertainty
        # of the value function, it's normal to have high uncertainty there.
        u_loss_costs = F.relu((pred_costs_var - model.u_costs_mean) / model.u_costs_std - model.opt.u_hinge)
        u_loss_states = F.relu((pred_states_var - model.u_states_mean) / model.u_states_std - model.opt.u_hinge)
        u_loss_images = F.relu((pred_images_var - model.u_images_mean) / model.u_images_std - model.opt.u_hinge)
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
        print('[estimating normal uncertainty ranges: {:2.1%}]'.format(float(i)/n_batches), end="\r")
        inputs, actions, targets, ids, car_sizes = dataloader.get_batch_fm('train', npred)
        pred_images_var, pred_states_var, pred_costs_var, pred_v_var, _, _, _ = compute_uncertainty_batch(model, inputs[0], inputs[1], actions, targets, car_sizes, npred=npred, n_models=10, detach=True)
        u_images.append(pred_images_var)
        u_states.append(pred_states_var)
        u_costs.append(pred_costs_var)
        u_values.append(pred_v_var)
        speeds.append(inputs[1][:, :, 2:].norm(2, 2))
        
    u_images = torch.stack(u_images).view(-1, npred)
    u_states = torch.stack(u_states).view(-1, npred)
    u_costs = torch.stack(u_costs).view(-1, npred)
    u_values = torch.stack(u_values)
    speeds = torch.stack(speeds)

    model.u_images_mean = u_images.mean(0)
    model.u_states_mean = u_states.mean(0)
    model.u_costs_mean = u_costs.mean(0)
    model.u_values_mean = u_values.mean()

    model.u_images_std = u_images.std(0)
    model.u_states_std = u_states.std(0)
    model.u_costs_std = u_costs.std(0)
    model.u_values_std = u_values.std()
    dataloader.opt.batch_size = data_bsize








def plan_actions_backprop(model, input_images, input_states, car_sizes, npred=50, n_futures=5, normalize=True, bprop_niter=5, bprop_lrt=1.0, u_reg=0.0, actions=None, use_action_buffer=True, n_models=10, save_opt_stats=True, nexec=1, lambda_l = 0.0):

    if use_action_buffer:
        actions = Variable(torch.cat((model.actions_buffer[nexec:, :], torch.zeros(nexec, model.opt.n_actions).cuda()), 0).cuda())
    elif actions is None:
        actions = Variable(torch.zeros(npred, model.opt.n_actions).cuda())

    if normalize:
        input_images = input_images.clone().float().div_(255.0)
        input_states -= model.stats['s_mean'].view(1, 4).expand(input_states.size())
        input_states /= model.stats['s_std'].view(1, 4).expand(input_states.size())
        input_images = Variable(input_images.cuda()).unsqueeze(0)
        input_states = Variable(input_states.cuda()).unsqueeze(0)
        
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

    gamma_mask = Variable(torch.from_numpy(numpy.array([0.99**t for t in range(npred + 1)])).float().cuda()).unsqueeze(0).expand(n_futures, npred + 1)

    for i in range(bprop_niter):
        optimizer_a.zero_grad()
        model.zero_grad()
            
        # first calculate proximity cost. Don't use dropout for this, it makes optimization difficult.
        model.eval()
        pred, _ = model.forward([input_images, input_states], actions_rep, None, sampling='fp', z_seq=Z)
        pred_images, pred_states = pred[0], pred[1]
        proximity_cost, _ = utils.proximity_cost(pred_images, Variable(pred_states.data), car_sizes.expand(n_futures, 2), unnormalize=True, s_mean=model.stats['s_mean'], s_std=model.stats['s_std'])


        if hasattr(model, 'value_function'):
            v = model.value_function(pred[0][:, -model.value_function.opt.ncond:].contiguous(), Variable(pred[1][:, -model.value_function.opt.ncond:].contiguous().data))
        else:
            v = Variable(torch.zeros(n_futures, 1).cuda())
        proximity_loss = torch.mean(torch.cat((proximity_cost, v), 1) * gamma_mask)
        loss = proximity_loss

        if u_reg > 0.0:
            model.train()
            _, _, _, _, _, _, uncertainty_loss = compute_uncertainty_batch(model, input_images, input_states, actions_rep, None, car_sizes, npred=npred, n_models=n_models, Z=Z.permute(1, 0, 2).clone(), detach=False, compute_total_loss=True)
            loss = loss + u_reg * uncertainty_loss
        else:
            uncertainty_loss = Variable(torch.zeros(1))
        
        lane_loss = torch.mean(utils.lane_cost(pred_images, car_sizes.expand(n_futures, 2)) * gamma_mask[:, :npred])
#        lane_loss = torch.mean(pred[2][:, :, 1] * gamma_mask[:, :npred])
#        lane_loss = torch.mean(pred[2][:, :, 1] * gamma_mask[:, :npred])
        loss = loss + lambda_l * lane_loss
        loss.backward()
        print('[iter {} | mean pred cost = {:.4f}, uncertainty = {:.4f}, grad = {}'.format(i, proximity_loss.item(), uncertainty_loss.item(), actions.grad.data.norm())) 
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









def train_policy_net_svg(model, inputs, targets, car_sizes, n_models=10, sampling_method='fp', lrt_z=0.1, n_updates_z = 10):
    
    input_images_orig, input_states_orig = inputs
    target_images, target_states, target_costs = targets
    
    input_images = input_images_orig.clone()
    input_states = input_states_orig.clone()
    bsize = input_images.size(0)
    npred = target_images.size(1)
    pred_images, pred_states, pred_costs, pred_actions = [], [], [], []
    pred_images_adv, pred_states_adv, pred_costs_adv, pred_actions_adv = None, None, None, None
    
    total_ploss = Variable(torch.zeros(1).cuda())
    # sample futures
    Z = model.sample_z(npred * bsize, method=sampling_method)
    if type(Z) is list: Z = Z[0]
    Z = Z.view(npred, bsize, model.opt.nz)
    # get initial action sequence
    model.eval()
    for t in range(npred):
        actions, _, _, _ = model.policy_net(input_images, input_states)
        pred_image, pred_state = model.forward_single_step(input_images, input_states, actions, Z[t])
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
        # optimize z vectors to be more difficult
        pred_actions = Variable(pred_actions.data.clone())
        Z.requires_grad = True
        optimizer_z = optim.Adam([Z], lrt_z)
        for k in range(n_updates_z):
            optimizer_z.zero_grad()
            pred, _ = model.forward([input_images, input_states], pred_actions, None, save_z=False, z_dropout=0.0, z_seq=Z, sampling='fixed')
            pred_cost = pred[2][:, :, 0].mean()
            _, _, _, _, _, _, total_u_loss = compute_uncertainty_batch(model, input_images, input_states, pred_actions, targets, car_sizes, npred=npred, n_models=n_models, detach=False, Z=Z.permute(1, 0, 2), compute_total_loss=True)
            loss_z = -pred_cost + total_u_loss.mean()
            loss_z.backward()
            torch.nn.utils.clip_grad_norm_([Z], 1)
            optimizer_z.step()
            print(f'[z opt | iter: {k} | pred cost: {pred_cost.item()}]')

        pred_images_adv, pred_states_adv, pred_costs_adv, pred_actions_adv = [], [], [], []
        for t in range(npred):
            actions, _, _, _ = model.policy_net(input_images, input_states)
            pred_image, pred_state = model.forward_single_step(input_images, input_states, actions, Z[t])
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images_adv.append(pred_image)
            pred_states_adv.append(pred_state)
            pred_costs_adv.append(pred_cost)
            pred_actions_adv.append(actions)

        pred_images_adv = torch.cat(pred_images_adv, 1)
        pred_states_adv = torch.stack(pred_states_adv, 1)
        pred_costs_adv = torch.stack(pred_costs_adv, 1)
        pred_actions_adv = torch.stack(pred_actions_adv, 1)
        # use the adversarial states
        pred_images = pred_images_adv
        pred_states = pred_states_adv


    proximity_cost, _ = utils.proximity_cost(pred_images, Variable(pred_states.data), car_sizes, unnormalize=True, s_mean=model.stats['s_mean'], s_std=model.stats['s_std'])
    if hasattr(model, 'value_function'):
        v = model.value_function(pred_images[:, -model.value_function.opt.ncond:].contiguous(), Variable(pred_states[:, -model.value_function.opt.ncond:].contiguous().data))
    else:
        v = Variable(torch.zeros(bsize, 1).cuda())
    gamma_mask = Variable(torch.from_numpy(numpy.array([0.99**t for t in range(npred+1)])).float().cuda()).unsqueeze(0)
    proximity_loss = torch.mean(torch.cat((proximity_cost, v), 1) * gamma_mask)
    lane_loss = torch.mean(utils.lane_cost(pred_images, car_sizes) * gamma_mask[:, :npred])

#    lane_loss = torch.mean(pred_costs[:, :, 1] * gamma_mask[:, :npred])

    _, _, _, _, _, _, total_u_loss = compute_uncertainty_batch(model, input_images, input_states, pred_actions, targets, car_sizes, npred=npred, n_models=n_models, detach=False, Z=Z.permute(1, 0, 2), compute_total_loss=True)

    return [pred_images, pred_states, proximity_loss, lane_loss, total_u_loss], pred_actions, [pred_images_adv, pred_states_adv, pred_costs_adv]



def train_policy_net_mbil(model, inputs, targets, targetprop=0, dropout=0.0, n_models=10, model_type = 'ten'):
    input_images, input_states = inputs
    target_images, target_states, target_costs = targets
    bsize = input_images.size(0)
    npred = target_images.size(1)
    pred_images, pred_states, pred_costs, pred_actions = [], [], [], []

    z = None
    total_ploss = Variable(torch.zeros(1).cuda())
    z_list = []
    for t in range(npred):
        actions, _, _, _ = model.policy_net(input_images, input_states)
        # encode the inputs
        h_x = model.encoder(input_images, input_states)
        # encode the targets into z
        h_y = model.y_encoder(target_images[:, t].unsqueeze(1).contiguous())
        if model_type == 'ten':
            z = model.z_network(utils.combine(h_x, h_y, model.opt.combine).view(bsize, -1))
        elif model_type == 'vae':
            mu_logvar = model.z_network(utils.combine(h_x, h_y, model.opt.combine).view(bsize, -1)).view(bsize, 2, model.opt.nz)
            mu = mu_logvar[:, 0]
            logvar = mu_logvar[:, 1]
            z = model.reparameterize(mu, logvar, True)
        z_ = z
        z_list.append(z_)
        z_exp = model.z_expander(z_).view(bsize, model.opt.nfeature, model.opt.h_height, model.opt.h_width)
        h_x = h_x.view(bsize, model.opt.nfeature, model.opt.h_height, model.opt.h_width)
        a_emb = model.a_encoder(actions).view(h_x.size())

        if model.opt.zmult == 0:
            h = utils.combine(h_x, z_exp, model.opt.combine)
            h = utils.combine(h, a_emb, model.opt.combine)
        elif model.opt.zmult == 1:
            a_emb = torch.sigmoid(a_emb)
            z_exp = torch.sigmoid(z_exp)
            h = h_x + a_emb + (1-a_emb) * z_exp
        if not model.disable_unet:
            h = h + model.u_network(h)
        pred_image, pred_state, pred_cost = model.decoder(h)
        pred_image = torch.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
        # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
        pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
        input_images = torch.cat((input_images[:, 1:], pred_image), 1)
        input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
        pred_images.append(pred_image)
        pred_states.append(pred_state)
        pred_costs.append(pred_cost)
        pred_actions.append(actions)

    pred_images = torch.cat(pred_images, 1)
    pred_states = torch.stack(pred_states, 1)
    pred_costs = torch.stack(pred_costs, 1)
    pred_actions = torch.stack(pred_actions, 1)

    return [pred_images, pred_states, pred_costs, total_ploss], pred_actions
