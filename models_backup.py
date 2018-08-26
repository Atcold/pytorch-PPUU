# forward model, deterministic (compatible with TEN3 model, use to initialize)
class FwdCNN3_STN(nn.Module):
    def __init__(self, opt, mfile):
        super(FwdCNN3_STN, self).__init__()
        self.opt = opt
        # If we are given a model file, use it to initialize this model. 
        # otherwise initialize from scratch
        if mfile == '':
            self.encoder = encoder(opt, 0, opt.ncond)
            self.decoder = decoder_stn(opt)
        else:
            print('[initializing encoder and decoder with: {}]'.format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)['model']
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.a_encoder = pretrained_model.a_encoder
            self.u_network = pretrained_model.u_network
            self.encoder.n_inputs = opt.ncond


        self.a_encoder = nn.Sequential(
            nn.Linear(self.opt.n_actions, self.opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.opt.hidden_size)
        )

        self.u_network = u_network(opt)


    def forward(self, inputs, actions, target, sampling=None, z_dropout=None):
        npred = actions.size(1)
        input_images, input_states = inputs
        pred_images, pred_states, pred_costs = [], [], []
        for t in range(npred):
            h = self.encoder(input_images, input_states)
            a_emb = self.a_encoder(actions[:, t]).view(h.size())
            h = utils.combine(h, a_emb, self.opt.combine)
            h = h + self.u_network(h)

            pred_image, pred_state, pred_cost = self.decoder(h, actions[:, t], input_images[:, -1].unsqueeze(1))
            pred_image = F.sigmoid(pred_image)
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        return [pred_images, pred_states, pred_costs, None], Variable(torch.zeros(1).cuda())

    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()





class FwdCNN2(nn.Module):
    def __init__(self, opt, mfile):
        super(FwdCNN2, self).__init__()
        self.opt = opt
        # If we are given a model file, use it to initialize this model. 
        # otherwise initialize from scratch
        if mfile == '':
            self.encoder = encoder(opt, 0, opt.ncond)
            self.decoder = decoder(opt)
            self.a_network = nn.Sequential(
                nn.Linear(self.opt.n_actions, self.opt.nfeature), 
                nn.LeakyReLU(0.2, inplace=True), 
                nn.Linear(self.opt.nfeature, self.opt.nfeature), 
                nn.LeakyReLU(0.2, inplace=True), 
                nn.Linear(self.opt.nfeature, self.opt.hidden_size)
            )

            self.h_network = nn.Sequential(
                nn.Linear(self.opt.hidden_size, self.opt.n_hidden), 
                nn.LeakyReLU(0.2, inplace=True), 
                nn.Linear(self.opt.n_hidden, self.opt.n_hidden), 
                nn.LeakyReLU(0.2, inplace=True), 
                nn.Linear(self.opt.n_hidden, self.opt.n_hidden), 
                nn.LeakyReLU(0.2, inplace=True), 
                nn.Linear(self.opt.n_hidden, self.opt.hidden_size)
            )

        else:
            print('[initializing encoder and decoder with: {}]'.format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.encoder.n_inputs = opt.ncond

    def forward(self, inputs, actions, target, sampling=None, z_dropout=None):
        t0 = time.time()
        bsize = actions.size(0)
        npred = actions.size(1)
        input_images, input_states = inputs
        pred_images, pred_states, pred_costs = [], [], []
        h = self.encoder(input_images, input_states)
        h = h.view(bsize, self.opt.hidden_size)
        for t in range(npred):
            h = h + self.a_network(actions[:, t])
            h = h + self.h_network(h)
            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + pred_images_images[:, -1].unsqueeze(1))
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        print(time.time() - t0)
        return [pred_images, pred_states, pred_costs], Variable(torch.zeros(1).cuda())

    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()









# forward LSTM model, deterministic
class FwdLSTM(nn.Module):
    def __init__(self, opt, mfile):
        super(FwdCNN, self).__init__()
        self.opt = opt
        # If we are given a model file, use it to initialize this model. 
        # otherwise initialize from scratch
        if mfile == '':
            self.encoder = encoder(opt, opt.n_actions, 1)
            self.decoder = decoder(opt)
        else:
            print('[initializing encoder and decoder with: {}]'.format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.encoder.n_inputs = opt.ncond

    def forward(self, inputs, actions, target, sampling=None):
        npred = actions.size(1)
        input_images, input_states = inputs
        pred_images, pred_states, pred_costs = [], [], []
        for t in range(npred):
            h = self.encoder(input_images, input_states, actions[:, t])
            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        return [pred_images, pred_states, pred_costs], Variable(torch.zeros(1).cuda())

    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()
















# forward model, Mixture Density Network. (tried this for comparison, but it's really hard to train). 
class FwdCNN_MDN(nn.Module):
    def __init__(self, opt, mfile):
        super(FwdCNN_MDN, self).__init__()
        self.opt = opt
        self.encoder = encoder(opt, opt.n_actions, opt.ncond)
        self.decoder = decoder(opt, n_out = 2*opt.n_mixture)

        self.pi_network = nn.Sequential(
            nn.Linear(self.opt.hidden_size, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.n_mixture)
        )

    def forward(self, inputs, actions, targets, sampling=None):
        npred = actions.size(1)
        input_images, input_states = inputs
        bsize = actions.size(0)
        pred_images_mu, pred_states_mu, pred_costs_mu = [], [], []
        pred_images_sigma, pred_states_sigma, pred_costs_sigma = [], [], []
        latent_probs = []
        for t in range(npred):
            h = self.encoder(input_images, input_states, actions[:, t])
            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = pred_image.view(bsize, self.opt.n_mixture, 2, 3, self.opt.height, self.opt.width)
            pred_image_mu = pred_image[:, :, 0].contiguous().view(bsize, -1)
            pred_image_sigma = pred_image[:, :, 1].contiguous().view(bsize, -1)
            pred_state = pred_state.view(bsize, self.opt.n_mixture, 2, 4)
            pred_state_mu = pred_state[:, :, 0].contiguous().view(bsize, -1)
            pred_state_sigma = pred_state[:, :, 1].contiguous().view(bsize, -1)
            pred_cost = pred_cost.view(bsize, self.opt.n_mixture, 2, 2)
            pred_cost_mu = pred_cost[:, :, 0].contiguous().view(bsize, -1)
            pred_cost_sigma = pred_cost[:, :, 1].contiguous().view(bsize, -1)

            pi = F.softmax(self.pi_network(h.view(bsize, -1)), dim=1)
            pred_image_mu = F.sigmoid(pred_image_mu)
            pred_image_sigma = F.softplus(pred_image_sigma)
            pred_state_sigma = F.softplus(pred_state_sigma)
            pred_cost_sigma = F.softplus(pred_cost_sigma)

            if targets is not None:
                target_images, target_states, target_costs = targets
                input_images = torch.cat((input_images[:, 1:], target_images[:, t].unsqueeze(1)), 1)
                input_states = torch.cat((input_states[:, 1:], target_states[:, t].unsqueeze(1)), 1)
            else:
                # sample, TODO
                pdb.set_trace()

            pred_images_mu.append(pred_image_mu)
            pred_states_mu.append(pred_state_mu)
            pred_costs_mu.append(pred_cost_mu)
            pred_images_sigma.append(pred_image_sigma)
            pred_states_sigma.append(pred_state_sigma)
            pred_costs_sigma.append(pred_cost_sigma)
            latent_probs.append(pi)


        pred_images_mu = torch.stack(pred_images_mu, 1)
        pred_states_mu = torch.stack(pred_states_mu, 1)
        pred_costs_mu = torch.stack(pred_costs_mu, 1)
        pred_images_sigma = torch.stack(pred_images_sigma, 1)
        pred_states_sigma = torch.stack(pred_states_sigma, 1)
        pred_costs_sigma = torch.stack(pred_costs_sigma, 1)
        latent_probs = torch.stack(latent_probs, 1)

        pred_images_sigma = torch.clamp(pred_images_sigma, min=1e-1)
        pred_states_sigma = torch.clamp(pred_states_sigma, min=1e-5)
        pred_costs_sigma = torch.clamp(pred_costs_sigma, min=1e-5)

        return [[pred_images_mu, pred_images_sigma], [pred_states_mu, pred_states_sigma], [pred_costs_mu, pred_costs_sigma], latent_probs], Variable(torch.zeros(1).cuda())


    def intype(self, t):
        if t == 'gpu':
            self.cuda()
        elif t == 'cpu':
            self.cpu()






# forward TEN model (autoregressive)
class FwdCNN_TEN(nn.Module):
    def __init__(self, opt, mfile=''):
        super(FwdCNN_TEN, self).__init__()
        self.opt = opt

        if mfile == '':
            self.encoder = encoder(opt, opt.n_actions, opt.ncond)
            self.decoder = decoder(opt)
        else:
            print('[initializing encoder and decoder with: {}]'.format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.encoder.n_inputs = opt.ncond
            self.decoder.n_out = 1

        self.y_encoder = encoder(opt, 0, 1, states=False)

        self.z_network = nn.Sequential(
            nn.Linear(opt.hidden_size, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nz)
        )

        self.z_zero = Variable(torch.zeros(self.opt.batch_size, self.opt.nz))


        # if beta > 0, it means we are training a prior network jointly
        if self.opt.beta > 0:
            self.prior_network = v_network(opt)

        self.z_expander = nn.Linear(opt.nz, opt.hidden_size)
        # this will store the non-parametric estimate of p(z)
        self.p_z = []

    # save a z vector in the memory buffer
    def save_z(self, z):
        if len(self.p_z) == 0:
            self.p_z = z.data.cpu()
        else:
            self.p_z = torch.cat((self.p_z, z.data.cpu()), 0)

    # compute the nearest-neighbor graph among z vectors
    def compute_z_graph(self):
        m = self.p_z.size(0)
        # do this in CUDA...
        self.knn_indx = torch.zeros(m, 4000).cuda() # indices
        self.knn_dist = torch.zeros(m, 4000).cuda() # distances
        self.p_z = self.p_z.cuda()
        for i in range(m):
            print('[computing z graph {}]'.format(float(i)/m), end="\r")
            d = torch.norm(self.p_z - self.p_z[i].view(1, self.opt.nz), 2, 1)
            nb_dist, nb_indx = torch.topk(d, 4000, largest=False)
            self.knn_indx[i].copy_(nb_indx)
            self.knn_dist[i].copy_(nb_dist)
        self.knn_dist = self.knn_dist.cpu()
        self.knn_indx = self.knn_indx.cpu().long()
            
    # sample a latent z vector, using uniform, KNN or learned prior methods. 
    def sample_z(self, bsize, method='fp', input_images=None, input_states=None, action=None, z_prev=None, t0=False):
        z_indx = None
        if method != 'knn':
            z = []
        # fixed prior, i.e. uniform
        if method == 'fp': 
            for b in range(bsize):
                z.append(random.choice(self.p_z))
            z = torch.stack(z).contiguous()
        # at each timestep, sample on of the K nearest neighbors of the current z vector
        elif method == 'knn':
            if t0:
                # we only have the initial z vector, so quantize it first to one in the training set
                self.p_z = self.p_z.cuda()
                d = torch.norm(self.p_z.view(-1, 1, self.opt.nz) - z_prev.view(1, -1, self.opt.nz).data.cuda(), 2, 2)
                z_prev_indx = torch.topk(d, 1, dim=0, largest=False)[1].squeeze()
            else:
                z_prev, z_prev_indx = z_prev
            z, z_indx = [], []
            for b in range(bsize):
                indx = random.choice(self.knn_indx[z_prev_indx[b]][:self.opt.topz_sample])
                z_indx.append(indx)
                z.append(self.p_z[indx])
            z = torch.stack(z).contiguous()            
        # use a learned likelihood model
        elif method == 'pdf':
            M = self.p_z.size(0)
            nz = self.p_z.size(1)
            pi, mu, sigma = self.prior_network(input_images, input_states, action)
            bsize = mu.size(0)
            # sample a z vector from the likelihood model
            pi_sample = torch.multinomial(pi, 1).squeeze()
            mu_sample = torch.gather(mu, dim=1, index=pi_sample.view(bsize, 1, 1).expand(bsize, 1, nz)).squeeze()
            sigma_sample = torch.gather(sigma, dim=1, index=pi_sample.view(bsize, 1, 1).expand(bsize, 1, nz)).squeeze()
            z_sample = mu_sample.data + torch.randn(bsize, nz).cuda()*sigma_sample.data
            # quantize it to its nearest neighbor
            dist = torch.norm(self.p_z.view(-1, 1, nz) - z_sample.view(1, -1, nz), 2, 2)
            _, z_ind = torch.topk(dist, 1, dim=0, largest=False)
            z = self.p_z.index(z_ind.squeeze().cuda())
        elif method == 'hinge_knn':
            M = self.p_z.size(0)
            nz = self.p_z.size(1)
            h_x = self.encoder(input_images, input_states, action)
            u = self.prior_network(h_x).data
            bsize = u.size(0)
            sim = torch.sum(self.p_z.view(-1, 1, nz)*u.view(1, -1, nz), 2)
            topsim, z_ind = torch.topk(sim, 100, dim=0, largest=True)
            pdb.set_trace()
            z = []
            for b in range(bsize):
                indx = random.choice(z_ind[:, b])
                z.append(self.p_z[indx])
            z = torch.stack(z).contiguous()
            
            
        if self.use_cuda: z = z.cuda()
        return [Variable(z), z_indx]

    def forward(self, inputs, actions, targets, save_z = False, sampling=None, z_dropout=0.0, z_seq=None):
        input_images, input_states = inputs
        bsize = input_images.size(0)
        actions = actions.view(bsize, -1, self.opt.n_actions)
        npred = actions.size(1)
        ploss = Variable(torch.zeros(1))
        if self.use_cuda:
            ploss = ploss.cuda()

        pred_images, pred_states, pred_costs = [], [], []
        self.Z = []
        self.z_top_list = []
            
        z = None
        for t in range(npred):
            # encode the inputs
            h_x = self.encoder(input_images, input_states, actions[:, t])
            if sampling is None:
                # we are training or estimating z distribution
                target_images, target_states, target_costs = targets
                # encode the targets into z
                h_y = self.y_encoder(target_images[:, t].unsqueeze(1).contiguous())
                if random.random() < z_dropout:
                    z = self.z_zero
                else:
                    z = self.z_network(utils.combine(h_x, h_y, self.opt.combine).view(bsize, -1))
                    if self.opt.z_sphere == 1:
                        z = z / torch.norm(z, 2, 1).unsqueeze(1)
                    if save_z:
                        self.save_z(z)
                    if self.opt.beta > 0:
                        u = self.prior_network(h_x)
                        ploss = utils.hinge_loss(u, z)
                        '''
                        pi, mu, sigma = self.prior_network(h_x)
                        ploss = utils.mdn_loss_fn(pi, sigma, mu, z)
                        '''
                        if math.isnan(ploss.data[0]):
                            pdb.set_trace()
            else:
                # we are doing inference
                if z_seq is not None:
                    z = [z_seq[t], None]
                else:
                    z = self.sample_z(bsize, sampling, input_images, input_states, actions[:, t], z, t0=False)

            z_ = z if sampling is None else z[0]
            z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp, self.opt.combine)

            pred_image, pred_state, pred_cost = self.decoder(h)
            if sampling is not None:
                pred_image.detach()
                pred_state.detach()
                pred_cost.detach()
            if self.opt.sigmoid_out == 1:
                pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            else:
                pred_image = torch.clamp(pred_image + input_images[:, -1].unsqueeze(1), min=0, max=1)
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        return [pred_images, pred_states, pred_costs], ploss


    def plan_actions_backprop(self, observation, args, verbose=True, normalize=True, optimize_z=False, optimize_a=True, gamma=0.99):
        input_images, input_states = observation
        input_images = input_images.float().clone()
        input_states = input_states.clone()
        if normalize:
            input_images.div_(255.0)
            input_states -= self.stats['s_mean'].view(1, 4).expand(input_states.size())
            input_states /= self.stats['s_std'].view(1, 4).expand(input_states.size())
            input_images = Variable(input_images.cuda()).unsqueeze(0)
            input_states = Variable(input_states.cuda()).unsqueeze(0)
        bsize = input_images.size(0)
        # repeat for multiple rollouts
        input_images = input_images.unsqueeze(1).expand(bsize, args.n_rollouts, args.ncond, 3, args.height, args.width)
        input_states = input_states.unsqueeze(1).expand(bsize, args.n_rollouts, args.ncond, 4)
        input_images = input_images.contiguous().view(bsize * args.n_rollouts, args.ncond, 3, args.height, args.width)
        input_states = input_states.contiguous().view(bsize * args.n_rollouts, args.ncond, 4)
        actions = Variable(torch.randn(bsize, args.npred, self.opt.n_actions).cuda().mul_(0.0))
        if optimize_a:
            actions.requires_grad = True
            optimizer_a = optim.Adam([actions], args.lrt)

        Z = self.sample_z(bsize * args.n_rollouts * args.npred, method='fp')[0]
        Z = Z.view(args.npred, bsize * args.n_rollouts, -1)
        Z0 = Z.clone()
        gamma_mask = Variable(torch.from_numpy(numpy.array([[gamma**t, gamma**t] for t in range(args.npred)])).float().cuda()).unsqueeze(0)
        if optimize_z:
            Z.requires_grad = True
            optimizer_z = optim.Adam([Z], args.lrt)
        pred = None
        for i in range(0, args.n_iter):
            optimizer_a.zero_grad()
            self.zero_grad()
            actions_rep = actions.unsqueeze(1).expand(bsize, args.n_rollouts, args.npred, 2).contiguous().view(bsize * args.n_rollouts, args.npred, 2)
            pred, _ = self.forward([input_images, input_states], actions_rep, None, sampling='fp', z_seq=Z)
            costs = pred[2]
            weighted_costs = costs * gamma_mask
            loss = weighted_costs[:, :, 0].mean() + 0.5*weighted_costs[:, :, 1].mean()
            loss.backward()
            if verbose:
                print('[iter {} | mean pred cost = ({:.4f}, {:.4f})], grad = {}'.format(i, costs.data[:, :, 0].mean(), costs.data[:, :, 1].mean(), actions.grad.data.norm()))
            if optimize_a:
                torch.nn.utils.clip_grad_norm([actions], 1)
                optimizer_a.step()
                actions.data.clamp_(min=-2, max=2)
                actions.data[:, :, 1].clamp_(min=-1, max=1)

            if optimize_z:
                torch.nn.utils.clip_grad_norm([Z], 1)
                Z.grad *= -1
                optimizer_z.step()


        # evaluate on new futures
        Z_test = self.sample_z(bsize * args.n_rollouts * args.npred, method='fp')[0]
        Z_test = Z_test.view(args.npred, bsize * args.n_rollouts, -1)
        actions_rep = actions.unsqueeze(1).expand(bsize, args.n_rollouts, args.npred, 2).contiguous().view(bsize * args.n_rollouts, args.npred, 2)
        pred, _ = self.forward([input_images, input_states], actions_rep, None, sampling='fp', z_seq=Z_test)
        costs = pred[2]
        loss_test = costs[:, :, 0].mean() + 0.5*costs[:, :, 1].mean()
        print('\n[pred test cost = {}]\n'.format(loss_test.data[0]))
        
        '''
        # also get predictions using zero actions
        pred_const, _ = self.forward([input_images, input_states], actions.clone().zero_(), None, sampling='fp', z_seq=Z0)
        '''
        pred_const = None
        actions = actions.data.cpu()
        if normalize:
            actions *= self.stats['a_std'].view(1, 1, 2).expand(actions.size())
            actions += self.stats['a_mean'].view(1, 1, 2).expand(actions.size())
        return actions.squeeze().numpy(), pred, pred_const, loss_test.data[0]


    def create_policy_net(self, opt):
        if opt.targetprop == 0:
            self.policy_net = StochasticPolicy(opt)
        elif opt.targetprop == 1:
            self.policy_net = PolicyMDN(opt)


    def train_policy_net(self, inputs, targets, targetprop=0):
        t0 = time.time()
        input_images, input_states = inputs
        target_images, target_states, target_costs = targets
        bsize = input_images.size(0)
        npred = target_images.size(1)
        pred_images, pred_states, pred_costs, pred_actions = [], [], [], []
            
        z = None
        for t in range(npred):
            # encode the inputs
            if targetprop == 0:
                actions = self.policy_net(input_images, input_states)
            elif targetprop == 1:
                _, _, _, actions = self.policy_net(input_images, input_states, sample=True)
                actions = Variable(actions)
            h_x = self.encoder(input_images, input_states, actions)
            target_images, target_states, target_costs = targets
            # encode the targets into z
            h_y = self.y_encoder(torch.cat((input_images, target_images[:, t].unsqueeze(1).contiguous()), 1), input_states, actions)
            z = self.z_network(utils.combine(h_x, h_y, self.opt.combine).view(bsize, -1))
            z_ = z
            z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp.squeeze(), self.opt.combine)
            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
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
#        print(time.time() - t0)
        return [pred_images, pred_states, pred_costs], pred_actions



    # assuming this is for a single input sample
    def compute_action_policy_net(self, inputs, opt, npred=50, n_actions=10, n_futures=5, normalize=True):
        input_images, input_states = inputs
        pred_images, pred_states, pred_costs = [], [], []

        if normalize:
            input_images = input_images.clone().float().div_(255.0)
            input_states -= self.stats['s_mean'].view(1, 4).expand(input_states.size())
            input_states /= self.stats['s_std'].view(1, 4).expand(input_states.size())
            input_images = Variable(input_images.cuda()).unsqueeze(0)
            input_states = Variable(input_states.cuda()).unsqueeze(0)


        # repeat for multiple rollouts
        bsize = n_futures * n_actions
        input_images = input_images.expand(bsize, opt.ncond, 3, opt.height, opt.width)
        input_states = input_states.expand(bsize, opt.ncond, 4)
        input_images = input_images.contiguous().view(bsize, opt.ncond, 3, opt.height, opt.width)
        input_states = input_states.contiguous().view(bsize, opt.ncond, 4)
            
        Z = self.sample_z(n_futures * npred, method='fp')[0]
        Z = Z.view(npred, 1, n_futures, -1)
        Z = Z.expand(npred, n_actions, n_futures, -1)
        Z0 = Z.clone()

        # sample initial actions: we will choose to execute one of these at the end
        actions_init = self.policy_net(input_images[0].unsqueeze(0), input_states[0].unsqueeze(0), sample=True, normalize=False, n_samples=n_actions)
        actions = actions.clone().unsqueeze(0).expand(n_futures, n_actions, 2)
        for t in range(npred):
            # encode the inputs
            actions = self.policy_net(input_images, input_states)
            h_x = self.encoder(input_images, input_states, actions)
            z_ = Z[t]
            z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp.squeeze(), self.opt.combine)
            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        return [pred_images, pred_states, pred_costs], None


        
    def intype(self, t):
        if t == 'gpu':
            self.cuda()
            self.z_zero = self.z_zero.cuda()
            self.use_cuda = True
            if len(self.p_z) > 0:
                self.p_z = self.p_z.cuda()
        elif t == 'cpu':
            self.cpu()
            self.use_cuda = False
            self.z_zero = self.z_zero.cpu()
            if len(self.p_z) > 0:
                self.p_z = self.p_z.cpu()














































# forward TEN model (autoregressive)
class FwdCNN_TEN2(nn.Module):
    def __init__(self, opt, mfile=''):
        super(FwdCNN_TEN2, self).__init__()
        self.opt = opt

        if mfile == '':
            self.encoder = encoder(opt, 0, opt.ncond)
            self.decoder = decoder(opt)


            self.a_network = nn.Sequential(
                nn.Linear(self.opt.n_actions, self.opt.nfeature), 
                nn.PReLU(self.opt.nfeature, init=1), 
                nn.Linear(self.opt.nfeature, self.opt.nfeature), 
                nn.PReLU(self.opt.nfeature, init=1), 
                nn.Linear(self.opt.nfeature, self.opt.hidden_size)
            )


            self.h_network_conv1 = nn.Sequential(
                nn.Conv2d(self.opt.nfeature, self.opt.nfeature, 1, 1), 
                nn.PReLU(self.opt.nfeature, init=1),
                nn.Conv2d(self.opt.nfeature, self.opt.nfeature, 1, 1)
            )


            self.h_network1 = nn.Sequential(
                nn.Linear(self.opt.hidden_size, self.opt.n_hidden), 
                nn.PReLU(self.opt.n_hidden, init=1), 
                nn.Linear(self.opt.n_hidden, self.opt.n_hidden), 
                nn.PReLU(self.opt.n_hidden, init=1), 
                nn.Linear(self.opt.n_hidden, self.opt.hidden_size)
            )


            self.h_network_conv2 = nn.Sequential(
                nn.Conv2d(self.opt.nfeature, self.opt.nfeature, 1, 1), 
                nn.PReLU(self.opt.nfeature, init=1),
                nn.Conv2d(self.opt.nfeature, self.opt.nfeature, 1, 1)
            )

            self.h_network2 = nn.Sequential(
                nn.Linear(self.opt.hidden_size, self.opt.n_hidden), 
                nn.PReLU(self.opt.n_hidden, init=1), 
                nn.Linear(self.opt.n_hidden, self.opt.n_hidden), 
                nn.PReLU(self.opt.n_hidden, init=1), 
                nn.Linear(self.opt.n_hidden, self.opt.hidden_size)
            )


        else:
            print('[initializing encoder and decoder with: {}]'.format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.encoder.n_inputs = opt.ncond
            self.decoder.n_out = 1

        self.y_encoder = encoder(opt, 0, 1)

        self.z_network = nn.Sequential(
            nn.Linear(opt.hidden_size, opt.nfeature),
            nn.PReLU(self.opt.nfeature, init=1),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.PReLU(self.opt.nfeature, init=1),
            nn.Linear(opt.nfeature, opt.nz)
        )

        # if beta > 0, it means we are training a prior network jointly
        if self.opt.beta > 0:
            self.prior_network = v_network(opt)

        self.z_expander = nn.Linear(opt.nz, opt.hidden_size)
        # this will store the non-parametric estimate of p(z)
        self.p_z = []

    # save a z vector in the memory buffer
    def save_z(self, z):
        if len(self.p_z) == 0:
            self.p_z = z.data.cpu()
        else:
            self.p_z = torch.cat((self.p_z, z.data.cpu()), 0)

            
    # sample a latent z vector, using uniform, KNN or learned prior methods. 
    def sample_z(self, bsize, method='fp', input_images=None, input_states=None, action=None, z_prev=None, t0=False):
        z_indx = None
        if method != 'knn':
            z = []
        # fixed prior, i.e. uniform
        if method == 'fp': 
            for b in range(bsize):
                z.append(random.choice(self.p_z))
            z = torch.stack(z).contiguous()
        # at each timestep, sample on of the K nearest neighbors of the current z vector
        elif method == 'knn':
            if t0:
                # we only have the initial z vector, so quantize it first to one in the training set
                self.p_z = self.p_z.cuda()
                d = torch.norm(self.p_z.view(-1, 1, self.opt.nz) - z_prev.view(1, -1, self.opt.nz).data.cuda(), 2, 2)
                z_prev_indx = torch.topk(d, 1, dim=0, largest=False)[1].squeeze()
            else:
                z_prev, z_prev_indx = z_prev
            z, z_indx = [], []
            for b in range(bsize):
                indx = random.choice(self.knn_indx[z_prev_indx[b]][:self.opt.topz_sample])
                z_indx.append(indx)
                z.append(self.p_z[indx])
            z = torch.stack(z).contiguous()            
        # use a learned likelihood model
        elif method == 'pdf':
            M = self.p_z.size(0)
            nz = self.p_z.size(1)
            pi, mu, sigma = self.prior_network(input_images, input_states, action)
            bsize = mu.size(0)
            # sample a z vector from the likelihood model
            pi_sample = torch.multinomial(pi, 1).squeeze()
            mu_sample = torch.gather(mu, dim=1, index=pi_sample.view(bsize, 1, 1).expand(bsize, 1, nz)).squeeze()
            sigma_sample = torch.gather(sigma, dim=1, index=pi_sample.view(bsize, 1, 1).expand(bsize, 1, nz)).squeeze()
            z_sample = mu_sample.data + torch.randn(bsize, nz).cuda()*sigma_sample.data
            # quantize it to its nearest neighbor
            dist = torch.norm(self.p_z.view(-1, 1, nz) - z_sample.view(1, -1, nz), 2, 2)
            _, z_ind = torch.topk(dist, 1, dim=0, largest=False)
            z = self.p_z.index(z_ind.squeeze().cuda())
        elif method == 'hinge_knn':
            M = self.p_z.size(0)
            nz = self.p_z.size(1)
            h_x = self.encoder(input_images, input_states, action)
            u = self.prior_network(h_x).data
            bsize = u.size(0)
            sim = torch.sum(self.p_z.view(-1, 1, nz)*u.view(1, -1, nz), 2)
            topsim, z_ind = torch.topk(sim, 100, dim=0, largest=True)
            pdb.set_trace()
            z = []
            for b in range(bsize):
                indx = random.choice(z_ind[:, b])
                z.append(self.p_z[indx])
            z = torch.stack(z).contiguous()
            
            
        if self.use_cuda: z = z.cuda()
        return [Variable(z), z_indx]

    def forward(self, inputs, actions, targets, save_z = False, sampling=None, z_dropout=0.0, z_seq=None):
        input_images, input_states = inputs
        bsize = input_images.size(0)
        actions = actions.view(bsize, -1, self.opt.n_actions)
        npred = actions.size(1)
        ploss = Variable(torch.zeros(1))
        if self.use_cuda:
            ploss = ploss.cuda()

        pred_images, pred_states, pred_costs = [], [], []
        self.Z = []
        self.z_top_list = []
            
        z = None
        # encode the inputs
        h = self.encoder(input_images, input_states).view(bsize, self.opt.hidden_size)

        for t in range(npred):
            h = h + self.a_network(actions[:, t]).view(bsize, self.opt.hidden_size)
            h = h.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = self.h_network_conv1(h)
            h = h.view(bsize, self.opt.hidden_size)
            h = h + self.h_network1(h)
            if sampling is None:
                # we are training or estimating z distribution
                target_images, target_states, target_costs = targets
                # encode the targets into z
                h_y = self.y_encoder(target_images[:, t].unsqueeze(1).contiguous()).view(bsize, self.opt.hidden_size)
                if random.random() < z_dropout:
                    z = Variable(torch.zeros(bsize, self.opt.nz).cuda())
                else:
                    z = self.z_network(utils.combine(h, h_y, self.opt.combine).view(bsize, -1))
                    if self.opt.z_sphere == 1:
                        z = z / torch.norm(z, 2, 1).unsqueeze(1)
                    if save_z:
                        self.save_z(z)
                    if self.opt.beta > 0:
                        u = self.prior_network(h_x)
                        ploss = utils.hinge_loss(u, z)
                        '''
                        pi, mu, sigma = self.prior_network(h_x)
                        ploss = utils.mdn_loss_fn(pi, sigma, mu, z)
                        '''
                        if math.isnan(ploss.data[0]):
                            pdb.set_trace()
            else:
                # we are doing inference
                if z_seq is not None:
                    z = [z_seq[t], None]
                else:
                    z = self.sample_z(bsize, sampling, input_images, input_states, actions[:, t], z, t0=False)

            z_ = z if sampling is None else z[0]
            z_exp = self.z_expander(z_).view(bsize, self.opt.hidden_size)
            h = utils.combine(h, z_exp, self.opt.combine)
            h = h.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = self.h_network_conv2(h)
            h = h.view(bsize, self.opt.hidden_size)
            h = h + self.h_network2(h)
            pred_image, pred_state, pred_cost = self.decoder(h.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width))
            if sampling is not None:
                pred_image.detach()
                pred_state.detach()
                pred_cost.detach()
            if t == 0:
                last_image = input_images[:, -1].unsqueeze(1)
                last_state = input_states[:, -1]
            else:
                last_image = pred_images[-1]
                last_state = pred_states[-1]
                
            pred_image = torch.clamp(pred_image + last_image, min=0, max=1)
            pred_state = torch.clamp(pred_state + last_state, min=-6, max=6)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        return [pred_images, pred_states, pred_costs], ploss


    def plan_actions_backprop(self, observation, args, verbose=True, normalize=True, optimize_z=False, optimize_a=True, gamma=0.99):
        input_images, input_states = observation
        input_images = input_images.float().clone()
        input_states = input_states.clone()
        if normalize:
            input_images.div_(255.0)
            input_states -= self.stats['s_mean'].view(1, 4).expand(input_states.size())
            input_states /= self.stats['s_std'].view(1, 4).expand(input_states.size())
            input_images = Variable(input_images.cuda()).unsqueeze(0)
            input_states = Variable(input_states.cuda()).unsqueeze(0)
        bsize = input_images.size(0)
        # repeat for multiple rollouts
        input_images = input_images.unsqueeze(1).expand(bsize, args.n_rollouts, args.ncond, 3, args.height, args.width)
        input_states = input_states.unsqueeze(1).expand(bsize, args.n_rollouts, args.ncond, 4)
        input_images = input_images.contiguous().view(bsize * args.n_rollouts, args.ncond, 3, args.height, args.width)
        input_states = input_states.contiguous().view(bsize * args.n_rollouts, args.ncond, 4)
        actions = Variable(torch.randn(bsize, args.npred, self.opt.n_actions).cuda().mul_(0.0))
        if optimize_a:
            actions.requires_grad = True
            optimizer_a = optim.Adam([actions], args.lrt)

        Z = self.sample_z(bsize * args.n_rollouts * args.npred, method='fp')[0]
        Z = Z.view(args.npred, bsize * args.n_rollouts, -1)
        Z0 = Z.clone()
        gamma_mask = Variable(torch.from_numpy(numpy.array([[gamma**t, gamma**t] for t in range(args.npred)])).float().cuda()).unsqueeze(0)
        if optimize_z:
            Z.requires_grad = True
            optimizer_z = optim.Adam([Z], args.lrt)
        pred = None
        for i in range(0, args.n_iter):
            optimizer_a.zero_grad()
            self.zero_grad()
            actions_rep = actions.unsqueeze(1).expand(bsize, args.n_rollouts, args.npred, 2).contiguous().view(bsize * args.n_rollouts, args.npred, 2)
            pred, _ = self.forward([input_images, input_states], actions_rep, None, sampling='fp', z_seq=Z)
            costs = pred[2]
            weighted_costs = costs * gamma_mask
            loss = weighted_costs[:, :, 0].mean() + 0.5*weighted_costs[:, :, 1].mean()
            loss.backward()
            if verbose:
                print('[iter {} | mean pred cost = ({:.4f}, {:.4f})], grad = {}'.format(i, costs.data[:, :, 0].mean(), costs.data[:, :, 1].mean(), actions.grad.data.norm()))
            if optimize_a:
                torch.nn.utils.clip_grad_norm([actions], 1)
                optimizer_a.step()
                actions.data.clamp_(min=-2, max=2)
                actions.data[:, :, 1].clamp_(min=-1, max=1)

            if optimize_z:
                torch.nn.utils.clip_grad_norm([Z], 1)
                Z.grad *= -1
                optimizer_z.step()


        # evaluate on new futures
        Z_test = self.sample_z(bsize * args.n_rollouts * args.npred, method='fp')[0]
        Z_test = Z_test.view(args.npred, bsize * args.n_rollouts, -1)
        actions_rep = actions.unsqueeze(1).expand(bsize, args.n_rollouts, args.npred, 2).contiguous().view(bsize * args.n_rollouts, args.npred, 2)
        pred, _ = self.forward([input_images, input_states], actions_rep, None, sampling='fp', z_seq=Z_test)
        costs = pred[2]
        loss_test = costs[:, :, 0].mean() + 0.5*costs[:, :, 1].mean()
        print('\n[pred test cost = {}]\n'.format(loss_test.data[0]))
        
        '''
        # also get predictions using zero actions
        pred_const, _ = self.forward([input_images, input_states], actions.clone().zero_(), None, sampling='fp', z_seq=Z0)
        '''
        pred_const = None
        actions = actions.data.cpu()
        if normalize:
            actions *= self.stats['a_std'].view(1, 1, 2).expand(actions.size())
            actions += self.stats['a_mean'].view(1, 1, 2).expand(actions.size())
        return actions.squeeze().numpy(), pred, pred_const, loss_test.data[0]


    def create_policy_net(self, opt):
        if opt.targetprop == 0:
            self.policy_net = StochasticPolicy(opt)
        elif opt.targetprop == 1:
            self.policy_net = PolicyMDN(opt)


    def train_policy_net(self, inputs, targets, targetprop=0):
        t0 = time.time()
        input_images, input_states = inputs
        target_images, target_states, target_costs = targets
        bsize = input_images.size(0)
        npred = target_images.size(1)
        pred_images, pred_states, pred_costs, pred_actions = [], [], [], []
            
        z = None
        for t in range(npred):
            # encode the inputs
            if targetprop == 0:
                actions = self.policy_net(input_images, input_states)
            elif targetprop == 1:
                _, _, _, actions = self.policy_net(input_images, input_states, sample=True)
                actions = Variable(actions)
            h_x = self.encoder(input_images, input_states, actions)
            target_images, target_states, target_costs = targets
            # encode the targets into z
            h_y = self.y_encoder(torch.cat((input_images, target_images[:, t].unsqueeze(1).contiguous()), 1), input_states, actions)
            z = self.z_network(utils.combine(h_x, h_y, self.opt.combine).view(bsize, -1))
            z_ = z
            z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp.squeeze(), self.opt.combine)
            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
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
#        print(time.time() - t0)
        return [pred_images, pred_states, pred_costs], pred_actions


    def train_policy_net_targetprop(self, inputs, targets, opt):
        input_images_, input_states_ = inputs
        target_images, target_states, target_costs = targets
        bsize = input_images_.size(0)
        npred = target_images.size(1)
        policy_input_images, policy_input_states = [], []

        gamma_mask = torch.Tensor([opt.gamma**t for t in range(opt.npred)]).view(1, -1).cuda()

            
        z = None
        loss_i, loss_s, loss_c = None, None, None
        self.actions = Variable(torch.randn(bsize, npred, self.opt.n_actions).cuda().mul_(0.1), requires_grad=True)
        self.optimizer_a = optim.Adam([self.actions], opt.lrt_traj)

        for i in range(opt.niter_traj):
            pred_images, pred_states, pred_costs = [], [], []
            policy_input_images, policy_input_states = [], []
            self.optimizer_a.zero_grad()
            self.zero_grad()
            for t in range(npred):
                # encode the inputs
                input_images = Variable(input_images_.data)
                input_states = Variable(input_states_.data)
                policy_input_images.append(input_images)
                policy_input_states.append(input_states)
                if i == 0:
                    _, _, _, a = self.policy_net(input_images, input_states, sample=True)
                    self.actions[:, t].data.copy_(a.squeeze())
                h_x = self.encoder(input_images, input_states, self.actions[:, t])
                target_images, target_states, target_costs = targets
                # encode the targets into z
                h_y = self.y_encoder(torch.cat((input_images, target_images[:, t].unsqueeze(1).contiguous()), 1), input_states, self.actions[:, t])
                z = self.z_network(utils.combine(h_x, h_y, self.opt.combine).view(bsize, -1))
                z_ = z
                z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
                h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
                h = utils.combine(h_x, z_exp.squeeze(), self.opt.combine)
                pred_image, pred_state, pred_cost = self.decoder(h)
                pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
                # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
                pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
                input_images = torch.cat((input_images[:, 1:], pred_image), 1)
                input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
                pred_images.append(pred_image)
                pred_states.append(pred_state)
                pred_costs.append(pred_cost)

            pred_images = torch.cat(pred_images, 1)
            pred_states = torch.stack(pred_states, 1)
            pred_costs = torch.stack(pred_costs, 1)
            loss_i = F.mse_loss(pred_images, target_images, reduce=False).mean(4).mean(3).mean(2)
            loss_s = F.mse_loss(pred_states, target_states, reduce=False).mean(2)

            proximity_cost = pred_costs[:, :, 0]
            lane_cost = pred_costs[:, :, 1]
            proximity_cost = proximity_cost * Variable(gamma_mask)
            lane_cost = lane_cost * Variable(gamma_mask)
            loss_c = proximity_cost.mean() + opt.lambda_lane * lane_cost.mean()
            loss_i = loss_i * Variable(gamma_mask)
            loss_s = loss_s * Variable(gamma_mask)
            loss_i = loss_i.mean()
            loss_s = loss_s.mean()
            loss = loss_i + loss_s + opt.lambda_c*loss_c
            loss.backward()
            self.optimizer_a.step()

        policy_input_images = torch.stack(policy_input_images).view(bsize*npred, self.opt.ncond, 3, self.opt.height, self.opt.width)
        policy_input_states = torch.stack(policy_input_states).view(bsize*npred, self.opt.ncond, 4)
        self.actions = self.actions.permute(1, 0, 2).contiguous().view(bsize*npred, 2)
        pi, mu, sigma, _ = self.policy_net(policy_input_images, policy_input_states)
        loss_mdn = utils.mdn_loss_fn(pi, sigma, mu, self.actions)
        self.optim_stats = self.optimizer_a.state_dict()
        return loss_i, loss_s, loss_c, loss_mdn



    # assuming this is for a single input sample
    def compute_action_policy_net(self, inputs, opt, npred=50, n_actions=10, n_futures=5, normalize=True):
        input_images, input_states = inputs
        pred_images, pred_states, pred_costs = [], [], []

        if normalize:
            input_images = input_images.clone().float().div_(255.0)
            input_states -= self.stats['s_mean'].view(1, 4).expand(input_states.size())
            input_states /= self.stats['s_std'].view(1, 4).expand(input_states.size())
            input_images = Variable(input_images.cuda()).unsqueeze(0)
            input_states = Variable(input_states.cuda()).unsqueeze(0)


        # repeat for multiple rollouts
        bsize = n_futures * n_actions
        input_images = input_images.expand(bsize, opt.ncond, 3, opt.height, opt.width)
        input_states = input_states.expand(bsize, opt.ncond, 4)
        input_images = input_images.contiguous().view(bsize, opt.ncond, 3, opt.height, opt.width)
        input_states = input_states.contiguous().view(bsize, opt.ncond, 4)
            
        Z = self.sample_z(n_futures * npred, method='fp')[0]
        Z = Z.view(npred, 1, n_futures, -1)
        Z = Z.expand(npred, n_actions, n_futures, -1)
        Z0 = Z.clone()

        # sample initial actions: we will choose to execute one of these at the end
        actions_init = self.policy_net(input_images[0].unsqueeze(0), input_states[0].unsqueeze(0), sample=True, normalize=False, n_samples=n_actions)
        actions = actions.clone().unsqueeze(0).expand(n_futures, n_actions, 2)
        for t in range(npred):
            # encode the inputs
            actions = self.policy_net(input_images, input_states)
            h_x = self.encoder(input_images, input_states, actions)
            z_ = Z[t]
            z_exp = self.z_expander(z_).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = utils.combine(h_x, z_exp.squeeze(), self.opt.combine)
            pred_image, pred_state, pred_cost = self.decoder(h)
            pred_image = F.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_costs.append(pred_cost)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_costs = torch.stack(pred_costs, 1)
        return [pred_images, pred_states, pred_costs], None


        
    def intype(self, t):
        if t == 'gpu':
            self.cuda()
            self.use_cuda = True
            if len(self.p_z) > 0:
                self.p_z = self.p_z.cuda()
        elif t == 'cpu':
            self.cpu()
            self.use_cuda = False
            if len(self.p_z) > 0:
                self.p_z = self.p_z.cpu()





