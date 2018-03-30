import numpy, random, pdb, math, pickle, glob, time, os
import torch

class DataLoader():
    def __init__(self, fname, opt, dataset='simulator', single_shard=False):
        self.opt = opt
        self.data = []
        self.dataset = dataset

        if dataset == 'i80':
            data_dir = '/misc/vlgscratch4/LecunGroup/nvidia-collab/data_i80/'
            if single_shard:
                # quick load for debugging
                data_files = ['trajectories-0500-0515.txt/']
            else:
                data_files = ['trajectories-0400-0415.txt/', 
                              'trajectories-0500-0515.txt/', 
                              'trajectories-0515-0530.txt/']

            self.images = []
            self.actions = []
            for df in data_files:
                combined_data_path = data_dir + f'{df}/all_data.pth'
                if os.path.isfile(combined_data_path):
                    print(f'[loading data {combined_data_path}]')
                    data = torch.load(combined_data_path)
                    self.images += data.get('images')
                    self.actions += data.get('actions')
                else:
                    print(data_dir)
                    for f in glob.glob(data_dir + f'{df}/car*.pkl'):
                        print(f'[loading {f}]')
                        fd = pickle.load(open(f, 'rb'))
                        self.data += [{'images': fd.get('images'), 'actions': fd.get('actions')}]
                        
                    images = []
                    actions = []
                    for run in self.data:
                        images.append(run.get('images'))
                        actions.append(run.get('actions'))
                    torch.save({'images': images, 'actions': actions}, combined_data_path)
                    self.images += images
                    self.actions += actions


            self.n_episodes = len(self.images)
            self.n_train = int(math.floor(self.n_episodes * 0.9))
            self.n_valid = int(math.floor(self.n_episodes * 0.05))
            self.n_test = int(math.floor(self.n_episodes * 0.05))
            splits_path = data_dir + '/splits.pth'
            if os.path.exists(splits_path):
                print('[loading data splits]')
                self.splits = torch.load(splits_path)                
                self.train_indx = self.splits.get('train_indx')
                self.valid_indx = self.splits.get('valid_indx')
                self.test_indx = self.splits.get('test_indx')
            else:
                print('[generating data splits]')
                perm = numpy.random.permutation(self.n_episodes)
                self.train_indx = perm[0:self.n_train]
                self.valid_indx = perm[self.n_train+1:self.n_train+self.n_valid]
                self.test_indx = perm[self.n_train+self.n_valid+1:self.n_train+self.n_valid+self.n_test]
                torch.save({'train_indx': self.train_indx, 'valid_indx': self.valid_indx, 'test_indx': self.test_indx}, splits_path)

        else:        
            for i in range(opt.nshards):
                f = f'{opt.data_dir}/traffic_data_lanes={opt.lanes}-episodes={opt.n_episodes}-seed={i+1}.pkl'
                print(f'[loading {f}]')
                self.data += pickle.load(open(f, 'rb'))

            self.images = []
            self.states = []
            self.actions = []
            self.masks = []
            for run in self.data:
                self.images.append(run.get('images'))
                self.actions.append(run.get('actions'))
                self.states.append(run.get('states'))
                self.masks.append(run.get('masks'))

        
            self.n_episodes = len(self.images)
            print(f'Number of episodes: {self.n_episodes}')
            self.n_train = int(math.floor(self.n_episodes * 0.9))
            self.n_valid = int(math.floor(self.n_episodes * 0.05))
            self.n_test = int(math.floor(self.n_episodes * 0.05))
            self.train_indx = range(0, self.n_train)
            self.valid_indx = range(self.n_train+1, self.n_train+self.n_valid)
            self.test_indx = range(self.n_train+self.n_valid+1, self.n_episodes)        

        print('[computing action stats]')
        all_actions = []
        for i in self.train_indx:
            all_actions.append(self.actions[i])
        print('[done]')
        all_actions = torch.cat(all_actions, 0)
        self.a_mean = torch.mean(all_actions, 0)
        self.a_std = torch.std(all_actions, 0)

    # get batch to use for imitation learning:
    # a sequence of ncond consecutive states, and a sequence of npred actions
    def get_batch_il(self, split):
        if split == 'train':
            indx = self.train_indx
        elif split == 'valid':
            indx = self.valid_indx
        elif split == 'test':
            indx = self.test_indx

        images, states, masks, actions = [], [], [], []
        nb = 0
        while nb < self.opt.batch_size:
            s = random.choice(indx)
            T = len(self.actions[s]) - 1
            if T > (self.opt.ncond + self.opt.npred):
                t = random.randint(0, T - (self.opt.ncond+self.opt.npred))
                images.append(self.images[s][t:t+(self.opt.ncond+self.opt.npred)])
#                states.append(self.states[s][t:t+(self.opt.ncond+self.opt.npred)])
#                masks.append(self.masks[s][t:t+(self.opt.ncond+self.opt.npred)])
                actions.append(self.actions[s][t:t+(self.opt.ncond+self.opt.npred)])
                nb += 1
        images = torch.stack(images)
#        states = torch.stack(states)
        actions = torch.stack(actions)
#        masks = torch.stack(masks)
        images = images[:, :self.opt.ncond].clone()
#        states = states[:, :self.opt.ncond, 0].clone()
#        masks = masks[:, :self.opt.ncond].clone()
        actions = actions[:, self.opt.ncond:(self.opt.ncond+self.opt.npred)].clone()
        images = images.float() / 255.0

        actions -= self.a_mean.view(1, 1, 2).expand(actions.size())
        actions /= (1e-8 + self.a_std.view(1, 1, 2).expand(actions.size()))
        states = torch.zeros(self.opt.batch_size, self.opt.ncond, self.opt.n_inputs).cuda()
        assert(images.max() <= 1 and images.min() >= 0)
        return images.float().cuda(), states.float().cuda(), actions.float().cuda()



    # get batch to use for forward modeling
    # a sequence of ncond given states, a sequence of npred actions, 
    # and a sequence of npred states to be predicted
    def get_batch_fm(self, split, npred=-1):
        if split == 'train':
            indx = self.train_indx
        elif split == 'valid':
            indx = self.valid_indx
        elif split == 'test':
            indx = self.test_indx

        if npred == -1:
            npred = self.opt.npred

        images, states, masks, actions = [], [], [], []
        nb = 0
        while nb < self.opt.batch_size:
            s = random.choice(indx)
            T = self.images[s].size(0)
            if T > (self.opt.ncond + npred + 1):
                t = random.randint(0, T - (self.opt.ncond+npred + 1))
                images.append(self.images[s][t:t+(self.opt.ncond+npred)+1].cuda())
                actions.append(self.actions[s][t:t+(self.opt.ncond+npred)].cuda())
                if self.dataset == 'simulator':
                    states.append(self.states[s][t:t+(self.opt.ncond+npred)+1])
                    masks.append(self.masks[s][t:t+(self.opt.ncond+npred)])
                nb += 1

        images = torch.stack(images)
        actions = torch.stack(actions)
        actions = actions[:, (self.opt.ncond-1):(self.opt.ncond+npred-1)].float().contiguous()
        input_images = images[:, :self.opt.ncond].float()
        target_images = images[:, self.opt.ncond:(self.opt.ncond+npred)].float()
        input_images.div_(255.0)
        target_images.div_(255.0)

        input_states, target_states, masks = None, None, None


#        states = torch.stack(states)
#        masks = torch.stack(masks)
#        input_states = states[:, :self.opt.ncond, 0].clone()
#        target_states = states[:, self.opt.ncond:(self.opt.ncond+npred), 0].clone()
#        masks = masks[:, :self.opt.ncond].clone()

        return input_images, actions, target_images, None, None #input_states.float().cuda(), target_states.float().cuda()

