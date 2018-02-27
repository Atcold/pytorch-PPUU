import numpy, random, pdb, math, pickle
import torch

class DataLoader():
    def __init__(self, fname, opt):
        self.opt = opt
        self.data = pickle.load(open(fname, 'rb'))
        self.states = []
        self.actions = []
        self.masks = []
        for run in self.data:
            self.states.append(run.get('states'))
            self.actions.append(run.get('actions'))
            self.masks.append(run.get('masks'))

        self.n_episodes = len(self.states)
        self.n_train = int(math.floor(self.n_episodes * 0.9))
        self.n_valid = int(math.floor(self.n_episodes * 0.05))
        self.n_test = int(math.floor(self.n_episodes * 0.05))
        self.train_indx = range(0, self.n_train)
        self.valid_indx = range(self.n_train+1, self.n_train+self.n_valid)
        self.test_indx = range(self.n_train+self.n_valid+1, self.n_episodes)


    # get batch to use for imitation learning:
    # a sequence of ncond consecutive states, and a sequence of npred actions
    def get_batch_il(self, split):
        if split == 'train':
            indx = self.train_indx
        elif split == 'valid':
            indx = self.valid_indx
        elif split == 'test':
            indx = self.test_indx

        states, masks, actions = [], [], []
        nb = 0
        while nb < self.opt.batch_size:
            s = random.choice(indx)
            T = len(self.states[s]) - 1
            if T > (self.opt.ncond + self.opt.npred):
                t = random.randint(0, T - (self.opt.ncond+self.opt.npred))
                states.append(self.states[s][t:t+(self.opt.ncond+self.opt.npred)+1])
                masks.append(self.masks[s][t:t+(self.opt.ncond+self.opt.npred)])
                actions.append(self.actions[s][t:t+(self.opt.ncond+self.opt.npred)])
                nb += 1

        states = torch.stack(states)
        actions = torch.stack(actions)
        masks = torch.stack(masks)
        states = states[:, :self.opt.ncond].clone()
        masks = masks[:, :self.opt.ncond].clone()
        actions = actions[:, self.opt.ncond:(self.opt.ncond+self.opt.npred)].clone()

        return states, masks, actions
