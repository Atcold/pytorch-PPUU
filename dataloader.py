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

    def get_batch(self, split):
        if split == 'train':
            indx = self.train_indx
        elif split == 'valid':
            indx = self.valid_indx
        elif split == 'test':
            indx = self.test_indx

        states, masks, actions = [], [], []
        for _ in range(self.opt.batch_size):
            s = random.choice(indx)
            T = len(self.states[s])
            t = random.randint(0, T - self.opt.T)
            states.append(self.states[s][t:t+self.opt.T+1])
            masks.append(self.masks[s][t:t+self.opt.T])
            actions.append(self.actions[s][t:t+self.opt.T])

        states = torch.stack(states)
        actions = torch.stack(actions)
        masks = torch.stack(masks)

        return states, masks, actions
