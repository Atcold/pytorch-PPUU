import sys
import numpy, random, pdb, math, pickle, glob, time, os, re
import torch
from torch.autograd import Variable


class DataLoader:
    def __init__(self, fname, opt, dataset='simulator', single_shard=False):
        if opt.debug:
            single_shard = True
        self.opt = opt
        self.random = random.Random()
        self.random.seed(12345)  # use this so that the same batches will always be picked

        if dataset == 'i80' or dataset == 'us101':
            data_dir = 'traffic-data/state-action-cost/data_{}_v0'.format(dataset)
            if single_shard:
                # quick load for debugging
                data_files = ['{}.txt/'.format(next(os.walk(data_dir))[1][0])]
            else:
                data_files = next(os.walk(data_dir))[1]

            self.images = []
            self.actions = []
            self.costs = []
            self.states = []
            self.ids = []
            for df in data_files:
                combined_data_path = f'{data_dir}/{df}/all_data.pth'
                if os.path.isfile(combined_data_path):
                    print('[loading data shard: {}]'.format(combined_data_path))
                    data = torch.load(combined_data_path)
                    self.images += data.get('images')
                    self.actions += data.get('actions')
                    self.costs += data.get('costs')
                    self.states += data.get('states')
                    self.ids += data.get('ids')
                else:
                    print(data_dir)
                    images = []
                    actions = []
                    costs = []
                    states = []
                    ids = glob.glob(f'{data_dir}/{df}/car*.pkl')
                    for f in ids:
                        print('[loading {}]'.format(f))
                        fd = pickle.load(open(f, 'rb'))
                        Ta = fd['actions'].size(0)
                        Tp = fd['pixel_proximity_cost'].size(0)
                        Tl = fd['lane_cost'].size(0)
                        # assert Ta == Tp == Tl  # TODO Check why there are more costs than actions
                        # if not(Ta == Tp == Tl): pdb.set_trace()
                        images.append(fd['images'])
                        actions.append(fd['actions'])
                        costs.append(torch.cat((
                            fd.get('pixel_proximity_cost')[:Ta].view(-1, 1),
                            fd.get('lane_cost')[:Ta].view(-1, 1),
                        ), 1),)
                        states.append(fd['states'])

                    print(f'Saving {combined_data_path} to disk')
                    torch.save({
                        'images': images,
                        'actions': actions,
                        'costs': costs,
                        'states': states,
                        'ids': ids,
                    }, combined_data_path)
                    self.images += images
                    self.actions += actions
                    self.costs += costs
                    self.states += states
                    self.ids += ids
        else:
            assert False, 'Data set not supported'

        self.n_episodes = len(self.images)
        print(f'Number of episodes: {self.n_episodes}')
        self.n_train = int(math.floor(self.n_episodes * 0.8))
        self.n_valid = int(math.floor(self.n_episodes * 0.1))
        self.n_test = int(math.floor(self.n_episodes * 0.1))
        splits_path = data_dir + '/splits.pth'
        if os.path.exists(splits_path):
            print('[loading data splits: {}]'.format(splits_path))
            self.splits = torch.load(splits_path)
            self.train_indx = self.splits.get('train_indx')
            self.valid_indx = self.splits.get('valid_indx')
            self.test_indx = self.splits.get('test_indx')
        else:
            print('[generating data splits]')
            numpy.random.seed(0)
            perm = numpy.random.permutation(self.n_episodes)
            self.train_indx = perm[0:self.n_train]
            self.valid_indx = perm[self.n_train + 1:self.n_train + self.n_valid]
            self.test_indx = perm[self.n_train + self.n_valid + 1:]
            torch.save(dict(
                train_indx=self.train_indx,
                valid_indx=self.valid_indx,
                test_indx=self.test_indx,
            ), splits_path)

        stats_path = data_dir + '/data_stats.pth'
        if os.path.isfile(stats_path):
            print('[loading data stats: {}]'.format(stats_path))
            stats = torch.load(stats_path)
            self.a_mean = stats.get('a_mean')
            self.a_std = stats.get('a_std')
            self.s_mean = stats.get('s_mean')
            self.s_std = stats.get('s_std')
        else:
            print('[computing action stats]')
            all_actions = []
            for i in self.train_indx:
                all_actions.append(self.actions[i])
            all_actions = torch.cat(all_actions, 0)
            self.a_mean = torch.mean(all_actions, 0)
            self.a_std = torch.std(all_actions, 0)
            print('[computing state stats]')
            all_states = []
            for i in self.train_indx:
                all_states.append(self.states[i][:, 0])
            all_states = torch.cat(all_states, 0)
            self.s_mean = torch.mean(all_states, 0)
            self.s_std = torch.std(all_states, 0)
            torch.save({'a_mean': self.a_mean,
                        'a_std': self.a_std,
                        's_mean': self.s_mean,
                        's_std': self.s_std}, stats_path)
        
        car_sizes_path = data_dir + '/car_sizes.pth'
        print('[loading car sizes: {}]'.format(car_sizes_path))
        self.car_sizes = torch.load(car_sizes_path)

    # get batch to use for forward modeling
    # a sequence of ncond given states, a sequence of npred actions,
    # and a sequence of npred states to be predicted
    def get_batch_fm(self, split, npred=-1, cuda=True):

        if split == 'train':
            indx = self.train_indx
        elif split == 'valid':
            indx = self.valid_indx
        elif split == 'test':
            indx = self.test_indx

        if npred == -1:
            npred = self.opt.npred

        images, states, actions, costs, ids, sizes = [], [], [], [], [], []
        nb = 0
        while nb < self.opt.batch_size:
            s = self.random.choice(indx)
            # min is important since sometimes numbers do not align causing issues in stack operation below
            T = min(self.images[s].size(0), self.states[s].size(0))  
            if T > (self.opt.ncond + npred + 1): 
                t = self.random.randint(0, T - (self.opt.ncond+npred+1))
                images.append(self.images[s][t:t+(self.opt.ncond+npred)+1].cuda())
                actions.append(self.actions[s][t:t+(self.opt.ncond+npred)].cuda())
                states.append(self.states[s][t:t+(self.opt.ncond+npred)+1].cuda())
                costs.append(self.costs[s][t:t+(self.opt.ncond+npred)+1].cuda())
                ids.append(self.ids[s])
                splits = self.ids[s].split('/')
                timeslot = splits[3]
                car_id = int(re.findall('car(\d+).pkl', splits[4])[0])
                size = self.car_sizes[timeslot][car_id]
                sizes.append([size[0], size[1]])
                nb += 1

        images = torch.stack(images).float()
        images.div_(255.0)

        states = torch.stack(states)
        states = states[:, :, 0].contiguous()

        actions = torch.stack(actions)
        sizes = torch.tensor(sizes)

        if not self.opt.debug:
            actions -= self.a_mean.view(1, 1, 2).expand(actions.size()).cuda()
            actions /= (1e-8 + self.a_std.view(1, 1, 2).expand(actions.size())).cuda()
            states -= self.s_mean.view(1, 1, 4).expand(states.size()).cuda()
            states /= (1e-8 + self.s_std.view(1, 1, 4).expand(states.size())).cuda()

        costs = torch.stack(costs)

        # |-----ncond-----||------------npred------------||
        # ^                ^                              ^
        # 0               t0                             t1
        t0 = self.opt.ncond
        t1 = t0 + npred
        input_images  = images [:,   :t0].float().contiguous()
        input_states  = states [:,   :t0].float().contiguous()
        target_images = images [:, t0:t1].float().contiguous()
        target_states = states [:, t0:t1].float().contiguous()
        target_costs  = costs  [:, t0:t1].float().contiguous()
        t0 -= 1; t1 -= 1
        actions       = actions[:, t0:t1].float().contiguous()
        # input_actions = actions[:, :t0].float().contiguous()

        if not cuda:
            input_images = input_images.cpu()
            actions = actions.cpu()
            target_images = target_images.cpu()

        return [input_images, input_states], actions, [target_images, target_states, target_costs], ids, sizes
