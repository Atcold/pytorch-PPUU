import torch, numpy, argparse, pdb, os, time, random
import models, utils
from dataloader import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from sklearn import decomposition

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-loss', type=str, default='sphere')
parser.add_argument('-data_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/data/')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/')
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-ncond', type=int, default=10)
parser.add_argument('-nz', type=int, default=32)
parser.add_argument('-u_sphere', type=int, default=0)
parser.add_argument('-nfeature', type=int, default=64)
parser.add_argument('-n_hidden', type=int, default=100)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-epoch_size', type=int, default=2000)
#parser.add_argument('-mfile', type=str, default='model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-zsphere=0-gclip=-1-warmstart=1.model')
parser.add_argument('-mfile', type=str, default='model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-warmstart=1.model')
#parser.add_argument('-mfile', type=str, default='model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-zsphere=1-gclip=-1-warmstart=1.model')
opt = parser.parse_args()
opt.model_dir += f'/dataset_{opt.dataset}/models/'


if opt.dataset == 'simulator':
    opt.height = 97
    opt.width = 20
    opt.h_height = 12
    opt.h_width = 2

elif opt.dataset == 'i80':
    opt.height = 117
    opt.width = 24
    opt.h_height = 14
    opt.h_width = 3

dataloader = DataLoader(None, opt, opt.dataset)


model = torch.load(opt.model_dir + opt.mfile)
opt.nz = model.opt.nz
if opt.loss == 'pdf':
    model.u_network = models.u_network_gaussian(opt, opt.ncond) #TODO: add actions?
else:
    model.u_network = models.u_network(opt, opt.ncond) #TODO: add actions?
model.intype('gpu')
optimizer = optim.Adam(model.u_network.parameters(), 0.001)

mfile_prior = f'{opt.model_dir}/{opt.mfile}-loss={opt.loss}-usphere={opt.u_sphere}-nfeature={opt.nfeature}.prior'
print(f'[will save prior model as: {mfile_prior}]')


def forward_u_network(model, inputs, actions, targets):
    bsize = inputs.size(0)
    inputs = inputs.view(bsize, model.opt.ncond, 3, model.opt.height, model.opt.width)
    actions = actions.view(bsize, -1, model.opt.n_actions)
    npred = actions.size(1)
    
    eye = Variable(torch.eye(bsize))
    loss = Variable(torch.zeros(1))
    if model.use_cuda:
        loss = loss.cuda()
        eye = eye.cuda()

    pred = []
    inputs_list, z_list, u_list = [], [], []
    for t in range(npred):
        h_x = model.encoder(inputs, actions[:, t])
        h_y = model.y_encoder(targets[:, t].unsqueeze(1).contiguous())
        z = model.z_network((h_x + h_y).view(bsize, -1))
        if opt.loss == 'pdf':
            mu, sigma = model.u_network(inputs)
            loss_t = utils.log_pdf(z, mu, sigma)
            loss += torch.mean(loss_t)
            u_list.append({'mu': mu, 'sigma': sigma})
        elif opt.loss == 'nll':
            u = model.u_network(inputs)
            if opt.u_sphere == 1:
                u = u / torch.norm(u, 2, 1).view(-1, 1).expand(u.size())
            e = torch.mm(u, z.t())
            log_p = F.log_softmax(e, dim=1)
            loss += F.nll_loss(log_p, Variable(torch.arange(bsize).cuda().long()))
            u_list.append(u)
        elif opt.loss == 'nll-w':
            us = model.u_network(inputs)
            u = us[:, :model.opt.nz]
            if opt.u_sphere == 1:
                u = u / torch.norm(u, 2, 1).view(-1, 1).expand(u.size())
            s = F.softplus(us[:, -1])
            e = torch.mm(u, z.t())
            e *= s.clone().view(-1, 1).expand(e.size())
            log_p = F.log_softmax(e, dim=1)
            loss += F.nll_loss(log_p, Variable(torch.arange(bsize).cuda().long()))
            u_list.append(u)
        elif opt.loss == 'sphere':
            u = model.u_network(inputs)
            e = torch.mm(u, z.t())
            loss += F.mse_loss(e, eye)
            u_list.append(u)


        z_exp = model.z_expander(z)
        h = h_x + z_exp.squeeze()
        pred_ = F.sigmoid(model.decoder(h) + inputs[:, -1].unsqueeze(1).clone())
        pred.append(pred_)
        z_list.append(z)
        inputs_list.append(inputs)
        inputs = torch.cat((inputs[:, 1:], pred_), 1)

    pred = torch.cat(pred, 1)
    loss /= npred
    return loss, inputs_list, z_list, u_list


def train(nbatches):
#    model.train()
    total_loss = 0
    for i in range(nbatches):
        inputs, actions, targets, _, _ = dataloader.get_batch_fm('train', 50)
        inputs = Variable(inputs)
        actions = Variable(actions)
        targets = Variable(targets)
        optimizer.zero_grad()
        loss, inputs_list, z_list, q_list = forward_u_network(model, inputs, actions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.data[0]
    return total_loss / nbatches

def test(nbatches):
#    model.eval()
    total_loss = 0
    for i in range(nbatches):
        inputs, actions, targets, _, _ = dataloader.get_batch_fm('valid', 20)
        inputs = Variable(inputs)
        actions = Variable(actions)
        targets = Variable(targets)
        loss, inputs_list, z_list, q_list = forward_u_network(model, inputs, actions, targets)
        total_loss += loss.data[0]
    return total_loss / nbatches
    

for i in range(500):
    loss_train = train(100)
    loss_test = test(100)
    log_string = f'epoch {i} | train loss: {loss_train:.5f}, test loss: {loss_test:.5f}'
    print(log_string)
    utils.log(mfile_prior + '.log', log_string)
    model.intype('cpu')
    model.u_network.cpu()
    torch.save(model.u_network, mfile_prior)
    model.intype('gpu')
    model.u_network.cuda()




'''
z_list, mu_list, sigma_list, inputs_list = [], [], [], []
model.intype('cpu')
for i in range(50):
    print(i)
    inputs, actions, targets, _, _ = dataloader.get_batch_fm('test', 20)
    inputs = Variable(inputs.cpu())
    actions = Variable(actions.cpu())
    targets = Variable(targets.cpu())
    loss, inputs_list_, mu_list_, sigma_list_, z_list_ = forward_z_network(model, inputs, actions, targets)
    z_list += [torch.stack(z_list_).cpu()]
    mu_list += [torch.stack(mu_list_).cpu()]
    sigma_list += [torch.stack(sigma_list_).cpu()]
    inputs_list += [torch.stack(inputs_list_).cpu()]
z_list = torch.stack(z_list).view(-1, opt.nz).data
mu_list = torch.stack(mu_list).view(-1, opt.nz).data
sigma_list = torch.stack(sigma_list).view(-1, opt.nz).data
inputs_list = torch.stack(inputs_list).data
z_list = z_list.numpy()
k = 200
m = 3
u = torch.randn(mu_list.size(0), k, opt.nz)
u *= sigma_list.view(-1, 1, opt.nz)
u += mu_list.view(-1, 1, opt.nz)
def sample():
    m = 5
    u_list = []
    for i in range(m):
        u_list.append(random.choice(u))
    U=torch.stack(u_list).view(-1, opt.nz)
    pca = decomposition.PCA(n_components=2)
    pca.fit(z_list)
    Z = pca.transform(z_list)
    U = pca.transform(U.numpy())
    torch.save({'Zpca': Z, 'Upca': U}, 'pca.pth')

torch.save(model, opt.model_dir + mfile + '.pz')
'''
