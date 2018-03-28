import torch, numpy, argparse, pdb, os, time, random
import models, utils
from dataloader import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from sklearn import decomposition

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-data_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/data/')
parser.add_argument('-model_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/')
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-ncond', type=int, default=10)
parser.add_argument('-nz', type=int, default=32)
parser.add_argument('-nfeature', type=int, default=64)
parser.add_argument('-n_hidden', type=int, default=100)
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-epoch_size', type=int, default=2000)
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

mfile = 'model=fwd-cnn-ae-fp-bsize=16-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-warmstart=1.model'
model = torch.load(opt.model_dir + mfile)
opt.nz = model.opt.nz
model.q_network = models.z_network_full(opt, opt.ncond) #TODO: add actions?
model.intype('gpu')
optimizer = optim.Adam(model.q_network.parameters(), 0.001)



def forward_z_network(model, inputs, actions, targets):
    bsize = inputs.size(0)
    inputs = inputs.view(bsize, model.opt.ncond, 3, model.opt.height, model.opt.width)
    actions = actions.view(bsize, -1, model.opt.n_actions)
    npred = actions.size(1)
    
    loss = Variable(torch.zeros(1))
    if model.use_cuda:
        loss = loss.cuda()

    pred = []
    inputs_list, mu_list, sigma_list, z_list = [], [], [], []
    for t in range(npred):
        h_x = model.encoder(inputs, actions[:, t])
        h_y = model.y_encoder(targets[:, t].unsqueeze(1).contiguous())
        z = model.z_network((h_x + h_y).view(bsize, -1))
        if opt.loss == 'pdf':
            mu, sigma = model.q_network(inputs)
            loss += utils.log_pdf(z, mu, sigma)
            mu_list.append(mu)
            sigma_list.append(sigma)
        elif opt.loss == 'nll':
            u = model.q_network(inputs)
            e = torch.mm(u, z.t())
            log_p = F.log_softmax(e, dim=1)
            loss += F.nll_loss(log_p, Variable(torch.eye(bsize).cuda()))

        z_exp = model.z_expander(z)
        h = h_x + z_exp.squeeze()
        pred_ = F.sigmoid(model.decoder(h) + inputs[:, -1].unsqueeze(1).clone())
        pred.append(pred_)
        z_list.append(z)
        inputs_list.append(inputs)
        inputs = torch.cat((inputs[:, 1:], pred_), 1)

    pred = torch.cat(pred, 1)
    return loss, inputs_list, mu_list, sigma_list, z_list


def train(nbatches):
#    model.train()
    total_loss = 0
    for i in range(nbatches):
        inputs, actions, targets, _, _ = dataloader.get_batch_fm('train', 20)
        inputs = Variable(inputs)
        actions = Variable(actions)
        targets = Variable(targets)
        optimizer.zero_grad()
        loss, inputs_list, mu_list, sigma_list, z_list = forward_z_network(model, inputs, actions, targets)
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
        loss, inputs_list, mu_list, sigma_list, z_list = forward_z_network(model, inputs, actions, targets)
        total_loss += loss.data[0]
    return total_loss / nbatches
    

for i in range(1000):
    loss_train = train(50)
    loss_test = test(50)
    print(f'epoch {i} | train: {loss_train:.5f}, test: {loss_test:.5f}')


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
