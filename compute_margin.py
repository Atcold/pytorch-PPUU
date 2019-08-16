from dataloader import DataLoader
from models import EnergyNet
import torch
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true', help='Disables cuda')
    parser.add_argument('-debug', action='store_true', help='Short execution + print energies')
    parser.add_argument('-state_stats', action='store_true', help='Print stats, if debugging')
    options = parser.parse_args()

    if torch.cuda.is_available() and options.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device('cpu' if options.no_cuda or not torch.cuda.is_available() else 'cuda:0')

    # Data set setting (which are not used)
    class DataSettings:
        debug = False
        batch_size = 4
        npred = 20
        ncond = 10


    # Instantiate data set object
    data_loader = DataLoader(None, opt=DataSettings, dataset='i80')
    # Ego car episodic position and velocity
    train_state_vectors = tuple(data_loader.states[idx][:, 0] for idx in data_loader.train_indx)
    # Episodic context images
    train_state_images = tuple(data_loader.images[idx] for idx in data_loader.train_indx)

    # Generate energy model
    energy_net = EnergyNet().to(device)

    print(' > Compute energies for training states')
    train_size = len(train_state_vectors)
    energy_tensor = torch.Tensor(train_size, 2).to(device)
    tot_nb_states = 0
    train_states = zip(train_state_vectors, train_state_images)

    if options.debug: tqdm = lambda iterable, total: iterable

    for episode, (state_vector, state_image) in tqdm(enumerate(train_states), total=train_size):
        episode_length = min(state_vector.size(0), state_image.size(0))
        tot_nb_states += episode_length
        state_vector = state_vector[:episode_length]  # uniform the length
        state_image = state_image[:episode_length]  # uniform the length
        state_image = data_loader.normalise_state_image(state_image.to(device))  # normalise [0, 255] -> [0., 1.]
        state_vector = data_loader.normalise_state_vector(state_vector.to(device))  # normalise -> N(0, 1)
        with torch.no_grad():
            energy_vct, energy_img = energy_net(state_vector=state_vector, state_image=state_image)
        energy_tensor[episode][0].copy_(energy_vct).mul_(episode_length)
        energy_tensor[episode][1].copy_(energy_img).mul_(episode_length)

        # Check state stats
        if options.debug:
            n = 20
            print(f'{episode:2d} (vct, img): ({energy_vct.item():.3f}, {energy_img.item():.3f})')
            if episode == n - 1:
                e = energy_tensor[:n].sum(0).div_(tot_nb_states)
                print(f'Average energy (vct, img): ({e[0]:.3f}, {e[1]:.3f})')
                quit()
        if options.debug and options.state_stats:
            vct = state_vector.mean(0).cpu(), state_vector.std(0).cpu()  # 4-dim vector each
            img = state_image.view(-1).mean().item(), state_image.view(-1).std().item()  # scalar each
            print(f'vct stats (mean, std): ({vct[0]}, {vct[1]}')
            print(f'img stats (mean, std): ({img[0]:.3f}, {img[1]:.3f})')

    print(' > Compute energies average and saving to disk')
    energy_average = energy_tensor.sum(0).div_(tot_nb_states).cpu()
    print(f'Average energy (vct, img): ({energy_average[0]:.3f}, {energy_average[1]:.3f})')
    torch.save(dict(
        energy_vct=energy_average[0],
        energy_img=energy_average[1],
    ), 'energy_average.pth')
