import scipy.misc
import cv2, pdb, numpy, os, argparse
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


width, height = 24, 117


parser = argparse.ArgumentParser()
# data params
parser.add_argument('-path_vae', type=str, default='/home/mbhenaff/scratch/models_v11/eval/model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step200000.model-nbatches=200-npred=200-nsample=10.eval/videos/')
parser.add_argument('-path_det', type=str, default='/home/mbhenaff/scratch/models_v11/eval/model=fwd-cnn-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-gclip=5.0-warmstart=0-seed=1.step200000.model-nbatches=200-npred=200-nsample=10.eval/videos/')
parser.add_argument('-x', type=int, default=0)
parser.add_argument('-z', type=int, default=0)
parser.add_argument('-T', type=int, default=200)
parser.add_argument('-fig', type=int, default=1)
opt = parser.parse_args()


if opt.fig == 0:
    video_dir = f'videos_website/vae/x{opt.x}-T{opt.T}/'
    os.system('mkdir -p ' + video_dir)
    os.system(f'rm {video_dir}/*')
    dir1 = f'{opt.path_vae}/x{opt.x}/y/'
    dir2 = f'{opt.path_det}/sampled_z/true_actions/x{opt.x}/z{opt.z}/'
    dir3 = f'{opt.path_vae}/sampled_z/true_actions/x{opt.x}/z{opt.z}/'
    dir4 = f'{opt.path_vae}/sampled_z/true_actions/x{opt.x}/z{opt.z+1}/'
elif opt.fig == 1:
    video_dir = f'videos_website/sens-zdropout/x{opt.x}-T{opt.T}/'
    os.system('mkdir -p ' + video_dir)
    os.system(f'rm {video_dir}/*')
    dir1 = f'{opt.path_vae}/x{opt.x}/y/'
    dir2 = f'{opt.path_vae}/sampled_z/true_actions/x{opt.x}/z{opt.z}/'
    dir3 = f'{opt.path_vae}/true_z/perm_actions/x{opt.x}/z{opt.z}/'
    dir4 = f'{opt.path_vae}/sampled_z/perm_actions/x{opt.x}/z{opt.z}/'


img_list1 = ['{}/im{:05d}.png'.format(dir1, t) for t in range(200)]
img_list2 = ['{}/im{:05d}.png'.format(dir2, t) for t in range(200)]
img_list3 = ['{}/im{:05d}.png'.format(dir3, t) for t in range(200)]
img_list4 = ['{}/im{:05d}.png'.format(dir4, t) for t in range(200)]


for t in range(opt.T):
    image1 = cv2.imread(img_list1[t])
    image2 = cv2.imread(img_list2[t])
    image3 = cv2.imread(img_list3[t])
    image4 = cv2.imread(img_list4[t])

    color1 = [101, 52, 152] # 'cause purple!
    b1 = 1
    b2 = 20

    image1 = cv2.copyMakeBorder(image1, b1, b1, b1, b1, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    image1 = cv2.copyMakeBorder(image1, b2, b2, b2, b2, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    image2 = cv2.copyMakeBorder(image2, b1, b1, b1, b1, cv2.BORDER_CONSTANT, value=color1)
    image2 = cv2.copyMakeBorder(image2, b2, b2, b2, b2, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    image3 = cv2.copyMakeBorder(image3, b1, b1, b1, b1, cv2.BORDER_CONSTANT, value=color1)
    image3 = cv2.copyMakeBorder(image3, b2, b2, b2, b2, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    image4 = cv2.copyMakeBorder(image4, b1, b1, b1, b1, cv2.BORDER_CONSTANT, value=color1)
    image4 = cv2.copyMakeBorder(image4, b2, b2, b2, b2, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    vis = numpy.concatenate((numpy.zeros((image1.shape[0], 30, 3)), image1, image2, image3, image4), axis=1)
    b3 = 20
    vis = cv2.copyMakeBorder(vis, b3, b3, b3, b3, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    cv2.putText(vis, 'Ground truth', (90, 20), 0, 0.4, (155, 155, 155), 1, cv2.LINE_AA)
    cv2.putText(vis, 'Sampled Z-True A', (220, 20), 0, 0.4, (155, 155, 155), 1, cv2.LINE_AA)
    cv2.putText(vis, 'Inferred Z-Sampled A', (380, 20), 0, 0.4, (155, 155, 155), 1, cv2.LINE_AA)
    cv2.putText(vis, 'Sampled Z-Sampled A', (550, 20), 0, 0.4, (155, 155, 155), 1, cv2.LINE_AA)
    cv2.putText(vis, 't={}'.format(t), (10, 350), 0, 0.5, (155, 155, 155), 1, cv2.LINE_AA)
    cv2.arrowedLine(vis, (500, 240), (500, 180), (150,150,150), 1)
    cv2.imwrite(f'{video_dir}/im{t:05d}.png', vis)
