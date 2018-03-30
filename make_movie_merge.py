import scipy.misc
import cv2, pdb, numpy, os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def add_text(img):
    img = Image.open("sample.png")
    draw = ImageDraw.Draw(img)
    draw.text((0, 0),"Sample Text",(255,255,255))
    img.save('sample-out.png')


width, height = 24, 117

mfile1 = 'all_videos/videos7/model=fwd-cnn-bsize=32-ncond=10-npred=30-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-warmstart=0.model'
#mfile2 = 'videos1/model=fwd-cnn-een-fp-bsize=32-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=16-warmstart=1.model'
mfile2 = 'all_videos/videos7/model=fwd-cnn-een-fp-bsize=32-ncond=10-npred=20-lrt=0.0001-nhidden=100-nfeature=96-tieact=0-nz=32-warmstart=1.model'

k = 2
m = 1
os.system('mkdir -p all_videos/movie_merge')
dir1 = f'{mfile1}/pred{k}/sample0/'


dir2 = f'{mfile2}/pred{k}/sample15/'
dir3 = f'{mfile2}/pred{k}/sample4/'
dir4 = f'{mfile2}/pred{k}/sample0/'

#dir2 = f'{mfile2}/pred{k}/sample9/'
#dir3 = f'{mfile2}/pred{k}/sample15/'
#dir4 = f'{mfile2}/pred{k}/sample18/'
dir5 = f'{mfile2}/pred{k}/sample3/'
dir6 = f'{mfile2}/pred{k}/sample4/'

img_list1 = ['{}/im{:05d}.png'.format(dir1, t) for t in range(200)]
img_list2 = ['{}/im{:05d}.png'.format(dir2, t) for t in range(200)]
img_list3 = ['{}/im{:05d}.png'.format(dir3, t) for t in range(200)]
img_list4 = ['{}/im{:05d}.png'.format(dir4, t) for t in range(200)]


for t in range(200):
    image1 = cv2.imread(img_list1[t])
    image2 = cv2.imread(img_list2[t])
    image3 = cv2.imread(img_list3[t])
    image4 = cv2.imread(img_list4[t])

    color1 = [101, 52, 152] # 'cause purple!
    b1 = 1
    b2 = 10

    M = cv2.getRotationMatrix2D((width/2,height/2),180,1)

    image1 = cv2.warpAffine(image1,M,(width,height))
    image1 = cv2.copyMakeBorder(image1, b1, b1, b1, b1, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    image1 = cv2.copyMakeBorder(image1, b2, b2, b2, b2, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    image2 = cv2.warpAffine(image2,M,(width,height))
    image2 = cv2.copyMakeBorder(image2, b1, b1, b1, b1, cv2.BORDER_CONSTANT, value=color1)
    image2 = cv2.copyMakeBorder(image2, b2, b2, b2, b2, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    image3 = cv2.warpAffine(image3,M,(width,height))
    image3 = cv2.copyMakeBorder(image3, b1, b1, b1, b1, cv2.BORDER_CONSTANT, value=color1)
    image3 = cv2.copyMakeBorder(image3, b2, b2, b2, b2, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    image4 = cv2.warpAffine(image4,M,(width,height))
    image4 = cv2.copyMakeBorder(image4, b1, b1, b1, b1, cv2.BORDER_CONSTANT, value=color1)
    image4 = cv2.copyMakeBorder(image4, b2, b2, b2, b2, cv2.BORDER_CONSTANT, value=[0, 0, 0])


    vis = numpy.concatenate((image1, image2, image3, image4), axis=1)
    b3 = 20
    vis = cv2.copyMakeBorder(vis, b3, b3, b3, b3, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    vis = cv2.resize(vis, (3*vis.shape[0], 2*vis.shape[1]))
    cv2.putText(vis, 'CNN', (80, 60), 0, 0.4, (155, 155, 155), 2, cv2.LINE_AA)
    cv2.putText(vis, 'EEN, z1', (190, 60), 0, 0.4, (155, 155, 155), 2, cv2.LINE_AA)
    cv2.putText(vis, 'EEN, z2', (300, 60), 0, 0.4, (155, 155, 155), 2, cv2.LINE_AA)
    cv2.putText(vis, 'EEN, z3', (410, 60), 0, 0.4, (155, 155, 155), 2, cv2.LINE_AA)
    cv2.putText(vis, 't={}'.format(t), (20, 220), 0, 0.5, (155, 155, 155), 2, cv2.LINE_AA)
    cv2.arrowedLine(vis, (500, 240), (500, 180), (150,150,150), 1)
    cv2.imwrite('all_videos/movie_merge/test{:05d}.png'.format(t), vis)
