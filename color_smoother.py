import cv2
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import matplotlib.image as Image
kernel_size = (7, 7)
trajectory_image = Image.imread('trajectory.jpg')/255
trajectory_image = rgb_to_hsv(trajectory_image)
trajectory_image[:, :, 0] = cv2.GaussianBlur(trajectory_image[:, :, 0], kernel_size, 2.33)
trajectory_image[:, :, 1] = cv2.GaussianBlur(trajectory_image[:, :, 1], kernel_size, 2.33)
trajectory_image[:, :, 2] = cv2.GaussianBlur(trajectory_image[:, :, 2], kernel_size, 2.33)
trajectory_image = hsv_to_rgb(trajectory_image)
Image.imsave("5gtrajectory.jpg", trajectory_image)
