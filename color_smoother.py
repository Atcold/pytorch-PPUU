import cv2
import matplotlib.image as Image
kernel_size = (7, 7)
trajectory_image = Image.imread('actrajectory.jpg')/255
trajectory_image[:, :, 0] = cv2.GaussianBlur(trajectory_image[:, :, 0], kernel_size, 2.33)
trajectory_image[:, :, 1] = cv2.GaussianBlur(trajectory_image[:, :, 1], kernel_size, 2.33)
trajectory_image[:, :, 2] = cv2.GaussianBlur(trajectory_image[:, :, 2], kernel_size, 2.33)
Image.imsave("7gactrajectory.jpg", trajectory_image)
