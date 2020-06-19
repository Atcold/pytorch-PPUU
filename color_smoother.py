import cv2
import numpy as np
kernel_size = (7,7)
trajectory_image=cv2.imread('1trajectory.jpg')
trajectory_image = trajectory_image.astype(np.uint8)
trajectory_image[:, :, 0] = cv2.GaussianBlur(trajectory_image[:, :, 0], kernel_size, 3)
trajectory_image[:, :, 1] = cv2.GaussianBlur(trajectory_image[:, :, 1], kernel_size, 3)
trajectory_image[:, :, 2] = cv2.GaussianBlur(trajectory_image[:, :, 2], kernel_size, 3)
cv2.imwrite("7gtrajectory.jpg",trajectory_image)