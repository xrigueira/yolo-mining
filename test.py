import cv2
import tensorflow as tf
print(cv2.__version__)
print(cv2.cuda.getCudaEnabledDeviceCount())

print(tf.__version__)
print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))