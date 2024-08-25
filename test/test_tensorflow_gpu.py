import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print('1', tf.config.list_physical_devices('GPU'))
print('2', tf.test.is_built_with_cuda())
print('3', tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())