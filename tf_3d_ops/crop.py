import tensorflow as tf
def _crop_unbatched(image, start, size):
    d, h, w = start
    i, j, k = size

    return image[d:d+i, h:h+j, w:w+k]

def _crop_batched(images, start, size):
    return tf.map_fn(lambda x: _crop_unbatched(x, start, size), images)