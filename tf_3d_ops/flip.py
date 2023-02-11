import tensorflow as tf
from tensorflow.keras import backend as K

def _flip_unbatched(image, flip_dimension):
    if flip_dimension == 'horizontal':
        flipped_image = tf.reverse(image, axis=[2])
    elif flip_dimension == 'vertical':
        flipped_image = tf.reverse(image, axis=[1])
    elif flip_dimension == 'depth':
        flipped_image = tf.reverse(image, axis=[0])
    else:
        raise ValueError("Invalid flip_dimension. Should be one of 'horizontal', 'vertical', 'depth'")
    return flipped_image

def _flip_batched(image, flip_dimension):
    return tf.map_fn(lambda x: _flip_unbatched(x, flip_dimension), image)

def flip(image, flip_dimension):
    assert K.image_data_format() == "channels_last", "Only channels_last format is supported"
    image_shape = K.int_shape(image)
    assert len(image_shape) in [4, 5], f"Invalid data format {image_shape}"
    if len(K.int_shape(image)) == 5:
        return _flip_batched(image, flip_dimension)
    else:
         return _flip_unbatched(image, flip_dimension=flip_dimension)


def flip_horizontal(image):
    return flip(image, flip_dimension='horizontal')

def flip_vertical(image):
    return flip(image, flip_dimension='vertical')

def flip_depth(image):
    return flip(image, flip_dimension='depth')