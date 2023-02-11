import tensorflow as tf
import tensorflow.keras.backend as K
import warnings




def resize(image, output_size, method='nearest'):
    assert len(image.shape) == 5, f"Unexpected shape {image.shape}, expected (b, d, h, w, c)" 
    assert K.image_data_format() == 'channels_last', "Only channels_last format is supported"
    d_out, h_out, w_out = output_size
    # Resize Along Depth
    unstack_img_depth_list = tf.unstack(image, axis=1)
    resized_list = []
    for i in unstack_img_depth_list:
        resized_list.append(tf.image.resize(i, [h_out, w_out], method=method))
    stack_img = tf.stack(resized_list, axis=1)

    unstack_img_height_list = tf.unstack(stack_img, axis=2)
    resized_list = []
    for i in unstack_img_height_list:
        resized_list.append(tf.image.resize(i, [d_out, w_out], method=method))
    stack_img = tf.stack(resized_list, axis=2)
    return stack_img