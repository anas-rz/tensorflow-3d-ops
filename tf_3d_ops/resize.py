import tensorflow as tf


def resize_by_axis(image, dim_1, dim_2, ax, is_grayscale):
# https://stackoverflow.com/questions/43814367/resize-3d-data-in-tensorflow-like-tf-image-resize-images
    resized_list = []


    if is_grayscale:
        unstack_img_depth_list = [tf.expand_dims(x,2) for x in tf.unstack(image, axis = ax)]
        for i in unstack_img_depth_list:
            resized_list.append(tf.image.resize(i, [dim_1, dim_2]))
        stack_img = tf.squeeze(tf.stack(resized_list, axis=ax))
        print(stack_img.get_shape())

    else:
        unstack_img_depth_list = tf.unstack(image, axis = ax)
        for i in unstack_img_depth_list:
            resized_list.append(tf.image.resize(i, [dim_1, dim_2]))
        stack_img = tf.stack(resized_list, axis=ax)
    return stack_img