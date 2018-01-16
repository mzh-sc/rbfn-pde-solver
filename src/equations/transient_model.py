from collections import namedtuple

import tensorflow as tf
import tf_utils as tf_ext

def restore_transient_model():
    return TransientModel(
        vr_weights=tf_ext.get_variable('weights'),
        vr_centers=tf_ext.get_variable('centers'),
        vr_parameters=tf_ext.get_variable('parameters'),
        op_loss=tf.get_collection('op_loss')[0],
        op_model_y=tf.get_collection('op_model_y')[0],
        pl_x_of_y=tf.get_collection('pl_x_of_y')[0]
    )

def store_transient_model(model):
    tf.add_to_collection('vr_weights', model.vr_weights)
    tf.add_to_collection('vr_centers', model.vr_centers)
    tf.add_to_collection('vr_parameters', model.vr_parameters)
    tf.add_to_collection('op_loss', model.op_loss)
    tf.add_to_collection('op_model_y', model.op_model_y)
    tf.add_to_collection('pl_x_of_y', model.pl_x_of_y)

TransientModel = namedtuple('TransientModel',
                            [
                                'vr_weights',
                                'vr_centers',
                                'vr_parameters',
                                'op_loss',
                                'op_model_y',
                                'pl_x_of_y'
                            ])
TransientModel.restore = restore_transient_model
TransientModel.save = store_transient_model
