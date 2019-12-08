# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:13:25 2019

@author: Raghav N G
"""
import tensorflow as tf
with tf.Session() as session:
    my_saver = tf.train.import_meta_graph('model.tflearn.meta')
    my_saver.restore(session, tf.train.latest_checkpoint('.'))
    
    # Rest of the code goes here
    output_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    #print(output_node_names)
    frozen_graph = tf.graph_util.convert_variables_to_constants(
    session,
    session.graph_def,
    ['my_output/Softmax']
    )
    
    with open('frozen_model.pb', 'wb') as f:
        f.write(frozen_graph.SerializeToString())
    