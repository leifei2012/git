
import tensorflow as tf
import numpy as np


# a=np.ones([2,3,5])
# b=tf.placeholder(dtype=tf.float32,shape=[2,3,5])

# c=tf.nn.rnn_cell.BasicLSTMCell(5)
# init_state = c.zero_state(2, dtype=tf.float32)
# out,state=tf.nn.dynamic_rnn(c,b,initial_state=init_state)
# sess = tf.Session()
# sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
# o=sess.run(out, feed_dict={b:a})
# print(np.array(o).shape,state)
batch_size = 4 
inp=np.ones([2,4,6])
input = tf.random_normal(shape=[2, 4, 6], dtype=tf.float32)
cell = tf.nn.rnn_cell.BasicLSTMCell(10, forget_bias=1.0, state_is_tuple=True)
init_state = cell.zero_state(batch_size, dtype=tf.float32)
output, final_state = tf.nn.dynamic_rnn(cell, input, initial_state=init_state, time_major=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    o,s=sess.run([output,final_state], feed_dict={input:inp})
    print(np.array(o).shape,final_state)