

import tensorflow as tf
import numpy as np


a=np.ones([2,3,5])
b=tf.placeholder(dtype=tf.float32,shape=[2,3,5])

c=tf.nn.rnn_cell.BasicLSTMCell(5)
init_state = c.zero_state(2, dtype=tf.float32)
out,state=tf.nn.dynamic_rnn(c,b,initial_state=init_state)
sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
o=sess.run([out], feed_dict={b:a})
print(np.array(o).shape,state)