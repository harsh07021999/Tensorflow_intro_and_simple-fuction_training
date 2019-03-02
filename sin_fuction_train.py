import matplotlib.pyplot as plt

import tensorflow as tf

import numpy as np

def add_layer(inputs,in_size,out_size,activation_function=None):
        with tf.name_scope("layers"):
                Weights=tf.Variable(tf.random_normal([in_size,out_size]))
                baises=tf.Variable(tf.zeros([1,out_size])+0.1)
                Wx_plus_b=tf.add(tf.matmul(inputs,Weights) ,baises)
                if activation_function is None:
                        outputs=Wx_plus_b
                else:
                        outputs = activation_function(Wx_plus_b)
                return outputs
x_data = np.linspace(-1.5,1.5,300, dtype=np.float32)[:,np.newaxis]
noise=np.random.normal(0,0.075,x_data.shape)
y_data= np.sin(x_data)+noise                                   ## simple sinusoidal curve as data to train
##plt.scatter(x_data,y_data)
##plt.show()
with tf.name_scope("inputs"):
        xs=tf.placeholder(tf.float32,[None,1],name="X_input")
        ys=tf.placeholder(tf.float32,[None,1],name="Y_input")
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
l2=add_layer(l1,10,1,activation_function=None)
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-l2),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()
sess=tf.Session()
#writer =tf.summary.FileWriter("logs/",sess.graph)
sess.run(init)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

for ri in range(1000):
        sess.run(train_step,feed_dict={xs: x_data , ys: y_data})
        if ri %50==0:
                try:
                        ax.lines.remove(lines[0])
                        #print('r')
                except Exception:
                       pass
                prediction_values=sess.run(l2,feed_dict={xs:x_data})
                lines=ax.plot(x_data,prediction_values,'r-',lw=4)
                plt.pause(0.35)
               #print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
