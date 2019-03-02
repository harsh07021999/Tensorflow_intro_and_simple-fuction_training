import matplotlib.pyplot as plt

import tensorflow as tf

import numpy as np

def add_layer(inputs,in_size,out_size,activation_function=None):                           ##function to add layers to the model
        with tf.name_scope("layers"):
                Weights=tf.Variable(tf.random_normal([in_size,out_size]))
                baises=tf.Variable(tf.zeros([1,out_size])+0.1)
                Wx_plus_b=tf.add(tf.matmul(inputs,Weights) ,baises)
                if activation_function is None:
                        outputs=Wx_plus_b
                else:
                        outputs = activation_function(Wx_plus_b)
                return outputs
x_data = np.linspace(-1.5,1.5,500, dtype=np.float32)[:,np.newaxis]                              ## creation of x data 
noise=np.random.normal(0,0.05,x_data.shape)
y_data= np.multiply(np.square(x_data),x_data)+noise+x_data     ##x^3+x is the function             also creation of y data
##plt.scatter(x_data,y_data)
##plt.show()
with tf.name_scope("inputs"):
        xs=tf.placeholder(tf.float32,[None,1],name="X_input")                       ## creating placeholders
        ys=tf.placeholder(tf.float32,[None,1],name="Y_input")                       ## creating placeholders
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)                                    ## first layer with activation function as relu function
l2=add_layer(l1,10,1,activation_function=None)                                            ## final layer
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-l2),reduction_indices=[1]))     ## loss function 
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)                          ## GD optimiser to minimise loss
init = tf.global_variables_initializer()                                                                                ## intialisation 
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
                prediction_values=sess.run(l2,feed_dict={xs:x_data})    ## feeding values to placeholders
                lines=ax.plot(x_data,prediction_values,'r-',lw=4)
                plt.pause(0.35)                                                                            ## to see the transition of learning curve 
               #print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
