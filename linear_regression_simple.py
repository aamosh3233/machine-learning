from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#values
train_X = np.asarray( [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]
#parameters
epoch=1000
batches=50
#Placeholders
X=tf.placeholder("float")
Y=tf.placeholder("float")
w=tf.Variable(np.random.rand(),name='weight')
b=tf.Variable(np.random.rand(),name='bias')
pred=tf.add(tf.multiply(w,X),b)
#optimizing loss
loss=tf.reduce_sum(tf.pow(pred-Y,2))/(2*n_samples)


optimizer=tf.train.GradientDescentOptimizer(0.01).minimize(loss=loss)
init=tf.global_variables_initializer()
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})

        if (i+1)%50==0:
            c=sess.run(loss,feed_dict={X:train_X,Y:train_Y})
            print('epoch=',(i+1),'c='.format(c),'b=',sess.run(b),'w=',sess.run(b))

            try:
                ax.lines.remove(lines[0])
            except:
                pass
            ax.plot(train_X,train_Y,'ro')
            lines=ax.plot(train_X,train_X*sess.run(w)+sess.run(b))
            plt.pause(0.5)


    print('finished!!!!')
    training_cost = sess.run(loss, feed_dict={X: train_X, Y: train_Y})
    print(training_cost,'w=',sess.run(w),'b=',sess.run(b))


    ax.plot(train_X,train_Y,'ro',label='original data')
    ax.plot(train_X,train_X*sess.run(w)+sess.run(b),label='fitted line')
    ax.legend(loc='best')
    plt.show()

