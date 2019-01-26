import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(x, in_size, out_size, n_layer, activation_func=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            w = tf.Variable(tf.random_normal([in_size, out_size]), name="W")
        tf.summary.histogram('weights', w)
        with tf.name_scope("biases"):
            b = tf.Variable(tf.zeros([1, out_size]) + 0.1, name="b")
        tf.summary.histogram('biases', b)
        with tf.name_scope("linear_result"):
            z = tf.matmul(x, w) + b
        if activation_func is None:
            a = z
        else:
            a = activation_func(z)
        tf.summary.histogram('outputs', a)
    return a


x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope("inputs"):
    x = tf.placeholder(tf.float32, [None, 1], name="x_in")
    y = tf.placeholder(tf.float32, [None, 1], name="y_in")
a1 = add_layer(x, 1, 10, n_layer=1, activation_func=tf.nn.relu)
yhat = add_layer(a1, 10, 1, n_layer=2, activation_func=None)
with tf.name_scope("loss"):
    l = tf.reduce_mean(tf.reduce_sum(tf.square(y - yhat), reduction_indices=[1]))
    tf.summary.scalar('loss', l)
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(l)

sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(tf.global_variables_initializer())

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(1000):
    sess.run(optimizer, feed_dict={x: x_data, y: y_data})
    if i % 50 == 0:
        rs = sess.run(merged, feed_dict={x: x_data, y: y_data})
        writer.add_summary(rs, i)
        prediction = sess.run(yhat, feed_dict={x: x_data, y: y_data})
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        lines = ax.plot(x_data, prediction, 'r-', lw=5)
        plt.pause(0.5)



