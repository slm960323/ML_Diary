import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
learn = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.ERROR)



#load the data
mnist = learn.datasets.load_dataset('mnist')
data = mnist.train.images
labels = np.asarray(mnist.train.labels, dtype = np.int32)
test_data = mnist.test.images
test_labels = np.asarray(mnist.test.labels, dtype = np.int32)

# #adjust the data
# x = tf.placeholder(tf.float32,[None,784])

# #parameter variables init with zeros
# w=tf.Variable(tf.zeros([784,10]))
# b=tf.Variable(tf.zeros([10]))

# #define the training model
# y = tf.nn.softmax(tf.matmul(x,w)+b)


# max_examples = 10000
# data = data[:max_examples]
# labels = labels[:max_examples]



max_examples = 10000 
data = data[:max_examples] 
labels = labels[:max_examples]


# display some digits

def display(i):
    img = test_data[i]
    plt.title('Example %d. Label: %d' % (i, test_labels[i]))
    plt.imshow(img.reshape((28,28)),cmap = plt.cm.gray_r)
    plt.show()
    
display(0)
display(1)


feature_columns = learn.infer_real_valued_columns_from_input(data)
classifier = learn.LinearClassifier(feature_columns = feature_columns, n_classes = 10)
classifier.fit(data,labels,batch_size = 100,steps = 1000)

classifier.evaluate(test_data,test_labels)
print classifier.evaluate(test_data,test_labels)["accuracy"]


weights = classifier.weights
f,axes = plt.subplots(2,5,figsize = (10,4))
axes = axes.reshape(-1)

for i in range (len(axis)):
    a = axes[i]
    a.imshow(weights.T[i].reshape(28,28), cmap = plt.cm.seismic)
    a.set_title(i)
    a.set_xticks(())
    a.set_yticks(())

plt.show()

# #training step

# #placeholder for correct answers
# y_ = tf.placeholder(tf.float32,[None,10])

# #define the loss function
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices = [1]))

# train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

# sess = tf.InteractiveSession()

# tf.global_variables_initializer().run()

# #run the training setp 1000 times
# for _ in range (1000):
#     batch_xs, batch_ys = minist.train.next_batch(100)
#     sess.run(train_step, feed_dict={x: batch_xs,y_:batch_ys})

    


# # In[ ]:


# #evaluating model

# correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_ , 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# #show accuracy
# print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_ : minist.test.labels})


# # In[ ]:




