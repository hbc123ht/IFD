import tensorflow.compat.v1 as tf
import keras
tf.disable_v2_behavior()
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from pywt import wavedec2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import cv2
from PIL import Image
import glob

def DWT(re):
    c = {}
    tmp = wavedec2(re, 'db1', level=3)
    c[0], (c[1], c[2], c[3]), (c[4], c[5], c[6]), (c[7], c[8], c[9]) = tmp
    x = 10
    tmp = wavedec2(re, 'db2', level=3)
    c[x + 0], (c[x + 1], c[x + 2], c[x + 3]), (c[x + 4], c[x + 5], c[x + 6]), (c[x + 7], c[x + 8], c[x + 9]) = tmp
    x = 20
    tmp = wavedec2(re, 'db3', level=3)
    c[x + 0], (c[x + 1], c[x + 2], c[x + 3]), (c[x + 4], c[x + 5], c[x + 6]), (c[x + 7], c[x + 8], c[x + 9]) = tmp
    x = 30
    tmp = wavedec2(re, 'db4', level=3)
    c[x + 0], (c[x + 1], c[x + 2], c[x + 3]), (c[x + 4], c[x + 5], c[x + 6]), (c[x + 7], c[x + 8], c[x + 9]) = tmp
    x = 40
    tmp = wavedec2(re, 'db5', level=3)
    c[x + 0], (c[x + 1], c[x + 2], c[x + 3]), (c[x + 4], c[x + 5], c[x + 6]), (c[x + 7], c[x + 8], c[x + 9]) = tmp
    for i in range(50):
        c[i] = c[i] * 10000000
    return c
def fearture_extract(data):
    re = [[0] * data.shape[1]] * data.shape[0]
    x = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            re[i][j] = data[i][j][1]
    b = DWT(re)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            re[i][j] = data[i][j][2]
    c = DWT(re)
    for i in range(50):
        x.append(np.std(b[i]))
        x.append(b[i].mean())
        x.append(b[i].sum())
        x.append(np.std(c[i]))
        x.append(c[i].mean())
        x.append(c[i].sum())
    return x

def loaddata():
    X = []
    y = []
    X_t = []
    y_t = []
    i = 0
    for filename in glob.glob('/Users/cong/Downloads/Casia2/au/images/*.jpg'): # paste the path of folder containing authentic images here
        i += 1
        if i<4000:
            image = cv2.imread(filename, cv2.COLOR_RGB2YCR_CB)
            X.append(fearture_extract(image))
            y.append(1)
        elif i<4100:
            image = cv2.imread(filename, cv2.COLOR_RGB2YCR_CB)
            X_t.append(fearture_extract(image))
            y_t.append(1)
        else:
            break
    i = 0
    for filename in glob.glob('/Users/cong/Downloads/Casia2/tp/images/*.jpg'): # paste the path of folder containing tampered images here
        i += 1
        if i<2000:
            image = cv2.imread(filename, cv2.COLOR_RGB2YCR_CB)
            X.append(fearture_extract(image))
            y.append(0)
        elif i<2100:
            image = cv2.imread(filename, cv2.COLOR_RGB2YCR_CB)
            X_t.append(fearture_extract(image))
            y_t.append(0)
        else:
            break
    X_t = np.array(X_t)
    y_t = np.array(y_t)
    X = np.array(X)
    y = np.array(y)
    print("finished loading data")
    return X,y,X_t,y_t

def onehot(y_train):
    b = np.zeros((y_train.size, y_train.max() + 1))
    b[np.arange(y_train.size), y_train] = 1
    b = np.array(b)
    return b
X_train,y_train,X_test,y_test = loaddata()
y_train = onehot(y_train)
y_test = onehot(y_test)
#initdata
starter_learning_rate = 0.0005
regularizer_rate = 0.05
input_X = tf.placeholder('float32', shape=(None, 300), name="input_X")
input_y = tf.placeholder('float32', shape=(None, 2), name='input_Y')
## for dropout layer
keep_prob = tf.placeholder(tf.float32)

#initlayers
weights_0 = tf.Variable(tf.random_normal([300, 300], stddev=(1 / tf.sqrt(float(300)))))
bias_0 = tf.Variable(tf.random_normal([300]))
weights_1 = tf.Variable(tf.random_normal([300, 200], stddev=(1 / tf.sqrt(float(300)))))
bias_1 = tf.Variable(tf.random_normal([200]))
weights_2 = tf.Variable(tf.random_normal([200, 150], stddev=(1 / tf.sqrt(float(200)))))
bias_2 = tf.Variable(tf.random_normal([150]))
weights_3 = tf.Variable(tf.random_normal([150, 100], stddev=(1 / tf.sqrt(float(150)))))
bias_3 = tf.Variable(tf.random_normal([100]))
weights_4 = tf.Variable(tf.random_normal([100, 2], stddev=(1 / tf.sqrt(float(100)))))
bias_4 = tf.Variable(tf.random_normal([2]))
## Initializing weigths and biases
hidden_output_0 = tf.nn.relu(tf.matmul(input_X, weights_0) + bias_0)
hidden_output_0_0 = tf.nn.dropout(hidden_output_0, keep_prob)
hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0_0, weights_1) + bias_1)
hidden_output_1_1 = tf.nn.dropout(hidden_output_1, keep_prob)
hidden_output_2 = tf.nn.relu(tf.matmul(hidden_output_1_1, weights_2) + bias_2)
hidden_output_2_2 = tf.nn.dropout(hidden_output_2, keep_prob)
hidden_output_3 = tf.nn.relu(tf.matmul(hidden_output_2_2, weights_3) + bias_3)
hidden_output_3_3 = tf.nn.dropout(hidden_output_3, keep_prob)
predicted_y = tf.sigmoid(tf.matmul(hidden_output_3_3, weights_4) + bias_4)
## Defining the loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_y, labels=input_y)) \
              + regularizer_rate * (tf.reduce_sum(tf.square(bias_0)) + tf.reduce_sum(tf.square(bias_1)) + tf.reduce_sum(tf.square(bias_2)) + tf.reduce_sum(tf.square(bias_3)) + tf.reduce_sum(tf.square(bias_4)))
## Variable learning rate
learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)
## Adam optimzer for finding the right weight
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,
                                                                  var_list=[weights_0, weights_1, weights_2, weights_3,weights_4,
                                                                            bias_0, bias_1, bias_2, bias_3, bias_4])


batch_size = 20 # batch size
epochs = 50 #epoch
dropout_prob = 0.8 #dropout
training_accuracy = []
training_loss = []
testing_accuracy = []
s = tf.InteractiveSession()

s.run(tf.global_variables_initializer())
## Metrics definition
correct_prediction = tf.equal(tf.argmax(y_train,1), tf.argmax(predicted_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for epoch in range(epochs):
    arr = np.arange(X_train.shape[0])
    np.random.shuffle(arr)
    for index in range(0, X_train.shape[0], batch_size):
        s.run(optimizer, {input_X: X_train[arr[index:index + batch_size]],
                          input_y: y_train[arr[index:index + batch_size]],
                          keep_prob: dropout_prob})
    training_accuracy.append(s.run(accuracy, feed_dict={input_X: X_train,
                                                        input_y: y_train, keep_prob: 1}))
    training_loss.append(s.run(loss, {input_X: X_train,
                                      input_y: y_train, keep_prob: 1}))

    ## Evaluation of model
    testing_accuracy.append(accuracy_score(y_test.argmax(1),
                                           s.run(predicted_y, {input_X: X_test, keep_prob: 1}).argmax(1)))
    print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}, Test acc:{3:.3f}".format(epoch,
                                                                                       training_loss[epoch],
                                                                                       training_accuracy[epoch],
                                                                                       testing_accuracy[epoch]))

#plot here
"""iterations = list(range(epochs))
plt.plot(iterations, training_accuracy, label='Train')
plt.plot(iterations, testing_accuracy, label='Test')
plt.ylabel('Accuracy')
plt.xlabel('iterations')
plt.show()
print("Train Accuracy: {0:.2f}".format(training_accuracy[-1]))
print("Test Accuracy:{0:.2f}".format(testing_accuracy[-1]))"""