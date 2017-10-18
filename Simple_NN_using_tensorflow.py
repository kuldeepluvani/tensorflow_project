#------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import random
random.seed(4)
np.random.seed(4)

#---------------------------------------------------------------
# function to get one-zero random numpy array
def get_one_zero_random_numpy_matrix(num_of_rows, num_of_cols, num_of_zeros):
    total_elements = num_of_rows * num_of_cols
    mask = np.concatenate((np.zeros(num_of_zeros), np.ones(total_elements - num_of_zeros)))
    np.random.shuffle(mask)
    mask = np.reshape(mask, newshape=[num_of_rows, num_of_cols])
    return (mask)

# function to reset random links (set to zero)
def reset_links(weights, what_fraction_to_reset):
    num_of_rows = weights.shape[0]
    num_of_cols = weights.shape[1]
    total_links = int(num_of_rows * num_of_cols)
    num_of_links_to_reset = int(total_links * what_fraction_to_reset)
    mask = get_one_zero_random_numpy_matrix(num_of_rows, num_of_cols, num_of_links_to_reset)
    return (weights * mask)

#-----------------------------------------------------------------------
# define architecture parameters
num_of_input_units  = 5
num_of_hidden_units = 20
num_of_output_units = 4

# network equations
x = tf.placeholder(tf.float32, shape=[1, num_of_input_units])
y = tf.placeholder(tf.float32, shape=[1, num_of_output_units])

weights_input_to_hidden = tf.Variable(tf.random_normal(shape=[num_of_input_units, num_of_hidden_units]))
biases_of_hidden_layer = tf.Variable(tf.zeros(shape=num_of_hidden_units))

weights_hidden_to_output = tf.Variable(tf.random_normal(shape=[num_of_hidden_units, num_of_output_units]))
biases_of_output_layer = tf.Variable(tf.zeros(shape=num_of_output_units))

# hidden layer output = sigmoid(Wx+b)
hidden_layer_output = tf.sigmoid(tf.add(tf.matmul(x, weights_input_to_hidden), biases_of_hidden_layer))

# output layer output = softmax(sigmoid(Wh+b))
network_output = tf.nn.softmax(tf.sigmoid(tf.add(tf.matmul(hidden_layer_output, weights_hidden_to_output), biases_of_output_layer)))

# loss function
loss = tf.nn.softmax_cross_entropy_with_logits(network_output, y)

# train step
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.4).minimize(loss)

#------------------------------------------------------------
# read dataset and divide into train and test
entire_dataset = pd.read_csv('user_knowledge_dataset.csv')
X_data = entire_dataset[['STG', 'SCG', 'STR', 'LPR', 'PEG']]
y_data = entire_dataset['UNS']
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
sss.get_n_splits(X=X_data, y=y_data)

for train_index, test_index in sss.split(X_data, y_data):
    X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]

# function to convert y to one_hot vector
def one_hot(label):
    if label == 'High':
        return (np.asarray([[1.0, 0, 0, 0]]))
    if label == 'Middle':
        return (np.asarray([[0, 1.0, 0, 0]]))
    if label == 'Low':
        return (np.asarray([[0, 0, 1.0, 0]]))
    if label == 'very_low':
        return (np.asarray([[0, 0, 0, 1.0]]))

# initialize variables
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# iterate over training dataset and update weights
n_epochs = 100
print "Training the network...."
for epoch in range(n_epochs):
    print "Epoch Number: " + str(epoch+1)
    for i in range(X_train.shape[0]):
        x_instance = np.asarray([X_train.iloc[i]])
        y_instance = one_hot(y_train.iloc[i])
        sess.run(train_op, feed_dict={x:x_instance, y:y_instance})

# function to evaluate
class_labels = ['High', 'Middle', 'Low', 'very_low']
def predictions(X, y):
    preds = []
    obs   = []
    for i in range(X.shape[0]):
        x_instance = np.asarray([X.iloc[i]])
        preds.append(y.iloc[i])
        obs.append(class_labels[sess.run(tf.argmax(network_output, 1), feed_dict={x:x_instance})[0]])
    return (pd.DataFrame({'pred':preds, 'obs': obs}))

# print training and test accuracies
print "\nPredicting on training and test set...."
training_predictions = predictions(X_train, y_train)
training_accuracy = sum(training_predictions.obs == training_predictions.pred) / float(training_predictions.shape[0])

test_predictions = predictions(X_test, y_test)
test_accuracy = sum(test_predictions.obs == test_predictions.pred) / float(test_predictions.shape[0])

print "Training Accuracy: " + str(training_accuracy)
print "Test Accuracy: " + str(test_accuracy)

# try out for various reset_percent values:
original_weights_input_to_hidden = sess.run(weights_input_to_hidden)
original_weights_hidden_to_output = sess.run(weights_hidden_to_output)
for reset_percent in [0.01, 0.02, 0.03, 0.05, 0.1, 0.15]:

    # print failure status
    print ("\n\nFailure of " + str(int(reset_percent*100)) + "% of links....")

    # reset some links randomly
    sess.run(weights_input_to_hidden.assign(reset_links(original_weights_input_to_hidden, reset_percent)))
    sess.run(weights_hidden_to_output.assign(reset_links(original_weights_hidden_to_output, reset_percent)))

    test_predictions = predictions(X_test, y_test)
    test_accuracy = sum(test_predictions.obs == test_predictions.pred) / float(test_predictions.shape[0])
    print "Test Accuracy: (after links failing randomly)" + str(test_accuracy)