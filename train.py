import tensorflow as tf
import random
from CNN import CNN
from data_util import *
from prep_data import *

root = '/Users/svenyan/Desktop/CS134-Machine Learning/Projects/Final/cs134-final-project/'
path_dict = {'train':'train/relations.json', 'dev':'dev/relations.json', 'test':'test/relations.json'}
output_f = '/Users/svenyan/Desktop/output8.json'

wdict, cdict = convert2id(root)

# A single training step
def train_step(x_batch, y_batch, cnn, sess):
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: 0.5
    }
    sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
        
# A single dev testing step
def dev_step(x_batch, y_batch, cnn, sess):
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: 1.0
    }
    step, loss, accuracy = sess.run(
        [global_step, cnn.loss, cnn.accuracy], feed_dict)
    print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))

# A single testing step, returns an argmax list of predicted sense labels
def test_step(x_batch, y_batch, cnn, sess):
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: 1.0
    }
    step, predictions, loss, accuracy = sess.run(
        [global_step, cnn.predictions,  cnn.loss, cnn.accuracy], feed_dict)
    print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
    return predictions

def train(train_data, cnn, iter_time):
    # Training loop, eval on dev after every loop of the entire training data
    for epoch in range(iter_time):
        random.shuffle(train_data)
        train_xs, train_ys = generateBatches(train_data, cnn.sen_len, cnn.num_class, wdict, cdict, 100)
        print('Iteration: ' + str(epoch + 1))
        for i in range(len(train_xs)):
            train_step(train_xs[i], train_ys[i], cnn, sess)
        for i in range(len(dev_xs)):
            dev_step(dev_xs[i], dev_ys[i], cnn, sess)
    train_data[:] = []

def test(cnn):
    test_data = load_data(root, path_dict['test'])
    test_xs, test_ys = generateBatches(test_data, cnn.sen_len, cnn.num_class, wdict, cdict, 100)
    count = 0
    rel_list = []
    for i in range(len(test_xs)):
        sense_ids = test_step(test_xs[i], test_ys[i], cnn, sess)
        for j in range(len(test_xs[i])):
            predict = make_json({}, test_data[count], sense_ids, j)
            rel_list.append(predict)
            count += 1
    with open(output_f, 'w') as f:
        for rel in rel_list:
            f.write(json.dumps(rel) + '\n')

def make_json(predict, gold, sense_ids, j):
    predict['Arg1'] = {'TokenList':[pos_list[2] for pos_list in gold['Arg1']['TokenList']]}
    predict['Arg2'] = {'TokenList':[pos_list[2] for pos_list in gold['Arg2']['TokenList']]}
    predict['DocID'] = gold['DocID']
    predict['Sense'] = [cdict[sense_ids[j]]]
    predict['Type'] = gold['Type']
    predict['Connective'] = {'TokenList':gold['Connective']['TokenList']}
    return predict

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        # Initialize a CNN instance, feeding hyper-parameters
        cnn = CNN(
            sen_len      = 50,
            emb_len      = 300,
            num_class    = len(cdict),
            vocab_len    = len(wdict),
            filter_sizes = [3, 4, 5],
            num_filter   = 3)
         
        # Define the training procedure
        global_step = tf.Variable(0, name = "global_step", trainable = False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
         
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        
        # Prepare data
        train_data = load_data(root, path_dict['train'])
        dev_data = load_data(root, path_dict['dev'])
        dev_xs, dev_ys = generateBatches(dev_data, cnn.sen_len, cnn.num_class, wdict, cdict, 100)
        dev_data[:] = []
        
        # Begin training
        train(train_data, cnn, iter_time = 1)
        
        # Begin testing
        test(cnn)
