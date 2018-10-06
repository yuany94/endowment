import sys
import tensorflow as tf
import numpy as np
import os
import shutil
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.metrics import roc_auc_score
import argparse

parser = argparse.ArgumentParser(description='Learning endowments vectors. By default we show the example of karate club network.')
parser.add_argument('--input', help='Path for input file. Each line represents an edge. We convert all directed graphs into undirected. By default we show the example of karate club network.')
parser.add_argument('--output', help='Path for output path.')
parser.add_argument('--dimension', help='Dimensionality for learned endowments. Default: 4.')
parser.add_argument('--bst', help='Dimensionality for beneficial endowments. Default: half of dimensions.')
parser.add_argument('--lamb_fp', help='Lamb_fp. Default: 0.1.')
parser.add_argument('--lamb_reg', help='Lamb_reg. Default: 0.1.')
parser.add_argument('--max_epoch', help='The iterations run in the learning process. Default: 5000.')
parser.add_argument('--lr', help='Learning rate. Default: 0.01.')
parser.add_argument('--negative_sample', help='The ratio of negative samples to positive samples. Default: 10.')
parser.add_argument('--gpu', help='GPU number when applicable.')
args = parser.parse_args()

# parser.print_help()

# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                    help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                    const=sum, default=max,
#                    help='sum the integers (default: find the max)')

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# we test karate club when no arg is input:
if args.input:
    input_file_name = args.input
else:
    input_file_name = 'data/karate.txt'
f = open(input_file_name)

if args.output:
    LOG_DIR = args.output
else:
    LOG_DIR = 'output/sample_%d' % (K)

if args.dimension:
    K = int(args.dimension)
else:
    K = 4 

if args.bst:
    midpoint = int(args.bst)
else:
    midpoint = K//2

assert 0 <= midpoint <= K

if args.lamb_fp:
    norm_fp = float(args.lamb_fp)  
else:
    norm_fp = 0.1    

if args.lamb_reg:
    norm_lambda = float(args['lamb_reg'])
else:
    norm_lambda = 0.1

if args.max_epoch:
    max_epoch = int(args.max_epoch)
else:
    max_epoch = 5000

if args.negative_sample:
    batch_multiplier = int(args.negative_sample)
else:
    batch_multiplier = 10

assert batch_multiplier >= 1

if args.lr:
    lr = float(args.lr)
else:
    lr = 0.01


variation = ''


connected_pairs = []
connected_values = []
line = f.readline()
set_pairs = set()

max_N = 0

while line:
    items = line.split(" ")
    vs = np.array([float(k) for k in items[2:]])
    i = int(items[0])
    j = int(items[1])
    set_pairs.add((i, j))
    set_pairs.add((j, i))
    max_N = max(i, max_N)
    max_N = max(j, max_N)
    line = f.readline()

N = max_N + 1
degrees = {}
for i, j in set_pairs:
    if i not in degrees:
        degrees[i] = 0
    degrees[i] += 1

for i, j in set_pairs:
    connected_pairs.append((i, j))
    connected_values.append([float(degrees[i])])

connected_pairs = np.array(connected_pairs)
connected_values = np.array(connected_values)

if variation == 'l1':
    ''' TODO '''
elif variation == 'nonsplit':
    ''' TODO '''
else:
    ori_c = tf.Variable(tf.random_normal([1, K-midpoint]))
    C = tf.exp(tf.minimum(tf.maximum(ori_c, -50), 50))
    ori_b = tf.Variable(tf.random_normal([1, midpoint]))
    B = tf.exp(tf.minimum(tf.maximum(ori_b, -50), 50))
    def utility(embeddings, length):
        b_tiled = tf.tile(tf.reshape(B, [1, midpoint]), [length, 1])
        c_tiled = tf.tile(tf.reshape(C, [1, K-midpoint]), [length, 1])
        benefits = tf.reduce_sum(tf.multiply(b_tiled, tf.nn.relu(embeddings[:, 1][:, :midpoint] - embeddings[:, 0][:, :midpoint])), axis=1)
        costs = np.exp(-20.0) + tf.sqrt(np.exp(-20.0) + tf.reduce_sum(tf.square(tf.multiply(c_tiled, (embeddings[:, 0][:, midpoint:K] - embeddings[:, 1][:, midpoint:K])) + np.exp(-20)), axis=1))
        return benefits, costs, benefits - costs + beta


ori_W = tf.Variable(tf.random_uniform(shape=(N, K)))
mean_W, var_W = tf.nn.moments(ori_W, axes=[0])
beta = tf.Variable(tf.random_uniform(shape=[]))

W = (ori_W - tf.tile(tf.reshape(mean_W, [1, K]), [N, 1])) / (tf.tile(tf.reshape(tf.sqrt(var_W+np.exp(-15)), [1, K]), [N, 1]) + np.exp(-15))
inputs_1 = tf.placeholder(tf.int32, [None, 2])
inputs_2 = tf.placeholder(tf.int32, [None, 2])
inputs_3 = tf.placeholder(tf.int32, [None, 2])

embeddings_1 = tf.nn.embedding_lookup(W, inputs_1)
embeddings_2 = tf.nn.embedding_lookup(W, inputs_2)
embeddings_3 = tf.nn.embedding_lookup(W, inputs_2)

batch_size_1 = tf.placeholder(tf.int32, [])
batch_size_2 = tf.placeholder(tf.int32, [])
batch_size_3 = tf.placeholder(tf.int32, [])

e_1 = tf.placeholder(tf.float32, [None])
e_2 = tf.placeholder(tf.float32, [None])
e_3 = tf.placeholder(tf.float32, [None])

period = tf.placeholder(tf.float32, [])


def generate_set(connected_pairs, values):
    setP = connected_pairs
    vs = values
    setN0 = np.random.randint(0, N, size=(len(setP) * batch_multiplier, 1))
    setN1 = np.random.randint(0, N - 1, size=(len(setP) * batch_multiplier, 1))
    setN1 += setN1 >= setN0
    setN = np.concatenate([setN0, setN1], axis=1)
    setN_reverse = np.concatenate([setN1, setN0], axis=1)
    return setP, setN, vs, np.zeros(len(setN)) + 0.00005, setN_reverse


b1, c1, u1 = utility(embeddings_1, batch_size_1)
b2, c2, u2 = utility(embeddings_2, batch_size_2)
b3, c3, u3 = utility(embeddings_3, batch_size_3)


sample_size = len(connected_pairs) * batch_multiplier
unit_matrix = np.identity(K) * N

ratio = 1.0 * len(connected_pairs) / (N * (N-1))
array_connected_pairs = np.array(connected_pairs)
array_connected_pairs_one = array_connected_pairs[:, 1] + N * array_connected_pairs[:, 0]


values, indices = tf.nn.top_k(u2, int(ratio*sample_size))        
bottom_value = values[-1]

positives = u2 >= bottom_value
true_mask_holder = tf.placeholder(tf.float32, [None])        
false_positives_mask = tf.multiply(1-true_mask_holder, tf.cast(positives, tf.float32))
negatives = u2 < bottom_value
false_negatives_mask = tf.multiply(true_mask_holder, tf.cast(negatives, tf.float32))


loss_11 = - tf.reduce_mean(tf.log(tf.sigmoid(tf.minimum(60., tf.maximum(-60., tf.multiply(e_1, u1))))))
loss_10 = - 1.0 / (1-ratio) * tf.reduce_mean(tf.multiply((1-true_mask_holder), tf.log(tf.sigmoid(-tf.minimum(u2, u3)))))
loss_21 = - tf.reduce_sum(tf.multiply(false_negatives_mask, tf.log(tf.sigmoid(u2)))) / ((1.-ratio) * sample_size)
loss_20 = - tf.reduce_sum(tf.multiply(false_positives_mask, tf.log(tf.sigmoid(-tf.minimum(u2, u3))))) / (ratio * sample_size)
BC = tf.concat((B, C), 1)
loss_norm = tf.norm(BC, ord=1) 
loss = loss_10 + loss_11 + norm_lambda * loss_norm + norm_fp * loss_20 

learning_rate = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)



sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

merged = tf.summary.merge_all()

training_summaries = [
    tf.summary.scalar("training_summary_tot", loss)
]

validating_summaries = [
    tf.summary.scalar("validating_summary_tot", loss)
]


connected_values = np.array(connected_values)

test_pairs_P, test_pairs_N, test_e_P, test_e_N, test_pairs_Nr = generate_set(connected_pairs, connected_values)
test_pairs_N_one = test_pairs_N[:, 1] + N * test_pairs_N[:, 0]
true_mask_test = np.isin(test_pairs_N_one, array_connected_pairs_one)

feeds_test = {
    inputs_1: test_pairs_P,
    inputs_2: test_pairs_N,
    inputs_3: test_pairs_Nr,
    batch_size_1: len(test_pairs_P),
    batch_size_2: len(test_pairs_N),
    batch_size_3: len(test_pairs_N),
    e_1: test_e_P[:, 0],
    e_2: test_e_N,
    e_3: test_e_N,
    true_mask_holder: true_mask_test
}

tf.nn.top_k(tf.minimum(u2, u3), tf.cast(tf.reduce_sum(true_mask_holder), tf.int32))
values, indices = tf.nn.top_k(tf.minimum(u2, u3), int(ratio*sample_size))
bottom_value = values[-1]
positives = tf.minimum(u2, u3) >= bottom_value

f_1 = 1.0 * tf.reduce_sum(tf.multiply(true_mask_holder, tf.cast(positives, tf.float32))) / tf.reduce_sum(true_mask_holder)

connected_values = np.array(connected_values)


from sklearn.metrics import roc_auc_score

pred_link = tf.placeholder(tf.float32, [])

pred_summaries = [tf.summary.scalar("pred_summary_tot", pred_link)]

    
while True:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    if np.isnan(sess.run(loss, feed_dict=feeds_test)):
        continue
    else:
        break


losses = []
for epoch in range(max_epoch):
    current_pairs_P, current_pairs_N, current_e_P, current_e_N, current_pairs_Nr = generate_set(connected_pairs, connected_values[:, 0])
    current_pairs_N_one = current_pairs_N[:, 1] + N * current_pairs_N[:, 0]
    true_mask = np.isin(current_pairs_N_one, array_connected_pairs_one)

    feeds = {
                inputs_1: current_pairs_P,
                inputs_2: current_pairs_N,
                inputs_3: current_pairs_Nr,
                batch_size_1: len(current_pairs_P),
                batch_size_2: len(current_pairs_N),
                batch_size_3: len(current_pairs_N),
                e_1: current_e_P,
                e_2: current_e_N,
                e_3: current_e_N,
                learning_rate: lr,
                true_mask_holder: true_mask
        }
    if epoch % 100 == 0:
        lr *= 0.95
    sess.run(optimizer, feed_dict=feeds)

    if epoch % 100 == 0:
        utility1 = sess.run(u1, feed_dict=feeds)
        utility2 = sess.run(u2, feed_dict=feeds)
        cc1 = sess.run(c1, feed_dict=feeds)
        bb1 = sess.run(b1, feed_dict=feeds)
        cc2 = sess.run(c2, feed_dict=feeds)
        bb2 = sess.run(b2, feed_dict=feeds)
        now_auc = roc_auc_score(sess.run(true_mask_holder, feed_dict=feeds_test), sess.run(tf.minimum(u2, u3), feed_dict=feeds_test))
        print("%d loss: %f \t auc: %f\t" % (
            epoch, 
            sess.run(loss, feed_dict=feeds_test),
            now_auc))
        train_sum = sess.run(training_summaries, feed_dict=feeds)
        valid_sum = sess.run(validating_summaries, feed_dict=feeds_test)
        now_auc = roc_auc_score(sess.run(true_mask_holder, feed_dict=feeds_test), sess.run(tf.minimum(u2, u3), feed_dict=feeds_test))
        losses.append([now_auc, sess.run(loss, feed_dict=feeds), now_auc])


np.savetxt(LOG_DIR + '_W_%d.txt' % (midpoint), sess.run(W))
np.savetxt(LOG_DIR +'_B_%d.txt' % (midpoint), sess.run(B))
np.savetxt(LOG_DIR + '_C_%d.txt' % (midpoint), sess.run(C))
np.savetxt(LOG_DIR + '_beta_%d.txt' % (midpoint), [sess.run(beta)])
np.savetxt(LOG_DIR + '_likelihood_%d.txt' % (midpoint), [sess.run(loss, feed_dict=feeds_test)])
np.savetxt(LOG_DIR + '_now_auc_%d.txt' % (midpoint), [now_auc])
np.savetxt(LOG_DIR + '_curve_%d.txt' % (midpoint), losses)


