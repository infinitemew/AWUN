from Utils import *


def get_input_layer(_input):
    input_tensor = tf.convert_to_tensor(_input)
    input_embedding = tf.Variable(input_tensor)
    _output = tf.nn.l2_normalize(input_embedding, 1)

    return _output


def gcn_layer(inlayer, dimension, M, act_func, dropout=0.0, init=ones):
    inlayer = tf.nn.dropout(inlayer, 1 - dropout)
    w0 = init([1, dimension])
    tosum = tf.sparse_tensor_dense_matmul(M, tf.multiply(inlayer, w0))

    if act_func is None:
        return tosum
    else:
        return act_func(tosum)


def full_gcn_layer(inlayer, dimension_in, dimension_out, M, act_func, dropout=0.0, init=glorot):
    inlayer = tf.nn.dropout(inlayer, 1 - dropout)
    print('adding a full layer...')
    w0 = init([dimension_in, dimension_out])
    tosum = tf.matmul(M, tf.matmul(inlayer, w0))
    if act_func is None:
        return tosum
    else:
        return act_func(tosum)


def highway(layer1, layer2, dimension):
    kernel_gate = glorot([dimension, dimension])
    bias_gate = zeros([dimension])
    transform_gate = tf.matmul(layer1, kernel_gate) + bias_gate
    transform_gate = tf.nn.sigmoid(transform_gate)
    carry_gate = 1.0 - transform_gate

    return transform_gate * layer2 + carry_gate * layer1


def self_att_layer(inlayer, adj_mat, act_func, hid_dim):
    in_fts = tf.layers.conv1d(tf.expand_dims(inlayer, 0), hid_dim, 1, use_bias=False)
    f_1 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
    f_2 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
    logits = f_1 + tf.transpose(f_2)
    adj_tensor = tf.constant(adj_mat, dtype=tf.float32)
    logits = tf.multiply(adj_tensor, logits)
    bias_mat = -1e9 * (1.0 - (adj_mat > 0))
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)
    vals = tf.matmul(coefs, inlayer)

    if act_func is None:
        return vals
    else:
        return act_func(vals)

def weight_update_layer(inlayer, adj_mat, hid_dim):
    in_fts = tf.layers.conv1d(tf.expand_dims(inlayer, 0), hid_dim, 1, use_bias=False)
    f_1 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
    f_2 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
    logits = f_1 + tf.transpose(f_2)
    adj_tensor = tf.constant(adj_mat, dtype=tf.float32)
    logits = tf.matmul(adj_tensor, logits)
    bias_mat = -1e9 * (1.0 - (adj_mat > 0))
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)
    return coefs