import math
import numpy as np
import tensorflow as tf
import json
import scipy.spatial


def loadfile(fn, num=1):
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))

    return ret


def load_json(file_path):
    with open(file=file_path, mode='r', encoding='utf-8') as f:
        data = json.load(f)
    f.close()

    return data


def save_json(file, path, i):
    with open(path, 'w') as file_obj:
        if i == 0:
            json.dump(file, file_obj)
        else:
            json.dump(file.tolist(), file_obj)


def get_mat(e, KG):
    du = [{e_id} for e_id in range(e)]
    for tri in KG:
        if tri[0] != tri[2]:
            du[tri[0]].add(tri[2])
            du[tri[2]].add(tri[0])

    du = [len(d) for d in du]
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 1
        else:
            pass
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = 1
        else:
            pass

    for i in range(e):
        M[(i, i)] = 1
    return M, du


def get_sparse_tensor(e, KG):
    print('getting a sparse tensor...')
    M, du = get_mat(e, KG)
    ind = []
    val = []
    M_arr = np.zeros((e, e))
    for fir, sec in M:
        ind.append((sec, fir))
        val.append(M[(fir, sec)] / math.sqrt(du[fir]) / math.sqrt(du[sec]))
        M_arr[fir][sec] = 1.0
    M = tf.SparseTensor(indices=ind, values=val, dense_shape=[e, e])

    return M, M_arr


def getAttr(inlayer, mat, e):
    ea_mat_sparse = mat
    ea_mat = np.zeros([e, 2000])
    for item in ea_mat_sparse:
        h = item[0]
        t = item[1]
        ea_mat[h][t] = 1.0

    a_mat = np.dot(ea_mat.T, ea_mat)

    a_cnt = np.ones(e)
    t1 = np.dot(a_cnt, ea_mat)
    e_cnt = np.ones(2000)
    t2 = np.dot(ea_mat, e_cnt)
    ea_mat_init = np.transpose(np.transpose(ea_mat) * t2) / [math.log(x, 2) for x in t1]

    head_l = tf.transpose(tf.constant(ea_mat, dtype=tf.float32))
    a_0 = tf.matmul(head_l, inlayer) / tf.expand_dims(tf.reduce_sum(head_l, axis=-1), -1)

    return a_0, a_mat, ea_mat_init


def uniform(shape, scale=0.05, name=None):
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)

    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)

    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    initial = tf.zeros(shape, dtype=tf.float32)

    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    initial = tf.ones(shape, dtype=tf.float32)

    return tf.Variable(initial, name=name)


def get_loss(outlayer, ILL, gamma, k):
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
    neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    D = A + gamma
    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
    neg_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))

    return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)


def get_neg(ILL, output_layer, k):
    neg = []
    t = len(ILL)
    ILL_vec = np.array([output_layer[e1] for e1 in ILL])
    KG_vec = np.array(output_layer)
    sim = scipy.spatial.distance.cdist(ILL_vec, KG_vec, metric='cityblock')
    for i in range(t):
        rank = sim[i, :].argsort()
        neg.append(rank[0:k])

    neg = np.array(neg)
    neg = neg.reshape((t * k,))
    return neg


def get_hits(vec, test_pair, top_k=(1, 10, 50, 100)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    top_lr = [0] * len(top_k)
    mrr_sum_l = 0
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))

    print("MRR: " + str(mrr_sum_l / len(test_pair)))