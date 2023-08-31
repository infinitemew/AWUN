from Layers import *



def build(e_input, mat, e_dim, a_dim, act_func, gamma, k, e, train, KG):
    tf.reset_default_graph()
    e_0 = get_input_layer(e_input)
    M, M_arr = get_sparse_tensor(e, KG)

    gcn_layer_1 = gcn_layer(e_0, e_dim, M, act_func, dropout=0.0)
    gcn_layer_1 = highway(e_0, gcn_layer_1, e_dim)
    gcn_layer_2 = gcn_layer(gcn_layer_1, e_dim, M, act_func, dropout=0.0)
    e_1 = highway(gcn_layer_1, gcn_layer_2, e_dim)

    a_0, a_mat, ea_mat = getAttr(e_1, mat, e)

    a_1 = self_att_layer(a_0, a_mat, tf.nn.relu, e_dim)
    a_update = weight_update_layer(a_1, ea_mat, e_dim)

    e_2 = full_gcn_layer(a_0, e_dim, a_dim, a_update, act_func)

    output_layer = tf.concat([0.2 * e_1, 0.8 * e_2], 1)

    loss = get_loss(output_layer, train, gamma, k)
    return output_layer, loss


def training(output_layer, loss, learning_rate, epochs, ILL, k, test):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    print('initializing...')
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print('running...')
    J = []
    t = len(ILL)
    ILL = np.array(ILL)
    L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
    neg_left = L.reshape((t * k,))
    L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
    neg2_right = L.reshape((t * k,))
    for i in range(epochs):
        if i <= 200:
            if i % 50 == 0:
                out = sess.run(output_layer)
                neg2_left = get_neg(ILL[:, 1], out, k)
                neg_right = get_neg(ILL[:, 0], out, k)
                feeddict = {"neg_left:0": neg_left,
                            "neg_right:0": neg_right,
                            "neg2_left:0": neg2_left,
                            "neg2_right:0": neg2_right}
            _, th = sess.run([train_step, loss], feed_dict=feeddict)
            if i % 50 == 0:
                th, outvec = sess.run([loss, output_layer], feed_dict=feeddict)
                J.append(th)
                get_hits(outvec, test)
            print('%d/%d' % (i + 1, epochs), 'epochs...', th)
        else:
            if i % 10 == 0:
                out = sess.run(output_layer)
                neg2_left = get_neg(ILL[:, 1], out, k)
                neg_right = get_neg(ILL[:, 0], out, k)
                feeddict = {"neg_left:0": neg_left,
                            "neg_right:0": neg_right,
                            "neg2_left:0": neg2_left,
                            "neg2_right:0": neg2_right}

            _, th = sess.run([train_step, loss], feed_dict=feeddict)
            if i % 10 == 0:
                th, outvec = sess.run([loss, output_layer], feed_dict=feeddict)
                J.append(th)
                get_hits(outvec, test)

            print('%d/%d' % (i + 1, epochs), 'epochs...', th)

    outvec = sess.run(output_layer)
    sess.close()
    return outvec, J