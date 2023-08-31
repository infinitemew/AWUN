import tensorflow as tf


class Config:
	epochs = 300
	e_dim = 300
	a_dim = 100
	act_func = tf.nn.relu
	gamma = 1.0  # margin based loss
	k = 125  # number of negative samples for each positive one
	seed = 3  # 30% of seeds
	rate = 0.001
