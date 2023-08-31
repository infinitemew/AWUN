from Config import Config
from Utils import *
from AWUN import *
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")


'''
Follow the code style of RDGCN:
https://github.com/StephanieWyt/RDGCN

@inproceedings{ijcai2019-733,
  title={Relation-Aware Entity Alignment for Heterogeneous Knowledge Graphs},
  author={Wu, Yuting and Liu, Xiao and Feng, Yansong and Wang, Zheng and Yan, Rui and Zhao, Dongyan},
  booktitle={Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, {IJCAI-19}},            
  pages={5278--5284},
  year={2019},
}
'''

seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/DBP15K/")
    parser.add_argument("--lang", default="zh_en")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    data_path = args.data_path
    lang = args.lang
    path = data_path + lang + '/'
    e1 = path + 'ent_ids_1'
    e2 = path + 'ent_ids_2'
    ill = path + 'ref_ent_ids'
    kg1 = path + 'triples_1'
    kg2 = path + 'triples_2'

    e = len(set(loadfile(e1, 1)) | set(loadfile(e2, 1)))
    ILL = loadfile(ill, 2)
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * Config.seed])
    test = ILL[illL // 10 * Config.seed:]
    KG1 = loadfile(kg1, 3)
    KG2 = loadfile(kg2, 3)
    KG = KG1 + KG2


    e_dim = Config.e_dim
    a_dim = Config.a_dim
    act_func = Config.act_func

    print('initial entity embedding loading ...')
    load_path = path + lang[0:2] + '_vectorList.json'
    e_input = load_json(load_path)
    print('load finish')

    print('entity-attribute adjacency matrix loading ...')
    load_path = path + 'ae_adj_sparse.json'
    mat = load_json(load_path)
    print('load finish')

    epochs = Config.epochs
    e_dim = Config.e_dim
    a_dim = Config.a_dim
    act_func = Config.act_func
    gamma = Config.gamma
    k = Config.k
    rate = Config.rate

    output, loss = build(e_input, mat, e_dim, a_dim, act_func, gamma, k, e, train, KG)
    vec, J = training(output, loss, rate, epochs, train, k, test)




