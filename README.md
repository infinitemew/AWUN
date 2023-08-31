# AWUN

西北工业大学学报论文《基于属性权重更新网络的跨语言实体对齐方法》的源码

## 环境

* Python = 3.6
* CUDA = 10.1
* Tensorflow = 1.14.0
* Scipy = 1.5.2
* Numpy = 1.19.2

> 在Conda下仅需手动安装Tensorflow包即可。 实验测试的计算卡为Nvidia Tesla T4 16G 和 Nvidia V100 32G。

## 数据集

实验使用DBP15K数据集，该数据集包含以下三个跨语言的子数据集：
- zh-en
- ja-en
- fr-en

> 默认的数据集路径为 `data/DBP15K/`

实验用到的数据集相关文件及信息如下（以zh-en数据集为例），若需更详细的信息您可以参考[RDGLite-A](https://github.com/infinitemew/RDGLite-A)。

* `ent_ids_1`: KG (ZH)中的实体ID和实体名
* `ent_ids_2`: KG (EN)中的实体ID和实体名
* `ref_ent_ids`: 预先对齐的实体对
* `triples_1`: KG (ZH)中的关系三元组
* `triples_2`: KG (EN)中的关系三元组
* `ae_adj_sparse.json`: KG的实体-属性邻接矩阵
* `zh_vectorList.json`: KG的实体初始词嵌入

> 您也可以在这里直接在[百度网盘](https://pan.baidu.com/s/1voXY4GqgNBBc4EdMYkRpZg?pwd=ytfq)下载所有的数据集文件。

## 运行

以DBP15K(zh-en)数据集为例。

```
python main.py --lang zh_en
```

> 您可以在`Config.py`中修改超参数。
