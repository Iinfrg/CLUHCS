import argparse
import sys

argv = sys.argv
# dataset = argv[1]           # 拿到数据集名称
dataset = 'self_imdb'           # 拿到数据集名称

def acm_params(parser):
    parser.add_argument('--save_emb', action="store_false")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--eva_lr', type=float, default=0.03)
    parser.add_argument('--eva_wd', type=float, default=0)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.0007)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--tau2', type=float, default=1)
    parser.add_argument('--feat_mask', type=float, default=0.3)
    parser.add_argument('--adj_mask', type=float, default=[0.3,0.2])
    parser.add_argument('--nei_max', type=int, default=[110,700])
    parser.add_argument('--num_cluster', default=[100,300], type=int, help='number of clusters')    
    parser.add_argument('--lam_proto', type=float, default=1)
    parser.add_argument('--final_num_cluster', type=int, default=3, help='最终聚类数量')
    parser.add_argument('--u', type=float, default=0.4, help='正样本采样的阈值')
    parser.add_argument('--t_hops', type=int, default=4, help='Transformer的hop数')
    parser.add_argument('--t_n_layers', type=int, default=4, help='Transformer的层数')
    parser.add_argument('--w', type=float, default=0.7, help='计算IEGS时的粒度')
    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]
    args.nei_num = 2
    return args


def dblp_params(parser):


    parser.add_argument('--save_emb', action="store_false", help='是否保存嵌入结果（action为store_false时，默认值为True，加参数则设为False）')
    parser.add_argument('--turn', type=int, default=0, help='实验轮次标识（用于多次实验区分）')
    parser.add_argument('--dataset', type=str, default="dblp", help='使用的数据集名称')
    parser.add_argument('--ratio', type=int, default=[20, 40, 60], help='数据集划分比例（如训练/验证/测试集比例）')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU设备编号（-1表示CPU）')
    parser.add_argument('--seed', type=int, default=1, help='随机数种子（确保实验可复现）')

    parser.add_argument('--nb_epochs', type=int, default=10000, help='最大训练轮次')
    # parser.add_argument('--nb_epochs', type=int, default=10, help='最大训练轮次')
    parser.add_argument('--hidden_dim', type=int, default=64, help='神经网络隐藏层维度')
    parser.add_argument('--embed_dim', type=int, default=64, help='节点嵌入向量维度，最终输出维度')

    parser.add_argument('--eva_lr', type=float, default=0.01, help='评估阶段的学习率')
    parser.add_argument('--eva_wd', type=float, default=0, help='评估阶段的权重衰减系数')

    parser.add_argument('--patience', type=int, default=35, help='早停机制耐心值（连续多少轮无改善则停止）')
    parser.add_argument('--l2_coef', type=float, default=0, help='L2正则化系数')
    parser.add_argument('--lr', type=float, default=0.0006, help='模型训练学习率')
    parser.add_argument('--dropout', type=float, default=0.2, help='神经网络Dropout概率')

    parser.add_argument('--tau', type=float, default=1, help='对比学习温度系数')
    parser.add_argument('--tau2', type=float, default=1, help='对比学习温度系数')
    parser.add_argument('--feat_mask', type=float, default=0.2, help='特征掩码率（用于数据增强）')
    parser.add_argument('--adj_mask', type=float, default=[0.2,0.5,0.6], help='邻接矩阵掩码率（可能对应不同层次或边的类型）')
    parser.add_argument('--lam_proto', type=float, default=1, help='原型对比损失的权重系数')
    parser.add_argument('--nei_max', type=int, default=[25,200,40], help='各类型节点的最大邻居采样数')
    parser.add_argument('--num_cluster', default=[200,700], help='聚类数量（可能用于多层级聚类）')
    parser.add_argument('--final_num_cluster', type=int, default=4, help='最终聚类数量')

    parser.add_argument('--u', type=float, default=0.1, help='正样本采样的阈值')
    parser.add_argument('--t_hops', type=int, default=2, help='Transformer的hop数')
    parser.add_argument('--t_n_layers', type=int, default=4, help='Transformer的层数')
    parser.add_argument('--w', type=float, default=0.5, help='计算IEGS时的粒度')

    args, _ = parser.parse_known_args()             # 解析已知参数（忽略未知参数）

    args.type_num = [4057, 14328, 7723, 20]  # 各类型节点的数量（如DBLP数据集的作者/论文/术语/会议节点数量），第一个类型的节点是主节点
    args.nei_num = 1
    return args


def imdb_params(parser):
    parser.add_argument('--save_emb', action="store_false")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--l2_coef', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--tau2', type=float, default=1)
    parser.add_argument('--feat_mask', type=float, default=0.3)
    parser.add_argument('--adj_mask', type=float, default=[0.1,0.1,0.1])
    parser.add_argument('--nei_max', type=int, default=[70,10,70])
    # parser.add_argument('--num_cluster', default=[500,700], type=int, help='number of clusters')
    parser.add_argument('--num_cluster', default=[2, 3], type=int, help='number of clusters')
    parser.add_argument('--lam_proto', type=float, default=1)
    parser.add_argument('--final_num_cluster', type=int, default=3, help='最终聚类数量')

    parser.add_argument('--u', type=float, default=0.9, help='正样本采样的阈值')
    parser.add_argument('--t_hops', type=int, default=2, help='Transformer的hop数')
    parser.add_argument('--t_n_layers', type=int, default=4, help='Transformer的层数')
    parser.add_argument('--w', type=float, default=0.1, help='计算IEGS时的粒度')
     
    args, _ = parser.parse_known_args()
    args.type_num = [4275, 5432, 2083, 7313]
    args.nei_num = 3
    return args


def self_imdb_params(parser):
    parser.add_argument('--save_emb', action="store_false")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="self_imdb")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)

    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--l2_coef', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--tau2', type=float, default=1)
    parser.add_argument('--lam_proto', type=float, default=1)

    parser.add_argument('--u', type=float, default=0.8, help='正样本采样的阈值')
    parser.add_argument('--t_hops', type=int, default=2, help='Transformer的hop数')
    parser.add_argument('--t_n_layers', type=int, default=4, help='Transformer的层数')
    parser.add_argument('--w', type=float, default=0.1, help='计算IEGS时的粒度')

    args, _ = parser.parse_known_args()
    args.type_num = [4278, 5257, 2081]
    return args


def freebase_params(parser):
    parser.add_argument('--save_emb', action="store_false")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="freebase")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=1000)

    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--l2_coef', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--tau2', type=float, default=1)
    parser.add_argument('--feat_mask', type=float, default=0.3)
    parser.add_argument('--adj_mask', type=float, default=[0.1, 0.1, 0.1])
    parser.add_argument('--nei_max', type=int, default=[70, 10, 70])
    # parser.add_argument('--num_cluster', default=[500,700], type=int, help='number of clusters')
    parser.add_argument('--num_cluster', default=[2, 3], type=int, help='number of clusters')
    parser.add_argument('--lam_proto', type=float, default=1)
    parser.add_argument('--final_num_cluster', type=int, default=3, help='最终聚类数量')

    parser.add_argument('--u', type=float, default=0.9, help='正样本采样的阈值')
    parser.add_argument('--t_hops', type=int, default=2, help='Transformer的hop数')
    parser.add_argument('--t_n_layers', type=int, default=4, help='Transformer的层数')
    parser.add_argument('--w', type=float, default=0.1, help='计算IEGS时的粒度')

    args, _ = parser.parse_known_args()
    args.type_num = [3492, 33401, 2502, 4459]
    return args

def set_params():
    parser = argparse.ArgumentParser()
    if dataset == "acm":
        args = acm_params(parser)
    elif dataset == "dblp":
        args = dblp_params(parser)
    elif dataset == "imdb":
        args = imdb_params(parser)
    elif dataset == "self_imdb":
        args = self_imdb_params(parser)
    elif dataset == "freebase":
        args = freebase_params(parser)

    return args
