import numpy
from utils import load_data, set_params, evaluate, run_kmeans
from module.CLUHCS import CLUHCS
import time
from module.preprocess import *
import warnings
import datetime
import dgl
import pickle as pkl
import random
from module.communitySearch import search_com
from module.communitySearch4 import search_com4

warnings.filterwarnings('ignore')       # 忽略所有警告信息，避免输出干扰
args = set_params()                     # 调用函数设置程序参数（例如：超参数、GPU编号等），返回参数对象args

device = torch.device("cpu")
if torch.cuda.is_available():
    # 如果有GPU，根据参数中的gpu编号创建对应的CUDA设备对象（例如："cuda:0"）
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def train():
    # 读取数据 特征矩阵列表 元路径矩阵列表 DGL图（里面有邻接矩阵） 输入特征维度列表 边(A, B)到类型的映射
    # e_feat:边的特征向量（是1维，是边类型） id2type:节点到类型的映射
    feats, adjs, g, in_dims, edge2type, e_feat, id2type = load_data(args.dataset, args.type_num, device)

    u = args.u                                      # 正负样本的阈值
    t_hops = args.t_hops                            # 处理的最大跳数       IMDB=2
    t_num_heads = 8                                 # Transformer中多头自主注意力的层数
    t_n_layers = args.t_n_layers                    # Transformer的层数

    num_etypes = len(set(edge2type.values()))       # 边的类型数量

    sub_num = int(len(adjs))
    print("数据集: ", args.dataset)
    print("元路径数量: ", sub_num)
    print("不同类型“节点”特征的维度: ", in_dims)

    feat = feats[0]                                         # 主节点类型特征

    start_time1 = time.time()
    # pos_adj: scipy.sparse.coo_matrix - 正样本邻接矩阵（亲和度≥u）
    adjs, pos_adj = pathsim(adjs, u)


    # 消融实验，所有元路径邻居都作为其正样本
    # adjs, pos_adj = xiaorong_pathsim(adjs, u)

    # 消融实验，无正样本
    # adjs, pos_adj = xiaorong_pathsim_zero(adjs, u)

    adjs_norm = [normalize_adj(adj) for adj in adjs]        # 元路径邻接矩阵进行GCN式的归一化 A' = D^{-1/2} (A) D^{-1/2}
    time1 = time.time() - start_time1

    model = CLUHCS(
        feats_dim_list=in_dims,
        sub_num=sub_num,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        tau=args.tau,
        tau2=args.tau2,
        adjs=adjs_norm,
        lam_proto=0.3,
        dropout=args.dropout,
        dataset=args.dataset,
        pos_adj=pos_adj,
        g=g,
        # 图编码器参数
        e_feat=e_feat,
        num_layers=2,
        heads=8,
        edge_dim=8,
        num_etypes=num_etypes,
        use_batchnorm=True,
        # Transformer 参数
        t_hops=t_hops,
        t_num_heads=t_num_heads,
        t_n_layers=t_n_layers
    )
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.l2_coef)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        feat = feat.cuda()
        feats = [f.cuda() for f in feats]

    cnt_wait = 0       # 早停计数器
    best = 1e9         # 最佳损失初始值
    starttime = datetime.datetime.now()  # 计时开始
    epoch_times = args.nb_epochs  # 总训练轮次

    t_start = time.time()
    for epoch in range(epoch_times):

        if not args.save_emb:   # 如果不保存嵌入则跳过训练
            print("不保留参数，所以跳过了训练。")
            break
        print("---------------------------------------------------")
        print("Epoch:",epoch)
        model.train()                           # 训练模式
        optimizer.zero_grad()                   # 梯度清零

        # 前向传播计算损失（包含对比损失、原型损失等）
        loss = model(feats, adjs_norm)
        z1_initial_embeds = model.get_z1_embeds()
        # 保存z1到本地（numpy格式，便于后续加载）
        # np.save("a.npy", z1_initial_embeds)

        loss.backward()             # 反向传播
        optimizer.step()            # 参数更新
        print('best:', best)

        if best > loss:             # 更新最佳损失
            best = loss
            cnt_wait = 0
        else:
            cnt_wait += 1
        # print('current patience: ', cnt_wait)
        if cnt_wait >= args.patience:
            print('Early stopping!')
            break
    print("训练时间: {:.4f}s".format(time.time() - t_start + time1))

    if args.save_emb:
        print("Start to save embeds.")
        embeds = model.get_embeds()         # 获取最终节点嵌入
        np.save(
            f"./embeds/{args.dataset}/PreEmb.npy",  # 路径 + .npy 后缀
            embeds.cpu().detach().numpy()  # 转换为 NumPy 数组
        )
        print("Save finish.")
    endtime = datetime.datetime.now()
    time_12 = (endtime - starttime).seconds
    print("Total time: ", time_12, "s")

    # w时计算IEGS时的细粒度
    for w in [0.2]:
        # 进行社区搜索
        # metrics = search_com_clustering1(f"./embeds/{args.dataset}","dbscan")
        # metrics = search_com_topk(f"./embeds/{args.dataset}", 500)
        metrics = search_com(f"./embeds/{args.dataset}", w)

        precision_list = metrics['precision']
        recall_list = metrics['recall']
        f1_list = metrics['f1']
        iou_list = metrics['iou']
        time_list = metrics['time']

        # 平均结果路径
        averesult = f"./embeds/{args.dataset}" + "/ave_result.txt"
        # 写入文件中
        write_results_to_file(precision_list, recall_list, f1_list, iou_list, time_list, averesult, w, u, t_hops,
                              t_n_layers)


def write_results_to_file(precisionList, recallList, f1_scoreList, iouList, query_timeList, averesult, w, u, t_hops, t_n_layers):
    # 计算平均指标
    average_precision = sum(precisionList) / len(precisionList)
    average_recall = sum(recallList) / len(recallList)
    average_f1_score = sum(f1_scoreList) / len(f1_scoreList)
    average_iou = sum(iouList) / len(iouList)
    average_query_time = sum(query_timeList) / len(query_timeList)

    # 打开文件，以追加模式（'a'）打开，如果文件不存在会创建新文件
    with open(averesult, 'a') as file:
        # 生成要写入文件的字符串
        result_str = f"precision={average_precision}, recall={average_recall}, f1_score={average_f1_score}, iou={average_iou}, query_time={average_query_time}\n"
        result_str1 = f"w={w}, u={u}, t_hops={t_hops}, t_n_layers={t_n_layers}\n\n"

        # 将结果写入文件
        file.write(result_str)
        file.write(result_str1)

if __name__ == '__main__':
    train()