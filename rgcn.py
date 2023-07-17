import torch
from torch import nn
import torch.nn.functional as F
# from nnet.spectral import SNLinear


class RGCN_Layer(nn.Module):
    """ A Relation GCN module operated on documents graphs. """

    def __init__(self, in_dim, mem_dim, num_layers, relation_cnt=4):
        super().__init__()
        self.layers = num_layers   #网络层数，3
        self.mem_dim = mem_dim  #
        self.relation_cnt = relation_cnt    #edge种类数，
        self.in_dim = in_dim    #输入特征维度：nodes的维度，emb_size+type_size
        self.in_drop = nn.Dropout(0.2)
        self.gcn_drop = nn.Dropout(0.2)
        self.W_0 = nn.ModuleList()
        self.W_r = nn.ModuleList()
        for i in range(relation_cnt):
            self.W_r.append(nn.ModuleList())

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W_0.append(nn.Linear(input_dim, self.mem_dim))
            for W in self.W_r:
                W.append(nn.Linear(input_dim, self.mem_dim))
        

    def forward(self, nodes, adj):
        gcn_inputs = self.in_drop(nodes)    #[batch_size, nodes_num, nodes_size]
        maskss = []
        denomss = []
        for batch in range(adj.shape[0]):
            masks = []
            denoms = []
            for i in range(self.relation_cnt):  #遍历每一种边类型
                if adj[batch, i]._nnz() == 0:
                    continue
                denom = torch.sparse.sum(adj[batch, i], dim=1).to_dense()
                t_g = denom + torch.sparse.sum(adj[batch, i], dim=0).to_dense()
                mask = t_g.eq(0)
                denoms.append(denom.unsqueeze(1))
                masks.append(mask)
            denoms = torch.sum(torch.stack(denoms), 0)
            denoms = denoms + 1
            masks = sum(masks)
            maskss.append(masks)
            denomss.append(denoms)
        denomss = torch.stack(denomss) #节点度数

        rgcn_hidden = []
        for l in range(self.layers):    #对于每层l，将gcn_inputs输入到每种关系的边权重W_r上
            gAxWs = []
            #遍历每一种关系下，叠加每一个点的邻居点的特征进行融合，
            # 最后加上一层的中心节点特征，经过一个激活函数输出作为中心节点的输出特征
            for j in range(self.relation_cnt):  #edge type =5，此时的adj应该为四维，[batch_size, edge_type_num, node_num, node_num]
                gAxW = []
                bxW = self.W_r[j][l](gcn_inputs) #用modulelist储存，依次将node_info送入到[j][l]，第j类边，第l层
                for batch in range(adj.shape[0]):   #batch中第一篇文档，
                    xW = bxW[batch]
                    '''
                    
                    '''
                    AxW = torch.sparse.mm(adj[batch][j], xW)    #前者为稀疏矩阵，后者为密集矩阵，adj[batch][j].size=len(node_num)*len(node_num),xW.size=len(node_num)*len(node)
                    gAxW.append(AxW)    #AxW.size=[len(node_num, len(node.size)],如：[27,7]
                gAxW = torch.stack(gAxW)
                gAxWs.append(gAxW)
            gAxWs = torch.stack(gAxWs, dim=1)   #将每一层所有的边类型的节点按照维度1拼接
            gAxWs = F.relu((torch.sum(gAxWs, 1) + self.W_0[l](gcn_inputs)) / denomss)  # self loop,
            gcn_inputs = self.gcn_drop(gAxWs)
            rgcn_hidden.append(gcn_inputs)
        return rgcn_hidden
