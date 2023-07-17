import torch
import torch.nn as nn
# from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss
from torch.nn.utils.rnn import pad_sequence    
from rgcn import RGCN_Layer                     
from utils import EmbedLayer                    


class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=512, num_labels=-1, max_entity=22):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()
        self.extractor_trans = nn.Linear(config.hidden_size, emb_size)
        self.ht_extractor = nn.Linear(emb_size*4, emb_size*2)
        self.MIP_Linear = nn.Linear(emb_size * 7, emb_size * 5)
        self.MIP_Linear1 = nn.Linear(emb_size * 5, emb_size * 4)
        self.MIP_Linear2 = nn.Linear(emb_size * 4, emb_size * 2)
        self.bilinear = nn.Linear(emb_size * 2, config.num_labels)
        self.emb_size = emb_size
        self.num_labels = num_labels
        self.max_entity = max_entity                                                                   
        self.type_dim = 20           
        self.rgcn = RGCN_Layer(emb_size + self.type_dim, emb_size, 3, 4)
        self.type_embed = EmbedLayer(num_embeddings=3, embedding_dim=self.type_dim, dropout=0.0)       
        inter_channel = int(emb_size // 2)
        self.fusion_s = nn.Sequential(
            nn.Conv2d(emb_size, inter_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, emb_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(emb_size)
        )
        self.fusion_c = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(emb_size, inter_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, emb_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(emb_size)
        )
        self.sigmoid = nn.Sigmoid()
        self.fusion_s_m = nn.Sequential(
            nn.Conv2d(emb_size, inter_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, emb_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(emb_size)
        )
        self.fusion_c_m = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(emb_size, inter_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, emb_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(emb_size)
        )
        self.conv_reason_e_l1 = nn.Sequential(
            nn.Conv2d(emb_size, inter_channel, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, inter_channel, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
        )
        self.conv_reason_e_l2 = nn.Sequential(
            nn.Conv2d(inter_channel, emb_size, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
        )
        self.conv_reason_m_l1 = nn.Sequential(
            nn.Conv2d(emb_size, inter_channel, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, inter_channel, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
        )
        self.conv_reason_m_l2 = nn.Sequential(
            nn.Conv2d(inter_channel, emb_size, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
        )

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def make_graph(self, sequence_output, attention, entity_pos, link_pos, nodes_info, sub_nodes):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()   #[batch_size, num_heads, sequence_length, sequence_length]
        nodes_batch = []
        new_nodes_batch = []
        entity_att_batch = []
        entity_node_batch = []
        mention_pos_batch = []
        #entity_node,mention_node,link_node分别为SIE部分三种结点，link_node缓解GCN过拟合问题
        mention_att_batch = []
        for i in range(len(entity_pos)):    #遍历一个batch_size中样本数，
            entity_nodes, mention_nodes, link_nodes = [], [], []
            entity_att = []
            mention_att = []
            mention_pos = []
            #取句子嵌入
            for start, end in link_pos[i]:
                if end + offset < c:    #没有越界，直接提取桥结点所link的两个句子的表示和注意力
                    link_rep = sequence_output[i, start + offset: end + offset]
                    link_att = attention[i, :, start + offset: end + offset, start + offset: end + offset]
                    link_att = torch.mean(link_att, dim=0)
                    link_rep = torch.mean(torch.matmul(link_att, link_rep), dim=0)
                elif start + offset < c:
                    link_rep = sequence_output[i, start + offset:]
                    link_att = attention[i, :, start + offset:, start + offset:]
                    link_att = torch.mean(link_att, dim=0)
                    link_rep = torch.mean(torch.matmul(link_att, link_rep), dim=0)
                else:
                    link_rep = torch.zeros(self.config.hidden_size).to(sequence_output)
                link_nodes.append(link_rep)
            #取实体嵌入
            for e in entity_pos[i]:
                mention_pos.append(len(mention_att))
                if len(e) > 1:
                    m_emb, e_att = [], []
                    #遍历每个实体的所有提及
                    for start, end, e_id, sid in e:
                    #for start, end, e_id, h_lid, t_lid, sid in e:
                        if start + offset < c:
                            mention_nodes.append(sequence_output[i, start + offset])    #提及结点rep
                            m_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                            mention_att.append(attention[i, :, start + offset])
                        else:
                            mention_nodes.append(torch.zeros(self.config.hidden_size).to(sequence_output))
                            m_emb.append(torch.zeros(self.config.hidden_size).to(sequence_output))
                            e_att.append(torch.zeros(h, c).to(attention))
                            mention_att.append(torch.zeros(h, c).to(attention))
                    if len(m_emb) > 0:
                        m_emb = torch.logsumexp(torch.stack(m_emb, dim=0), dim=0)   #得到实体嵌入，
                        e_att = torch.stack(e_att, dim=0).mean(0)   #实体注意力，这个张量表示了所有实体的平均注意力分布
                    else:
                        m_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    #start, end, e_id, h_lid, t_lid, sid = e[0]
                    start, end, e_id, sid = e[0]
                    if start + offset < c:
                        mention_nodes.append(sequence_output[i, start + offset])
                        m_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                        mention_att.append(attention[i, :, start + offset])
                    else:
                        m_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                        mention_att.append(torch.zeros(h, c).to(attention))
                entity_nodes.append(m_emb)  #实体结点
                entity_att.append(e_att)    #实体注意力
            mention_pos.append(len(mention_att))
            entity_att = torch.stack(entity_att, dim=0)
            entity_att_batch.append(entity_att) ###
            entity_nodes = torch.stack(entity_nodes, dim=0)
            mention_nodes = torch.stack(mention_nodes, dim=0)   #提及结点
            mention_att = torch.stack(mention_att, dim=0)
            link_nodes = torch.stack(link_nodes, dim=0)
            nodes = torch.cat([entity_nodes, mention_nodes, link_nodes], dim=0)     #SIE部分三种节点
            nodes_type = self.type_embed(nodes_info[i][:, 6].to(sequence_output.device))
            nodes = torch.cat([nodes, nodes_type], dim=1)   #将节点与类型拼接
            nodes_batch.append(nodes)   ###
            '''
            取子图中的节点嵌入表示
            new_nodes
            new_nodes_batch
            
            sub_node_id = [node[5] for node in sub_nodes[i]]
            #print(sub_node_id)
            #sub_nodess = torch.LongTensor(sub_nodes[i]).to(sequence_output.device)
            node_id = torch.tensor(sub_node_id).to(sequence_output.device)
            '''

            sub_nodess = torch.LongTensor(sub_nodes[i]).to(sequence_output.device)
            new_nodes = torch.index_select(nodes, 0, sub_nodess[:, 5])   #根据子图节点id从全节点nodes中取相应节点表示
            new_nodes_batch.append(new_nodes)   #保存所有batch信息

            entity_node_batch.append(entity_nodes)  ###
            mention_att_batch.append(mention_att)   ###
            mention_pos_batch.append(mention_pos)   ###
        nodes_batch = pad_sequence(nodes_batch, batch_first=True, padding_value=0.0)    #nodes_batch是一个列表，包含了多个tensor，每个tensor表示一个图节点的特征，这些节点的特征需要进行padding
        new_nodes_batch = pad_sequence(new_nodes_batch, batch_first=True, padding_value=0.0)
        return nodes_batch, new_nodes_batch, entity_att_batch, entity_node_batch, mention_att_batch, mention_pos_batch

    def relation_map(self, gcn_nodes, entity, entity_att, entity_pos, sequence_output, mention_att):
        entity_s, mention_s = [], []
        entity_c, mention_c = [], []
        ec_rep = [] #存放Ec
        nodes = gcn_nodes[-1]   #最后一层节点输出
        m_num_max = 0
        e_num_max = 0
        for i in range(len(entity_pos)):
            m_num, _, _ = mention_att[i].size()
            m_num_max = m_num if m_num > m_num_max else m_num_max
            e_num = len(entity_pos[i])
            e_num_max = e_num if e_num > e_num_max else e_num_max
        for i in range(len(entity_pos)):
            e_num = len(entity_pos[i])
            entity_stru = nodes[i][: e_num] #得到Es，第i篇文档中的所有实体嵌入，nodes结构为ent_num+m_num+bridge_num
            m_num, head_num, dim = mention_att[i].size()
            mention_stru = nodes[i][e_num: e_num+m_num] #得到Ms
            #CIE模块
            e_att = entity_att[i].mean(1)
            e_att = e_att / (e_att.sum(1, keepdim=True) + 1e-5)
            e_context = torch.einsum('ij, jl->il', e_att, sequence_output[i])   #Ec

            m_att = mention_att[i].mean(1)
            m_att = m_att / (m_att.sum(1, keepdim=True) + 1e-5) #
            m_context = torch.einsum('ij,jl->il', m_att, sequence_output[i])    #Mc

            n, h = entity_stru.size()   #分别为实体数量，实体嵌入维度为d_hid=d_emb
            e_s = torch.zeros([e_num_max, h]).to(sequence_output)
            e_s[:n] = entity_stru
            e_s_map = torch.einsum('ij, jk->jik', e_s, e_s.t()) #论文中公式11    Es*Es转置
            entity_s.append(e_s_map)
            m, h = mention_stru.size()
            m_s = torch.zeros([m_num_max, h]).to(sequence_output)
            m_s[:m] = mention_stru
            m_s_map = torch.einsum('ij,jk->jik', m_s, m_s.t())  #论文中公式10    Ms*Ms转置
            mention_s.append(m_s_map)
            n, h_2 = e_context.size()
            e_c = torch.zeros([e_num_max, h_2]).to(sequence_output)
            e_c[:n] = e_context
            ec_rep.append(e_c)
            e_c_map = torch.einsum('ij, jk->jik', e_c, e_c.t()) #论文中公式9,Ec*Ec转置
            entity_c.append(e_c_map)
            m, h = m_context.size() #m和h分别为提及数和每个提及上下文向量大小demb
            m_c = torch.zeros([m_num_max, h]).to(sequence_output)   #创建一个形状为(m_num_max, h)的0张量m_c，其中m_num_max是所有mentions中最大的mention数量
            m_c[:m] = m_context #将之前得到的m_context复制到m_c的前m行
            m_c_map = torch.einsum('ij,jk->jik', m_c, m_c.t())  #论文中公式8 ， Mc*Mc转置，
            mention_c.append(m_c_map)
        #经处理后得到的特征，
        ec_rep = torch.stack(ec_rep, dim=0)
        entity_c = torch.stack(entity_c, dim=0)
        entity_s = torch.stack(entity_s, dim=0)
        mention_c = torch.stack(mention_c, dim=0)
        mention_s = torch.stack(mention_s, dim=0)
        return entity_c, entity_s, mention_c, mention_s, ec_rep

    def fusion_feature_e(self, feat_struc, feat_context):   #MDIF处理带有文本上下文信息和结构信息的实体表示
        feat_mix = feat_struc + feat_context
        feat_s = self.fusion_s(feat_mix)    #MDIF模块中对结构信息进行重构
        feat_c = self.fusion_c(feat_mix)
        feat_sc = feat_s + feat_c       #
        w_feat_sc = self.sigmoid(feat_sc)   #公式14
        feat_fusion = feat_struc * w_feat_sc + feat_context * (1-w_feat_sc) #得到Esc
        return feat_fusion

    def fusion_feature_m(self, feat_struc, feat_context):
        feat_mix = feat_struc + feat_context
        feat_s = self.fusion_s_m(feat_mix)  #对于包含结构信息的
        feat_c = self.fusion_c_m(feat_mix)
        feat_sc = feat_s + feat_c   #
        w_feat_sc = self.sigmoid(feat_sc)
        feat_fusion = feat_struc * w_feat_sc + feat_context * (1-w_feat_sc)
        return feat_fusion

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                adjacency=None,         
                link_pos=None,      
                nodes_info=None,
                sub_nodes = None,
                sub_adjacency = None,
                instance_mask=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        sequence_output = self.extractor_trans(sequence_output)
        nodes, new_nodes, entity_att, entity_node_batch, mention_att, mentions_pos = self.make_graph(sequence_output, attention, entity_pos, link_pos, nodes_info, sub_nodes)
        gcn_nodes = self.rgcn(nodes, adjacency)
        '''
            子图推理层
        '''
        new_gcn_nodes = self.rgcn(new_nodes, sub_adjacency) #子图推理

        entity_c, entity_s, mention_c, mention_s, ec_rep = self.relation_map(gcn_nodes, entity_node_batch, entity_att, entity_pos, sequence_output, mention_att)    #MDIF
        feature_e = self.fusion_feature_e(entity_s, entity_c)   #MDIF模块中得到Esc
        feature_m = self.fusion_feature_m(mention_s, mention_c) #得到Msc  #(bs, emb_size, num_ents, num_ents)
        r_rep_e = self.conv_reason_e_l1(feature_e)
        r_rep_e = self.conv_reason_e_l2(r_rep_e)    #Ers,结构为demb*Ne*Ne，(num_rels, hidden_size, num_ents, num_ents)
        r_rep_m = self.conv_reason_m_l1(feature_m)
        r_rep_m = self.conv_reason_m_l2(r_rep_m)    #Mrs

        #分类层，要获得relation的rep
        relation = []
        entity_h = []
        entity_t = []

        #子图
        new_entity_h = []
        new_entity_t = []

        #Ec中取得头尾实体
        eh = []
        et = []
        e_tou = []
        e_wei = []

        sc_feature_e = []
        nodes_re = torch.cat([gcn_nodes[0], gcn_nodes[-1]], dim=-1)
        '''
            子图推理特征
        '''
        new_nodes_re = torch.cat([new_gcn_nodes[0], new_gcn_nodes[-1]], dim=-1) #子图特征
        for i in range(len(entity_pos)):    #遍历每一篇文章
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device) #将将列表转换为一个长整型的张量，
            r_v1 = r_rep_e[i, :, ht_i[:, 0], ht_i[:, 1]].transpose(1, 0)   #得到在Ers中的实体关系推理表示 #其中ht_i[:, 0], ht_i[:, 1]表示当前样本中所有实体对的头尾实体索引，这个关系表示矩阵转置成形状为(num_pairs, hidden_size)的关系向量
            r_v2 = []
            for j in range(ht_i.shape[0]):  #遍历行
                h_e_pos = ht_i[j, 0]    #头实体索引
                t_e_pos = ht_i[j, 1]    #尾实体索引
                e_m_feat = r_rep_m[i, :, mentions_pos[i][h_e_pos]:mentions_pos[i][h_e_pos+1], mentions_pos[i][t_e_pos]:mentions_pos[i][t_e_pos+1]]
                e_feat = torch.mean(e_m_feat, dim=[1, 2]).reshape([1, -1])  #论文公式18
                r_v2.append(e_feat)
            r_v2 = torch.cat(r_v2, dim=0)   #对于尾实体按维度1进行拼接，
            relation.append(torch.cat([r_v1, r_v2], dim=-1))    #R  rrs,    对于头尾实体按维度1进行拼接，得到relation形状，numeht*(emb_size*2)
            f_e = feature_e[i, :, ht_i[:, 0], ht_i[:, 1]].transpose(1, 0)   #论文中的rf
            sc_feature_e.append(f_e)

            #取Es中的头尾实体表示
            e_h = torch.index_select(nodes_re[i], 0, ht_i[:, 0])    #取所有实体对的头实体
            e_t = torch.index_select(nodes_re[i], 0, ht_i[:, 1])    #取所有尾实体
            entity_h.append(e_h)
            entity_t.append(e_t)

            '''
                取子图中头尾实体表示
            '''
            new_e_h = torch.index_select(new_nodes_re[i], 0, ht_i[:, 0])
            new_e_t = torch.index_select(new_nodes_re[i], 0, ht_i[:, 1])
            new_entity_h.append(new_e_h)
            new_entity_t.append(new_e_t)


            #从Ec中获取ht_rep
            eh = torch.index_select(ec_rep[i], 0, ht_i[:, 0])
            et = torch.index_select(ec_rep[i], 0, ht_i[:, 1])
            e_tou.append(eh)
            e_wei.append(et)



        relation = torch.cat(relation, dim=0)       #rrs
        sc_feature_e = torch.cat(sc_feature_e, dim=0)   #rf


        #取Es中的实体对表示
        entity_h = torch.cat(entity_h, dim=0)
        entity_t = torch.cat(entity_t, dim=0)
        entity_ht = self.ht_extractor(torch.cat([entity_h, entity_t], dim=-1))  #rht，来自Es的,得到实体对    emb_size*4 -> emb_size*2
        #取Ec中的实体对表示
        e_tou = torch.cat(e_tou, dim=0)
        e_wei = torch.cat(e_wei, dim=0)
        e_tw = torch.cat([e_tou, e_wei], dim=-1)
        #取子图中实体对表示
        new_entity_h = torch.cat(new_entity_h, dim=0)
        new_entity_t = torch.cat(new_entity_t, dim=0)
        new_entity_ht = self.ht_extractor(torch.cat([new_entity_h, new_entity_t], dim=-1))
        relation_rep = self.MIP_Linear(torch.cat([entity_ht, e_tw, sc_feature_e, new_entity_ht], dim=-1))   #emb_size * 7 -> emb_size * 5
        relation_rep = self.MIP_Linear1(relation_rep)   #emb_size * 5 -> emb_size * 4
        relation_rep = torch.tanh(self.MIP_Linear2(relation_rep))
        logits = self.bilinear(relation_rep)
        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)

        '''
        relation_rep = self.MIP_Linear(torch.cat([relation, sc_feature_e, entity_ht], dim=-1))  #r = [rrs;rht;rf],  三者最后一个维度分别为
        relation_rep = torch.tanh(self.MIP_Linear2(relation_rep))   #r先过一层线性，再激活函数tanH,
        logits = self.bilinear(relation_rep)
        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)
        '''



        '''
        #取Ec中的实体对表示用于分类
        e_tou = torch.cat(e_tou, dim=0)
        e_wei = torch.cat(e_wei, dim=0)
        e_tw = torch.cat([e_tou, e_wei], dim=-1)
        logit = self.bilinear(torch.tanh(e_tw))
        outputs = (self.loss_fnt.get_label(logit, num_labels=self.num_labels),)
        '''

        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            a = loss.to(sequence_output)
            output = (a,) + output
        return output
