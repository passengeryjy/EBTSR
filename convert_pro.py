import os
from tqdm import tqdm
import ujson as json
import numpy as np
from adj_utils import sparse_mxs_to_torch_sparse_tensor
from transformers import AutoConfig, AutoModel, AutoTokenizer
import scipy.sparse as sp
from subGraph import preprocess_data


docred_rel2id = json.load(open('meta/rel2id.json', 'r'))
cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}


def read_docred_con(file_in, tokenizer, max_seq_length=1024):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    max_entity = 0  # lry
    maxlen = 0  # lry
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []

        doc_path = preprocess_data(sample)
        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        mention_pos = []
        ent_num = len(entities)     #实体数量
        men_num = 0
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((pos[0]))
                entity_end.append((pos[1]))
                mention_pos.append((pos[0], pos[1]))
                men_num += 1

        sent_pos = {}
        i_t = 0
        i_s = 0
        sent_map = {}
        for sent in sample['sents']:
            sent_pos[i_s] = len(sents)
            for token in sent:
                tokens_wordpiece = tokenizer.tokenize(token)
                for start, end in mention_pos:
                    if start == i_t:
                        tokens_wordpiece = ["*"] + tokens_wordpiece
                    if end == i_t + 1:
                        tokens_wordpiece = tokens_wordpiece + ["*"]
                sent_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
                i_t += 1
            sent_map[i_t] = len(sents)
            i_s += 1
        sent_pos[i_s] = len(sents)


        sentl_node = []
        # 句子节点
        for l in range(len(sent_pos) - 1):
            sentl_node += [[l, l, l, l, l,l+ent_num+men_num, 2]]
        mention_pos = []

        # 取处理后的句子token首尾位置
        sentl_pos = []
        for i in range(len(sent_pos) - 1):
            sentl_pos.append((sent_pos[i], sent_pos[i + 1]))

        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                # evidence = label['evidence']
                # r = int(cdr_rel2id_rel2id[label['r']])
                # r = int(cdr_rel2id[label['r']])
                r = label['r']
                dist = label['dist']
                if (label['h'], label['t']) not in train_triple:
                    # train_triple[(label['h'], label['t'])] = [{'relation': r, 'evidence': evidence}]
                    train_triple[(label['h'], label['t'])] = [{'relation': r, 'dist': dist}]
                else:
                    # train_triple[(label['h'], label['t'])].append({'relation': r, 'evidence': evidence})
                    train_triple[(label['h'], label['t'])] = [{'relation': r, 'dist': dist}]

        entity_pos = []
        men_id = 0
        for e_id, e in enumerate(entities):
            entity_pos.append([])
            for m in e:
                # start = sent_map[m["sent_id"]][m["pos"][0]]
                # end = sent_map[m["sent_id"]][m["pos"][1]]
                start = sent_map[m["pos"][0]]
                end = sent_map[m["pos"][1]]
                s_id = m["sent_id"]

                # entity_pos[-1].append((start, end,))
                # entity_pos[-1].append((start, end, e_id, h_lid, t_lid, s_id))
                entity_pos[-1].append((start, end, e_id, s_id, men_id, men_id+ent_num))
                men_id += 1

        entity_node = []
        mention_node = []
        for idx in range(len(entity_pos)):
            # entity_node += [[idx, idx, idx, idx, idx, idx, 0]]
            entity_node += [[idx, idx, idx, idx, idx, idx, 0]]
            for item in entity_pos[idx]:
                mention_node += [list(item) + [1]]

        nodes = entity_node + mention_node + sentl_node
        nodes = np.array(nodes)
        # xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')

        xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')
        l_type, r_type = nodes[xv, 6], nodes[yv, 6]
        l_eid, r_eid = nodes[xv, 2], nodes[yv, 2]
        l_sid, r_sid = nodes[xv, 3], nodes[yv, 3]

        adj_temp = np.full((l_type.shape[0], r_type.shape[0]), 0,
                           'i')  # adj_temp是一个临时的邻接矩阵，用来存储边的情况，
        # adjacency = np.full((5, l_type.shape[0], r_type.shape[0]), 0.0) #adjacency是一个5维矩阵，其中adjacency[i]表示关系类型i（五类边）的邻接矩阵
        adjacency = np.full((4, l_type.shape[0], r_type.shape[0]), 0.0)
        # adjacency = np.full((3, l_type.shape[0], r_type.shape[0]), 0.0)

        # mention-mention edge
        # 左右节点类型都为1 且出现在相同句子，则进行连接
        adj_temp = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adj_temp)
        adjacency[0] = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adjacency[0])

        # mention-entity
        adj_temp = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adj_temp)
        adj_temp = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adj_temp)
        adjacency[1] = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adjacency[1])
        adjacency[1] = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adjacency[1])

        # mention-sentl
        adj_temp = np.where((l_type == 1) & (r_type == 2) & (l_sid == r_sid), 1, adj_temp)
        adj_temp = np.where((l_type == 2) & (r_type == 1) & (l_sid == r_sid), 1, adj_temp)
        adjacency[2] = np.where((l_type == 1) & (r_type == 2) & (l_sid == r_sid), 1, adjacency[2])
        adjacency[2] = np.where((l_type == 2) & (r_type == 1) & (l_sid == r_sid), 1, adjacency[2])

        # sentl-sentl
        # adj_temp = np.where((l_type == 2) & (r_type == 2), 1, adj_temp)
        # adjacency[3] = np.where((l_type == 2) & (r_type == 2), 1, adjacency[3])

        # 按顺序相连
        adj_temp = np.where((l_type == 2) & (r_type == 2) & (l_sid + 1 == r_sid), 1, adj_temp)
        adjacency[3] = np.where((l_type == 2) & (r_type == 2) & (l_sid + 1 == r_sid), 1,adjacency[3])

        # adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(adjacency[i]) for i in range(5)])
        adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(adjacency[i]) for i in range(4)])

        '''
            根据路径选择子图节点，并构建其邻接矩阵
            new_ent_node = []
            new_men_node = []
            new_sen_node = []
        '''
        #path分为：e-m-s-m-e; e-m-s-s-m-e; e-m-s-m-e-m-s-m-e
        new_ent_id = []
        new_men_id = []
        new_sen_id = []
        for i in range(len(doc_path)):
            if len(doc_path[i]) == 5:
                new_ent_id.append(doc_path[i][0])
                new_ent_id.append(doc_path[i][-1])
                new_men_id.append(doc_path[i][1])
                new_men_id.append(doc_path[i][-2])
                new_sen_id.append(doc_path[i][2])
            elif len(doc_path[i]) == 6:

                new_ent_id.append(doc_path[i][0])
                new_ent_id.append(doc_path[i][-1])
                new_men_id.append(doc_path[i][1])
                new_men_id.append(doc_path[i][-2])
                new_sen_id.append(doc_path[i][2])
                new_sen_id.append(doc_path[i][3])
            elif len(doc_path[i]) == 9:
                new_ent_id.append(doc_path[i][0])
                new_ent_id.append(doc_path[i][4])
                new_ent_id.append(doc_path[i][-1])
                new_men_id.append(doc_path[i][1])
                new_men_id.append(doc_path[i][3])
                new_men_id.append(doc_path[i][5])
                new_men_id.append(doc_path[i][7])
                new_sen_id.append(doc_path[i][2])
                new_sen_id.append(doc_path[i][6])
            elif len(doc_path[i]) == 13:
                new_ent_id.append(doc_path[i][0])
                new_ent_id.append(doc_path[i][4])
                new_ent_id.append(doc_path[i][8])
                new_ent_id.append(doc_path[i][12])
                new_men_id.append(doc_path[i][1])
                new_men_id.append(doc_path[i][3])
                new_men_id.append(doc_path[i][5])
                new_men_id.append(doc_path[i][7])
                new_men_id.append(doc_path[i][9])
                new_men_id.append(doc_path[i][11])
                new_sen_id.append(doc_path[i][2])
                new_sen_id.append(doc_path[i][6])
                new_sen_id.append(doc_path[i][10])

        new_ent_id = list(set(new_ent_id))
        new_men_id = list(set(new_men_id))
        new_sen_id = list(set(new_sen_id))
        new_ent_id.sort()
        new_men_id.sort()
        new_sen_id.sort()
        new_ent_node = [node for node in entity_node if node[0] in new_ent_id]
        new_men_node = [node for node in mention_node if node[4] in new_men_id]
        new_sen_node = [node for node in sentl_node if node[0] in new_sen_id]

        #子图构建
        new_nodes = new_ent_node + new_men_node + new_sen_node
        new_nodes = np.array(new_nodes)

        new_xv, new_yv = np.meshgrid(np.arange(new_nodes.shape[0]), np.arange(new_nodes.shape[0]), indexing='ij')
        new_l_type, new_r_type = new_nodes[new_xv, 6], new_nodes[new_yv, 6]
        new_l_eid, new_r_eid = new_nodes[new_xv, 2], new_nodes[new_yv, 2]
        new_l_sid, new_r_sid = new_nodes[new_xv, 3], new_nodes[new_yv, 3]

        new_adj_temp = np.full((new_l_type.shape[0], new_r_type.shape[0]), 0,'i')  # adj_temp是一个临时的邻接矩阵，用来存储边的情况，

        new_adjacency = np.full((4, new_l_type.shape[0], new_r_type.shape[0]), 0.0)

        # mention-mention edge
        # 左右节点类型都为1 且出现在相同句子，则进行连接
        new_adj_temp = np.where((new_l_type == 1) & (new_r_type == 1) & (new_l_sid == new_r_sid), 1, new_adj_temp)
        new_adjacency[0] = np.where((new_l_type == 1) & (new_r_type == 1) & (new_l_sid == new_r_sid), 1, new_adjacency[0])

        # mention-entity
        new_adj_temp = np.where((new_l_type == 0) & (new_r_type == 1) & (new_l_eid == new_r_eid), 1, new_adj_temp)
        new_adj_temp = np.where((new_l_type == 1) & (new_r_type == 0) & (new_l_eid == new_r_eid), 1, new_adj_temp)
        new_adjacency[1] = np.where((new_l_type == 0) & (new_r_type == 1) & (new_l_eid == new_r_eid), 1, new_adjacency[1])
        new_adjacency[1] = np.where((new_l_type == 1) & (new_r_type == 0) & (new_l_eid == new_r_eid), 1, new_adjacency[1])

        # mention-sentl
        new_adj_temp = np.where((new_l_type == 1) & (new_r_type == 2) & (new_l_sid == new_r_sid), 1, new_adj_temp)
        new_adj_temp = np.where((new_l_type == 2) & (new_r_type == 1) & (new_l_sid == new_r_sid), 1, new_adj_temp)
        new_adjacency[2] = np.where((new_l_type == 1) & (new_r_type == 2) & (new_l_sid == new_r_sid), 1, new_adjacency[2])
        new_adjacency[2] = np.where((new_l_type == 2) & (new_r_type == 1) & (new_l_sid == new_r_sid), 1, new_adjacency[2])

        # sentl-sentl
        # adj_temp = np.where((l_type == 2) & (r_type == 2), 1, adj_temp)
        # adjacency[3] = np.where((l_type == 2) & (r_type == 2), 1, adjacency[3])

        # 按顺序相连
        new_adj_temp = np.where((new_l_type == 2) & (new_r_type == 2) & (new_l_sid + 1 == new_r_sid), 1, new_adj_temp)
        new_adjacency[3] = np.where((new_l_type == 2) & (new_r_type == 2) & (new_l_sid + 1 == new_r_sid), 1, new_adjacency[3])

        new_adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(new_adjacency[i]) for i in range(4)])


        relations, hts, dists = [], [], []
        for h, t in train_triple.keys():
            # relation = [0] * len(docred_rel2id)
            relation = [0] * len(cdr_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                # evidence = mention["evidence"]
                if mention["dist"] == "CROSS":
                    dist =1
                elif mention["dist"] == "NON-CROSS":
                    dist = 0
            relations.append(relation)
            hts.append([h, t])
            dists.append(dist)
            # pos_samples += 1

        maxlen = max(maxlen, len(sents))
        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        i_line += 1
        if len(entity_pos) > max_entity:
            max_entity = len(entity_pos)
            if max_entity > 50:
                print(i_line, max_entity)
        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'title': sample['title'],
                   'dists': dists,
                   'adjacency': adjacency,
                   'link_pos': sentl_pos,
                   'nodes_info': nodes,
                   'sub_nodes': new_nodes,
                   'sub_adjacency': new_adjacency,
                   }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    # print("# of positive examples {}.".format(pos_samples))
    # print("# of negative examples {}.".format(neg_samples))
    print(f'maxlen:{maxlen}')
    print(f'max_entity_num:{max_entity}')
    return features


if __name__ == '__main__':
    data_dir = 'dataset\cdr\convert_CDR'
    tokenizer = AutoTokenizer.from_pretrained('./biobert_base')
    #rel2id = json.load(open(os.path.join(data_dir, 'rel2id.json'), 'r'))
    rel2id = {'1:NR:2': 0, '1:CID:2': 1}
    id2rel = {v: k for k, v in rel2id.items()}
    #word2id = json.load(open(os.path.join(data_dir, 'word2id.json'), 'r'))
    #ner2id = json.load(open(os.path.join(data_dir, 'ner2id.json'), 'r'))
    train_in_file = os.path.join(data_dir, 'convert_train.json')
    dev_in_file = os.path.join(data_dir, 'convert_dev.json')
    test_in_file = os.path.join(data_dir, 'convert_test.json')
    train_feature = read_docred_con(train_in_file, tokenizer)