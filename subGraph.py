import os
from collections import defaultdict
import json
import dgl
from tqdm import tqdm
import numpy as np
import torch
import networkx as nx
from transformers import AutoTokenizer
from utils import weighted_path_score
from torch.utils.data import Dataset


class REDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item


def preprocess_data(in_file):

    doc_path_ent = []  #记录文档得证据路径
    doc_path_men = []
    doc_path_sen = []
    #with open(in_file, 'r', encoding='utf-8') as f:
    #    raw_data = json.load(f)
    #for doc_id, doc in tqdm(enumerate(raw_data), total=len(raw_data)):
    doc = in_file

    title, sents, entities, labels = doc['title'], doc['sents'], doc['vertexSet'], doc['labels']
    num_sents = len(sents)

    num_entities = len(entities)

    '''
        每一篇文档证据路径所涉及到得节点        
    '''

    mention2sent = defaultdict(int)
    entity2mention = defaultdict(list)
    mention2pos = dict()
    mention_id = 0
    for entity_id, entity in enumerate(entities):
        for mention in entity:
            pos, ner, sent_id = mention['pos'], mention['type'], mention['sent_id']
            # ner_ids[pos[0]:pos[1]] = ner2id[ner]  # 对提及位置进行标注
            entity2mention[entity_id].append(mention_id)
            mention2sent[mention_id] = sent_id
            mention2pos[mention_id] = pos
            mention_id += 1

    num_mentions = mention_id

    # 原始图
    # 输入为实体集及对应提及，提及数及对应句子id，句子数
    graph = build_graph(entity2mention, mention2sent, num_sents)
    node_type_seg = [0, num_entities, num_entities + num_mentions,
                     num_entities + num_mentions + num_sents]
    nx_graph = dgl.to_networkx(dgl.to_homogeneous(graph))
    nx_graph = nx.Graph(nx_graph.to_undirected())
    # 句子-句子之间的边权重设为4
    for nx_sent_id in range(node_type_seg[2], node_type_seg[3] - 1):
        nx_graph.edges[nx_sent_id, nx_sent_id + 1]['weight'] = 4

    head_tail_entity_pairs = []
    evidence_ent_nodes = []  # 记录labels中实体对所有证据路径上涉及到的节点（实体，提及，句子）
    evidence_men_nodes = []
    evidence_sen_nodes = []
    evidence_path = []
    # 遍历每一个实体对，找其路径，有多条路径存在
    for label in labels:
        head_entity_id = label['h']
        tail_entity_id = label['t']
        head_tail_entity_pairs.append([head_entity_id, tail_entity_id])
        cutoff = 4
        paths = [path for path in
                 nx.all_simple_paths(nx_graph, head_entity_id, tail_entity_id, cutoff=cutoff) if
                 weighted_path_score(nx_graph, path) <= cutoff]
        while not paths:
            cutoff += 4
            paths = [path for path in
                     nx.all_simple_paths(nx_graph, head_entity_id, tail_entity_id,
                                         cutoff=cutoff) if
                     weighted_path_score(nx_graph, path) <= cutoff]
        node_set = {node for path in paths for node in path}

        entity, mention, sentence = [], [], []
        for node in node_set:
            if node >= node_type_seg[2]:
                sentence.append(node - node_type_seg[2])
            elif node >= node_type_seg[1]:
                mention.append(node - node_type_seg[1])
            else:
                entity.append(node)
        entity.sort()
        mention.sort()
        sentence.sort()

        for path in paths:
            for i in range(len(path)):
                #转换句子序号
                if path[i] >= node_type_seg[2]:
                    path[i] -= node_type_seg[2]
                #转换提及序号
                elif path[i] >= node_type_seg[1]:
                    path[i] -= node_type_seg[1]
                else:
                    continue

        evidence_ent_nodes.append(entity)
        evidence_men_nodes.append(mention)
        evidence_sen_nodes.append(sentence)
        for i in range(len(paths)):
            evidence_path.append(paths[i])
    '''
    doc_path_ent.append(evidence_ent_nodes)
    doc_path_men.append(evidence_men_nodes)
    doc_path_men.append(evidence_sen_nodes)
    doc_path.append(evidence_path)
    '''

    #torch.save(data, out_file)

    return evidence_path


def build_graph(entity2mention, mention2sent, num_sents):
    num_nodes_dict = {'entity': len(entity2mention), 'mention': len(mention2sent),
                      'sentence': num_sents, 'context': 0}
    data_dict = defaultdict(list)

    data_dict[('context', 'cc', 'context')] = []
    data_dict[('context', 'ce', 'entity')] = []
    data_dict[('entity', 'ec', 'context')] = []

    for entity_id, mentions in entity2mention.items():
        data_dict[('entity', 'ee', 'entity')].append((entity_id, entity_id))
        for mention_id in mentions:
            data_dict[('mention', 'mm', 'mention')].append((mention_id, mention_id))
            data_dict[('mention', 'me', 'entity')].append((mention_id, entity_id))
            data_dict[('entity', 'em', 'mention')].append((entity_id, mention_id))

    for mention_id, sent_id in mention2sent.items():
        data_dict[('mention', 'ms', 'sentence')].append((mention_id, sent_id))
        data_dict[('sentence', 'sm', 'mention')].append((sent_id, mention_id))

    for i in range(num_sents - 1):
        data_dict[('sentence', 'ss', 'sentence')].append((i, i + 1))
        data_dict[('sentence', 'ss', 'sentence')].append((i + 1, i))

    for i in range(num_sents):
        data_dict[('sentence', 'ss', 'sentence')].append((i, i))

    graph = dgl.heterograph(data_dict, num_nodes_dict)

    return graph




if __name__ == '__main__':
    data_dir = 'dataset\cdr\convert_CDR'

    #rel2id = json.load(open(os.path.join(data_dir, 'rel2id.json'), 'r'))
    rel2id = {'1:NR:2': 0, '1:CID:2': 1}
    id2rel = {v: k for k, v in rel2id.items()}
    #word2id = json.load(open(os.path.join(data_dir, 'word2id.json'), 'r'))
    #ner2id = json.load(open(os.path.join(data_dir, 'ner2id.json'), 'r'))
    tokenizer = AutoTokenizer.from_pretrained
    train_in_file = os.path.join(data_dir, 'convert_train.json')
    dev_in_file = os.path.join(data_dir, 'convert_dev.json')
    test_in_file = os.path.join(data_dir, 'convert_test.json')
    train_out_file = os.path.join(data_dir, 'prepro_data', 'train.pt')
    dev_out_file = os.path.join(data_dir, 'prepro_data', 'dev.pt')
    test_out_file = os.path.join(data_dir, 'prepro_data', 'test.pt')

    preprocess_data(train_in_file)

    preprocess_data(dev_in_file)

    preprocess_data(test_in_file)
