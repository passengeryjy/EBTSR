from collections import defaultdict
import json
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

'''
默认路径，两个句子，一个包含头实体，一个包含尾实体（是指实体提及）
连续路径，两个实体（考虑的所有提及），只要存在相差不超过三句，则将这三句都作为证据句
多跳路径，不超过三跳

'''
def extract_path(data_path, keep_sent_order=True):
    with open(data_path, 'r') as fh:
        docu = json.load(fh)
    result = [] #存放所有处理好的文章证据句

    #处理证据路径
    for id, data in tqdm(enumerate(docu), desc = "Example", total = len(docu)):
        sents = data["sents"]
        nodes = [[] for _ in range(len(data['sents']))] #用来存放每个句子中涉及的实体，以实体在顶点集中的序号存入
        e2e_sent = defaultdict(dict)    #存放头尾实体共现句子

        # create mention's list for each sentence
        for ns_no, ns in enumerate(data['vertexSet']):
            for n in ns:    #遍历实体提及，将提及所在的句子添加该实体id
                sent_id = int(n['sent_id'])
                nodes[sent_id].append(ns_no)

        for sent_id in range(len(sents)):
            for n1 in nodes[sent_id]:   #遍历句子中的实体
                for n2 in nodes[sent_id]:
                    if n1 == n2:
                        continue
                    if n2 not in e2e_sent[n1]:
                        e2e_sent[n1][n2] = set()
                    e2e_sent[n1][n2].add(sent_id)

        # 两跳路径
        path_two = defaultdict(dict)    #保存桥接实体和两跳路径
        entityNum = len(data['vertexSet'])
        for n1 in range(entityNum):
            for n2 in range(entityNum):
                if n1 == n2:
                    continue
                for n3 in range(entityNum):
                    if n3 == n1 or n3 == n2:
                        continue
                    #n3非桥接实体
                    if not (n3 in e2e_sent[n1] and n2 in e2e_sent[n3]):
                        continue
                    for s1 in e2e_sent[n1][n3]:
                        for s2 in e2e_sent[n3][n2]:
                            if s1 == s2:
                                continue
                            if n2 not in path_two[n1]:
                                path_two[n1][n2] = []
                            cand_sents = [s1, s2]   #n1和n3共现s1，n2和n3共现s3，则n1和n2的证据路径为s1,s3
                            if keep_sent_order == True:
                                cand_sents.sort()
                            path_two[n1][n2].append((cand_sents, n3))
        print("two_path:",path_two)

        # 三跳，用了两个桥接实体
        #n1-n3-nn(桥接n3和n2且n1不在的句子)-nn-n2
        path_three = defaultdict(dict)
        for n1 in range(entityNum):
            for n2 in range(entityNum):
                if n1 == n2:
                    continue
                for n3 in range(entityNum):
                    if n3 == n1 or n3 == n2:
                        continue
                    if n3 in e2e_sent[n1] and n2 in path_two[n3]:
                        for cand1 in e2e_sent[n1][n3]:
                            for cand2 in path_two[n3][n2]:
                                if cand1 in cand2[0]:
                                    continue
                                if cand2[1] == n1:
                                    continue
                                if n2 not in path_three[n1]:
                                    path_three[n1][n2] = []
                                cand_sents = [cand1] + cand2[0]
                                if keep_sent_order:
                                    cand_sents.sort()
                                path_three[n1][n2].append((cand_sents, [n3, cand2[1]]))
        print("three_path:", path_three)

        # Consecutive Path
        consecutive = defaultdict(dict)
        for h in range(entityNum):
            for t in range(h + 1, entityNum):
                for n1 in data['vertexSet'][h]:
                    for n2 in data['vertexSet'][t]:
                        gap = abs(n1['sent_id'] - n2['sent_id'])
                        if gap > 2:
                            continue
                        if t not in consecutive[h]:
                            consecutive[h][t] = []
                            consecutive[t][h] = []
                        if n1['sent_id'] < n2['sent_id']:
                            beg, end = n1['sent_id'], n2['sent_id']
                        else:
                            beg, end = n2['sent_id'], n1['sent_id']

                        consecutive[h][t].append([[i for i in range(beg, end + 1)]])
                        consecutive[t][h].append([[i for i in range(beg, end + 1)]])
        print("consecutive:", consecutive)
        # Merge
        merge = defaultdict(dict)
        for n1 in range(entityNum):
            for n2 in range(entityNum):
                if n2 in path_two[n1]:
                    merge[n1][n2] = path_two[n1][n2]
                if n2 in path_three[n1]:
                    if n2 in merge[n1]:
                        merge[n1][n2] += path_three[n1][n2]
                    else:
                        merge[n1][n2] = path_three[n1][n2]

                if n2 in consecutive[n1]:
                    if n2 in merge[n1]:
                        merge[n1][n2] += consecutive[n1][n2]
                    else:
                        merge[n1][n2] = consecutive[n1][n2]

        # Default Path
        for h in range(len(data['vertexSet'])):
            for t in range(len(data['vertexSet'])):
                if h == t:
                    continue
                if t in merge[h]:
                    continue
                merge[h][t] = []
                for n1 in data['vertexSet'][h]:
                    for n2 in data['vertexSet'][t]:
                        cand_sents = [n1['sent_id'], n2['sent_id']]
                        if keep_sent_order:
                            cand_sents.sort()
                        merge[h][t].append([cand_sents])

        # Remove redundency
        tp_set = set()
        for n1 in merge.keys():
            for n2 in merge[n1].keys():
                hash_set = set()
                new_list = []
                for t in merge[n1][n2]:
                    if tuple(t[0]) not in hash_set:
                        hash_set.add(tuple(t[0]))
                        new_list.append(t[0])
                merge[n1][n2] = new_list
        result.append(merge)

    #根据证据路径构造伪文档
    return result

def pro_docu(file_in, result, tokenizer):
    with open(file_in, 'r') as fh:
        docu = json.load(fh)

    for sample in tqdm(enumerate(docu)):
        entities = sample["vertexSet"]

'''
统计result中的句子数量，去重后再对原文档句子进行抽取

'''






#利用证据句构造伪文档，再以此构造结构图


if  __name__ == "__main__":
    result = extract_path('./dataset/docred/single.json')
    print(result)












