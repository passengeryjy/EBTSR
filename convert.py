from tqdm import tqdm
import ujson as json
import numpy as np
from adj_utils import sparse_mxs_to_torch_sparse_tensor
import scipy.sparse as sp
import  json

docred_rel2id = json.load(open('meta/rel2id.json', 'r'))
cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}


def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res

def convert(file_in):
    pmids = set()
    features = []
    maxlen = 0
    conv_res = []   #存放最终的结果写入json文件

    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        # 处理每篇文档
        for i_l, line in enumerate(tqdm(lines)):
            result = {}  # 存放每篇文档的结果，包括四个键："vertexSet", "labels", "title", "sents"
            line = line.rstrip().split('\t')    #分成['pmid', 'sents', ''....]
            sent = []   #存放句子id
            mention_pos = []    #存放提及开始结束位置，实体id,所在句子id
            entity_pos = []     #存放提及开始结束位置，及实体类型
            ent2idx = {}    #存放实体和id之间映射
            train_triples = {}  #结构{(h_id, t_id):[{'relation': r}],},存放头尾实体id对应关系类型，用于构造labels
            # 添加文章标题
            pmid = line[0]  # 得到文章id，相当于title
            if 'title' not in result:
                result['title'] = int(pmid)

            #每篇文档result中添加句子
            #获得句子集
            text = line[1]
            sents = [t.split(' ') for t in text.split('|')] #[[]..]
            if  'sents' not in result:
                result['sents'] = sents
            #取一个句子id信息
            for i in range(len(sents)):
                sent.append(i)

            #转换labels和vertexSet
            # 按17的size进行切分，prs中每一个元素相当于实体所有提及和疾病所构成的关系，给出了头尾实体，以及相应关系
            prs = chunks(line[2:], 17)
            #p:1:CID:2	R2L	NON-CROSS	58-61	51-52	D008750	alpha - methyldopa|alpha - methyldopa	Chemical	58:203	61:206	2:6	D007022	hypotensive	Disease	51	52	2
            for p in prs:
                #取化学物质实体的出现开始结束位置
                es = list(map(int, p[8].split(':')))
                ed = list(map(int, p[9].split(':')))
                tpy = p[7]
                for start, end in zip(es, ed):
                    entity_pos.append([start, end, tpy])

                #取疾病实体的位置信息
                es = list(map(int, p[14].split(':')))
                ed = list(map(int, p[15].split(':')))
                tpy = p[13]
                for start, end in zip(es, ed):
                    entity_pos.append([start, end, tpy])


            #处理化学物质疾病关系三元组信息
            #得到mention_pos,包含开始结束位置，实体id, 句子id, 提及名称
            #train_triple包含头尾实体，所属关系r，
            for p in prs:
                if p[0] == "not_include":
                    continue
                if p[1] == "L2R":
                    h_id, t_id = p[5], p[11]
                    h_start, t_start = p[8], p[14]
                    h_end, t_end = p[9], p[15]
                    h_sid, t_sid = p[10], p[16]
                    mention_h = [t.split(' ') for t in p[6].split('|')]
                    mention_t = [t.split(' ') for t in p[12].split('|')]
                else:
                    t_id, h_id = p[5], p[11]
                    t_start, h_start = p[8], p[14]
                    t_end, h_end = p[9], p[15]
                    t_sid, h_sid = p[10], p[16]
                    mention_t = [t.split(' ') for t in p[6].split('|')]
                    mention_h = [t.split(' ') for t in p[12].split('|')]
                h_start = list(map(int, h_start.split(':')))
                h_end = list(map(int, h_end.split(':')))

                t_start = list(map(int, t_start.split(':')))
                t_end = list(map(int, t_end.split(':')))

                h_sid = map(int, h_sid.split(':'))     #在句子中被标记为实体的token的id，这些token可能不止一个
                t_sid = map(int, t_sid.split(':'))
                h_sid = [idx for idx in h_sid]
                t_sid = [idx for idx in t_sid]
                if h_id not in ent2idx:
                    h_eid = [len(ent2idx)] * len(h_start)
                    ent2idx[h_id] = len(ent2idx)
                    mention_pos.append(list(zip(h_start, h_end, h_eid, h_sid, mention_h)))
                if t_id not in ent2idx:
                    t_eid = [len(ent2idx)] * len(t_start)
                    ent2idx[t_id] = len(ent2idx)
                    mention_pos.append(list(zip(t_start, t_end, t_eid, t_sid, mention_t)))

                h_id, t_id = ent2idx[h_id], ent2idx[t_id]   #此时id为实体编号了，不再是专业id
                r = gda_rel2id[p[0]]
                dist = p[2]     #实体对距离
                if (h_id, t_id) not in train_triples:
                    train_triples[(h_id, t_id)] = [{'relation': r, 'dist': dist}]
                else:
                    train_triples[(h_id, t_id)].append({'relation': r, 'dist': dist})

            vertex = []  # 存放顶点集

            label = []  # 存放三元组
            for h, t in train_triples:
                rel_label = {}  #
                if "h" not in rel_label:
                    rel_label["h"] = h
                if "t" not in rel_label:
                    rel_label["t"] = t
                for rel in train_triples[h, t]:
                    if "r" not in rel_label:
                        rel_label["r"] = rel["relation"]
                        rel_label["dist"] = rel["dist"]
                label.append(rel_label)
            result["labels"] = label

            for entity in mention_pos:
                ent_men = []
                for mention in entity:
                    ent_dict = {}
                    ent_dict["pos"] = [mention[0], mention[1]]
                    ent_dict["type"] =  mention[2]
                    ent_dict["sent_id"] = mention[3]
                    ent_dict["name"] = mention[4]
                    ent_men.append(ent_dict)
                vertex.append(ent_men)
            result["vertexSet"] = vertex

            print(result)
            conv_res.append(result)
        return conv_res

if __name__ == "__main__":
    file = "./dataset/gda/train.data"
    res = convert(file)
    with open("./dataset/gda/convert_gda/convert_train" + ".json", "w") as fh:
        json.dump(res, fh)
    file = "./dataset/gda/dev.data"
    res = convert(file)
    with open("./dataset/gda/convert_gda/convert_dev" + ".json", "w") as fh:
        json.dump(res, fh)
    file = "./dataset/gda/test.data"
    res = convert(file)
    with open("./dataset/gda/convert_gda/convert_test" + ".json", "w") as fh:
        json.dump(res, fh)










