from tqdm import tqdm
import ujson as json
import numpy as np                                          
from adj_utils import sparse_mxs_to_torch_sparse_tensor     
import scipy.sparse as sp                                   

docred_rel2id = json.load(open('meta/rel2id.json', 'r'))
cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}


def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res

def read_docred(file_in, tokenizer, max_seq_length=1024):

    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    max_entity = 0              # lry
    maxlen = 0                  # lry
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []

        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))
        sent_pos = {}                                       
        for i_s, sent in enumerate(sample['sents']):
            sent_pos[i_s] = len(sents)                      
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)
        sent_pos[i_s + 1] = len(sents)                      

        link_node = []                                      
        for l in range(len(sent_pos) - 2):                  
            link_node += [[l, l, l, l, l, l, 2]]            
        link_pos = []                                       
        for i in range(len(sent_pos) - 2):                  
            link_pos.append((sent_pos[i], sent_pos[i+2]))   

        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = []
        for e_id, e in enumerate(entities):
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                s_id = m["sent_id"]                                 
                if s_id < 1:                                        
                    h_lid = -1                                      
                    t_lid = s_id                                    
                elif s_id > len(link_pos) - 1:                      
                    h_lid = s_id - 1                                
                    t_lid = -1                                      
                else:                                               
                    h_lid = s_id - 1                                
                    t_lid = s_id                                    
                # entity_pos[-1].append((start, end,))
                entity_pos[-1].append((start, end, e_id, h_lid, t_lid, s_id))       

        entity_node = []            
        mention_node = []           
        for idx in range(len(entity_pos)):                  
            entity_node += [[idx, idx, idx, idx, idx, idx, 0]]      
            for item in entity_pos[idx]:                            
                mention_node += [list(item) + [1]]                  

        nodes = entity_node + mention_node + link_node              
        nodes = np.array(nodes)  
        xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')  
        l_type, r_type = nodes[xv, 6], nodes[yv, 6]  
        l_eid, r_eid = nodes[xv, 2], nodes[yv, 2]  
        l_h_lid, r_h_lid = nodes[xv, 3], nodes[yv, 3]  
        l_t_lid, r_t_lid = nodes[xv, 4], nodes[yv, 4]  
        l_sid, r_sid = nodes[xv, 5], nodes[yv, 5]  

        adj_temp = np.full((l_type.shape[0], r_type.shape[0]), 0, 'i')
        adjacency = np.full((5, l_type.shape[0], r_type.shape[0]), 0.0)  
        # mention-mention(intra-sentence)
        adj_temp = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adj_temp)
        adjacency[0] = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adjacency[0])  
        # mention-entity
        adj_temp = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adj_temp)
        adj_temp = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adj_temp)
        adjacency[1] = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adjacency[1])  
        adjacency[1] = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adjacency[1])   
        # mention-link
        adj_temp = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1,
                            adj_temp)
        adj_temp = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1,
                            adj_temp)
        adjacency[2] = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid),
                                1, adjacency[2])  
        adjacency[2] = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid),
                                1, adjacency[2])  
        # link-link
        adj_temp = np.where((l_type == 2) & (r_type == 2), 1, adj_temp)
        adjacency[3] = np.where((l_type == 2) & (r_type == 2), 1, adjacency[3])  
        # entity-link
        for x, y in zip(xv.ravel(), yv.ravel()):  
            if nodes[x, 5] == 0 and nodes[y, 5] == 2:  
                z = np.where(
                    (l_eid == nodes[x, 2]) & (l_type == 1) & (r_type == 2) & np.logical_or(r_h_lid == nodes[y, 3], r_t_lid == nodes[y, 4]))  
                temp_ = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid),
                                 1, adj_temp)  
                temp_ = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid),
                                 1, temp_)  
                adjacency[4][x, y] = 1 if (temp_[z] == 1).any() else 0  
                adjacency[4][y, x] = 1 if (temp_[z] == 1).any() else 0

        adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(adjacency[i]) for i in range(5)])  

        relations, hts = [], []
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)

        maxlen = max(maxlen, len(sents))
        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        i_line += 1
        max_entity = max(max_entity, len(entity_pos))           
        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'title': sample['title'],
                   'adjacency': adjacency,  
                   'link_pos': link_pos,  
                   'nodes_info': nodes,  
                   }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    print(f'maxlen:{maxlen}')                                   
    print(f'max_entity_num:{max_entity}')                       
    return features

def read_cdr(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                #按17的size进行切分，prs中每一个元素相当于实体所有提及和疾病所构成的关系，给出了头尾实体，以及相应关系
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()

                #读取实体位置，得到初步的entity_pos,结构为：嵌套集合
                for p in prs:
                    #1:CID:2	R2L	NON-CROSS	58-61	51-52	D008750	alpha - methyldopa|alpha - methyldopa	Chemical	58:203	61:206	2:6	D007022	hypotensive	Disease	51	52	2
                    #
                    es = list(map(int, p[8].split(':')))    #[58, 203]
                    ed = list(map(int, p[9].split(':')))    #[61, 206]
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                i_s = 0     
                sent_pos = {}
                #在文本中对实体位置进行标记，
                for sent in sents:
                    sent_pos[i_s] = len(new_sents)
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)  #原每个token位置对应分词之后的位置映射
                    i_s += 1        
                sent_pos[i_s] = len(new_sents)     #原每个句子pos对应分词之后的位置映射
                sents = new_sents

                '''
                link_node = []
                #len(sent_pos)-2 为链接数，如七句话，sent_pos为8，减2，遍历得到6种链接：1-2-3-4-5-6-7，每种链接句子数为2
                for l in range(len(sent_pos) - 2):          
                    link_node += [[l, l, l, l, l, l, 2]]          
                mention_pos = []
                link_pos = []                               
                for i in range(len(sent_pos) - 2):          #[(0,88),(23, 122), (88,139),(122,249),(139,281),(249,354)],表示每链接的两句子的所有经分词后的token,如第一句和第二句其所有token数为0-88的token
                    link_pos.append((sent_pos[i], sent_pos[i+2]))
                '''


                sentl_node = []
                #句子节点
                for l in range(len(sent_pos)-1):
                    sentl_node += [[l, l, l, l, 2]]
                mention_pos = []

                #取处理后的句子token首尾位置
                sentl_pos = []
                for i in range(len(sent_pos)-1):
                    sentl_pos.append((sent_pos[i],sent_pos[i+1]))


                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                        h_sid, t_sid = p[10], p[16]     
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                        t_sid, h_sid = p[10], p[16]     
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_sid = map(int, h_sid.split(':'))     #在句子中被标记为实体的token的id，这些token可能不止一个
                    t_sid = map(int, t_sid.split(':'))
                    #解映射，得到分词后的实体提及首尾pos
                    h_start = [sent_map[idx] for idx in h_start]    #将提及原来所在位置映射到经tokenizer处理后的句子中对应位置
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    h_sid = [idx for idx in h_sid]      
                    t_sid = [idx for idx in t_sid]
                    '''
                    h_eid,t_eid：长度为实体提及个数，值为实体id
                    '''

                    #h_h_lids, h_t_lids, t_h_lids, t_t_lids = [], [], [], []
                    if h_id not in ent2idx:
                        h_eid = [len(ent2idx)] * len(h_start)   
                        ent2idx[h_id] = len(ent2idx)
                        '''
                        for id in h_sid:                                    
                            if id < 1:                 #实体链中头部单词在句子中的位置，如果头部单词不在实体链所在的句子中，则为 -1；
                                h_h_lid = -1                                
                                h_t_lid = id                                
                            elif id > len(link_pos) - 1:    #实体链中尾部单词在句子中的位置，如果尾部单词不在实体链所在的句子中，则为 -1；
                                h_h_lid = id - 1                            
                                h_t_lid = -1                               
                            else:    #实体提及首尾位置
                                h_h_lid = id - 1                           
                                h_t_lid = id                                
                            h_h_lids.append(h_h_lid)                        
                            h_t_lids.append(h_t_lid)
                        '''
                        #mention_pos.append(list(zip(h_start, h_end, h_eid, h_h_lids, h_t_lids, h_sid)))
                        mention_pos.append(list(zip(h_start, h_end, h_eid, h_sid)))

                    if t_id not in ent2idx:
                        t_eid = [len(ent2idx)] * len(t_start)       
                        ent2idx[t_id] = len(ent2idx)
                        '''
                        for id in t_sid:                                    
                            if id < 1:                                      
                                t_h_lid = -1                                
                                t_t_lid = id                                
                            elif id > len(link_pos) - 1:                    
                                t_h_lid = id - 1                            
                                t_t_lid = -1                               
                            else:                                           
                                t_h_lid = id - 1                            
                                t_t_lid = id                                
                            t_h_lids.append(t_h_lid)                        
                            t_t_lids.append(t_t_lid)
                        '''
                        #mention_pos.append(list(zip(t_start, t_end, t_eid, t_h_lids, t_t_lids, t_sid)))
                        mention_pos.append((list(zip(t_start, t_end, t_eid, t_sid))))

                    #把实体类型转为数字id
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = cdr_rel2id[p[0]]
                    dist = p[2]         
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r, 'dist': dist}]               
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r, 'dist': dist})           

                entity_node = []        
                mention_node = []
                #mention_pos：(start, end, eid, h_lids, t_lids, sid),[[()]...[()]],嵌套存储每个实体所有提及表示
                for idx in range(len(mention_pos)):     
                    entity_node += [[idx, idx, idx, idx, 0]]
                    for item in mention_pos[idx]:           
                        mention_node += [list(item)+[1]]        

                #nodes = entity_node + mention_node + link_node
                nodes = entity_node + mention_node + sentl_node
                nodes = np.array(nodes)         
                if nodes.shape == (34, 5):
                    print("第{}篇文章".format(i_l), len(entity_node), len(mention_node), len(sentl_node))
                #二维矩阵
                #xv, yv分别表示左右节点的索引值
                xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')       
                '''
                #l_type[i,j] 表示左侧第 i 个节点的类型，r_type[i,j] 表示右侧第 j 个节点的类型。
                l_type, r_type = nodes[xv, 6], nodes[yv, 6]    #以 xv 矩阵的行索引为下标，6 为列索引所对应的节点类型,以 yv 矩阵的行索引为下标，6 为列索引所对应的节点类型
                #左右节点在不同属性上的关系矩阵
                l_eid, r_eid = nodes[xv, 2], nodes[yv, 2]
                l_h_lid, r_h_lid = nodes[xv, 3], nodes[yv, 3]       
                l_t_lid, r_t_lid = nodes[xv, 4], nodes[yv, 4]       
                l_sid, r_sid = nodes[xv, 5], nodes[yv, 5]           
                '''
                l_type, r_type = nodes[xv, 4], nodes[yv, 4]
                l_eid, r_eid = nodes[xv, 2], nodes[yv, 2]
                l_sid, r_sid = nodes[xv, 3], nodes[yv, 3]

                adj_temp = np.full((l_type.shape[0], r_type.shape[0]), 0, 'i')  #adj_temp是一个临时的邻接矩阵，用来存储边的情况，
                #adjacency = np.full((5, l_type.shape[0], r_type.shape[0]), 0.0) #adjacency是一个5维矩阵，其中adjacency[i]表示关系类型i（五类边）的邻接矩阵
                #adjacency = np.full((4, l_type.shape[0], r_type.shape[0]), 0.0)
                adjacency = np.full((3, l_type.shape[0], r_type.shape[0]), 0.0)
                '''
                #mention-mention edge
                #左右节点类型都为1 且出现在相同句子，则为1
                adj_temp = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adj_temp)  #将矩阵adj_temp的(l_sid, r_sid)位置赋值为1，如果左实体和右实体都是实体类型（l_type == 1, r_type == 1），并且它们在句子中的位置（l_sid, r_sid）相同
                adjacency[0] = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adjacency[0])

                #mention-entity
                adj_temp = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adj_temp)
                adj_temp = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adj_temp)
                adjacency[1] = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adjacency[1])      
                adjacency[1] = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adjacency[1])

                #mention-bridge
                adj_temp = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)
                adj_temp = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)
                adjacency[2] = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adjacency[2])      
                adjacency[2] = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adjacency[2])      

                #bridge-bridge
                adj_temp = np.where((l_type == 2) & (r_type == 2), 1, adj_temp)
                adjacency[3] = np.where((l_type == 2) & (r_type == 2), 1, adjacency[3])

                #entity-bridge edge
                for x, y in zip(xv.ravel(), yv.ravel()):
                    if nodes[x, 5] == 0 and nodes[y, 5] == 2:                                                   
                        z = np.where((l_eid == nodes[x, 2]) & (l_type == 1) & (r_type == 2) & np.logical_or(r_h_lid == nodes[y, 3], r_t_lid == nodes[y, 4]))   
                        temp_ = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)                
                        temp_ = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, temp_)                   
                        adjacency[4][x, y] = 1 if (temp_[z] == 1).any() else 0                                  
                        adjacency[4][y, x] = 1 if (temp_[z] == 1).any() else 0                                  
                '''

                # mention-mention edge
                # 左右节点类型都为1 且出现在相同句子，则进行连接
                #adj_temp = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adj_temp)
                #adjacency[0] = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adjacency[0])

                # mention-entity
                adj_temp = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adj_temp)
                adj_temp = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adj_temp)
                adjacency[0] = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adjacency[0])
                adjacency[0] = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adjacency[0])

                # mention-sentl
                adj_temp = np.where((l_type == 1) & (r_type == 2) & (l_sid == r_sid), 1, adj_temp)
                adj_temp = np.where((l_type == 2) & (r_type == 1) & (l_sid == r_sid), 1, adj_temp)
                adjacency[1] = np.where((l_type == 1) & (r_type == 2) & (l_sid == r_sid), 1, adjacency[1])
                adjacency[1] = np.where((l_type == 2) & (r_type == 1) & (l_sid == r_sid), 1, adjacency[1])

                # sentl-sentl
                #adj_temp = np.where((l_type == 2) & (r_type == 2), 1, adj_temp)
                #adjacency[3] = np.where((l_type == 2) & (r_type == 2), 1, adjacency[3])

                #按顺序相连
                adj_temp = np.where((l_type == 2) & (r_type == 2) & (l_sid + 1 == r_sid), 1, adj_temp)
                adjacency[2] = np.where((l_type == 2) & (r_type == 2) & (l_sid + 1 == r_sid), 1, adjacency[2])

                # entity-bridge edge
                '''
                for x, y in zip(xv.ravel(), yv.ravel()):
                    if nodes[x, 5] == 0 and nodes[y, 5] == 2:
                        z = np.where((l_eid == nodes[x, 2]) & (l_type == 1) & (r_type == 2) & np.logical_or(r_h_lid == nodes[y, 3], r_t_lid == nodes[y, 4]))
                        temp_ = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)
                        temp_ = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, temp_)
                        adjacency[4][x, y] = 1 if (temp_[z] == 1).any() else 0
                        adjacency[4][y, x] = 1 if (temp_[z] == 1).any() else 0
                '''
                #adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(adjacency[i]) for i in range(5)])
                adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(adjacency[i]) for i in range(3)])

                relations, hts, dists = [], [], []           
                for h, t in train_triples.keys():
                    relation = [0] * len(cdr_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                        if mention["dist"] == "CROSS":
                            dist = 1                            
                        elif mention["dist"] == "NON-CROSS":
                            dist = 0                            
                    relations.append(relation)
                    hts.append([h, t])
                    dists.append(dist)                  

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
            #print(len(mention_pos))
            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': mention_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           'dists': dists,
                           'adjacency': adjacency,
                           'link_pos': sentl_pos,
                           'nodes_info': nodes,         
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features


def read_gda(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]
            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)
                ent2idx = {}
                train_triples = {}
                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))
                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))
                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                i_s = 0                    
                sent_pos = {}    #每个键表示一个句子，对应的值则表示该句子在处理后的文本中的起始位置
                for sent in sents:
                    sent_pos[i_s] = len(new_sents)    
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                    i_s += 1       
                sent_pos[i_s] = len(new_sents)     
                sents = new_sents

                if len(sents) > max_seq_length:         
                    pmids.remove(pmid)                  
                    continue
                '''
                link_node = []  
                link_pos = []
                if len(sent_pos) > 2:
                    for l in range(len(sent_pos) - 2):  
                        link_node += [[l, l, l, l, l, l, 2]]    #表示每个实体和关系节点的属性，每个元素都是一个列表，包含7个值。其中前6个值是节点的属性，第7个值是节点的类型
                        link_pos.append((sent_pos[l], sent_pos[l + 2]))    #
                else:
                    link_node = [[0, 0, 0, 0, 0, 0, 2]]             
                    link_pos.append((sent_pos[0], sent_pos[1]))     
                '''

                sentl_node = []
                #句子节点
                for l in range(len(sent_pos)-1):
                    sentl_node += [[l, l, l, l, 2]]
                mention_pos = []

                #取处理后的句子token首尾位置
                sentl_pos = []
                for i in range(len(sent_pos)-1):
                    sentl_pos.append((sent_pos[i],sent_pos[i+1]))

                #mention_pos = []
                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                        h_sid, t_sid = p[10], p[16]     
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                        t_sid, h_sid = p[10], p[16]     
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_sid = map(int, h_sid.split(':'))  
                    t_sid = map(int, t_sid.split(':'))  
                    h_start = [sent_map[idx] for idx in h_start]    #映射到经处理后的sent_map上，
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    h_sid = [idx for idx in h_sid]  
                    t_sid = [idx for idx in t_sid]  
                    h_h_lids, h_t_lids, t_h_lids, t_t_lids = [], [], [], []    #
                    if h_id not in ent2idx:
                        h_eid = [len(ent2idx)] * len(h_start)    
                        ent2idx[h_id] = len(ent2idx)
                        '''
                        for id in h_sid:                                   
                            if id < 1:                                      
                                h_h_lid = -1                               
                                h_t_lid = id                               
                            elif id > len(link_pos) - 1:                  
                                h_h_lid = id - 1                           
                                h_t_lid = -1                                
                            else:                                          
                                h_h_lid = id - 1                          
                                h_t_lid = id                                
                            h_h_lids.append(h_h_lid)                        
                            h_t_lids.append(h_t_lid)                        
                        mention_pos.append(list(zip(h_start, h_end, h_eid, h_h_lids, h_t_lids, h_sid)))
                        '''
                        mention_pos.append((list(zip(h_start, h_end, h_eid, h_sid))))

                    if t_id not in ent2idx:
                        t_eid = [len(ent2idx)] * len(t_start)       
                        ent2idx[t_id] = len(ent2idx)
                        '''
                        for id in t_sid:                                    
                            if id < 1:                                      
                                t_h_lid = -1                                
                                t_t_lid = id                                
                            elif id > len(link_pos) - 1:                    
                                t_h_lid = id - 1                            
                                t_t_lid = -1                                
                            else:                                           
                                t_h_lid = id - 1                            
                                t_t_lid = id                               
                            t_h_lids.append(t_h_lid)                        
                            t_t_lids.append(t_t_lid)                        
                        mention_pos.append(list(zip(t_start, t_end, t_eid, t_h_lids, t_t_lids, t_sid)))  
                        '''
                        mention_pos.append((list(zip(t_start, t_end, t_eid, t_sid))))

                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = gda_rel2id[p[0]]
                    dist = p[2]             
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r, "dist": dist}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r, "dist": dist})

                entity_node = []  
                mention_node = []  
                for idx in range(len(mention_pos)):  
                    entity_node += [[idx, idx, idx, idx, idx, idx, 0]]  
                    for item in mention_pos[idx]:  
                        mention_node += [list(item) + [1]]     #提及的起始位置、结束位置、实体id、头实体的位置、尾实体的位置、句子id和节点类型（1表示提及）

                #nodes = entity_node + mention_node + link_node
                nodes = entity_node + mention_node + sentl_node
                nodes = np.array(nodes) #一个二维数组
                #一个二维矩阵，其中每个元素都是一个坐标点，这个坐标点的x坐标来自于向量xv，y坐标来自于向量yv，生成的矩阵的第i行j列坐标为(xv[i][j], yv[i][j])
                #创建一个二维网格，其中第一个维度表示每个数据点在nodes数组中的行索引，第二个维度表示每个数据点在nodes数组中的列索引，这样可以用于后续计算每个数据点之间的关系或距离
                xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')
                '''
                l_type, r_type = nodes[xv, 6], nodes[yv, 6]  #取左右节点类型
                l_eid, r_eid = nodes[xv, 2], nodes[yv, 2]  #实体id,mention_node中提及所对应的实体id，
                l_h_lid, r_h_lid = nodes[xv, 3], nodes[yv, 3]  #左节点和右节点头实体的链id
                l_t_lid, r_t_lid = nodes[xv, 4], nodes[yv, 4]  #左节点和右节点尾实体的链id
                l_sid, r_sid = nodes[xv, 5], nodes[yv, 5]  #所在句子id
                '''
                l_type, r_type = nodes[xv, 4], nodes[yv, 4]
                l_eid, r_eid = nodes[xv, 2], nodes[yv, 2]
                l_sid, r_sid = nodes[xv, 3], nodes[yv, 3]

                adj_temp = np.full((l_type.shape[0], r_type.shape[0]), 0, 'i')
                adjacency = np.full((4, l_type.shape[0], r_type.shape[0]), 0.0)

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
                adj_temp = np.where((l_type == 2) & (r_type == 2), 1, adj_temp)
                adjacency[3] = np.where((l_type == 2) & (r_type == 2), 1, adjacency[3])

                '''
                adj_temp = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adj_temp)
                adjacency[0] = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adjacency[0])
                adj_temp = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adj_temp)
                adj_temp = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adj_temp)
                adjacency[1] = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adjacency[1])  
                adjacency[1] = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adjacency[1]) 
                adj_temp = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)
                adj_temp = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)
                adjacency[2] = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adjacency[2])  
                adjacency[2] = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adjacency[2])  
                adj_temp = np.where((l_type == 2) & (r_type == 2), 1, adj_temp)
                adjacency[3] = np.where((l_type == 2) & (r_type == 2), 1, adjacency[3])
                
                for x, y in zip(xv.ravel(), yv.ravel()):  
                    if nodes[x, 5] == 0 and nodes[y, 5] == 2:  
                        z = np.where((l_eid == nodes[x, 2]) & (l_type == 1) & (r_type == 2) & np.logical_or(
                            r_h_lid == nodes[y, 3], r_t_lid == nodes[y, 4]))  
                        temp_ = np.where(
                            (l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1,
                            adj_temp)  
                        temp_ = np.where(
                            (l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1,
                            temp_)  
                        adjacency[4][x, y] = 1 if (temp_[z] == 1).any() else 0  
                        adjacency[4][y, x] = 1 if (temp_[z] == 1).any() else 0  
                '''

                adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(adjacency[i]) for i in range(5)])  

                relations, hts, dists = [], [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(gda_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                        if mention["dist"] == "CROSS":
                            dist = 1                           
                        elif mention["dist"] == "NON-CROSS":
                            dist = 0                           
                    relations.append(relation)
                    hts.append([h, t])
                    dists.append(dist)                          

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)    

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': mention_pos,
                           'labels': relations,
                           'dists': dists,          
                           'hts': hts,
                           'title': pmid,
                           'adjacency': adjacency,  
                           'link_pos': sentl_pos,
                           'nodes_info': nodes,  
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features

#按docred格式读取
def read_docred_con(file_in, tokenizer, max_seq_length=1024):

    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    max_entity = 0              # lry
    maxlen = 0                  # lry
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []

        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        mention_pos = []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((pos[0]))
                entity_end.append((pos[1]))
                mention_pos.append((pos[0], pos[1]))
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
        '''
                new_sents = []
                sent_map = {}
                i_t = 0
                i_s = 0     
                sent_pos = {}
                #在文本中对实体位置进行标记，
                for sent in sents:
                    sent_pos[i_s] = len(new_sents)
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)  #原每个token位置对应分词之后的位置映射
                    i_s += 1        
                sent_pos[i_s] = len(new_sents)     #原每个句子pos对应分词之后的位置映射
                sents = new_sents
        '''
        '''
        link_node = []
        for l in range(len(sent_pos) - 2):
            link_node += [[l, l, l, l, l, l, 2]]
        link_pos = []
        for i in range(len(sent_pos) - 2):
            link_pos.append((sent_pos[i], sent_pos[i+2]))
        '''

        sentl_node = []
        # 句子节点
        for l in range(len(sent_pos) - 1):
            sentl_node += [[l, l, l, l, 2]]
        mention_pos = []

        # 取处理后的句子token首尾位置
        sentl_pos = []
        for i in range(len(sent_pos) - 1):
            sentl_pos.append((sent_pos[i], sent_pos[i + 1]))


        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                #evidence = label['evidence']
                #r = int(cdr_rel2id_rel2id[label['r']])
                #r = int(cdr_rel2id[label['r']])
                r = label['r']
                if (label['h'], label['t']) not in train_triple:
                    #train_triple[(label['h'], label['t'])] = [{'relation': r, 'evidence': evidence}]
                    train_triple[(label['h'], label['t'])] = [{'relation': r}]
                else:
                    #train_triple[(label['h'], label['t'])].append({'relation': r, 'evidence': evidence})
                    train_triple[(label['h'], label['t'])] = [{'relation': r}]

        entity_pos = []
        for e_id, e in enumerate(entities):
            entity_pos.append([])
            for m in e:
                #start = sent_map[m["sent_id"]][m["pos"][0]]
                #end = sent_map[m["sent_id"]][m["pos"][1]]
                start = sent_map[m["pos"][0]]
                end = sent_map[m["pos"][1]]
                s_id = m["sent_id"]
                '''
                if s_id < 1:
                    h_lid = -1
                    t_lid = s_id
                elif s_id > len(link_pos) - 1:
                    h_lid = s_id - 1
                    t_lid = -1
                else:
                    h_lid = s_id - 1
                    t_lid = s_id
                '''
                # entity_pos[-1].append((start, end,))
                #entity_pos[-1].append((start, end, e_id, h_lid, t_lid, s_id))
                entity_pos[-1].append((start, end, e_id, s_id))

        entity_node = []
        mention_node = []
        for idx in range(len(entity_pos)):
            #entity_node += [[idx, idx, idx, idx, idx, idx, 0]]
            entity_node += [[idx, idx, idx, idx, 0]]
            for item in entity_pos[idx]:
                mention_node += [list(item) + [1]]

        nodes = entity_node + mention_node + sentl_node
        nodes = np.array(nodes)
        #xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')
        '''
        l_type, r_type = nodes[xv, 6], nodes[yv, 6]
        l_eid, r_eid = nodes[xv, 2], nodes[yv, 2]
        l_h_lid, r_h_lid = nodes[xv, 3], nodes[yv, 3]
        l_t_lid, r_t_lid = nodes[xv, 4], nodes[yv, 4]
        l_sid, r_sid = nodes[xv, 5], nodes[yv, 5]
        
        
        adj_temp = np.full((l_type.shape[0], r_type.shape[0]), 0, 'i')
        adjacency = np.full((5, l_type.shape[0], r_type.shape[0]), 0.0)
        # mention-mention(intra-sentence)
        adj_temp = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adj_temp)
        adjacency[0] = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adjacency[0])
        # mention-entity
        adj_temp = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adj_temp)
        adj_temp = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adj_temp)
        adjacency[1] = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adjacency[1])
        adjacency[1] = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adjacency[1])
        # mention-link
        adj_temp = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1,
                            adj_temp)
        adj_temp = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1,
                            adj_temp)
        adjacency[2] = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid),
                                1, adjacency[2])
        adjacency[2] = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid),
                                1, adjacency[2])
        # link-link
        adj_temp = np.where((l_type == 2) & (r_type == 2), 1, adj_temp)
        adjacency[3] = np.where((l_type == 2) & (r_type == 2), 1, adjacency[3])
        # entity-link
        for x, y in zip(xv.ravel(), yv.ravel()):
            if nodes[x, 5] == 0 and nodes[y, 5] == 2:
                z = np.where(
                    (l_eid == nodes[x, 2]) & (l_type == 1) & (r_type == 2) & np.logical_or(r_h_lid == nodes[y, 3], r_t_lid == nodes[y, 4]))
                temp_ = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid),
                                 1, adj_temp)
                temp_ = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid),
                                 1, temp_)
                adjacency[4][x, y] = 1 if (temp_[z] == 1).any() else 0
                adjacency[4][y, x] = 1 if (temp_[z] == 1).any() else 0

        adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(adjacency[i]) for i in range(5)])
        '''

        xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')
        l_type, r_type = nodes[xv, 4], nodes[yv, 4]
        l_eid, r_eid = nodes[xv, 2], nodes[yv, 2]
        l_sid, r_sid = nodes[xv, 3], nodes[yv, 3]

        adj_temp = np.full((l_type.shape[0], r_type.shape[0]), 0, 'i')  # adj_temp是一个临时的邻接矩阵，用来存储边的情况，
        # adjacency = np.full((5, l_type.shape[0], r_type.shape[0]), 0.0) #adjacency是一个5维矩阵，其中adjacency[i]表示关系类型i（五类边）的邻接矩阵
        adjacency = np.full((4, l_type.shape[0], r_type.shape[0]), 0.0)
        #adjacency = np.full((3, l_type.shape[0], r_type.shape[0]), 0.0)

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
        adjacency[3] = np.where((l_type == 2) & (r_type == 2) & (l_sid + 1 == r_sid), 1, adjacency[3])


        # adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(adjacency[i]) for i in range(5)])
        adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(adjacency[i]) for i in range(4)])

        relations, hts = [], []
        for h, t in train_triple.keys():
            #relation = [0] * len(docred_rel2id)
            relation = [0] * len(cdr_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                #evidence = mention["evidence"]
            relations.append(relation)
            hts.append([h, t])
            #pos_samples += 1
        '''
        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    #relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relation = [1] + [0] * (len(cdr_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)
        '''
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
                   'adjacency': adjacency,
                   'link_pos': sentl_pos,
                   'nodes_info': nodes,
                   }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    #print("# of positive examples {}.".format(pos_samples))
    #print("# of negative examples {}.".format(neg_samples))
    print(f'maxlen:{maxlen}')
    print(f'max_entity_num:{max_entity}')
    return features