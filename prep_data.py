import json, codecs
import numpy as np

def load_data(root, path):
    '''Load data from path'''
    data = codecs.open(root + path, encoding = 'utf8')
    relations = [json.loads(x) for x in data]
    data.close()
    return relations

def convert2id(root):
    '''Convert training data into indices'''
    wdict = {'<pad>':0, '<unk>':1}
    cdict = {}
    with codecs.open(root + 'train/relations.json', encoding = 'utf8') as pdtb:
        relations = [json.loads(x) for x in pdtb]
        wid = 2
        cid = 0
        for rel in relations:
            sen_cat = rel['Arg1']['RawText'].split() + rel['Arg2']['RawText'].split()
            sense = rel['Sense'][0]
            if sense not in cdict:
                cdict[sense] = cid
                cdict[cid] = sense
                cid += 1
            for word in sen_cat:
                if word not in wdict:
                    wdict[word] = wid
                    wdict[wid] = word
                    wid += 1
    return wdict, cdict

def generateBatches(relations, sen_len, num_class, wdict, cdict, batch_size):
    '''Generate train/dev/test batches for the model'''
    per_len = int(sen_len / 2)
    emb_list = []
    class_list = []
    for i in range(0, len(relations), batch_size):
        batch = relations[i:i+batch_size]
        emb = np.ndarray((len(batch), sen_len))
        cla = np.zeros((len(batch), num_class))
        for j in range(len(batch)):
            rel = batch[j]
            sense = rel['Sense'][0]
            sen1 = rel['Arg1']['RawText'].split()
            sen2 = rel['Arg2']['RawText'].split()
            sen = word2id(sen1, per_len, wdict) + word2id(sen2, per_len, wdict)
            if len(sen) < sen_len:
                sen = sen + [wdict['<pad>']] * (sen_len - len(sen))
            emb[j] = sen
            cla[j][cdict[sense]] = 1
        emb_list.append(emb)
        class_list.append(cla)
    return emb_list, class_list

def word2id(sen, per_len, wdict):
    '''Convert a sequence of word into a sequence of indices'''
    if len(sen) > per_len:
        sen = sen[:per_len]
    w2id_sen = []
    for w in sen:
        if w not in wdict:
            w2id_sen.append(wdict['<unk>'])
        else:
            w2id_sen.append(wdict[w])
    return w2id_sen