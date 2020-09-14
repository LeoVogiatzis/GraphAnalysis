import os
import json
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import numpy as np
import ujson
from sklearn.metrics import jaccard_score

results = dict()
info = dict()
for _file in os.listdir(os.getcwd()):
    if not _file.endswith('.txt') or \
        not _file.startswith('comm'): continue

    with open(_file, 'r') as f:
        content = f.readlines()
    content = [x.rstrip() for x in content]

    info[_file] = content

with open('sc_res_test200.txt', 'r') as f:
    sc = json.loads(f.read())

cnt = 0
temp= dict()
with open('similarty_sc_communities.txt', 'a') as f:
    for comm in sc:
        for item in info:
            cou = 0
            tl= list()
            for i in info[item]:
                
                temp_l = i.split(" ")
                
                cou+=1
                day = item[12:].replace('.txt', '')
                sim = difflib.SequenceMatcher(None, sc[comm], temp_l)
                
                tl.append(sim.ratio())
                temp[comm] = max(tl)
                
                results[day+'_sc'+comm] = temp[comm]
                
    print(results)
    f.write(ujson.dumps(results))