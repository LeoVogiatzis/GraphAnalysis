import json
import ujson
import ast
import os
import pandas as pd
from datetime import datetime

def intersection(l1, l2):
    tup1 = map(tuple, l1)
    tup2 = map(tuple, l2)
    return list(map(list, set(tup1).intersection(tup2)))

def cosine_similarity(a, b):
    res = 0
    for index, value in enumerate(a):
        res += len(set(value) & set(b[index])) / float(len(set(value) | set(b[index])))
    return res/len(a)

def calc_overlaps(df1, df2):
    time_now = datetime.now()
    current_overlap = 0
    df1_tmp = []
    df2_tmp = []
    for index, item in enumerate(df2):
        df1_tmp.append(df1[index])
        df2_tmp.append(item)
        current_overlap = current_overlap + len(intersection(df1_tmp, df2_tmp)) / len(df1_tmp)
        if index == (len(df1)-1): overlap = len(intersection(df1_tmp, df2_tmp)) / len(df1_tmp)
    rbo = current_overlap / len(df1)
    return overlap, rbo, datetime.now()-time_now

for _file in os.listdir(os.getcwd()):

    if not _file.startswith('nx_lp'): continue

    with open(_file, 'r') as f:
        print('Reading - ' + _file)
        s = f.read()
        data = json.loads(s)
        df = pd.DataFrame(data, columns= ['source_node', 'target_node', 'score'])

        if 'AA' in _file:
            print('Converting - ' + _file)
            aa = df.sort_values('score', ascending=False).head(100).sort_values(['score'], 
                        ascending=False).reset_index(drop=True).drop(['score'], axis=1).values.tolist()
        elif 'JC' in _file:
            print('Converting - ' + _file)
            jc = df.sort_values('score', ascending=False).head(100).sort_values(['score'], 
                        ascending=False).reset_index(drop=True).drop(['score'], axis=1).values.tolist()
        elif 'PA' in _file:
            print('Converting - ' + _file)
            pa = df.sort_values('score', ascending=False).head(100).sort_values(['score'], 
                        ascending=False).reset_index(drop=True).drop(['score'], axis=1).values.tolist()
        elif _file.endswith('RA_community.txt'):
            print('Converting - ' + _file)
            ra_comm = df.sort_values('score', ascending=False).head(100).sort_values(['score'], 
                        ascending=False).reset_index(drop=True).drop(['score'], axis=1).values.tolist()
        elif 'RA' in _file:
            print('Converting - ' + _file)
            ra = df.sort_values('score', ascending=False).head(100).sort_values(['score'], 
                        ascending=False).reset_index(drop=True).drop(['score'], axis=1).values.tolist()
        elif 'CN_community' in _file:
            print('Converting - ' + _file)
            cn_comm = df.sort_values('score', ascending=False).head(100).sort_values(['score'], 
                        ascending=False).reset_index(drop=True).drop(['score'], axis=1).values.tolist()
        else: continue

print('Reading finished.....')

print('Cosine Similarity Computation.....')
cos_aa_jc = cosine_similarity(aa, jc)
cos_aa_pa = cosine_similarity(aa, pa)
cos_aa_ra = cosine_similarity(aa, ra)
cos_aa_ra_comm = cosine_similarity(aa, ra_comm)
cos_aa_cn_comm = cosine_similarity(aa, cn_comm)
cos_jc_pa = cosine_similarity(jc, pa)
cos_jc_ra = cosine_similarity(jc, ra)
cos_jc_ra_comm = cosine_similarity(jc, ra_comm)
cos_jc_cn_comm = cosine_similarity(jc, cn_comm)
cos_pa_ra = cosine_similarity(pa, ra)
cos_pa_ra_comm = cosine_similarity(pa, ra_comm)
cos_pa_cn_comm = cosine_similarity(pa, cn_comm)
cos_ra_ra_comm = cosine_similarity(ra, ra_comm)
cos_ra_cn_comm = cosine_similarity(ra, cn_comm)
cos_raComm_cnComm = cosine_similarity(ra_comm, cn_comm) 
print('Cosine Similarity Computation Finished.....')

print('Calculate Overlap and Rank Biased Overlap.....')
aa_jc_ol, aa_jc_rbo, aa_jc_time_taken = calc_overlaps(aa, jc)
aa_pa_ol, aa_pa_rbo, aa_pa_time_taken = calc_overlaps(aa, pa)
aa_ra_ol, aa_ra_rbo, aa_ra_time_taken = calc_overlaps(aa, ra)
aa_raC_ol, aa_raC_rbo, aa_raC_time_taken = calc_overlaps(aa, ra_comm)
aa_cnC_ol, aa_cnC_rbo, aa_cnC_time_taken = calc_overlaps(aa, cn_comm)
jc_pa_ol, jc_pa_rbo, jc_pa_time_taken = calc_overlaps(jc, pa) 
jc_ra_ol, jc_ra_rbo, jc_ra_time_taken = calc_overlaps(jc, ra)
jc_raC_ol, jc_raC_rbo, jc_raC_time_taken = calc_overlaps(jc, ra_comm)
jc_cnC_ol, jc_cnC_rbo, jc_cnC_time_taken = calc_overlaps(jc, cn_comm)
pa_ra_ol, pa_ra_rbo, pa_ra_time_taken = calc_overlaps(pa, ra)
pa_raC_ol, pa_raC_rbo, pa_raC_time_taken = calc_overlaps(pa, ra_comm)
pa_cnC_ol, pa_cnC_rbo, pa_cnC_time_taken = calc_overlaps(pa, cn_comm)
ra_raC_ol, ra_raC_rbo, ra_raC_time_taken = calc_overlaps(ra, ra_comm)
ra_cnC_ol, ra_cnC_rbo, ra_cnC_time_taken = calc_overlaps(ra, cn_comm)
raC_cnC_ol, raC_cnC_rbo, raC_cnC_time_taken = calc_overlaps(ra_comm, cn_comm)
print('Overlap and Rank Biased Overlap Calculation finished.....')

print('File Writing.....')
with open('comparison_nx_algos.txt', 'a') as f:
    f.write('Comparison Results---------------------\n')

    f.write('Cosine Similarity Results-----------------\n')
    f.write('AdamicAdar-JaccardCoefficient CS: ' + str(cos_aa_jc) + '\n')
    f.write('AdamicAdar-PreferentialAttachment CS: ' + str(cos_aa_pa) + '\n')
    f.write('AdamicAdar-ResourceAllocation CS: ' + str(cos_aa_ra) + '\n')
    f.write('AdamicAdar-ResourceAllocationCommunity CS: ' + str(cos_aa_ra_comm) + '\n')
    f.write('AdamicAdar-CommonNeighborsCommunity CS: ' + str(cos_aa_cn_comm) + '\n')
    f.write('JaccardCoefficient-PreferentialAttachment CS: ' + str(cos_jc_pa) + '\n')
    f.write('JaccardCoefficient-ResourceAllocation CS: ' + str(cos_jc_ra) + '\n')
    f.write('JaccardCoefficient-ResourceAllocationCommunity CS: ' + str(cos_jc_ra_comm) + '\n')
    f.write('JaccardCoefficient-CommonNeighborsCommunity CS: ' + str(cos_jc_cn_comm) + '\n')
    f.write('PreferentialAttachment-ResourceAllocation CS: ' + str(cos_pa_ra) + '\n')
    f.write('PreferentialAttachment-ResourceAllocationCommunity CS: ' + str(cos_pa_ra_comm) + '\n')
    f.write('PreferentialAttachment-CommonNeighborsCommunity CS: ' + str(cos_pa_cn_comm) + '\n')
    f.write('ResourceAllocation-ResourceAllocationCommunity CS: ' + str(cos_ra_ra_comm) + '\n')
    f.write('ResourceAllocation-CommonNeighborsCOmmunity CS: ' + str(cos_ra_cn_comm) + '\n')
    f.write('ResourceAllocationCommunity-CommonNeighborsCommunity CS: ' + str(cos_raComm_cnComm) + '\n')

    f.write('Overlap Results-----------------\n')
    f.write('AdamicAdar-JaccardCoefficient OL: ' + str(aa_jc_ol) + '\n')
    f.write('AdamicAdar-PreferentialAttachment OL: ' + str(aa_pa_ol) + '\n')
    f.write('AdamicAdar-ResourceAllocation OL: ' + str(aa_ra_ol) + '\n')
    f.write('AdamicAdar-ResourceAllocationCommunity OL: ' + str(aa_raC_ol) + '\n')
    f.write('AdamicAdar-CommonNeighborsCommunity OL: ' + str(aa_cnC_ol) + '\n')
    f.write('JaccardCoefficient-PreferentialAttachment OL: ' + str(jc_pa_ol) + '\n')
    f.write('JaccardCoefficient-ResourceAllocation OL: ' + str(jc_ra_ol) + '\n')
    f.write('JaccardCoefficient-ResourceAllocationCommunity OL: ' + str(jc_raC_ol) + '\n')
    f.write('JaccardCoefficient-CommonNeighborsCommunity OL: ' + str(jc_cnC_ol) + '\n')
    f.write('PreferentialAttachment-ResourceAllocation OL: ' + str(pa_ra_ol) + '\n')
    f.write('PreferentialAttachment-ResourceAllocationCommunity OL: ' + str(pa_raC_ol) + '\n')
    f.write('PreferentialAttachment-CommonNeighborsCommunity OL: ' + str(pa_cnC_ol) + '\n')
    f.write('ResourceAllocation-ResourceAllocationCommunity OL: ' + str(ra_raC_ol) + '\n')
    f.write('ResourceAllocation-CommonNeighborsCOmmunity OL: ' + str(ra_cnC_ol) + '\n')
    f.write('ResourceAllocationCommunity-CommonNeighborsCommunity OL: ' + str(raC_cnC_ol) + '\n')

    f.write('Rank Biased Overlap Results--------------------\n')
    f.write('AdamicAdar-JaccardCoefficient RBO: ' + str(aa_jc_rbo) + '\n')
    f.write('AdamicAdar-PreferentialAttachment RBO: ' + str(aa_pa_rbo) + '\n')
    f.write('AdamicAdar-ResourceAllocation RBO: ' + str(aa_ra_rbo) + '\n')
    f.write('AdamicAdar-ResourceAllocationCommunity RBO: ' + str(aa_raC_rbo) + '\n')
    f.write('AdamicAdar-CommonNeighborsCommunity RBO: ' + str(aa_cnC_rbo) + '\n')
    f.write('JaccardCoefficient-PreferentialAttachment RBO: ' + str(jc_pa_rbo) + '\n')
    f.write('JaccardCoefficient-ResourceAllocation RBO: ' + str(jc_ra_rbo) + '\n')
    f.write('JaccardCoefficient-ResourceAllocationCommunity RBO: ' + str(jc_raC_rbo) + '\n')
    f.write('JaccardCoefficient-CommonNeighborsCommunity RBO: ' + str(jc_cnC_rbo) + '\n')
    f.write('PreferentialAttachment-ResourceAllocation RBO: ' + str(pa_ra_rbo) + '\n')
    f.write('PreferentialAttachment-ResourceAllocationCommunity RBO: ' + str(pa_raC_rbo) + '\n')
    f.write('PreferentialAttachment-CommonNeighborsCommunity RBO: ' + str(pa_cnC_rbo) + '\n')
    f.write('ResourceAllocation-ResourceAllocationCommunity RBO: ' + str(ra_raC_rbo) + '\n')
    f.write('ResourceAllocation-CommonNeighborsCOmmunity RBO: ' + str(ra_cnC_rbo) + '\n')
    f.write('ResourceAllocationCommunity-CommonNeighborsCommunity RBO: ' + str(raC_cnC_rbo) + '\n')

    f.write('Time taken to compute overlaps--------------------\n')
    f.write('AdamicAdar-JaccardCoefficient Time: ' + str(aa_jc_time_taken) + '\n')
    f.write('AdamicAdar-PreferentialAttachment Time: ' + str(aa_pa_time_taken) + '\n')
    f.write('AdamicAdar-ResourceAllocation Time: ' + str(aa_ra_time_taken) + '\n')
    f.write('AdamicAdar-ResourceAllocationCommunity Time: ' + str(aa_raC_time_taken) + '\n')
    f.write('AdamicAdar-CommonNeighborsCommunity Time: ' + str(aa_cnC_time_taken) + '\n')
    f.write('JaccardCoefficient-PreferentialAttachment TimeL: ' + str(jc_pa_time_taken) + '\n')
    f.write('JaccardCoefficient-ResourceAllocation Time: ' + str(jc_ra_time_taken) + '\n')
    f.write('JaccardCoefficient-ResourceAllocationCommunity TimeL: ' + str(jc_raC_time_taken) + '\n')
    f.write('JaccardCoefficient-CommonNeighborsCommunity Time: ' + str(jc_cnC_time_taken) + '\n')
    f.write('PreferentialAttachment-ResourceAllocation Time: ' + str(pa_ra_time_taken) + '\n')
    f.write('PreferentialAttachment-ResourceAllocationCommunity Time: ' + str(pa_raC_time_taken) + '\n')
    f.write('PreferentialAttachment-CommonNeighborsCommunity Time: ' + str(pa_cnC_time_taken) + '\n')
    f.write('ResourceAllocation-ResourceAllocationCommunity Time: ' + str(ra_raC_time_taken) + '\n')
    f.write('ResourceAllocation-CommonNeighborsCOmmunity Time: ' + str(ra_cnC_time_taken) + '\n')
    f.write('ResourceAllocationCommunity-CommonNeighborsCommunity Time: ' + str(raC_cnC_time_taken) + '\n')

print('Process finished.....')