from collections import defaultdict
import ujson
import ast
import pandas as pd
import json
import os
import glob
import numpy as np
from datetime import datetime
import re

d = datetime.now()
cou = 0
_all = dict()

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

def split_df(df):
    attacks =[re.findall('[0-9]+',i[0]) for i in df.iloc[:, [0]]['attacks_score'].nlargest(n=100).index.tolist()]
    messages = [re.findall('[0-9]+',i[0]) for i in df.iloc[:, [1]]['messages_score'].nlargest(n=100).index.tolist()]
    trades = [re.findall('[0-9]+',i[0]) for i in df.iloc[:, [2]]['trades_score'].nlargest(n=100).index.tolist()]
    return attacks, messages, trades

for _file in os.listdir(os.getcwd()):
    if 'best_in' in _file and \
         'comparison' not in _file:
        
        with open(_file, 'rb') as f:
            
            print('Reading - ' + _file)
            s = f.read()
            user_dict = json.loads(s)

            if 'Common' in _file:
                print('Converting - ' + _file)
                c = pd.DataFrame.from_dict({(i,j): user_dict[i][j]
                           for i in user_dict.keys()
                           for j in user_dict[i].keys()},
                            orient='index')
                common_attacks, common_messages, common_trades = split_df(c)
                del c
            elif 'AA' in _file:
                print('Converting - ' + _file)
                aa = pd.DataFrame.from_dict({(i,j): user_dict[i][j]
                           for i in user_dict.keys()
                           for j in user_dict[i].keys()},
                            orient='index')
                aa_attacks, aa_messages, aa_trades = split_df(aa)
                del aa
            elif 'total' in _file:
                print('Converting - ' + _file)
                total = pd.DataFrame.from_dict({(i,j): user_dict[i][j]
                           for i in user_dict.keys()
                           for j in user_dict[i].keys()},
                            orient='index')
                total_attacks, total_messages, total_trades = split_df(total)
                del total
            elif 'Preferential' in _file:
                print('Converting - ' + _file)
                pref = pd.DataFrame.from_dict({(i,j): user_dict[i][j]
                           for i in user_dict.keys()
                           for j in user_dict[i].keys()},
                            orient='index')           
                pref_attacks, pref_messages, pref_trades = split_df(pref)                
                del pref
            elif 'Resource' in _file:
                print('Converting - ' + _file)
                resource = pd.DataFrame.from_dict({(i,j): user_dict[i][j]
                           for i in user_dict.keys()
                           for j in user_dict[i].keys()},
                            orient='index')
                resource_attacks, resource_messages, resource_trades = split_df(resource)
                del resource
            elif 'same' in _file:
                print('Converting - ' + _file)
                same = pd.DataFrame.from_dict({(i,j): user_dict[i][j]
                           for i in user_dict.keys()
                           for j in user_dict[i].keys()},
                            orient='index')
                same_attacks, same_messages, same_trades = split_df(same)
                del same
print('Reading done.....')

print('Calculate Cosine Similarities......')
cos_attacks_aa_common = cosine_similarity(aa_attacks, common_attacks)
cos_attacks_aa_total = cosine_similarity(aa_attacks, total_attacks)
cos_attacks_aa_pref = cosine_similarity(aa_attacks, pref_attacks)
cos_attacks_aa_resource = cosine_similarity(aa_attacks, resource_attacks)
cos_attacks_aa_same = cosine_similarity(aa_attacks, same_attacks)
cos_attacks_common_total = cosine_similarity(common_attacks, total_attacks)
cos_attacks_common_pref = cosine_similarity(common_attacks, pref_attacks)
cos_attacks_common_resource = cosine_similarity(common_attacks, resource_attacks)
cos_attacks_common_same = cosine_similarity(common_attacks, same_attacks)
cos_attacks_total_pref = cosine_similarity(total_attacks, pref_attacks)
cos_attacks_total_resource = cosine_similarity(total_attacks, resource_attacks)
cos_attacks_total_same = cosine_similarity(total_attacks, same_attacks)
cos_attacks_pref_resource = cosine_similarity(pref_attacks, resource_attacks)
cos_attacks_pref_same = cosine_similarity(pref_attacks, same_attacks)
cos_attacks_resource_same = cosine_similarity(resource_attacks, same_attacks)

cos_messages_aa_common = cosine_similarity(aa_messages, common_messages)
cos_messages_aa_total = cosine_similarity(aa_messages, total_messages)
cos_messages_aa_pref = cosine_similarity(aa_messages, pref_messages)
cos_messages_aa_resource = cosine_similarity(aa_messages, resource_messages)
cos_messages_aa_same = cosine_similarity(aa_messages, same_messages)
cos_messages_common_total = cosine_similarity(common_messages, total_messages)
cos_messages_common_pref = cosine_similarity(common_messages, pref_messages)
cos_messages_common_resource = cosine_similarity(common_messages, resource_messages)
cos_messages_common_same = cosine_similarity(common_messages, same_messages)
cos_messages_total_pref = cosine_similarity(total_messages, pref_messages)
cos_messages_total_resource = cosine_similarity(total_messages, resource_messages)
cos_messages_total_same = cosine_similarity(total_messages, same_messages)
cos_messages_pref_resource = cosine_similarity(pref_messages, resource_messages)
cos_messages_pref_same = cosine_similarity(pref_messages, same_messages)
cos_messages_resource_same = cosine_similarity(resource_messages, same_messages)

cos_trades_aa_common = cosine_similarity(aa_trades, common_trades)
cos_trades_aa_total = cosine_similarity(aa_trades, total_trades)
cos_trades_aa_pref = cosine_similarity(aa_trades, pref_trades)
cos_trades_aa_resource = cosine_similarity(aa_trades, resource_trades)
cos_trades_aa_same = cosine_similarity(aa_trades, same_trades)
cos_trades_common_total = cosine_similarity(common_trades, total_trades)
cos_trades_common_pref = cosine_similarity(common_trades, pref_trades)
cos_trades_common_resource = cosine_similarity(common_trades, resource_trades)
cos_trades_common_same = cosine_similarity(common_trades, same_trades)
cos_trades_total_pref = cosine_similarity(total_trades, pref_trades)
cos_trades_total_resource = cosine_similarity(total_trades, resource_trades)
cos_trades_total_same = cosine_similarity(total_trades, same_trades)
cos_trades_pref_resource = cosine_similarity(pref_trades, resource_trades)
cos_trades_pref_same = cosine_similarity(pref_trades, same_trades)
cos_trades_resource_same = cosine_similarity(resource_trades, same_trades)

print('Calculate Overlap and Rank Biased Overlap......')
aa_c_overlap_attacks, aa_c_rbo_attacks, aa_c_attacks_time_taken = calc_overlaps(aa_attacks, common_attacks)
aa_t_overlap_attacks, aa_t_rbo_attacks, aa_t_attacks_time_taken = calc_overlaps(aa_attacks, total_attacks)
aa_p_overlap_attacks, aa_p_rbo_attacks, aa_p_attacks_time_taken = calc_overlaps(aa_attacks, pref_attacks)
aa_r_overlap_attacks, aa_r_rbo_attacks, aa_r_attacks_time_taken = calc_overlaps(aa_attacks, resource_attacks)
aa_s_overlap_attacks, aa_s_rbo_attacks, aa_s_attacks_time_taken = calc_overlaps(aa_attacks, same_attacks)
common_t_overlap_attacks, common_t_rbo_attacks, common_t_attacks_time_taken = calc_overlaps(common_attacks, total_attacks)
common_p_overlap_attacks, common_p_rbo_attacks, common_p_attacks_time_taken = calc_overlaps(common_attacks, pref_attacks)
common_r_overlap_attacks, common_r_rbo_attacks, common_r_attacks_time_taken = calc_overlaps(common_attacks, resource_attacks)
common_s_overlap_attacks, common_s_rbo_attacks, common_s_attacks_time_taken = calc_overlaps(common_attacks, same_attacks)
total_p_overlap_attacks, total_p_rbo_attacks, total_p_attacks_time_taken = calc_overlaps(total_attacks, pref_attacks)
total_r_overlap_attacks, total_r_rbo_attacks, total_r_attacks_time_taken = calc_overlaps(total_attacks, resource_attacks)
total_s_overlap_attacks, total_s_rbo_attacks, total_s_attacks_time_taken = calc_overlaps(total_attacks, same_attacks)
pref_r_overlap_attacks, pref_r_rbo_attacks, pref_r_attacks_time_taken = calc_overlaps(pref_attacks, resource_attacks)
pref_s_overlap_attacks, pref_s_rbo_attacks, pref_s_attacks_time_taken = calc_overlaps(pref_attacks, same_attacks)
resource_s_overlap_attacks, resource_s_rbo_attacks, resource_s_attacks_time_taken = calc_overlaps(resource_attacks, same_attacks)

aa_c_overlap_messages, aa_c_rbo_messages, aa_c_messages_time_taken = calc_overlaps(aa_messages, common_messages)
aa_t_overlap_messages, aa_t_rbo_messages, aa_t_messages_time_taken = calc_overlaps(aa_messages, total_messages)
aa_p_overlap_messages, aa_p_rbo_messages, aa_p_messages_time_taken = calc_overlaps(aa_messages, pref_messages)
aa_r_overlap_messages, aa_r_rbo_messages, aa_r_messages_time_taken = calc_overlaps(aa_messages, resource_messages)
aa_s_overlap_messages, aa_s_rbo_messages, aa_s_messages_time_taken = calc_overlaps(aa_messages, same_messages)
common_t_overlap_messages, common_t_rbo_messages, common_t_messages_time_taken = calc_overlaps(common_messages, total_messages)
common_p_overlap_messages, common_p_rbo_messages, common_p_messages_time_taken = calc_overlaps(common_messages, pref_messages)
common_r_overlap_messages, common_r_rbo_messages, common_r_messages_time_taken = calc_overlaps(common_messages, resource_messages)
common_s_overlap_messages, common_s_rbo_messages, common_s_messages_time_taken = calc_overlaps(common_messages, same_messages)
total_p_overlap_messages, total_p_rbo_messages, total_p_messages_time_taken = calc_overlaps(total_messages, pref_messages)
total_r_overlap_messages, total_r_rbo_messages, total_r_messages_time_taken = calc_overlaps(total_messages, resource_messages)
total_s_overlap_messages, total_s_rbo_messages, total_s_messages_time_taken = calc_overlaps(total_messages, same_messages)
pref_r_overlap_messages, pref_r_rbo_messages, pref_r_messages_time_taken = calc_overlaps(pref_messages, resource_messages)
pref_s_overlap_messages, pref_s_rbo_messages, pref_s_messages_time_taken = calc_overlaps(pref_messages, same_messages)
resource_s_overlap_messages, resource_s_rbo_messages, resource_s_messages_time_taken = calc_overlaps(resource_messages, same_messages)

aa_c_overlap_trades, aa_c_rbo_trades, aa_c_trades_time_taken = calc_overlaps(aa_trades, common_trades)
aa_t_overlap_trades, aa_t_rbo_trades, aa_t_trades_time_taken = calc_overlaps(aa_trades, total_trades)
aa_p_overlap_trades, aa_p_rbo_trades, aa_p_trades_time_taken = calc_overlaps(aa_trades, pref_trades)
aa_r_overlap_trades, aa_r_rbo_trades, aa_r_trades_time_taken = calc_overlaps(aa_trades, resource_trades)
aa_s_overlap_trades, aa_s_rbo_trades, aa_s_trades_time_taken = calc_overlaps(aa_trades, same_trades)
common_t_overlap_trades, common_t_rbo_trades, common_t_trades_time_taken = calc_overlaps(common_trades, total_trades)
common_p_overlap_trades, common_p_rbo_trades, common_p_trades_time_taken = calc_overlaps(common_trades, pref_trades)
common_r_overlap_trades, common_r_rbo_trades, common_r_trades_time_taken = calc_overlaps(common_trades, resource_trades)
common_s_overlap_trades, common_s_rbo_trades, common_s_trades_time_taken = calc_overlaps(common_trades, same_trades)
total_p_overlap_trades, total_p_rbo_trades, total_p_trades_time_taken = calc_overlaps(total_trades, pref_trades)
total_r_overlap_trades, total_r_rbo_trades, total_r_trades_time_taken = calc_overlaps(total_trades, resource_trades)
total_s_overlap_trades, total_s_rbo_trades, total_s_trades_time_taken = calc_overlaps(total_trades, same_trades)
pref_r_overlap_trades, pref_r_rbo_trades, pref_r_trades_time_taken = calc_overlaps(pref_trades, resource_trades)
pref_s_overlap_trades, pref_s_rbo_trades, pref_s_trades_time_taken = calc_overlaps(pref_trades, same_trades)
resource_s_overlap_trades, resource_s_rbo_trades, resource_s_trades_time_taken = calc_overlaps(resource_trades, same_trades)

with open('index_comparison_best_in_results.txt', 'a') as f:
    f.write('BEST-IN RESULTS----------------\n')
    f.write('Results: \n')
    
    f.write('ATTACKS SCORE COMPARISON-----------\n')
    
    f.write('Cosine Similarities------------\n')
    f.write('Adamic-CommonNeighbors CS: ' + str(cos_attacks_aa_common) + '\n')
    f.write('Adamic-TotalNeighbors CS: ' + str(cos_attacks_aa_total) + '\n')
    f.write('Adamic-Preferential CS: ' + str(cos_attacks_aa_pref) + '\n')
    f.write('Adamic-Resource CS: ' + str(cos_attacks_aa_resource) + '\n')
    f.write('Adamic-Same CS: ' + str(cos_attacks_aa_same) + '\n')
    f.write('CommonNeighbors-TotalNeighbors CS: ' + str(cos_attacks_common_total) + '\n')
    f.write('CommonNeighbors-Preferential CS: ' + str(cos_attacks_common_pref) + '\n')
    f.write('CommonNeighbors-Resource CS: ' + str(cos_attacks_common_resource) + '\n')
    f.write('CommonNeighbors-Same CS: ' + str(cos_attacks_common_same) + '\n')
    f.write('TotalNeighbors-Preferential CS: ' + str(cos_attacks_total_pref) + '\n')
    f.write('TotalNeighbors-Resource CS: ' + str(cos_attacks_total_resource) + '\n')
    f.write('TotalNeighbors-Same CS: ' + str(cos_attacks_total_same) + '\n')
    f.write('Preferential-Resource CS: ' + str(cos_attacks_pref_resource) + '\n')
    f.write('Preferential-Same CS: ' + str(cos_attacks_pref_same) + '\n')
    f.write('Resource-Same CS: ' + str(cos_attacks_resource_same) + '\n')
    
    f.write('Overlap------------\n')
    f.write('Adamic-CommonNeighbors OL: ' + str(aa_c_overlap_attacks) + '\n')
    f.write('Adamic-TotalNeighbors OL: ' + str(aa_t_overlap_attacks) + '\n')
    f.write('Adamic-Preferential OL: ' + str(aa_p_overlap_attacks) + '\n')
    f.write('Adamic-Resource OL: ' + str(aa_r_overlap_attacks) + '\n')
    f.write('Adamic-Same OL: ' + str(aa_s_overlap_attacks) + '\n')
    f.write('CommonNeighbors-TotalNeighbors OL: ' + str(common_t_overlap_attacks) + '\n')
    f.write('CommonNeighbors-Preferential OL: ' + str(common_p_overlap_attacks) + '\n')
    f.write('CommonNeighbors-Resource OL: ' + str(common_r_overlap_attacks) + '\n')
    f.write('CommonNeighbors-Same OL: ' + str(common_s_overlap_attacks) + '\n')
    f.write('TotalNeighbors-Preferential OL: ' + str(total_p_overlap_attacks) + '\n')
    f.write('TotalNeighbors-Resource OL: ' + str(total_r_overlap_attacks) + '\n')
    f.write('TotalNeighbors-Same OL: ' + str(total_s_overlap_attacks) + '\n')
    f.write('Preferential-Resource OL: ' + str(pref_r_overlap_attacks) + '\n')
    f.write('Preferential-Same OL: ' + str(pref_s_overlap_attacks) + '\n')
    f.write('Resource-Same OL: ' + str(resource_s_overlap_attacks) + '\n')

    f.write('Overlap Ranked Biased--------------\n')
    f.write('Adamic-CommonNeighbors RBO: ' + str(aa_c_rbo_attacks) + '\n')
    f.write('Adamic-TotalNeighbors RBO: ' + str(aa_t_rbo_attacks) + '\n')
    f.write('Adamic-Preferential RBO: ' + str(aa_p_rbo_attacks) + '\n')
    f.write('Adamic-Resource RBO: ' + str(aa_r_rbo_attacks) + '\n')
    f.write('Adamic-Same RBO: ' + str(aa_s_rbo_attacks) + '\n')
    f.write('CommonNeighbors-TotalNeighbors RBO: ' + str(common_t_rbo_attacks) + '\n')
    f.write('CommonNeighbors-Preferential RBO: ' + str(common_p_rbo_attacks) + '\n')
    f.write('CommonNeighbors-Resource RBO: ' + str(common_r_rbo_attacks) + '\n')
    f.write('CommonNeighbors-Same RBO: ' + str(common_s_rbo_attacks) + '\n')
    f.write('TotalNeighbors-Preferential RBO: ' + str(total_p_rbo_attacks) + '\n')
    f.write('TotalNeighbors-Resource RBO: ' + str(total_r_rbo_attacks) + '\n')
    f.write('TotalNeighbors-Same RBO: ' + str(total_s_rbo_attacks) + '\n')
    f.write('Preferential-Resource RBO: ' + str(pref_r_rbo_attacks) + '\n')
    f.write('Preferential-Same RBO: ' + str(pref_s_rbo_attacks) + '\n')
    f.write('Resource-Same RBO: ' + str(resource_s_rbo_attacks) + '\n')

    f.write('Time taken to compute overlaps-------------\n')
    f.write('Adamin-Common Time: ' + str(aa_c_attacks_time_taken) + '\n')
    f.write('Adamin-Total Time: ' + str(aa_t_attacks_time_taken) + '\n')
    f.write('Adamin-Preferential Time: ' + str(aa_p_attacks_time_taken) + '\n')
    f.write('Adamin-Resource Time: ' + str(aa_r_attacks_time_taken) + '\n')
    f.write('Adamin-Same Time: ' + str(aa_s_attacks_time_taken) + '\n')
    f.write('Common-Total Time: ' + str(common_t_attacks_time_taken) + '\n')
    f.write('Common-Preferential Time: ' + str(common_p_attacks_time_taken) + '\n')
    f.write('Common-Resource Time: ' + str(common_r_attacks_time_taken) + '\n')
    f.write('Common-Same Time: ' + str(common_s_attacks_time_taken) + '\n')
    f.write('Total-Preferential Time: ' + str(total_p_attacks_time_taken) + '\n')
    f.write('Total-Resource Time: ' + str(total_r_attacks_time_taken) + '\n')
    f.write('Total-Same Time: ' + str(total_s_attacks_time_taken) + '\n')
    f.write('Preferential-Resource Time: ' + str(pref_r_attacks_time_taken) + '\n')
    f.write('Preferential-Same Time: ' + str(pref_s_attacks_time_taken) + '\n')
    f.write('Resource-Same Time: ' + str(resource_s_attacks_time_taken) + '\n')

    f.write('MESSAGES SCORE COMPARISON-----------\n')
    
    f.write('Cosine Similarities------------\n')
    f.write('Adamic-CommonNeighbors CS: ' + str(cos_messages_aa_common) + '\n')
    f.write('Adamic-TotalNeighbors CS: ' + str(cos_messages_aa_total) + '\n')
    f.write('Adamic-Preferential CS: ' + str(cos_messages_aa_pref) + '\n')
    f.write('Adamic-Resource CS: ' + str(cos_messages_aa_resource) + '\n')
    f.write('Adamic-Same CS: ' + str(cos_messages_aa_same) + '\n')
    f.write('CommonNeighbors-TotalNeighbors CS: ' + str(cos_messages_common_total) + '\n')
    f.write('CommonNeighbors-Preferential CS: ' + str(cos_messages_common_pref) + '\n')
    f.write('CommonNeighbors-Resource CS: ' + str(cos_messages_common_resource) + '\n')
    f.write('CommonNeighbors-Same CS: ' + str(cos_messages_common_same) + '\n')
    f.write('TotalNeighbors-Preferential CS: ' + str(cos_messages_total_pref) + '\n')
    f.write('TotalNeighbors-Resource CS: ' + str(cos_messages_total_resource) + '\n')
    f.write('TotalNeighbors-Same CS: ' + str(cos_messages_total_same) + '\n')
    f.write('Preferential-Resource CS: ' + str(cos_messages_pref_resource) + '\n')
    f.write('Preferential-Same CS: ' + str(cos_messages_pref_same) + '\n')
    f.write('Resource-Same CS: ' + str(cos_messages_resource_same) + '\n')
    
    f.write('Overlap------------\n')
    f.write('Adamic-CommonNeighbors OL: ' + str(aa_c_overlap_messages) + '\n')
    f.write('Adamic-TotalNeighbors OL: ' + str(aa_t_overlap_messages) + '\n')
    f.write('Adamic-Preferential OL: ' + str(aa_p_overlap_messages) + '\n')
    f.write('Adamic-Resource OL: ' + str(aa_r_overlap_messages) + '\n')
    f.write('Adamic-Same OL: ' + str(aa_s_overlap_messages) + '\n')
    f.write('CommonNeighbors-TotalNeighbors OL: ' + str(common_t_overlap_messages) + '\n')
    f.write('CommonNeighbors-Preferential OL: ' + str(common_p_overlap_messages) + '\n')
    f.write('CommonNeighbors-Resource OL: ' + str(common_r_overlap_messages) + '\n')
    f.write('CommonNeighbors-Same OL: ' + str(common_s_overlap_messages) + '\n')
    f.write('TotalNeighbors-Preferential OL: ' + str(total_p_overlap_messages) + '\n')
    f.write('TotalNeighbors-Resource OL: ' + str(total_r_overlap_messages) + '\n')
    f.write('TotalNeighbors-Same OL: ' + str(total_s_overlap_messages) + '\n')
    f.write('Preferential-Resource OL: ' + str(pref_r_overlap_messages) + '\n')
    f.write('Preferential-Same OL: ' + str(pref_s_overlap_messages) + '\n')
    f.write('Resource-Same OL: ' + str(resource_s_overlap_messages) + '\n')

    f.write('Overlap Ranked Biased--------------\n')
    f.write('Adamic-CommonNeighbors RBO: ' + str(aa_c_rbo_messages) + '\n')
    f.write('Adamic-TotalNeighbors RBO: ' + str(aa_t_rbo_messages) + '\n')
    f.write('Adamic-Preferential RBO: ' + str(aa_p_rbo_messages) + '\n')
    f.write('Adamic-Resource RBO: ' + str(aa_r_rbo_messages) + '\n')
    f.write('Adamic-Same RBO: ' + str(aa_s_rbo_messages) + '\n')
    f.write('CommonNeighbors-TotalNeighbors RBO: ' + str(common_t_rbo_messages) + '\n')
    f.write('CommonNeighbors-Preferential RBO: ' + str(common_p_rbo_messages) + '\n')
    f.write('CommonNeighbors-Resource RBO: ' + str(common_r_rbo_messages) + '\n')
    f.write('CommonNeighbors-Same RBO: ' + str(common_s_rbo_messages) + '\n')
    f.write('TotalNeighbors-Preferential RBO: ' + str(total_p_rbo_messages) + '\n')
    f.write('TotalNeighbors-Resource RBO: ' + str(total_r_rbo_messages) + '\n')
    f.write('TotalNeighbors-Same RBO: ' + str(total_s_rbo_messages) + '\n')
    f.write('Preferential-Resource RBO: ' + str(pref_r_rbo_messages) + '\n')
    f.write('Preferential-Same RBO: ' + str(pref_s_rbo_messages) + '\n')
    f.write('Resource-Same RBO: ' + str(resource_s_rbo_messages) + '\n')

    f.write('Time taken to compute overlaps-------------\n')
    f.write('Adamin-Common Time: ' + str(aa_c_messages_time_taken) + '\n')
    f.write('Adamin-Total Time: ' + str(aa_t_messages_time_taken) + '\n')
    f.write('Adamin-Preferential Time: ' + str(aa_p_messages_time_taken) + '\n')
    f.write('Adamin-Resource Time: ' + str(aa_r_messages_time_taken) + '\n')
    f.write('Adamin-Same Time: ' + str(aa_s_messages_time_taken) + '\n')
    f.write('Common-Total Time: ' + str(common_t_messages_time_taken) + '\n')
    f.write('Common-Preferential Time: ' + str(common_p_messages_time_taken) + '\n')
    f.write('Common-Resource Time: ' + str(common_r_messages_time_taken) + '\n')
    f.write('Common-Same Time: ' + str(common_s_messages_time_taken) + '\n')
    f.write('Total-Preferential Time: ' + str(total_p_messages_time_taken) + '\n')
    f.write('Total-Resource Time: ' + str(total_r_messages_time_taken) + '\n')
    f.write('Total-Same Time: ' + str(total_s_messages_time_taken) + '\n')
    f.write('Preferential-Resource Time: ' + str(pref_r_messages_time_taken) + '\n')
    f.write('Preferential-Same Time: ' + str(pref_s_messages_time_taken) + '\n')
    f.write('Resource-Same Time: ' + str(resource_s_messages_time_taken) + '\n')

    f.write('TRADES SCORE COMPARISON-----------\n')
    
    f.write('Cosine Similarities------------\n')
    f.write('Adamic-CommonNeighbors CS: ' + str(cos_trades_aa_common) + '\n')
    f.write('Adamic-TotalNeighbors CS: ' + str(cos_trades_aa_total) + '\n')
    f.write('Adamic-Preferential CS: ' + str(cos_trades_aa_pref) + '\n')
    f.write('Adamic-Resource CS: ' + str(cos_trades_aa_resource) + '\n')
    f.write('Adamic-Same CS: ' + str(cos_trades_aa_same) + '\n')
    f.write('CommonNeighbors-TotalNeighbors CS: ' + str(cos_trades_common_total) + '\n')
    f.write('CommonNeighbors-Preferential CS: ' + str(cos_trades_common_pref) + '\n')
    f.write('CommonNeighbors-Resource CS: ' + str(cos_trades_common_resource) + '\n')
    f.write('CommonNeighbors-Same CS: ' + str(cos_trades_common_same) + '\n')
    f.write('TotalNeighbors-Preferential CS: ' + str(cos_trades_total_pref) + '\n')
    f.write('TotalNeighbors-Resource CS: ' + str(cos_trades_total_resource) + '\n')
    f.write('TotalNeighbors-Same CS: ' + str(cos_trades_total_same) + '\n')
    f.write('Preferential-Resource CS: ' + str(cos_trades_pref_resource) + '\n')
    f.write('Preferential-Same CS: ' + str(cos_trades_pref_same) + '\n')
    f.write('Resource-Same CS: ' + str(cos_trades_resource_same) + '\n')
    
    f.write('Overlap------------\n')
    f.write('Adamic-CommonNeighbors OL: ' + str(aa_c_overlap_trades) + '\n')
    f.write('Adamic-TotalNeighbors OL: ' + str(aa_t_overlap_trades) + '\n')
    f.write('Adamic-Preferential OL: ' + str(aa_p_overlap_trades) + '\n')
    f.write('Adamic-Resource OL: ' + str(aa_r_overlap_trades) + '\n')
    f.write('Adamic-Same OL: ' + str(aa_s_overlap_trades) + '\n')
    f.write('CommonNeighbors-TotalNeighbors OL: ' + str(common_t_overlap_trades) + '\n')
    f.write('CommonNeighbors-Preferential OL: ' + str(common_p_overlap_trades) + '\n')
    f.write('CommonNeighbors-Resource OL: ' + str(common_r_overlap_trades) + '\n')
    f.write('CommonNeighbors-Same OL: ' + str(common_s_overlap_trades) + '\n')
    f.write('TotalNeighbors-Preferential OL: ' + str(total_p_overlap_trades) + '\n')
    f.write('TotalNeighbors-Resource OL: ' + str(total_r_overlap_trades) + '\n')
    f.write('TotalNeighbors-Same OL: ' + str(total_s_overlap_trades) + '\n')
    f.write('Preferential-Resource OL: ' + str(pref_r_overlap_trades) + '\n')
    f.write('Preferential-Same OL: ' + str(pref_s_overlap_trades) + '\n')
    f.write('Resource-Same OL: ' + str(resource_s_overlap_trades) + '\n')

    f.write('Overlap Ranked Biased--------------\n')
    f.write('Adamic-CommonNeighbors RBO: ' + str(aa_c_rbo_trades) + '\n')
    f.write('Adamic-TotalNeighbors RBO: ' + str(aa_t_rbo_trades) + '\n')
    f.write('Adamic-Preferential RBO: ' + str(aa_p_rbo_trades) + '\n')
    f.write('Adamic-Resource RBO: ' + str(aa_r_rbo_trades) + '\n')
    f.write('Adamic-Same RBO: ' + str(aa_s_rbo_trades) + '\n')
    f.write('CommonNeighbors-TotalNeighbors RBO: ' + str(common_t_rbo_trades) + '\n')
    f.write('CommonNeighbors-Preferential RBO: ' + str(common_p_rbo_trades) + '\n')
    f.write('CommonNeighbors-Resource RBO: ' + str(common_r_rbo_trades) + '\n')
    f.write('CommonNeighbors-Same RBO: ' + str(common_s_rbo_trades) + '\n')
    f.write('TotalNeighbors-Preferential RBO: ' + str(total_p_rbo_trades) + '\n')
    f.write('TotalNeighbors-Resource RBO: ' + str(total_r_rbo_trades) + '\n')
    f.write('TotalNeighbors-Same RBO: ' + str(total_s_rbo_trades) + '\n')
    f.write('Preferential-Resource RBO: ' + str(pref_r_rbo_trades) + '\n')
    f.write('Preferential-Same RBO: ' + str(pref_s_rbo_trades) + '\n')
    f.write('Resource-Same RBO: ' + str(resource_s_rbo_trades) + '\n')

    f.write('Time taken to compute overlaps-------------\n')
    f.write('Adamin-Common Time: ' + str(aa_c_trades_time_taken) + '\n')
    f.write('Adamin-Total Time: ' + str(aa_t_trades_time_taken) + '\n')
    f.write('Adamin-Preferential Time: ' + str(aa_p_trades_time_taken) + '\n')
    f.write('Adamin-Resource Time: ' + str(aa_r_trades_time_taken) + '\n')
    f.write('Adamin-Same Time: ' + str(aa_s_trades_time_taken) + '\n')
    f.write('Common-Total Time: ' + str(common_t_trades_time_taken) + '\n')
    f.write('Common-Preferential Time: ' + str(common_p_trades_time_taken) + '\n')
    f.write('Common-Resource Time: ' + str(common_r_trades_time_taken) + '\n')
    f.write('Common-Same Time: ' + str(common_s_trades_time_taken) + '\n')
    f.write('Total-Preferential Time: ' + str(total_p_trades_time_taken) + '\n')
    f.write('Total-Resource Time: ' + str(total_r_trades_time_taken) + '\n')
    f.write('Total-Same Time: ' + str(total_s_trades_time_taken) + '\n')
    f.write('Preferential-Resource Time: ' + str(pref_r_trades_time_taken) + '\n')
    f.write('Preferential-Same Time: ' + str(pref_s_trades_time_taken) + '\n')
    f.write('Resource-Same Time: ' + str(resource_s_trades_time_taken) + '\n')

    f.write('Total Time Taken: ' + str(datetime.now() - d))

print('Comparison Finished')