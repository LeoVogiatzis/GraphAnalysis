import os
import pandas as pd


results = {}
cnt_a = 0
cnt_m = 0
cnt_t = 0
with open('dataset_info_results.txt', 'a') as f:
    f.write('Dataset Info Results.....\n')
    for _file in os.listdir(os.getcwd()):
        
        if not _file.endswith('.csv') and (not \
            'attacks' in _file or not \
                'message' in _file or not \
                    'trade' in _file): continue
        
        df = pd.read_csv(_file, header=None)
        df.columns = ['Timestamp', 'id1', 'id2']

        info = _file.replace('.csv', '').split('-')
        day = info[4]
        
        num_of_rels = len(df)
        num_of_sources = len(set(df['id1']))
        num_of_destinations = len(set(df['id2']))
        rels_per_source = num_of_rels / num_of_sources
        rels_per_destinations = num_of_rels / num_of_destinations

        if 'attacks' in _file:
            rel = 'attacks'
            cnt_a += num_of_rels
        elif 'messages' in _file:
            rel = 'messages'
            cnt_m += num_of_rels
        else:
            rel = 'trades'
            cnt_t += num_of_rels

        f.write(_file + '\n')
        f.write(rel.upper() + '_' + str(day) + '\n')
        f.write('Number of relationships: ' + str(num_of_rels) + '\n')
        f.write('Number of sources: ' + str(num_of_sources) + '\n')
        f.write('Number of destinations: ' + str(num_of_destinations) + '\n')
        f.write('Relationships per source: ' + str(rels_per_source) + '\n')
        f.write('Relationships per destination: ' + str(rels_per_destinations) + '\n')


    f.write('*'*30)
    f.write('Total Resutls.....\n')
    f.write('Total Attacks: ' + str(cnt_a) + '\n')
    f.write('Total Messages: ' + str(cnt_m) + '\n')
    f.write('Total Trades: ' + str(cnt_t) + '\n')
    f.write('Attacks per day: ' + str(cnt_a/30) + '\n')
    f.write('Messages per day: ' + str(cnt_m/30) + '\n')
    f.write('Trades per day: ' + str(cnt_t/30) + '\n')