import os
import ast

total_avg = 0

with open('dataset_info222.txt', 'a') as info:
    file_avg = 0
    for _file in os.listdir(os.getcwd()):
        if not _file.endswith('.txt') or \
            not _file.startswith('comm'): continue

        print(_file)
        with open(_file, 'r') as f:
            content = f.readlines()
            for x in content: 
                x = x.rstrip()
                x_list = x.split(' ')
                file_avg = len(x_list)
         
        content = [x.rstrip() for x in content]
    
        info.write('-----------------------')
        info.write(_file + " has " + str(len(content)) + ' communities...\n')
        info.write(_file + " has " + str(file_avg) + ' nodes in comms...\n')
        info.write(_file + " has " + str(file_avg/(len(content))) + ' nodes per comms...\n')
        total_avg += file_avg/(len(content))
    
    info.write('total avg nodes per comm = ' + str(total_avg/30))