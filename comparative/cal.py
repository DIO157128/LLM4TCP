import os

import pandas as pd

algos = ['FAST-sqrt', 'FAST-log', 'FAST-one', 'FAST-pw','STR']
for algo in algos:
    path = '/hxd/ybw/LLM4TCP/comparative/data/'+algo
    for project in os.listdir(path):
        time_path = '/hxd/ybw/LLM4TCP/comparative/data/outputs/{}_time/similarity/'.format(algo)
        if not os.path.exists(time_path):
            os.makedirs(time_path)
        times = []
        for i in os.listdir('/hxd/ybw/LLM4TCP/comparative/data/TestMethods_txt/'+project):
            i = i.replace('.txt','')
            file = path+'/'+project+'/'+i+'_sigtime.txt'
            f = open(file)
            times.append(f.readlines()[0])
        df = pd.DataFrame()
        df['Similarity Time'] = times
        df.to_csv(time_path+project+'.csv')

