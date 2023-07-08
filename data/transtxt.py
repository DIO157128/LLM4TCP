# -*- coding: utf-8 -*-
import pandas as pd
import os
if __name__ == '__main__':
    for d in os.listdir('./TestMethods'):
        folder_name = "./TestMethods/"+d
        os.mkdir("./TestMethods_txt/"+d)
        for f in os.listdir(folder_name):
            file_name = folder_name+"/"+f
            df = pd.read_csv(file_name,encoding='utf-8')
            df1 = pd.DataFrame()
            txt = open("./TestMethods_txt/"+d+"/"+f.replace("csv","txt"),'a',encoding='utf-8')
            codes = df['test_code'].tolist()
            for c in codes :
                txt.write(c+'\n')
            txt.close()