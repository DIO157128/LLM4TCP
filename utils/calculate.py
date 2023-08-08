import os
import pandas as pd
def calculatemodel():
    models = ['codebert','graphcodebert','unixcoder']
    for m in models:
        root = '/hxd/ybw/LLM4TCP/data/outputs/{}_APFD'.format(m)
        result_dir = os.path.join('/hxd/ybw/LLM4TCP/result',m)
        for dir_name in os.listdir(root):
            path = os.path.join(root,dir_name)
            apfds_cosine = []
            apfds_euclidean=[]
            project = pd.read_csv(os.path.join(path,'{}_cosine.csv'.format(0)))['Project Version']
            for i in range(50):
                path1 = os.path.join(path,'{}_cosine.csv'.format(i))
                path2 = os.path.join(path,'{}_euclidean.csv'.format(i))
                df1 = pd.read_csv(path1)
                df2 = pd.read_csv(path2)
                apfds_cosine.append(df1['APFD'].tolist())
                apfds_euclidean.append(df2['APFD'].tolist())
            result_dir1 = os.path.join(result_dir,'cosine')
            result_dir2 = os.path.join(result_dir,'euclidean')
            if not os.path.exists(result_dir1):
                os.makedirs(result_dir1)
            if not os.path.exists(result_dir2):
                os.makedirs(result_dir2)
            res_df1 = pd.DataFrame()
            res_df2 = pd.DataFrame()
            res_df1['Project Version'] = project
            res_df2['Project Version'] = project
            for i in range(50):
                res_df1['APFD_{}'.format(i)] = apfds_cosine[i]
                res_df2['APFD_{}'.format(i)] = apfds_euclidean[i]
            res_df1.to_csv(os.path.join(result_dir1,'{}.csv'.format(dir_name)),index=False)
            res_df2.to_csv(os.path.join(result_dir2, '{}.csv'.format(dir_name)),index=False)
def calculatecompartive():
    models = ['FAST-all','FAST-log','FAST-one','FAST-pw','FAST-sqrt','STR']
    for m in models:
        root = '/hxd/ybw/LLM4TCP/comparative/data/outputs/{}_APFD'.format(m)
        result_dir = os.path.join('/hxd/ybw/LLM4TCP/result',m)
        for dir_name in os.listdir(root):
            path = os.path.join(root,dir_name)
            apfds_cosine = []
            project = pd.read_csv(os.path.join(path,'{}_{}.csv'.format(m,0)))['Project Version']
            for i in range(50):
                path1 = os.path.join(path,'{}_{}.csv'.format(m,i))
                df1 = pd.read_csv(path1)
                apfds_cosine.append(df1['APFD'].tolist())
            result_dir1 = os.path.join(result_dir,m)
            if not os.path.exists(result_dir1):
                os.makedirs(result_dir1)
            res_df1 = pd.DataFrame()
            res_df1['Project Version'] = project
            for i in range(50):
                res_df1['APFD_{}'.format(i)] = apfds_cosine[i]
            res_df1.to_csv(os.path.join(result_dir1,'{}.csv'.format(dir_name)),index=False)
if __name__ == '__main__':
    calculatecompartive()