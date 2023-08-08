import os
models = ['codebert','graphcodebert','unixcoder']
for m in models:
    root = '/hxd/ybw/LLM4TCP/data/outputs/{}_APFD'.format(m)
    for dir_name in os.listdir(root):
        path = os.path.join(root,dir_name)
        for i in range(50):
            path1 = os.path.join(path,'{}_cosine.csv'.format(i))
            path2 = os.path.join(path,'{}_euclidean.csv'.format(i))
            if not os.path.exists(path1):
                print(path1)
            if not os.path.exists(path2):
                print(path2)