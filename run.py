import os
import sys
import os
import csv
import time
import random

import pandas as pd
from collections import defaultdict
import numpy as np
from itertools import combinations
import torch
import torch.nn as nn
from scipy.spatial import distance
from torch.nn.modules import transformer
from torch.utils.data import Dataset, RandomSampler, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.models import roberta
import argparse
from model import LLM

models = {'codebert': 'microsoft/codebert-base',
          'graphcodebert': 'microsoft/graphcodebert-base',
          'unixcoder': 'microsoft/unixcoder-base'
          }

similarity_measures = {
    'cosine': distance.cosine,
    'euclidean': distance.euclidean
}


def generate_vectors(input_path, output_path, output_folder ,model, device):
    print('   Generating vectors..')

    print('       Reading data file..')
    df = pd.read_csv(input_path)
    df = df.dropna()
    code = df['test_code']
    print('       Tokenize..')
    tokens = model.tokenize(code.tolist(), mode='<encoder-only>')  # tokens_ids
    print('       Embeddings..')
    embeddings = []
    with torch.no_grad():
        for token_id in tqdm(tokens):
            embeddings.append(model(torch.tensor([token_id]).to(device))[1].tolist()[0])
    print('       To dataframe..')
    vectors = pd.DataFrame()
    vectors['test_case'] = list(df['test_case'])
    vectors['vector'] = embeddings
    #####generate vectors file
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    vectors.to_csv(output_path, index=False)

    return vectors

def parseVec(vec):
    res = []
    for v in vec:
        v = v.replace('[',"")
        v = v.replace(']', "")
        tem_v = v.split(',')
        tem_v = np.array([float(x) for x in tem_v])
        res.append(tem_v)
    return np.array(res)
def ART(distance_matrix, args):
    """
    优化后的ART排序算法
    """
    num = len(distance_matrix)
    select = []
    suite = np.arange(num)

    if args.random_first:
        first = np.random.randint(0, num)
    else:
        first = 0

    select.append(first)
    suite = np.delete(suite, first)

    while len(suite) != 0:
        r2_values = distance_matrix[select]  # 从距离矩阵中选择已选项目
        candidate = np.min(distance_matrix[suite][:, select], axis=1)  # 计算剩余项目的最小距离
        candidate_idx = suite[np.argwhere(candidate == candidate.max())].flatten()

        if args.random_select:
            select_case = np.random.choice(candidate_idx)
        else:
            select_case = candidate_idx[0]

        select.append(select_case)
        suite = np.delete(suite, np.where(suite == select_case))

    return select

def calculate_similarity_optimized(vectors, args, proj_ver, output_path):
    vecs = vectors['vector'].tolist()
    tcs = vectors['test_case'].tolist()
    vecs = parseVec(vecs)
    dstnc = distance.pdist(vecs, args.sim_name)
    dist_matrix = distance.squareform(dstnc)
    if args.save_sim_matrix:
        filename = output_path + '_similarity/' + args.sim_name + '/' + proj_ver + '.csv'
        output_folder = output_path + '_similarity/' + args.sim_name + '/' + proj_ver.split('/')[0]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(dist_matrix)
    return dist_matrix
def read_mutant_matrix(path):
    df = pd.read_csv(path)
    data = df['faulty'].tolist()
    data = [int(d) for d in data]
    return np.array(data).reshape(-1, 1)
def get_apfd(testList, mutant_matrix):
    """
    :param testList: 排序后测试用例列表
    :param mutant_matrix: 变异矩阵
    :return:  apfd
    """
    mutantkillmatrix = mutant_matrix
    testno_testkill_dict = {}
    for i in range(len(mutantkillmatrix)):
        thisline = mutantkillmatrix[i]
        killlist = []
        for j in range(len(thisline)):
            if thisline[j] == 1:
                killlist.append(j)
        testno_testkill_dict[i] = killlist
    thissequence = testList
    apfd = 0.0
    is_killed = []

    for i in range(len(mutantkillmatrix[0])):
        is_killed.append(False)
    for i in range(len(thissequence)):
        thistestno = thissequence[i]
        killlist = testno_testkill_dict[thistestno]
        for j in killlist:
            if is_killed[j] == False:
                is_killed[j] = True
                apfd += float(i + 1)
    apfd = apfd / float(len(mutantkillmatrix) * len(mutantkillmatrix[0]))
    apfd = 1 - apfd + 1 / float(2 * len(thissequence))
    return apfd
def main():
    parser = argparse.ArgumentParser()
    # Params
    parser.add_argument("--input_path", default="./data/TestMethods", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_path", default="./data/outputs/", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--use_cuda", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--model_name", default="codebert", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--project_name", default="Chart", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--generate_vector", action='store_true', default=False,
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--calculate_similarity", action='store_true', default=False,
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--calculate_apfd", action='store_true', default=False,
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--sorting", action='store_true', default=False,
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--save_sim_matrix", action='store_true', default=False,
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--sim_name", default="cosine", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--repeat", default=50, type=int, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--random_first", default=1, type=int, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--random_select", default=1, type=int, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--start", default=0, type=int, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--single_version", default=False, action='store_true',
                        help="Whether to write raw predictions on test data.")
    args = parser.parse_args()
    print("language model:", args.model_name)
    print('Load the model..')
    huggingface_path = models[args.model_name]
    model = LLM(huggingface_path)
    print("Finish loading!")
    device = torch.device("cuda:0" if args.use_cuda else "cpu")
    print(device)
    output_path = args.output_path + args.model_name
    project = args.project_name + '/'
    versions = os.listdir(args.input_path + '/' + project)
    versions = sorted(versions, key=lambda x: int(x[:-4]))
    project_versions = [project + version.replace('.csv', '') for version in versions]
    for i in range(args.start,args.repeat):
        apfd_time = []
        art_time = []
        sim_time = []
        vec_time = []
        print("Project:{} length:{} repeat{}".format(args.project_name, len(project_versions),i))

        df_apfd = pd.DataFrame()
        res_versions = []
        res_s = []
        res_apfd = []
        if args.single_version:
            project_versions = project_versions[:1]
        for project_version in project_versions:
            print(project_version)
            project_version_input_path = args.input_path + '/' + project_version + '.csv'
            project_version_output_folder = output_path + '_vectors/' + project_version.split('/')[0]
            project_version_output_path = output_path + '_vectors/' + project_version + '.csv'
            start_time = time.time()
            if args.generate_vector:
                generate_vectors(project_version_input_path, project_version_output_path, project_version_output_folder,
                                 model, device)
                end_time = time.time()
                execution_time = end_time - start_time
                vec_time.append(execution_time)
            mutant_matrix = read_mutant_matrix(project_version_input_path)
            if args.calculate_similarity:
                vectors = pd.read_csv(project_version_output_path)
                print('   Calculating ' + args.sim_name + '..')
                start_time = time.time()
                calculate_similarity_optimized(vectors, args, project_version, output_path)
                end_time = time.time()
                execution_time = end_time - start_time
                sim_time.append(execution_time)
                print(f"相似度计算时间为: {execution_time} 秒")
            if args.sorting:
                dist_matrix = np.genfromtxt(output_path + '_similarity/' + args.sim_name + '/' + project_version + '.csv',
                                            delimiter=',')
                start_time = time.time()
                s = ART(dist_matrix,args)
                end_time = time.time()
                execution_time = end_time - start_time
                art_time.append(execution_time)
                start_time = time.time()
                apfd = get_apfd(s, mutant_matrix)
                end_time = time.time()
                execution_time = end_time - start_time
                apfd_time.append(execution_time)
                print(f"排序时间为: {execution_time} 秒")
                res_versions.append(project_version)
                res_s.append(s)
                res_apfd.append(apfd)
        if args.generate_vector:
            vector_folder_path = args.output_path + args.model_name + "_time/vector"
            if not os.path.exists(vector_folder_path):
                os.makedirs(vector_folder_path)
            df_time = pd.DataFrame()
            df_time['Vector Time'] = vec_time
            df_time.to_csv(vector_folder_path + "/" + args.project_name + ".csv", index=False)
        if args.calculate_similarity:
            sim_folder_path = args.output_path + args.model_name + "_time/similarity"
            if not os.path.exists(sim_folder_path):
                os.makedirs(sim_folder_path)
            df_time = pd.DataFrame()
            df_time['Similarity Time'] = sim_time
            df_time.to_csv( sim_folder_path+ "/" + args.project_name +'_'+args.sim_name+ ".csv", index=False)
        if args.sorting:
            art_folder_path = args.output_path + args.model_name + "_time/art/" + args.project_name
            if not os.path.exists(art_folder_path):
                os.makedirs(art_folder_path)
            df_time = pd.DataFrame()
            df_time['Project Version'] = res_versions
            df_time['ART Time'] = art_time
            df_time['APFD Time'] = apfd_time
            df_time.to_csv(art_folder_path+"/{}_".format(i)+args.sim_name+".csv",index=False)
        if args.calculate_apfd:
            df_apfd['Project Version'] = res_versions
            df_apfd['Sort'] = res_s
            df_apfd['APFD'] = res_apfd
            apfd_folder_path = args.output_path + args.model_name+"_APFD/"+args.project_name
            if not os.path.exists(apfd_folder_path):
                os.makedirs(apfd_folder_path)
            df_apfd.to_csv(apfd_folder_path+"/{}_".format(i)+args.sim_name+".csv",index=False)


        args.calculate_similarity = False
        args.generate_vector = False



if __name__ == '__main__':
    main()
