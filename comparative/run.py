import argparse
import math
import os
import time

import numpy as np
import pandas as pd

import fast, competitors


def read_mutant_matrix(path):
    df = pd.read_csv(path)
    data = df['faulty'].tolist()
    data = [int(d) for d in data]
    return np.array(data).reshape(-1, 1)


def get_apfd(testList, mutant_matrix):
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


def getS(fin, algoname):
    k, n, r, b = 5, 10, 1, 10
    prioritization = None
    if algoname == "FAST-all":
        def all_(x):
            return x

        selsize = all_
        stime, ptime, prioritization = fast.fast_(
            fin, selsize, r=r, b=b, bbox=True, k=k, memory=False)
    elif algoname == "FAST-sqrt":
        def sqrt_(x): return int(math.sqrt(x)) + 1
        selsize = sqrt_
        stime, ptime, prioritization = fast.fast_(
            fin, selsize, r=r, b=b, bbox=True, k=k, memory=False)
    elif algoname == "FAST-log":
        def log_(x): return int(math.log(x, 2)) + 1
        selsize = log_
        stime, ptime, prioritization = fast.fast_(
            fin, selsize, r=r, b=b, bbox=True, k=k, memory=False)
    elif algoname == "FAST-one":
        def one_(x): return 1
        selsize = one_
        stime, ptime, prioritization = fast.fast_(
            fin, selsize, r=r, b=b, bbox=True, k=k, memory=False)
    elif algoname == "STR":
        stime, ptime, prioritization = competitors.str_(fin)
    elif algoname == "I-TSD":
        stime, ptime, prioritization = competitors.i_tsd(fin)
    else:
        def pw(x): pass
        stime, ptime, prioritization = fast.fast_pw(
            fin, r=r, b=b, bbox=True, k=k, memory=False)
    return [i - 1 for i in prioritization]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="./data/TestMethods_txt", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_path", default="./data/outputs/", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--project_name", default="Chart", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--algo_name", default="STR", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--mutant_path", default="../data/TestMethods", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--repeat", default=50, type=int, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    args = parser.parse_args()
    # FAST parameters

    algoname = args.algo_name
    # FAST-f sample size
    project = args.project_name + '/'
    versions = os.listdir(args.input_path + '/' + project)
    versions = sorted(versions, key=lambda x: int(x[:-4]))
    project_versions = [project + version.replace('.txt', '') for version in versions]
    for i in range(args.repeat):
        print("Project:{} length:{} repeat{}".format(args.project_name, len(project_versions),i))

        df_apfd = pd.DataFrame()
        res_versions = []
        res_s = []
        res_apfd = []
        for project_version in project_versions:
            print(project_version)
            fin_path = args.input_path + '/' + project_version + '.txt'
            mutant_input_path = args.mutant_path + '/' + project_version + '.csv'
            mutant_matrix = read_mutant_matrix(mutant_input_path)
            start_time = time.time()
            print('   Calculating ' + algoname + '..')
            s = getS(fin_path, algoname)
            apfd = get_apfd(s, mutant_matrix)
            end_time = time.time()
            execution_time = end_time - start_time
            print(execution_time)
            res_versions.append(project_version)
            res_s.append(s)
            res_apfd.append(apfd)
        df_apfd['Project Version'] = res_versions
        df_apfd['Sort'] = res_s
        df_apfd['APFD'] = res_apfd
        apfd_folder_path = args.output_path + algoname + "_APFD/" + args.project_name
        if not os.path.exists(apfd_folder_path):
            os.makedirs(apfd_folder_path)
        df_apfd.to_csv(apfd_folder_path + "/" + algoname + "_{}".format(i) + ".csv", index=False)


if __name__ == '__main__':
    main()
