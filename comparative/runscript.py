import os
import multiprocessing
import subprocess


def run_command(command, output_file):
    with open(output_file, 'w') as f:
        subprocess.call(command, stdout=f, shell=True)
if __name__ == '__main__':
    algos = ['FAST-all','FAST-sqrt', 'FAST-log', 'FAST-one', 'FAST-pw','STR']
    algo = 'FAST-all'
    commands = []
    for dir_name in os.listdir('./data/TestMethods_txt'):
        commands.append('python run.py --project_name {} --algo_name {} --repeat 50 --single_version'.format(dir_name, algo))
    max_concurrent_processes = 7

    pool = multiprocessing.Pool(processes=max_concurrent_processes)

    for command in commands:
        if not os.path.exists('./data/{}'.format(algo)):
            os.makedirs('./data/{}'.format(algo))
        output_file = './data/{}/{}.txt'.format(algo, command.split()[3])
        pool.apply_async(run_command, args=(command, output_file))

    pool.close()

    pool.join()