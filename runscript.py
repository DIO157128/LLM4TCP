import os
import multiprocessing
import subprocess


def run_command(command, output_file):
    with open(output_file, 'w') as f:
        subprocess.call(command, stdout=f, shell=True)
if __name__ == '__main__':
    commands = []
    sim_name = 'cosine'
    model_name = 'codebert'
    for dir_name in os.listdir('./data/TestMethods'):

        print('''os.system('python run.py --project_name {} --model_name {} --calculate_apfd --save_sim_matrix --calculate_similarity --sim_name {} --repeat 50')'''.format(dir_name,model_name,sim_name))
        commands.append('python run.py --project_name {} --model_name {} --calculate_apfd --save_sim_matrix --calculate_similarity --sim_name {} --repeat 50'.format(dir_name,model_name,sim_name))

    # max_concurrent_processes = 7
    #
    # pool = multiprocessing.Pool(processes=max_concurrent_processes)
    #
    # for command in commands:
    #     if not os.path.exists('./data/{}/{}'.format(model_name,sim_name)):
    #         os.makedirs('./data/{}/{}'.format(model_name,sim_name))
    #     output_file = './data/{}/{}/{}.txt'.format(model_name,sim_name,command.split()[3])
    #     pool.apply_async(run_command, args=(command, output_file))
    #
    # pool.close()
    #
    # pool.join()