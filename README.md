#LLM4TCP
python run.py --project_name Chart --generate_vector --calculate_similarity --calculate_apfd --sim_name cosine 

python run.py --project_name Chart  --calculate_similarity --calculate_apfd --sim_name euclidean
##comparative
对于对比实验而言，fast是用python2编写的，所以需要切换python2.7的conda环境
