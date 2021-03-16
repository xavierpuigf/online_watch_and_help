import glob
from multiprocessing import Pool
import pickle as pkl
from pathlib import Path

files = glob.glob('../../data_scratch/train_env_task_set_20_full_reduced_tasks/*/*')
def convert_file(file_name):
    size_1 = Path(file_name).stat().st_size
    with open(file_name, 'rb') as f:
        content = pkl.load(f)
        if 'graph' not in content:
            print(file_name)
            size_2 = size_1
        else:
            for graph in content['graph']:
                nodes = graph['nodes']
                for node in nodes:
                    if 'prefab_name' in node:
                        del node['prefab_name']
                    if 'obj_transform' in node:
                        del node['obj_transform']

                    if 'properties' in node:
                        del node['properties']
    with open(file_name, 'wb+') as f:
        pkl.dump(content, f)
    size_2 = Path(file_name).stat().st_size
    print(size_2 - size_1)
    return (size_2 - size_1)

with Pool(10) as p:
    size = p.map(convert_file, files)
    print(sizediff)
