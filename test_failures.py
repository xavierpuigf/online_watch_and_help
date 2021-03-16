import pickle as pkl
import ipdb
import sys
sys.path.append('../')
from virtualhome.simulation.unity_simulator import comm_unity
file_name = '../path_sim_dev/linux_exec.x86_64'
comm = comm_unity.UnityCommunication(file_name=file_name, no_graphics=True)
file_name = 'dataset/train_env_task_set_20_full_reduced_tasks.pik'
with open(file_name, 'rb') as f:
	content = pkl.load(f)


episode_id = 17

epi = content[episode_id]
comm.reset(epi['env_id'])
s, g = comm.environment_graph()
max_id = max(node['id'] for node in g['nodes'])

for node in epi['init_graph']['nodes']:
    if node['id'] > max_id:
        node['id'] += 1000
for edge in epi['init_graph']['edges']:
    if edge['from_id'] > max_id:
        edge['from_id'] += 1000

    if edge['to_id'] > max_id:
        edge['to_id'] += 1000

expand = comm.expand_scene(epi['init_graph'], transfer_transform=True)
print(expand)
comm.add_character()

script = [
        '[walktowards] <kitchencabinet> (238)'
        ,'[walktowards] <kitchencabinet> (238)'
        ,'[walktowards] <kitchencabinet> (238)'
        ,'[open] <kitchencabinet> (238)'
        ,'[walktowards] <kitchencabinet> (235)'
        ,'[open] <kitchencabinet> (235)'
        ,'[grab] <cutleryfork> (458)'
        ,'[grab] <cutleryfork> (459)'
        ,'[walktowards] <kitchentable> (232)'
        ,'[walktowards] <kitchentable> (232)'
        ,'[putback] <cutleryfork> (458) <kitchentable> (232)'
        ,'[walktowards] <bedroom> (75)'
        ,'[walktowards] <bedroom> (75)'
        ,'[walktowards] <bedroom> (75)'
        ,'[walktowards] <bedroom> (75)'
        ,'[walktowards] <bedroom> (75)'
        ,'[walktowards] <bedroom> (75)'
        ,'[walktowards] <plate> (455)'
]
for scr_item in script:
    scr = ['<char0> {}'.format(scr_item)]
    s, m = comm.render_script(script=scr, skip_animation=True, image_synthesis=[])
    print(scr, s, m)
