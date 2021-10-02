import json
import sys
import random
import time
import pdb
import ipdb
import numpy as np
import time
sys.path.append('../virtualhome/')
from simulation.unity_simulator import comm_unity

times = {
    'reset': [],
    'char_time': [],
    'graph': [],
    'walktowards': [],
    'open': [],
    'close': [],
    'grab': [],
    'putback': []
}


comm = comm_unity.UnityCommunication()
comm.reset(6)
s, g = comm.environment_graph()
cid = 1000
list_ids = [259, 271, 33, 168, 175, 177,
 178, 179, 180, 181, 182, 183, 184,
 185, 186, 187, 191, 192, 331, 332,
 99, 100, 101, 102, 104, 105, 106,
 107, 108, 109, 110, 111, 112, 360, 367, 372, 373,
 250, 251]

with open('tests/graph.json', 'r') as f:
    cont = json.load(f)
cont = g
script = ['<char0> [walktowards] <bedroom> (294)|<char1> [walktowards] <kitchen> (56)']
red_list = [list_ids[33]]

cids = [node['id'] for node in cont['nodes']]
nodes_trashcan = [node for node in g['nodes'] if node['id'] == 360]
edges_trash = [edge for edge in g['edges'] if edge['from_id'] == 360 and edge['to_id'] in cids or  edge['to_id'] == 360 and edge['from_id'] in cids]
cont['edges'] += edges_trash
cont['nodes'] += nodes_trash

# cont['nodes'] = [node for node in cont['nodes'] if node['id'] < cid and node['id'] not in red_list]
# cont['edges'] = [edge for edge in cont['edges'] if edge['from_id'] < cid and edge['to_id'] < cid and edge['from_id'] not in red_list and edge['to_id'] not in red_list]
comm.expand_scene(cont)
comm.add_character(initial_room='livingroom')
comm.add_character(initial_room='bedroom')
comm.render_script(script, skip_animation=True)

s, g = comm.environment_graph()
nodes_plate = [node['id'] for node in g['nodes'] if node['class_name'] == 'plate']
nodes_table = [node['id'] for node in g['nodes'] if node['class_name'] == 'kitchentable']

nodes_cabinet = [node['id'] for node in g['nodes'] if node['class_name'] == 'kitchencabinet']


for i in range(200):
    script = []
    plates = random.choices(nodes_plate, k=4)
    for plate_id in plates[:-1]:
        script += ['[walktowards] <plate> ({})'.format(plate_id)]*5
    
    
    script += ['[walktowards] <kitchencabinet> ({})'.format(nodes_cabinet[0])]*3
    script += [
        '[open] <kitchencabinet> ({})'.format(nodes_cabinet[0]),
        '[close] <kitchencabinet> ({})'.format(nodes_cabinet[0]),
    ]

    last_plate_id = plates[-1]
    script += [

        '[walktowards] <plate> ({})'.format(last_plate_id),
        '[walktowards] <plate> ({})'.format(last_plate_id),
        '[grab] <plate> ({})'.format(last_plate_id),
        '[walktowards] <table> ({})'.format(nodes_table[0]),
        '[putback] <plate> ({}) <table> ({})'.format(last_plate_id, nodes_table[0]),
    ]

    print("START")
    t0= time.time()
    comm.reset(0)
    print(t0)
    reset_time = time.time() - t0
    t0 = time.time()
    comm.add_character()
    add_char_time = time.time() - t0
    
    times['reset'].append(reset_time)
    times['char_time'].append(add_char_time)
    
    for j in range(len(script)):
        t0= time.time()
        s, g = comm.environment_graph()
        graph_time = time.time() - t0
        times['graph'].append(graph_time)
        t0= time.time()
        s, m = comm.render_script(['<char0> '+ script[j]], recording=False, skip_animation=True, gen_vid=False, image_synthesis=[])
        # print(m)
        exec_time = time.time() - t0
        action_name = script[j].split(']')[0][1:]
        times[action_name].append(exec_time)

    for k in times.keys():
        print(k, np.mean(times[k]))
    print('----')
ipdb.set_trace()
        