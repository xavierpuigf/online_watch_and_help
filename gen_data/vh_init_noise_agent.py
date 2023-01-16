import pickle
import pdb
import ipdb
import sys
import os
import random
import json
import numpy as np
import copy
import argparse

curr_dir = os.path.dirname(os.path.abspath(__file__))
home_path = '../../'
sys.path.insert(0, f'{curr_dir}/../../virtualhome/')

print('path', sys.path[0])

sys.path.insert(0, f'{curr_dir}/..')
from simulation.unity_simulator import comm_unity
print(comm_unity.__file__)
from init_goal_setter.init_goal_base import SetInitialGoal
from init_goal_setter.tasks_noise import Task


from utils import utils_goals

parser = argparse.ArgumentParser()
parser.add_argument('--num-per-apartment', type=int, default=2, help='Maximum #episodes/apartment')
parser.add_argument('--seed', type=int, default=10, help='Seed for the apartments')

parser.add_argument('--split', type=str, default='train', help='split')
parser.add_argument('--task', type=str, default='all', help='Task name')
parser.add_argument('--apt_str', type=str, default='0,1,2,3,4', help='The apartments where we will generate the data')
parser.add_argument('--port', type=str, default='8092', help='Task name')
parser.add_argument('--display', type=int, default=0, help='Task name')
parser.add_argument('--mode', type=str, default='full', choices=['simple', 'full'], help='Task name')
parser.add_argument('--use-editor', action='store_true', default=False, help='Use unity editor')
parser.add_argument('--exec_file', type=str,
        default='/data/vision/torralba/frames/data_acquisition/SyntheticStories/website/release/simulator/v2.0/v2.2.5_beta4/linux_exec.v2.2.5_beta4.x86_64',
                    help='Use unity editor')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.seed == 0:
        rand = random.Random()
    else:
        rand = random.Random(args.seed)


    with open(f'{curr_dir}/data/init_pool_noise.json') as file:
        init_pool = json.load(file)
    # comm = comm_unity.UnityCommunication()
    if args.use_editor:
        comm = comm_unity.UnityCommunication()
    else:
        print(comm_unity)
        comm = comm_unity.UnityCommunication(port=args.port,
                                             file_name=args.exec_file,
                                             no_graphics=True,
                                             logging=False,
                                             x_display=args.display)
    comm.reset()

    ## -------------------------------------------------------------
    ## step3 load object size
    with open(f'{curr_dir}/data/class_name_size.json', 'r') as file:
        class_name_size = json.load(file)

    ## -------------------------------------------------------------
    ## gen graph
    ## -------------------------------------------------------------
    task_names = {
        1: ["setup_table", "put_fridge", "prepare_food"],
        2: ["setup_table", "put_fridge", "prepare_food", "put_dishwasher"],
        3: ["setup_table", "put_fridge", "prepare_food", "put_dishwasher"],
        4: ["setup_table", "put_fridge", "prepare_food", "put_dishwasher"],
        5: ["setup_table", "put_fridge", "prepare_food", "put_dishwasher"],
        6: ["setup_table", "put_fridge", "prepare_food"],
        7: ["setup_table", "put_fridge", "prepare_food", "put_dishwasher"]
    }

    bad_containers = {
        '1': ['dishwasher'],
        '6': ['dishwasher']
    }

    success_init_graph = []

    apartment_ids = [int(apt_id) for apt_id in args.apt_str.split(',')]
    if args.task == 'all':
        tasks =  ["setup_table", "put_fridge", "prepare_food", "put_dishwasher"]
    else:
        raise Exception

    num_per_apartment = args.num_per_apartment

    for task in tasks:
        # for apartment in range(6,7):
        for apartment in apartment_ids:
            print('apartment', apartment)

            if 'toy' not in task and task not in task_names[apartment + 1]: continue
            # if apartment != 4: continue
            # apartment = 3

            with open(f'{curr_dir}/data/object_info_final.json', 'r') as file:
                obj_position = json.load(file)

            # pdb.set_trace()bathroomcounter

            # filtering out certain locations
            old_obj_position = copy.deepcopy(obj_position)
            for obj, pos_list in old_obj_position.items():
                positions = [pos for pos in pos_list if pos[1] != 'kitchencounter']
                pos_list = positions
                if obj in ['book', 'remotecontrol']:
                    positions = [pos for pos in pos_list if \
                                 pos[0] == 'INSIDE' and pos[1] in ['kitchencabinet', 'cabinet'] or \
                                 pos[0] == 'ON' and pos[1] in \
                                 (['cabinet', 'bench', 'nightstand'] + ([] if apartment == 2 else ['kitchentable']))]
                elif obj == 'remotecontrol':
                    # TODO: we never get here
                    positions = [pos for pos in pos_list if pos[0] == 'ON' and pos[1] in \
                                 ['tvstand']]
                else:
                    positions = [pos for pos in pos_list if \
                                 pos[0] == 'INSIDE' and pos[1] in ['fridge', 'kitchencabinet', 'cabinet', 'microwave',
                                                                   'dishwasher', 'stove'] or \
                                 pos[0] == 'ON' and pos[1] in \
                                 (['cabinet', 'coffeetable', 'bench', 'kitchencounter', 'sofa'] + ['kitchentable'])]
                obj_position[obj] = positions
            

            num_test = 100000
            count_success = 0
            for i in range(num_test):
                comm.reset(apartment)
                s, original_graph = comm.environment_graph()
                graph = copy.deepcopy(original_graph)

                task_name = task

                print('------------------------------------------------------------------------------')
                print('testing %d/%d: %s. apartment %d' % (i, num_test, task_name, apartment))
                print('------------------------------------------------------------------------------')

                ## -------------------------------------------------------------
                ## setup goal based on currect environment
                ## -------------------------------------------------------------
                set_init_goal = SetInitialGoal(obj_position, class_name_size, init_pool, 
                                               task_name, same_room=False, rand=rand, set_random_goal=True)
                

                task_name_red = task_name
                if 'toy' in task_name:
                    task_name_red = task_name.replace('_1', '').replace('_2', '')


                init_graph, env_goal, success_setup = getattr(Task, task_name_red)(set_init_goal, graph)
                env_goal_key = list(env_goal[task_name][0].keys())[0]
                

                # ipdb.set_trace()
                # if env_goal is None:
                #     pdb.set_trace()
                if success_setup:
                    # If all objects were well added
                    success, message = comm.expand_scene(init_graph, transfer_transform=False)
                    print('----------------------------------------------------------------------')
                    print(task_name, success, message, set_init_goal.num_other_obj)
                    # print(env_goal)
                    if not success:
                        goal_objs = []
                        goal_names = []
                        for k, goals in env_goal.items():
                            goal_objs += [int(list(goal.keys())[0].split('_')[-1]) for goal in goals if
                                          list(goal.keys())[0].split('_')[-1] not in ['book', 'remotecontrol']]
                            goal_names += [list(goal.keys())[0].split('_')[1] for goal in goals]
                        print(message)
                        obj_names = [obj.split('.')[0] for obj in message['unplaced']]
                        obj_ids = [int(obj.split('.')[1]) for obj in message['unplaced']]
                        id2node = {node['id']: node for node in init_graph['nodes']}

                        for obj_id in obj_ids:
                            print("Objects unplaced")
                            print([id2node[edge['to_id']]['class_name'] for edge in init_graph['edges'] if
                                   edge['from_id'] == obj_id])
                            ipdb.set_trace()
                        if task_name != 'read_book' and task_name != 'watch_tv':
                            intersection = set(obj_names) & set(goal_names)
                        else:
                            intersection = set(obj_ids) & set(goal_objs)

                        ## goal objects cannot be placed



                        if len(intersection) != 0:
                            success2 = False
                        else:
                            init_graph = set_init_goal.remove_obj(init_graph, obj_ids)
                            comm.reset(apartment)
                            success2, message2 = comm.expand_scene(init_graph, transfer_transform=False)
                            success = True

                    else:
                        success2 = True

                    if success2 and success:


                        success = set_init_goal.check_goal_achievable(init_graph, comm, env_goal, apartment)

                            # ipdb.set_trace()
                        if success:
                            init_graph0 = copy.deepcopy(init_graph)
                            comm.reset(apartment)
                            comm.expand_scene(init_graph)
                            s, init_graph = comm.environment_graph()
                            print('final s:', s)
                            # ipdb.set_trace()
                            if s:

                                for subgoal in env_goal[task_name_red]:
                                    for k, v in subgoal.items():
                                        elements = k.split('_')
                                        # print(elements)
                                        # pdb.set_trace()
                                        if len(elements) == 4:
                                            obj_class_name = elements[1]
                                            ids = [node['id'] for node in init_graph['nodes'] if
                                                   node['class_name'] == obj_class_name]
                                            print(obj_class_name, v, ids)


                                count_success += s
                                check_result = set_init_goal.check_graph(init_graph, apartment + 1, original_graph)
                                assert check_result == True

                                # ipdb.set_trace()
                                env_goal_key = list(env_goal[task_name][0].keys())[0]
                                #if '72' in env_goal_key:
                                #    node_ids_from = [edge['from_id'] for edge in init_graph['edges'] if edge['to_id'] == 72 and edge['relation_type'] != 'CLOSE']
                                #    id2node = {node['id']: node for node in init_graph['nodes']}
                                #    print([id2node[idi]['class_name'] for idi in node_ids_from])
                                #    node_ids_from = [edge['from_id'] for edge in init_graph0['edges'] if edge['to_id'] == 72 and edge['relation_type'] != 'CLOSE']

                                #    id2node = {node['id']: node for node in init_graph0['nodes']}
                                #    #print([id2node[idi]['class_name'] for idi in node_ids_from])
                                #    #ipdb.set_trace()
                                success_init_graph.append({'id': count_success,
                                                           'apartment': (apartment + 1),
                                                           'task_name': task_name,
                                                           'init_graph': init_graph,
                                                           'original_graph': original_graph,
                                                           'goal': env_goal})
                else:
                    pass
                    # pdb.set_trace()
                print('apartment: %d: success %d over %d (total: %d)' % (apartment, count_success, i + 1, num_test))
                if count_success >= num_per_apartment:
                    break

    
    data = success_init_graph
    env_task_set = []

    # for task in ['setup_table', 'put_fridge', 'put_dishwasher', 'prepare_food', 'read_book']:
        
    for task_id, problem_setup in enumerate(data):
        env_id = problem_setup['apartment'] - 1
        task_name = problem_setup['task_name']
        task_name_red = task_name
        if 'toy' in task_name:
            task_name_red = task_name.replace('_1', '').replace('_2', '')
        init_graph = problem_setup['init_graph']
        goal = problem_setup['goal'][task_name_red]

        goals = utils_goals.convert_goal_spec(task_name, goal, init_graph,
                                              exclude=['cutleryknife'])

        goal_noise = problem_setup['goal']['noise']

        goals_noise = utils_goals.convert_goal_spec('noise', goal_noise, init_graph,
                                              exclude=['cutleryknife'])
        print('env_id:', env_id)
        print('task_name:', task_name)
        print('goals:', goals)

        task_goal = {}
        task_goal[0] = goals
        task_goal[1] = goals_noise


        env_task_set.append({'task_id': task_id, 'task_name': task_name, 'env_id': env_id, 'init_graph': init_graph,
                             'task_goal': task_goal,
                             'level': 0, 'init_rooms': rand.sample(['kitchen', 'bedroom', 'livingroom', 'bathroom'], 2)})

    pickle.dump(env_task_set, open(f'{curr_dir}/../dataset/third_agent/{args.split}_env_task_set_{args.num_per_apartment}_{args.mode}.pik', 'wb'))





