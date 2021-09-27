
import pdb
import ipdb
import copy

def get_container_task(init_goal_manager, graph, containers):

    containers = list(set(containers))

    

    container_id_map = {}
    container_ids = []
    container_preds = []
    kitchen = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchen'][0]
    ids_kichen = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == kitchen and edge['relation_type'] == 'INSIDE']
    
    for cont, pred in containers:
        cont_ids = [node['id'] for node in graph['nodes'] if (cont == node['class_name'])]
        if cont == 'sink':
            cont_ids = [ctid for ctid in cont_ids if ctid in ids_kichen]
        if len(cont_ids) == 0:
            ipdb.set_trace()
        cont_id = init_goal_manager.rand.choice(cont_ids)
        container_id_map[cont] = cont_id
        container_ids.append(cont_id)
        container_preds.append(pred)

    
    if len(container_ids) == 0:
        ipdb.set_trace()



    # if 'stove' not in container_id_map:
    #     ipdb.set_trace()
    return container_ids, container_preds, container_id_map


def remove_objects_from_ids(init_goal_manager, graph, container_ids, rel_dict):
    # Remove the obejcts inside the index

    rels_cont = ['ON', 'INSIDE']
    ids_in_container = []
    for edge in graph['edges']:
        if edge['to_id'] in container_ids:
            curr_rel = rel_dict[edge['to_id']]
            if edge['relation_type'] in curr_rel:
                ids_in_container.append(edge['from_id'])

    # ids_in_container = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] in container_ids and edge['relation_type'] in rels_cont]
    graph = init_goal_manager.remove_obj(graph, ids_in_container)
    return graph
    

def cleanup_graph(init_goal_manager, graph, start):
    if not start:
        return graph

    # ipdb.set_trace()
    # Clean the containers where we will place stuff
    objects_rel_clean = [("kitchentable", ["ON"]), ("dishwasher", ["INSIDE"]), ("fridge", ["INSIDE"]), 
                         ("stove", ["INSIDE"]), ("microwave", ["INSIDE"]), ("coffeetable", ["ON"])]
    objects_clean = [x[0] for x in objects_rel_clean]
    rel_class_dict = {x[0]: x[1] for x in objects_rel_clean}

    objects_grab_clean = list(init_goal_manager.init_pool_tasks['obj_random'])


    container_ids_clean = []
    rel_dict = {}
    for node in graph['nodes']:
        if node['id'] in objects_clean:
            rel_dict[node['id']] = rel_class_dict[node['class_name']]
    graph = remove_objects_from_ids(init_goal_manager, graph, container_ids_clean, rel_dict)

    # Remove the objects that have to do with the goal
    ids_obj = [node['id'] for node in graph['nodes'] if node['class_name'] in objects_grab_clean]
    graph = init_goal_manager.remove_obj(graph, ids_obj)
    id2node = {node['id']: node for node in graph['nodes']}
    # print(container_ids_clean)
    # ipdb.set_trace()
    # print(ids_obj)
    # print([id2node[edge['from_id']]['class_name'] for edge in graph['edges'] if edge['to_id'] == 72 and edge['relation_type'] != 'CLOSE'])
    return graph


def build_env_goal(task_name, init_goal_manager, container_ids, container_pred=[], container_ids_random=[], container_random_pred=[]):
    env_goal = {task_name: []}
    for k, v in init_goal_manager.goal.items():
        env_goal[task_name].append({'put_{}_{}_{}'.format(k, container_pred[0], container_ids[0]): v})

    ## get goal
    env_goal['noise'] = []
    for k, v in init_goal_manager.goal_random_agent.items():
        env_goal['noise'].append({'put_{}_{}_{}'.format(k, container_random_pred[0], container_ids_random[0]): v})
    return env_goal

class Task:




    @staticmethod
    def setup_table(init_goal_manager, graph, start=True):
        # ipdb.set_trace()

        # Make sure candidates are available
        candidates = init_goal_manager.init_pool['candidates']
        
        class_names = [node['class_name'] for node in graph['nodes']]
        
        candidates = [cand for cand in candidates if cand[0] in class_names]
        container_name, pred_name = init_goal_manager.rand.choice(candidates)
        min_count, max_count = init_goal_manager.init_pool['counts']['min'],  init_goal_manager.init_pool['counts']['max']
        
        pr_graph = copy.deepcopy(graph)
        graph = cleanup_graph(init_goal_manager, graph, start)
        print([node['id'] for node in pr_graph['nodes'] if node['class_name'] == 'kitchencounter'])
        print([node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchencounter'])
        # ipdb.set_trace()

        container_ids, container_pred, container_id_map = get_container_task(init_goal_manager, graph, [(container_name, pred_name)])

        container_ids_random, container_random_pred, container_ids_random_map = [], [], {}

        id2node = {node['id']: node for node in graph['nodes']}

        # for how many is the table
        counts_objects = init_goal_manager.rand.randint(min_count, max_count)
        
        init_goal_manager.goal = {}
        object_dict = init_goal_manager.init_pool['objects']

        extra_object = init_goal_manager.rand.choice(["wineglass", "waterglass"])
        objects_select = [extra_object] + ["plate", "cutleryfork"]
        for object_name in objects_select:
            init_goal_manager.goal[object_name] = counts_objects



        if len(container_ids) == 0:
            ipdb.set_trace()

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(container_ids[0])
        else:
            objs_in_room = None

        except_position_ids = [node['id'] for node in graph['nodes'] if ('floor' in node['class_name'])]
        except_position_ids += container_ids + container_ids_random

        # place objects and random objects
        for k, v in init_goal_manager.goal.items():
            # obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            # graph = init_goal_manager.remove_obj(graph, obj_ids)

            num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool['objects'][k]['env_max_num'] + 1)  # random select objects >= goal
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
                                                                objs_in_room=objs_in_room, except_position=except_position_ids,
                                                                goal_obj=True)
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                # ipdb.set_trace()
                return None, None, False

        # # place objects and random objects
        # for k, v in init_goal_manager.goal_random_agent.items():


        #     num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool_tasks['noise'][k]['env_max_num'] + 1)  # random select objects >= goal
        #     init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
        #                                                         objs_in_room=objs_in_room, except_position=except_position_ids,
        #                                                         goal_obj=True)
        #     # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
        #     if not success:
        #         ipdb.set_trace()
        #         return None, None, False


        if start:
            init_goal_manager.object_id_count, graph = init_goal_manager.setup_other_objs(graph, init_goal_manager.object_id_count, objs_in_room=objs_in_room,
                                                                                          except_position=except_position_ids)

        # ipdb.set_trace()
        # # pdb.set_trace()
        # if 72 in container_ids:
        #     node_ids_from = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == 72 and edge['relation_type'] != 'CLOSE']
        #     id2node = {node['id']: node for node in graph['nodes']}
        #     print([id2node[idi]['class_name'] for idi in node_ids_from])
        #     ipdb.set_trace()

        assert len(container_ids) == 1
        # assert len(container_ids_random) > 0

        ## get goal
        env_goal = build_env_goal("setup_table", init_goal_manager, container_ids, container_pred, container_ids_random, container_random_pred)        

        return graph, env_goal, True




    @staticmethod
    def put_dishwasher(init_goal_manager, graph, start=True):
        candidates = init_goal_manager.init_pool['candidates']
        
        class_names = [node['class_name'] for node in graph['nodes']]
        candidates = [cand for cand in candidates if cand[0] in class_names]
        container_name, pred_name = init_goal_manager.rand.choice(candidates)


        graph = cleanup_graph(init_goal_manager, graph, start)
        container_ids, container_pred, container_id_map = get_container_task(init_goal_manager, graph, [(container_name, pred_name)])

        container_ids_random, container_random_pred, container_ids_random_map = [], [], {}

        id2node = {node['id']: node for node in graph['nodes']}


        object_candidates = ["plate", "waterglass", "wineglass", "cutleryfork"]
        different_classes = init_goal_manager.rand.randint(1,len(object_candidates))
        objects_selected = init_goal_manager.rand.choices(object_candidates, k=different_classes)
        how_many_objects = init_goal_manager.rand.randint(3,7)
        all_object_pool = []
        for obj_name in objects_selected:
            all_object_pool += [obj_name] * how_many_objects
        # Get a list of the number of objecs we want to add
        obj_final = init_goal_manager.rand.choices(all_object_pool, k=how_many_objects)

        init_goal_manager.goal = {}
        for ob_name in obj_final:
            if not ob_name in init_goal_manager.goal.keys():
                init_goal_manager.goal[ob_name] = 0
            init_goal_manager.goal[ob_name] += 1
                
        

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(container_ids[0])
        else:
            objs_in_room = None

        except_position_ids = [node['id'] for node in graph['nodes'] if ('floor' in node['class_name'])]
        except_position_ids += container_ids + container_ids_random

        # place objects and random objects
        for k, v in init_goal_manager.goal.items():
            # obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            # graph = init_goal_manager.remove_obj(graph, obj_ids)

            num_obj = init_goal_manager.rand.randint(v, v)  # random select objects >= goal
            try:
                init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
                                                                    objs_in_room=objs_in_room, except_position=except_position_ids,
                                                                    goal_obj=True)
            except:
                ipdb.set_trace()
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                ipdb.set_trace()
                return None, None, False



        # pdb.set_trace()
        if start:
            init_goal_manager.object_id_count, graph = init_goal_manager.setup_other_objs(graph, init_goal_manager.object_id_count, objs_in_room=objs_in_room,
                                                                                          except_position=except_position_ids, except_objects=object_candidates)

        assert len(container_ids) == 1

        ## get goal
        env_goal = build_env_goal("put_dishwasher", init_goal_manager, container_ids, container_pred)        

        return graph, env_goal, True



    @staticmethod
    def put_fridge(init_goal_manager, graph, start=True):
        
        candidates = init_goal_manager.init_pool['candidates']
        
        class_names = [node['class_name'] for node in graph['nodes']]
        candidates = [cand for cand in candidates if cand[0] in class_names]
        container_name, pred_name = init_goal_manager.rand.choice(candidates)


        graph = cleanup_graph(init_goal_manager, graph, start)
        container_ids, container_pred, container_id_map = get_container_task(init_goal_manager, graph, [(container_name, pred_name)])

        container_ids_random, container_random_pred, container_ids_random_map = [], [], {}

        id2node = {node['id']: node for node in graph['nodes']}


        object_candidates = ["pudding", "cupcake", "salmon", "apple"]
        different_classes = init_goal_manager.rand.randint(1,len(object_candidates))
        objects_selected = init_goal_manager.rand.choices(object_candidates, k=different_classes)
        how_many_objects = init_goal_manager.rand.randint(3,7)
        all_object_pool = []
        for obj_name in objects_selected:
            all_object_pool += [obj_name] * how_many_objects
        # Get a list of the number of objecs we want to add
        obj_final = init_goal_manager.rand.choices(all_object_pool, k=how_many_objects)

        init_goal_manager.goal = {}
        for ob_name in obj_final:
            if not ob_name in init_goal_manager.goal.keys():
                init_goal_manager.goal[ob_name] = 0
            init_goal_manager.goal[ob_name] += 1
                
        

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(container_ids[0])
        else:
            objs_in_room = None

        except_position_ids = [node['id'] for node in graph['nodes'] if ('floor' in node['class_name'])]
        except_position_ids += container_ids + container_ids_random

        # place objects and random objects
        for k, v in init_goal_manager.goal.items():
            # obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            # graph = init_goal_manager.remove_obj(graph, obj_ids)

            num_obj = init_goal_manager.rand.randint(v, v)  # random select objects >= goal
            try:
                init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
                                                                    objs_in_room=objs_in_room, except_position=except_position_ids,
                                                                    goal_obj=True)
            except:
                ipdb.set_trace()
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                ipdb.set_trace()
                return None, None, False



        # pdb.set_trace()
        if start:
            init_goal_manager.object_id_count, graph = init_goal_manager.setup_other_objs(graph, init_goal_manager.object_id_count, objs_in_room=objs_in_room,
                                                                                          except_position=except_position_ids, except_objects=object_candidates)

        assert len(container_ids) == 1

        ## get goal
        env_goal = build_env_goal("put_fridge", init_goal_manager, container_ids, container_pred)        

        return graph, env_goal, True


    @staticmethod
    def prepare_food(init_goal_manager, graph, start=True):

        candidates = init_goal_manager.init_pool['candidates']
        
        class_names = [node['class_name'] for node in graph['nodes']]
        candidates = [cand for cand in candidates if cand[0] in class_names]
        container_name, pred_name = init_goal_manager.rand.choice(candidates)
        min_count, max_count = init_goal_manager.init_pool['counts']['min'],  init_goal_manager.init_pool['counts']['max']
        

        graph = cleanup_graph(init_goal_manager, graph, start)
        container_ids, container_pred, container_id_map = get_container_task(init_goal_manager, graph, [(container_name, pred_name)])

        container_ids_random, container_random_pred, container_ids_random_map = [], [], {}

        id2node = {node['id']: node for node in graph['nodes']}

        # for how many is the table
        counts_objects = init_goal_manager.rand.randint(min_count, max_count)
        
        init_goal_manager.goal = {}
        object_dict = init_goal_manager.init_pool['objects']

        extra_object = init_goal_manager.rand.choice(["cupcake", "pudding"])
        objects_select = [extra_object] + ["salmon", "apple"]
        for object_name in objects_select:
            init_goal_manager.goal[object_name] = counts_objects


        if len(container_ids) == 0:
            ipdb.set_trace()

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(container_ids[0])
        else:
            objs_in_room = None

        except_position_ids = [node['id'] for node in graph['nodes'] if ('floor' in node['class_name'])]
        except_position_ids += container_ids + container_ids_random

        # place objects and random objects
        for k, v in init_goal_manager.goal.items():
            # obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            # graph = init_goal_manager.remove_obj(graph, obj_ids)

            num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool['objects'][k]['env_max_num'] + 1)  # random select objects >= goal
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
                                                                objs_in_room=objs_in_room, except_position=except_position_ids,
                                                                goal_obj=True)
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                ipdb.set_trace()
                return None, None, False

        # # place objects and random objects
        # for k, v in init_goal_manager.goal_random_agent.items():


        #     num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool_tasks['noise'][k]['env_max_num'] + 1)  # random select objects >= goal
        #     init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
        #                                                         objs_in_room=objs_in_room, except_position=except_position_ids,
        #                                                         goal_obj=True)
        #     # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
        #     if not success:
        #         ipdb.set_trace()
        #         return None, None, False


        if start:
            init_goal_manager.object_id_count, graph = init_goal_manager.setup_other_objs(graph, init_goal_manager.object_id_count, objs_in_room=objs_in_room,
                                                                                          except_position=except_position_ids)

        # ipdb.set_trace()
        # # pdb.set_trace()
        # if 72 in container_ids:
        #     node_ids_from = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == 72 and edge['relation_type'] != 'CLOSE']
        #     id2node = {node['id']: node for node in graph['nodes']}
        #     print([id2node[idi]['class_name'] for idi in node_ids_from])
        #     ipdb.set_trace()

        assert len(container_ids) == 1
        # assert len(container_ids_random) > 0

        ## get goal
        env_goal = build_env_goal("prepare_food", init_goal_manager, container_ids, container_pred, container_ids_random, container_random_pred)        

        return graph, env_goal, True


    @staticmethod
    def watch_tv(init_goal_manager, graph, start=True):

        candidates = init_goal_manager.init_pool['candidates']
        
        class_names = [node['class_name'] for node in graph['nodes']]
        candidates = [cand for cand in candidates if cand[0] in class_names]
        container_name, pred_name = init_goal_manager.rand.choice(candidates)
        min_count, max_count = init_goal_manager.init_pool['counts']['min'],  init_goal_manager.init_pool['counts']['max']
        

        graph = cleanup_graph(init_goal_manager, graph, start)
        container_ids, container_pred, container_id_map = get_container_task(init_goal_manager, graph, [(container_name, pred_name)])

        container_ids_random, container_random_pred, container_ids_random_map = [], [], {}

        id2node = {node['id']: node for node in graph['nodes']}

        # for how many is the table
        counts_objects = init_goal_manager.rand.randint(min_count, max_count)
        
        init_goal_manager.goal = {}
        object_dict = init_goal_manager.init_pool['objects']

        for object_name in object_dict.keys():
            init_goal_manager.goal[object_name] = counts_objects



        if len(container_ids) == 0:
            ipdb.set_trace()

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(container_ids[0])
        else:
            objs_in_room = None

        except_position_ids = [node['id'] for node in graph['nodes'] if ('floor' in node['class_name'])]
        except_position_ids += container_ids + container_ids_random

        # place objects and random objects
        init_graph = graph
        for k, v in init_goal_manager.goal.items():
            # obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            # graph = init_goal_manager.remove_obj(graph, obj_ids)

            num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool['objects'][k]['env_max_num'] + 1)  # random select objects >= goal
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
                                                                objs_in_room=objs_in_room, except_position=except_position_ids,
                                                                goal_obj=True)
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                ipdb.set_trace()
                return None, None, False

        # # place objects and random objects
        # for k, v in init_goal_manager.goal_random_agent.items():


        #     num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool_tasks['noise'][k]['env_max_num'] + 1)  # random select objects >= goal
        #     init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
        #                                                         objs_in_room=objs_in_room, except_position=except_position_ids,
        #                                                         goal_obj=True)
        #     # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
        #     if not success:
        #         ipdb.set_trace()
        #         return None, None, False


        if start:
            init_goal_manager.object_id_count, graph = init_goal_manager.setup_other_objs(graph, init_goal_manager.object_id_count, objs_in_room=objs_in_room,
                                                                                          except_position=except_position_ids)

        # ipdb.set_trace()
        # # pdb.set_trace()
        # if 72 in container_ids:
        #     node_ids_from = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == 72 and edge['relation_type'] != 'CLOSE']
        #     id2node = {node['id']: node for node in graph['nodes']}
        #     print([id2node[idi]['class_name'] for idi in node_ids_from])
        #     ipdb.set_trace()

        assert len(container_ids) == 1
        # assert len(container_ids_random) > 0

        ## get goal
        env_goal = build_env_goal("watch_tv", init_goal_manager, container_ids, container_pred, container_ids_random, container_random_pred)        

        return graph, env_goal, True


