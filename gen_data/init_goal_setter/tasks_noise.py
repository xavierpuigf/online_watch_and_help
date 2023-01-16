
import pdb
import ipdb

def get_container_task(init_goal_manager, graph):
    # Get the containers to be used for this task
    objects = [name_obj for name_obj, count in init_goal_manager.goal.items() if count > 0]
    containers = []
    container_preds = []
    for obj in objects:
        containers += [(elem, 'inside') for elem in init_goal_manager.init_pool[obj]['in']] +  [(elem, 'on') for elem in init_goal_manager.init_pool[obj]['on']]
    containers = list(set(containers))

    

    container_id_map = {}
    container_ids = []
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

    # The containers where the random agent will be doing stuff
    containers_r = []
    container_preds_random = []
    for obj in init_goal_manager.init_pool_tasks['noise'].keys():
        info_obj = init_goal_manager.init_pool_tasks['noise'][obj]
        containers_r += [(elem, 'inside') for elem in info_obj['in']] + [(elem, 'on') for elem in info_obj['on']]


    containers_random = list(set(containers_r) - set(containers))
    container_id_random_map = {}
    container_ids_random = []
    for cont, pred in containers_random:
        cont_ids = [node['id'] for node in graph['nodes'] if (cont == node['class_name'])]
        if cont == 'sink':
            cont_ids = [ctid for ctid in cont_ids if ctid in ids_kichen]
        
        if len(cont_ids) > 0:
            cont_id = init_goal_manager.rand.choice(cont_ids)
            container_id_random_map[cont] = cont_id
            container_ids_random.append(cont_id)
            container_preds_random.append(pred)

    # For now we always palce on same thing
    container_preds_random = container_preds_random[:1]
    container_ids_random = container_ids_random[:1]

    # if 'stove' not in container_id_map:
    #     ipdb.set_trace()
    return container_ids, container_preds, container_id_map, container_ids_random, container_preds_random, container_id_random_map


def remove_objects_from_ids(init_goal_manager, graph, container_ids):
    # Remove the obejcts inside the index

    rels_cont = ['ON', 'INSIDE']
    ids_in_container = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] in container_ids and edge['relation_type'] in rels_cont]
    graph = init_goal_manager.remove_obj(graph, ids_in_container)
    return graph
    

def cleanup_graph(init_goal_manager, graph, start):
    if not start:
        return graph

    # Clean the containers where we will place stuff
    objects_clean = ["kitchentable", "dishwasher", "fridge", "stove", "microwave", "coffeetable"]
    objects_grab_clean = list(init_goal_manager.init_pool_tasks['noise'].keys())
    container_ids_clean = [node['id'] for node in graph['nodes'] if node['class_name'] in objects_clean]
    graph = remove_objects_from_ids(init_goal_manager, graph, container_ids_clean)

    # Remove the objects that have to do with the goal
    ids_obj = [node['id'] for node in graph['nodes'] if node['class_name'] in objects_grab_clean]
    graph = init_goal_manager.remove_obj(graph, ids_obj)
    id2node = {node['id']: node for node in graph['nodes']}
    # print(container_ids_clean)
    # print(ids_obj)
    # print([id2node[edge['from_id']]['class_name'] for edge in graph['edges'] if edge['to_id'] == 72 and edge['relation_type'] != 'CLOSE'])
    return graph


def build_env_goal(task_name, init_goal_manager, container_ids, container_pred, container_ids_random, container_random_pred):
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
        graph = cleanup_graph(init_goal_manager, graph, start)
        container_ids, container_pred, container_id_map, container_ids_random, container_random_pred, container_ids_random_map = get_container_task(init_goal_manager, graph)
        id2node = {node['id']: node for node in graph['nodes']}


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

            num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool[k]['env_max_num'] + 1)  # random select objects >= goal
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
                                                                objs_in_room=objs_in_room, except_position=except_position_ids,
                                                                goal_obj=True)
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                ipdb.set_trace()
                return None, None, False

        # place objects and random objects
        for k, v in init_goal_manager.goal_random_agent.items():


            num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool_tasks['noise'][k]['env_max_num'] + 1)  # random select objects >= goal
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
                                                                objs_in_room=objs_in_room, except_position=except_position_ids,
                                                                goal_obj=True)
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                ipdb.set_trace()
                return None, None, False


        if start:
            init_goal_manager.object_id_count, graph = init_goal_manager.setup_other_objs(graph, init_goal_manager.object_id_count, objs_in_room=objs_in_room,
                                                                                          except_position=except_position_ids)

        # # pdb.set_trace()
        # if 72 in container_ids:
        #     node_ids_from = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == 72 and edge['relation_type'] != 'CLOSE']
        #     id2node = {node['id']: node for node in graph['nodes']}
        #     print([id2node[idi]['class_name'] for idi in node_ids_from])
        #     ipdb.set_trace()

        assert len(container_ids) == 1
        assert len(container_ids_random) > 0

        ## get goal
        env_goal = build_env_goal("setup_table", init_goal_manager, container_ids, container_pred, container_ids_random, container_random_pred)        

        return graph, env_goal, True




    @staticmethod
    def put_dishwasher(init_goal_manager, graph, start=True):
        graph = cleanup_graph(init_goal_manager, graph, start)
        container_ids, container_pred, container_id_map, container_ids_random, container_random_pred, container_ids_random_map = get_container_task(init_goal_manager, graph)
        id2node = {node['id']: node for node in graph['nodes']}


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

            num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool[k]['env_max_num'] + 1)  # random select objects >= goal
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
                                                                objs_in_room=objs_in_room, except_position=except_position_ids,
                                                                goal_obj=True)
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                ipdb.set_trace()
                return None, None, False

        # place objects and random objects
        for k, v in init_goal_manager.goal_random_agent.items():


            num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool_tasks['noise'][k]['env_max_num'] + 1)  # random select objects >= goal
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
                                                                objs_in_room=objs_in_room, except_position=except_position_ids,
                                                                goal_obj=True)
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                ipdb.set_trace()
                return None, None, False

        # pdb.set_trace()
        if start:
            init_goal_manager.object_id_count, graph = init_goal_manager.setup_other_objs(graph, init_goal_manager.object_id_count, objs_in_room=objs_in_room,
                                                                                          except_position=except_position_ids)

        assert len(container_ids) == 1
        assert len(container_ids_random) > 0

        ## get goal
        env_goal = build_env_goal("put_dishwasher", init_goal_manager, container_ids, container_pred, container_ids_random, container_random_pred)        

        return graph, env_goal, True




    @staticmethod
    def put_fridge(init_goal_manager, graph, start=True):
        graph = cleanup_graph(init_goal_manager, graph, start)
        container_ids, container_pred, container_id_map, container_ids_random, container_random_pred, container_ids_random_map = get_container_task(init_goal_manager, graph)
        id2node = {node['id']: node for node in graph['nodes']}


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

            num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool[k]['env_max_num'] + 1)  # random select objects >= goal
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
                                                                objs_in_room=objs_in_room, except_position=except_position_ids,
                                                                goal_obj=True)
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                ipdb.set_trace()
                return None, None, False

        # place objects and random objects
        for k, v in init_goal_manager.goal_random_agent.items():


            num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool_tasks['noise'][k]['env_max_num'] + 1)  # random select objects >= goal
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
                                                                objs_in_room=objs_in_room, except_position=except_position_ids,
                                                                goal_obj=True)
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                ipdb.set_trace()
                return None, None, False

        # pdb.set_trace()
        if start:
            init_goal_manager.object_id_count, graph = init_goal_manager.setup_other_objs(graph, init_goal_manager.object_id_count, objs_in_room=objs_in_room,
                                                                                          except_position=except_position_ids)

        assert len(container_ids) == 1
        assert len(container_ids_random) > 0

        ## get goal
        env_goal = build_env_goal("put_fridge", init_goal_manager, container_ids, container_pred, container_ids_random, container_random_pred)        

        return graph, env_goal, True


    @staticmethod
    def prepare_food(init_goal_manager, graph, start=True):
        graph = cleanup_graph(init_goal_manager, graph, start)
        container_ids, container_pred, container_id_map, container_ids_random, container_random_pred, container_ids_random_map = get_container_task(init_goal_manager, graph)
        id2node = {node['id']: node for node in graph['nodes']}


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

            num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool[k]['env_max_num'] + 1)  # random select objects >= goal
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
                                                                objs_in_room=objs_in_room, except_position=except_position_ids,
                                                                goal_obj=True)
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                ipdb.set_trace()
                return None, None, False

        # place objects and random objects
        for k, v in init_goal_manager.goal_random_agent.items():


            num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool_tasks['noise'][k]['env_max_num'] + 1)  # random select objects >= goal
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
                                                                objs_in_room=objs_in_room, except_position=except_position_ids,
                                                                goal_obj=True)
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                ipdb.set_trace()
                return None, None, False

        # pdb.set_trace()
        if start:
            init_goal_manager.object_id_count, graph = init_goal_manager.setup_other_objs(graph, init_goal_manager.object_id_count, objs_in_room=objs_in_room,
                                                                                          except_position=except_position_ids)

        assert len(container_ids) == 1
        assert len(container_ids_random) > 0

        ## get goal
        env_goal = build_env_goal("prepare_food", init_goal_manager, container_ids, container_pred, container_ids_random, container_random_pred)        

        return graph, env_goal, True



