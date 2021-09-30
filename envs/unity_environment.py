from utils import utils_environment as utils
import math
import sys
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{curr_dir}/../../virtualhome/simulation/')
sys.path.append(f'{curr_dir}/..')

from environment.unity_environment import UnityEnvironment as BaseUnityEnvironment
from evolving_graph import utils as utils_env
from utils import utils_environment as utils_env2
import pdb
import traceback
import numpy as np
from scipy.spatial.transform import Rotation as R

import copy
import ipdb


class UnityEnvironment(BaseUnityEnvironment):
    def __init__(
        self,
        num_agents=2,
        max_episode_length=200,
        env_task_set=None,
        observation_types=None,
        agent_goals=None,
        use_editor=False,
        base_port=8080,
        convert_goal=False,
        port_id=0,
        executable_args={},
        recording_options={
            'recording': False,
            'output_folder': None,
            'file_name_prefix': None,
            'cameras': 'PERSON_FROM_BACK',
            'modality': 'normal',
        },
        seed=123,
    ):

        if agent_goals is not None:
            self.agent_goals = agent_goals
        else:
            self.agent_goals = ['full' for _ in range(num_agents)]

        self.convert_goal = convert_goal
        self.task_goal, self.goal_spec = {0: {}, 1: {}}, {0: {}, 1: {}}
        self.env_task_set = env_task_set
        self.agent_object_touched = []
        super(UnityEnvironment, self).__init__(
            num_agents=num_agents,
            max_episode_length=max_episode_length,
            observation_types=observation_types,
            use_editor=use_editor,
            base_port=base_port,
            port_id=port_id,
            executable_args=executable_args,
            recording_options=recording_options,
            seed=seed,
        )
        self.full_graph = None



    def get_graph(self):
        graph = super(UnityEnvironment, self).get_graph()
        objects_seen = self.agent_object_touched
        for node in graph['nodes']:
            if (
                'TOUCHED' not in [st.upper() for st in node['states']]
                and node['id'] in objects_seen
            ):
                node['states'].append('TOUCHED')
        return graph

    def reward(self):
        reward = 0.0
        done = True
        # print(self.goal_spec)
        if self.convert_goal:
            satisfied, unsatisfied = utils.check_progress2(
                self.get_graph(), self.goal_spec[0]
            )

        else:
            satisfied, unsatisfied = utils.check_progress(
                self.get_graph(), self.goal_spec[0]
            )
        for key, value in satisfied.items():
            if self.convert_goal:
                resp = self.goal_spec[0][key]
                preds_needed, mandatory, reward_per_pred = (
                    resp['count'],
                    resp['final'],
                    resp['reward'],
                )
            else:
                preds_needed, mandatory, reward_per_pred = self.goal_spec[0][key]
            # How many predicates achieved
            value_pred = min(len(value), preds_needed)
            reward += value_pred * reward_per_pred
            if self.convert_goal:
                if mandatory and unsatisfied[key] > 0:
                    done = False
            else:
                if mandatory and unsatisfied[key] > 0:
                    done = False

        self.prev_reward = reward
        return reward, done, {'satisfied_goals': satisfied}

    def get_goal2(self, task_spec, agent_goal):
        if agent_goal == 'full':
            # pred = [x for x, y in task_spec.items() if y['count'] > 0 and x.split('_')[0] in ['on', 'inside']]
            # object_grab = [pr.split('_')[1] for pr in pred]
            # predicates_grab = {'holds_{}_1'.format(obj_gr): [1, False, 2] for obj_gr in object_grab}
            res_dict = {
                goal_k: copy.deepcopy(goal_c)
                for goal_k, goal_c in task_spec.items()
                if goal_c['count'] > 0
            }
            for goal_k, goal_dict in res_dict.items():
                goal_dict.update({'final': True, 'reward': 2})
            # res_dict.update(predicates_grab)
            return res_dict
        elif agent_goal == 'grab':
            candidates = [
                x.split('_')[1]
                for x, y in task_spec.items()
                if y > 0 and x.split('_')[0] in ['on', 'inside']
            ]
            object_grab = self.rnd.choice(candidates)
            # print('GOAL', candidates, object_grab)
            return {
                'holds_'
                + object_grab
                + '_'
                + '1': {
                    'count': 1,
                    'final': True,
                    'reward': 10,
                    'grab_obj_ids': object_grab,
                    'container_ids': [1],
                },
                'close_'
                + object_grab
                + '_'
                + '1': {
                    'count': 1,
                    'final': False,
                    'reward': 0.1,
                    'grab_obj_ids': object_grab,
                    'container_ids': [1],
                },
            }
        elif agent_goal == 'put':
            pred = self.rnd.choice(
                [
                    (x, y)
                    for x, y in task_spec.items()
                    if y['count'] > 0 and x.split('_')[0] in ['on', 'inside']
                ]
            )
            object_grab = [pred[0].split('_')[1]]
            ctid = pred[1]['container_ids']
            return {
                pred: {
                    'count': 1,
                    'final': True,
                    'reward': 60,
                    'grab_obj_ids': object_grab,
                    'container_ids': ctid,
                },
                'holds_'
                + object_grab
                + '_'
                + '1': {
                    'count': 1,
                    'final': False,
                    'reward': 2,
                    'grab_obj_ids': object_grab,
                    'container_ids': [1],
                },
                'close_'
                + object_grab
                + '_'
                + '1': {
                    'count': 1,
                    'final': False,
                    'reward': 0.05,
                    'grab_obj_ids': object_grab,
                    'container_ids': [1],
                },
            }
        else:
            raise NotImplementedError

    def get_goal(self, task_spec, agent_goal):
        if agent_goal == 'full':
            pred = [
                x
                for x, y in task_spec.items()
                if y > 0 and x.split('_')[0] in ['on', 'inside']
            ]
            # object_grab = [pr.split('_')[1] for pr in pred]
            # predicates_grab = {'holds_{}_1'.format(obj_gr): [1, False, 2] for obj_gr in object_grab}
            res_dict = {
                goal_k: [goal_c, True, 2] for goal_k, goal_c in task_spec.items()
            }
            # res_dict.update(predicates_grab)
            return res_dict
        elif agent_goal == 'grab':
            candidates = [
                x.split('_')[1]
                for x, y in task_spec.items()
                if y > 0 and x.split('_')[0] in ['on', 'inside']
            ]
            object_grab = self.rnd.choice(candidates)
            # print('GOAL', candidates, object_grab)
            return {
                'holds_' + object_grab + '_' + '1': [1, True, 10],
                'close_' + object_grab + '_' + '1': [1, False, 0.1],
            }
        elif agent_goal == 'put':
            pred = self.rnd.choice(
                [
                    x
                    for x, y in task_spec.items()
                    if y > 0 and x.split('_')[0] in ['on', 'inside']
                ]
            )
            object_grab = pred.split('_')[1]
            return {
                pred: [1, True, 60],
                'holds_' + object_grab + '_' + '1': [1, False, 2],
                'close_' + object_grab + '_' + '1': [1, False, 0.05],
            }
        else:
            raise NotImplementedError

    def reset(self, environment_graph=None, task_id=None):

        # Make sure that characters are out of graph, and ids are ok
        # ipdb.set_trace()
        if task_id is None:
            task_id = self.rnd.choice(list(range(len(self.env_task_set))))
        env_task = self.env_task_set[task_id]

        self.agent_object_touched = []

        self.task_id = env_task['task_id']
        self.init_graph = copy.deepcopy(env_task['init_graph'])
        self.init_rooms = env_task['init_rooms']
        self.task_goal = env_task['task_goal']

        if self.convert_goal:
            self.task_goal = {
                agent_id: utils_env2.convert_goal(task_goal, self.init_graph)
                for agent_id, task_goal in self.task_goal.items()
            }
        # ipdb.set_trace()
        # TODO: remove
        self.task_name = env_task['task_name']

        old_env_id = self.env_id
        self.env_id = env_task['env_id']
        print(
            "Resetting... Envid: {}. Taskid: {}. Index: {}".format(
                self.env_id, self.task_id, task_id
            )
        )

        # TODO: in the future we may want different goals
        if self.convert_goal:
            self.goal_spec = {
                agent_id: self.get_goal2(
                    self.task_goal[agent_id], self.agent_goals[agent_id]
                )
                for agent_id in range(self.num_agents)
            }

        else:
            self.goal_spec = {
                agent_id: self.get_goal(
                    self.task_goal[agent_id], self.agent_goals[agent_id]
                )
                for agent_id in range(self.num_agents)
            }

        if False:  # old_env_id == self.env_id:
            print("Fast reset")
            self.comm.fast_reset()
        else:
            self.comm.reset(self.env_id)

        s, g = self.comm.environment_graph()
        edge_ids = set(
            [edge['to_id'] for edge in g['edges']]
            + [edge['from_id'] for edge in g['edges']]
        )
        node_ids = set([node['id'] for node in g['nodes']])
        if len(edge_ids - node_ids) > 0:
            pdb.set_trace()

        if self.env_id not in self.max_ids.keys():
            max_id = max([node['id'] for node in g['nodes']])
            self.max_ids[self.env_id] = max_id

        max_id = self.max_ids[self.env_id]

        # ipdb.set_trace()
        if environment_graph is not None:
            updated_graph = environment_graph
            s, g = self.comm.environment_graph()
            updated_graph = utils.separate_new_ids_graph(updated_graph, max_id)
            success, m = self.comm.expand_scene(updated_graph)
        else:
            updated_graph = self.init_graph
            s, g = self.comm.environment_graph()
            updated_graph = utils.separate_new_ids_graph(updated_graph, max_id)
            success, m = self.comm.expand_scene(updated_graph)



        if not success:
            ipdb.set_trace()
            print("Error expanding scene")
            ipdb.set_trace()
            return None

        self.offset_cameras = self.comm.camera_count()[1]
        if self.init_rooms[0] not in ['kitchen', 'bedroom', 'livingroom', 'bathroom']:
            rooms = self.rnd.sample(['kitchen', 'bedroom', 'livingroom', 'bathroom'], 2)
        else:
            rooms = list(self.init_rooms)

        for i in range(self.num_agents):
            if i in self.agent_info:
                # rooms[i] = 'kitchen'
                self.comm.add_character(self.agent_info[i], initial_room=rooms[i])
            else:
                self.comm.add_character()

        self.changed_graph = True
        graph = self.get_graph()
        self.init_unity_graph = graph
        self.rooms = [
            (node['class_name'], node['id'])
            for node in graph['nodes']
            if node['category'] == 'Rooms'
        ]
        self.id2node = {node['id']: node for node in graph['nodes']}

        obs = self.get_observations()
        self.steps = 0
        self.prev_reward = 0.0
        return obs

    def step(self, action_dict):
        script_list = utils.convert_action(action_dict)
        failed_execution = False
        if len(script_list[0]) > 0:
            print(script_list)
            if self.recording_options['recording']:
                success, message = self.comm.render_script(
                    script_list,
                    recording=True,
                    skip_animation=False,
                    camera_mode=self.recording_options['cameras'],
                    file_name_prefix='task_{}'.format(self.task_id),
                    image_synthesis=self.recording_optios['modality'],
                )
            else:

                if 'touch' in script_list[0]:
                    objid = int(action_dict[0].split('(')[1].strip()[:-1])
                    self.agent_object_touched.append(objid)
                    success, message = True, {}
                else:
                    success, message = self.comm.render_script(
                        script_list,
                        recording=False,
                        image_synthesis=[],
                        skip_animation=True,
                    )
            if not success:
                print("NO SUCCESS")
                print(message, script_list)
                failed_execution = True
            else:
                self.changed_graph = True

        # Obtain reward
        reward, done, info = self.reward()

        graph = self.get_graph()
        self.steps += 1

        obs = self.get_observations()

        info['finished'] = done
        info['graph'] = graph
        info['failed_exec'] = failed_execution
        if self.steps == self.max_episode_length:
            done = True
        return obs, reward, done, info

    def get_angle(self, rot):
        rot = R.from_quat(rot)
        euler = rot.as_euler('xzy')
        # dchange = np.sin(euler[1])*np.cos(euler[0]), np.cos(euler[1])*np.sin(euler[0])
        # dchange = np.sin(euler[1]+euler[0]), np.cos(euler[1]+euler[0])
        x = np.cos(euler[2]) * np.cos(euler[1])
        y = np.sin(euler[2]) * np.cos(euler[1])
        z = np.sin(euler[1])
        dchange = y, x
        return np.arctan2(x, y) * 180 / math.pi

    def get_observation(self, agent_id, obs_type, info={}):
        if obs_type == 'partial':
            # agent 0 has id (0 + 1)
            curr_graph = self.get_graph()
            curr_graph = utils.clean_house_obj(curr_graph)
            curr_graph = utils.inside_not_trans(curr_graph)
            self.full_graph = copy.deepcopy(curr_graph)
            obs = utils_env.get_visible_nodes(curr_graph, agent_id=(agent_id + 1))
            return obs

        elif obs_type == 'full':
            curr_graph = self.get_graph()
            curr_graph = utils.clean_house_obj(curr_graph)
            curr_graph = utils.inside_not_trans(curr_graph)
            self.full_graph = copy.deepcopy(curr_graph)
            return curr_graph

        elif obs_type == 'cone':
            curr_graph = self.get_graph()
            curr_graph = utils.clean_house_obj(curr_graph)
            curr_graph = utils.inside_not_trans(curr_graph)
            self.full_graph = copy.deepcopy(curr_graph)
            obs = utils_env.get_visible_nodes(curr_graph, agent_id=(agent_id + 1))

            # TODO: implement a real coen here, with unity
            # s, obs_cone = self.comm.get_visible_objects(camera_index)
            agent_node = [node for node in obs['nodes'] if node['id'] == agent_id + 1][
                0
            ]
            position, rotation = (
                agent_node['obj_transform']['position'],
                agent_node['obj_transform']['rotation'],
            )
            rotation_char = self.get_angle(rotation)
            rotation_all = [
                (
                    node['id'],
                    180.0
                    / math.pi
                    * np.arctan2(
                        node['obj_transform']['position'][2] - position[2],
                        node['obj_transform']['position'][0] - position[0],
                    ),
                )
                for node in obs['nodes']
            ]
            rot = [
                rot_id for rot_id in rotation_all if abs(rot_id[1] - rotation_char) < 20
            ]
            rotation_ids = [r[0] for r in rot]
            room_doors = [
                node['id']
                for node in obs['nodes']
                if node['category'] in ['Rooms', 'Doors']
            ]
            rotation_ids = set(rotation_ids + room_doors + [agent_id + 1])
            all_ids = [node['id'] for node in obs['nodes']]
            missing_ids = list(set(all_ids) - set(rotation_ids))
            # print("Removed:")
            # print([node['class_name'] for node in obs['nodes'] if node['id'] in missing_ids])
            new_obs = {
                'nodes': [node for node in obs['nodes'] if node['id'] in rotation_ids],
                'edges': [
                    edge
                    for edge in obs['edges']
                    if edge['from_id'] in rotation_ids and edge['to_id'] in rotation_ids
                ],
            }
            return new_obs

        elif obs_type == 'visible':
            # Only objects in the field of view of the agent
            raise NotImplementedError

        elif obs_type == 'image':
            camera_ids = [
                self.offset_cameras
                + agent_id * self.num_camera_per_agent
                + self.CAMERA_NUM
            ]
            if 'image_width' in info:
                image_width = info['image_width']
                image_height = info['image_height']
            else:
                image_width, image_height = (
                    self.default_image_width,
                    self.default_image_height,
                )
            if 'obs_type' in info:
                curr_obs_type = info['obs_type']
            else:
                curr_obs_type = self.default_obs_type

            s, images = self.comm.camera_image(
                camera_ids,
                mode=curr_obs_type,
                image_width=image_width,
                image_height=image_height,
            )
            if not s:
                pdb.set_trace()
            return images[0]
        else:
            raise NotImplementedError

        return updated_graph
