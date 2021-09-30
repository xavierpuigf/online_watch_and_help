import sys
import cv2
import pickle as pkl
sys.path.append('../../virtualhome/simulation')
from unity_simulator import UnityCommunication

with open('logs_episode.39_iter.0.pik', 'rb') as f:
    cont = pkl.load(f)

actions = cont['action'][0]
exec_name = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/website/release/simulator/v2.0/v2.2.5_beta4/linux_exec.v2.2.5_beta4.x86_64'
comm = UnityCommunication(file_name=exec_name, no_graphics=False, x_display='0')
comm.reset(cont['env_id'])
comm.expand_scene(cont['init_unity_graph'])

cam_id = comm.camera_count()[1] - 7

obs = cont['graph'][0]
pos = [node['bounding_box']['center'] for node in obs['nodes'] if node['id'] == 1][0]
pos[1] = 0
init_pos_char = comm.add_character(position=pos)


index = 0
for act in actions:
    act_str = [f'<char0> {act}']
    comm.render_script(act_str, skip_animation=True)
    s, aux = comm.camera_image([cam_id])
    s, g = comm.environment_graph()
    cv2.imwrite(f'test{index}.png', aux)
    index += 1
    print(g['nodes'][0]['bounding_box']['center'])