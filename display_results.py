import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import utils_plot2
import ipdb
import pickle as pkl

def display_res(file_in, file_out):
    with open(file_inp, 'r') as f:
        content = json.load(f)
    html_str = []
    #ipdb.set_trace()
    info_dict = {}
    fnames = list(set(sorted([ct['filename'].split('/')[-1] for ct in content['info']])))
    fnames_dict = {fn: it for it, fn in enumerate(fnames)}
    fnames_info = {it: [None, None, None] for it in range(len(fnames))}
    
    dict_inf = {5: 0, 11: 1, 13: 2}
    for info in content['info']:
        fname = info['filename'].split('/')[-1]
        fname_index = fnames_dict[fname]
        num = int(info['filename'].split('/')[-2].split('_')[0])
        index_mem = dict_inf[num]
        fnames_info[fname_index][index_mem] = info

    html_str.append('<table>')
    
    for it, finfo in tqdm(fnames_info.items()):
        fname = fnames[it]
        strs = []
        for i in range(3):
            info = finfo[i]
            if info is None:
                curr_str = ''
            else:

                with open(info['filename'], 'rb') as f:
                    content_graph = pkl.load(f)

                actions = content_graph['action'][0]
                goals = content_graph['goals'][0]
                goal_names = {x.split('_')[1] for x,y in goals.items() if y > 0}
                goal_str = str(goals)
                if len(strs) == 0:
                    strs.append(goal_str)
                gs = content_graph['graph']
                goal_obj = [node for node in gs[0]['nodes'] if node['class_name'] in goal_names]
                belief_ids = goal_obj
                
                fig = plt.figure(constrained_layout=True, figsize=(10, 6))
                spec2 = fig.add_gridspec(2,3)
                axes = [fig.add_subplot(spec2[:, :2]), fig.add_subplot(spec2[0, 2]), fig.add_subplot(spec2[1,2])]
                #fig, axes = plt.subplots(1,3, figsize=(27, 7))
                utils_plot2.plot_episode(gs, axes[0], goal_obj, belief_ids, actions)
                values1 = [info[f'{name}_belief_room'] for name in ['gt', 'pred']]
                values2 = [info[f'{name}_belief_container'] for name in ['gt', 'pred']]
                utils_plot2.bar_plot(values1, [n.replace('cabinet', 'cab.') for n in info['names_belief_room']], axes[1])
                utils_plot2.bar_plot(values2, [n.replace('cabinet', 'cab.') for n in info['names_belief_container']], axes[2])
                
                fname = '_'.join(info['filename'].split('/')[-2:])
                img_name = f'plots/{fname}.png'
                curr_str = '<img src={} style="height: 300px"></img>'.format(img_name)
                print(img_name)
                fig.savefig(img_name)
                plt.close(fig)
            strs.append(curr_str)
        strs = ''.join(['<td>{}</td>'.format(st) for st in strs])
        final_str = '<tr>'+strs+'</tr>'
        html_str.append(final_str)
    html_str.append('</table>')

    #for info in tqdm(content['info']):
    #    with open(info['filename'], 'rb') as f:
    #        content_graph = pkl.load(f)
    #    actions = content_graph['action'][0]
    #    goals = content_graph['goals'][0]
    #    goal_names = {x.split('_')[1] for x,y in goals.items() if y > 0}
    #    goal_str = str(goals)
    #    gs = content_graph['graph']
    #    goal_obj = [node for node in gs[0]['nodes'] if node['class_name'] in goal_names]
    #    belief_ids = goal_obj

    #    #fig, axes = plt.subplots(1,3, figsize=(27, 7))
    #    #utils_plot2.plot_episode(gs, axes[0], goal_obj, belief_ids, actions)
    #    values1 = [info[f'{name}_belief_room'] for name in ['gt', 'pred']]
    #    values2 = [info[f'{name}_belief_container'] for name in ['gt', 'pred']]
    #    #utils_plot2.bar_plot(values1, info['names_belief_room'], axes[1])
    #    #utils_plot2.bar_plot(values2, info['names_belief_container'], axes[2])
    #    
    #    fname = '_'.join(info['filename'].split('/')[-2:])
    #    img_name = f'plots/{fname}.png'
    #    #fig.savefig(img_name)
    #    plt.close()
    #    html_str.append('<p>{}</p><p>{}</p><img src={} style="height: 300px"></img><br><br>'.format(goal_str, fname, img_name))

    with open(file_out, 'w+') as f:
        f.writelines(html_str)


def kldiv(p, q):
    kld = np.array(p)*np.log(np.array(p)/(np.array(q)+1e-7))
    return np.mean(kld)

def kl_vs_dist(file_inp):
    with open(file_inp, 'r') as f:
        content = json.load(f)
    actions = [[], [], []]
    k1 =  [[], [], []]
    k2 =  [[], [], []]
    dict_inf = {5: 0, 11: 1, 13: 2}
    namest = ['belief1', 'belief2', 'belief3']
    for info in tqdm(content['info']):
        if 'len' not in info:
            with open(info['filename'], 'rb') as f:
                content_graph = pkl.load(f)
            info['len'] = len(content_graph['action'][0])

        label_type = int(info['filename'].split('/')[-2].split('_')[0])
        index = dict_inf[label_type]
        kl1 = kldiv(info['pred_belief_room'], info['gt_belief_room'])
        kl2 = kldiv(info['pred_belief_container'], info['gt_belief_container'])
        k1[index].append(kl1)
        k2[index].append(kl2)
        actions[index].append(info['len'])

    with open(file_inp, 'w+') as f:
        f.write(json.dumps(content))
    fig, ax = plt.subplots(1,2, figsize=(6, 3))
    for i in range(3):
        ax[0].scatter(actions[i], k1[i], label=namest[i])
        ax[1].scatter(actions[i], k2[i], label=namest[i])
        ax[0].grid()
        ax[1].grid()
        ax[0].set_title("Room KLDiv")
        ax[0].set_xlabel("Episode Length")
        ax[1].set_xlabel("Episode Length")
        ax[0].set_ylabel("KLDiv")
        ax[1].set_ylabel("KLDiv")
        ax[1].set_title("Object KLDiv")
        plt.legend()
        plt.tight_layout()
    fig.savefig('kl_dist.png')
    plt.close(fig)
    fig, ax = plt.subplots()
    x = np.array(range(3))
    y = [np.mean(act) for act in actions]
    err = [np.std(act)/np.sqrt(len(act)) for act in actions]
    ax.bar(x, y, yerr=err)
    ax.set_xticks(x)
    ax.set_ylabel("Episode Length")
    ax.set_xticklabels(namest)
    ax.yaxis.grid(True)
    fig.savefig('barplot.png')


if __name__ == '__main__':
    file_inp = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/agent_preferences/data_scratch/large_data_touch_v2/results_model_belief/results.json'
    file_out = 'out.html'
    kl_vs_dist(file_inp)
    #display_res(file_inp, file_out)
