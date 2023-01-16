import glob
from multiprocessing import Queue, Process
from omegaconf import DictConfig, OmegaConf
import yaml
import os
import random
import wandb
import tensorflow as tf
import tensorflow as tf
from hydra.utils import get_original_cwd, to_absolute_path
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import torch
import wandb
import json
import ipdb
import pdb
import torch.nn as nn
import matplotlib.pyplot as plt
from .utils_plot import Plotter
from utils import utils_rl_agent
import time



class ThreadedPlotter():
  # plot func receives a dict and gets what it needs to plot
  def __init__(self, plot_func, use_threading=True, queue_size=10, force_except=False):
    self.queue = Queue(queue_size)
    self.plot_func = plot_func
    self.use_threading = use_threading
    self.force_except = force_except
    def plot_results_process(queue, plot_func):
        # to avoid wasting time making videos
        while True:
            try:
                if queue.empty():
                    time.sleep(1)
                    if queue.full():
                        print("Plotting queue is full!")
                else:
                    actual_plot_dict = queue.get()
                    time_put_on_queue = actual_plot_dict.pop('time_put_on_queue')
                    print("Plotting...")
                    plot_func(**actual_plot_dict)
                    continue
            except Exception as e:
                if self.force_except:
                  raise e
                print('Plotting failed wiht exception: ')
                print(e)
    if self.use_threading:
      Process(target=plot_results_process, args=[self.queue, self.plot_func]).start()

  def _detach_tensor(self, tensor):
    if tensor.is_cuda:
      tensor = tensor.detach().cpu()
    tensor = np.array(tensor.detach())
    return tensor

  def _detach_dict_or_list_torch(self, list_or_dict):
    # We put things to cpu here to avoid er
    if type(list_or_dict) is dict:
      to_iter = list(list_or_dict.keys())
    elif type(list_or_dict) is list:
      to_iter = list(range(len(list_or_dict)))
    else:
      return list_or_dict
    for k in to_iter:
      if type(list_or_dict[k]) is torch.Tensor:
        list_or_dict[k] = self._detach_tensor(list_or_dict[k])
      else:
        list_or_dict[k] = self._detach_dict_or_list_torch(list_or_dict[k])
    return list_or_dict

  def clear_queue(self):
    while not self.queue.empty():
      self.queue.get()

  def is_queue_full(self):
    if not self.use_threading:
      return False
    else:
      return self.queue.full()

  def n_queue_elements(self):
      if not self.use_threading:
        return 0
      else:
        return self.queue.qsize()

  def put_plot_dict(self, plot_dict):
    # try:
    if True:
      assert type(plot_dict) is dict
      # assert 'env' in plot_dict, 'Env to plot not found in plot_dict!'
      plot_dict = self._detach_dict_or_list_torch(plot_dict)
      if self.use_threading:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M:%S")
        plot_dict['time_put_on_queue'] = timestamp
        self.queue.put(plot_dict)
      else:
        self.plot_func(**plot_dict)
    
    try:
        pass
    except Exception as e:
      if self.force_except:
        raise e
      print('Putting onto plot queue failed with exception:')
      print(e)






plt.switch_backend('agg')
def build_table(rows, header_names):
    html_str = '<table>'
    html_str += '<tr>'+''.join([f'<th>{nh}</th>' for nh in header_names])+'</tr>'
    for row in rows:
        td_cont = ''.join([f'<td>{rc}</td>' for rc in row])
        html_str += f'<tr>{td_cont}</tr>'
    html_str += '</table>'
    # ipdb.set_trace()
    return html_str



def build_task_pred_str(from_id, to_id, edge_type, obj_names, tstep, graph_helper, new_nodes=None):
    from_id = from_id[tstep]
    to_id = to_id[tstep]
    edge_type = edge_type[tstep]
    if new_nodes is not None:
        new_nodes = list(new_nodes)
    # else:

    from_edge = {}
    # ipdb.set_trace()
    for elem_from, elem_to in zip(from_id.tolist(), to_id.tolist()):
        if int(elem_to) not in from_edge:
            from_edge[int(elem_to)] = []
        from_edge[int(elem_to)].append(int(elem_from))

    all_elems = sorted(list(set(list(from_edge.keys()))))
    # obj_names = gt_graph['nodes']

    graph_str = ''
    def convert_name(name, itt):
        if new_nodes is None or new_nodes[itt] == 0:
            return name
        else:
            return '<span style="color:blue">'+name+'</span>'
            # return name÷
    for elem in all_elems:
        on_curr = []
        if elem in from_edge:
            on_curr = from_edge[elem]
        # ipdb.set_trace()
        on_str = ' '.join([convert_name(obj_names[itt].strip(), itt) for itt in on_curr])
        elem2 = obj_names[elem]
        graph_str += f'<span style="white-space: nowrap"><b>{elem2}</b>: [{on_str}]</span><br>'
    # print("==========")


    return graph_str

def build_graph_str(from_id, to_id, edge_type, obj_names, tstep, graph_helper, new_nodes=None):
    from_id = from_id[tstep]
    to_id = to_id[tstep]
    edge_type = edge_type[tstep]
    if new_nodes is not None:
        new_nodes = list(new_nodes)
    # else:

    from_edge = {}
    # ipdb.set_trace()
    for elem_from, elem_to in zip(from_id.tolist(), to_id.tolist()):
        if int(elem_to) not in from_edge:
            from_edge[int(elem_to)] = []
        from_edge[int(elem_to)].append(int(elem_from))

    all_elems = sorted(list(set(list(from_edge.keys()))))
    # obj_names = gt_graph['nodes']

    graph_str = ''
    def convert_name(name, itt):
        if new_nodes is None or new_nodes[itt] == 0:
            return name
        else:
            return '<span style="color:blue">'+name+'</span>'
            # return name÷
    for elem in all_elems:
        on_curr = []
        if elem in from_edge:
            on_curr = from_edge[elem]
        # ipdb.set_trace()
        on_str = ' '.join([convert_name(obj_names[itt].strip(), itt) for itt in on_curr])
        elem2 = obj_names[elem]
        graph_str += f'<span style="white-space: nowrap"><b>{elem2}</b>: [{on_str}]</span><br>'
    # print("==========")


    return graph_str

def get_predicates(pred, t, source='pred'):
    # pred_edge_prob = pred['edge_prob']
    # print(len(pred['edge_input'][t]), len(pred['edge_pred'][t]))
    t = min(t, len(pred['edge_pred']) - 1)
    edge_pred = pred['edge_pred'][t] if source == 'pred' else pred['edge_input'][t]
    pred_edge_names = pred['edge_names']
    pred_nodes = pred['nodes']
    pred_from_ids = pred['from_id'] if source == 'pred' else pred['from_id_input']
    pred_to_ids = pred['to_id'] if source == 'pred' else pred['to_id_input']

    # edge_prob = pred_edge_prob[t]
    # edge_pred = np.argmax(edge_prob, 1)

    edge_pred_ins = {}

    num_edges = len(edge_pred)
    # print(pred_from_ids[t], num_edges)
    for edge_id in range(num_edges):
        from_id = pred_from_ids[t][edge_id]
        to_id = pred_to_ids[t][edge_id]
        from_node_name = pred_nodes[from_id]
        to_node_name = pred_nodes[to_id]
        # if object_name in from_node_name or object_name in to_node_name:
        edge_name = pred_edge_names[edge_pred[edge_id]]
        if to_node_name.split('.')[1] == '-1':
            continue
        if edge_name in ['inside', 'on']:  # disregard room locations + plate
            if to_node_name.split('.')[0] in [
                'kitchen',
                'livingroom',
                'bedroom',
                'bathroom',
                'plate',
            ]:
                continue
            if from_node_name.split('.')[0] in [
                'kitchen',
                'livingroom',
                'bedroom',
                'bathroom',
                'character',
            ]:
                continue
        else:
            continue

        edge_class = '{}_{}_{}'.format(
            edge_name, from_node_name.split('.')[0], to_node_name.split('.')[1]
        )

        # print(from_node_name, to_node_name, edge_name)
        if edge_class not in edge_pred_ins:
            edge_pred_ins[edge_class] = {
                'count': 0,
                'grab_obj_ids': [],
                'container_ids': [int(to_node_name.split('.')[1])],
            }
        edge_pred_ins[edge_class]['count'] += 1
        edge_pred_ins[edge_class]['grab_obj_ids'].append(
            int(from_node_name.split('.')[1])
        )
    return edge_pred_ins

def get_difference_pred(pred1_dict, pred2_dict):
    # Compute pred1 - pred2
    pred_diff = {}
    pred_names = list(set(list(pred1_dict.keys()) + list(pred2_dict.keys())))
    for pred_name in pred_names:
        if pred_name in pred1_dict:
            if pred_name in pred2_dict:
                pred_diff[pred_name] = {'count': pred1_dict[pred_name]['count'] - pred2_dict[pred_name]['count']}
            else:
                pred_diff[pred_name] = {'count': pred1_dict[pred_name]['count']}
        else:
            pred_diff[pred_name] = {'count': -pred2_dict[pred_name]['count']}
    pred_diff = {x: y for x,y in pred_diff.items() if y['count'] != 0}
    return pred_diff

def get_pred_str(pred_dict):
    l_items = ['<li>{}: {}</li>'.format(key, pred_dict[key]['count']) for key in sorted(pred_dict.keys())]
    return ''.join(l_items)

def get_pred_task_str(str_task, str_mask=None, correct=None, remove=False):

    l_items = ['<li><span style="color: blue">{}</span><span style="color: {}">{}</span></li>'.format('[New] ' if (str_mask != None and str_mask[i] == 1) else '', 
        'black' if correct is None or not correct[i] else 'green', 
        str_item) for i, str_item in enumerate(str_task)]
    if remove:
        l_items = ['<li><span style="color: blue">{}</span><span style="color: {}">{}</span></li>'.format(
            '[New] ', 
            'black' if not correct or not correct[i] else 'green', 
            str_item) for i, str_item in enumerate(str_task) if (str_mask != None and str_mask[i] == 1)]
    return ''.join(l_items)


def get_html_task(results, graph_helper):
    other_info = results['other_info']

    gt_task, gt_graph_tensor = results['gt_task']
    pred_task, pred_task_tensor_list = results['pred_task']
    # ipdb.set_trace()
    
    program_gt = other_info['prog_gt']
    index = other_info['index']

    scores_step = other_info['metrics_tstep'] 
    

    script_js = '''
            <script>
            var show_graph = false;
            function switchView(){
                var preds = document.getElementsByClassName('preds');
                var graphs = document.getElementsByClassName('graph'); 
                for (var j = 0; j < preds.length; j++){
                    if (show_graph){

                        preds[j].style.display = 'none';
                        graphs[j].style.display = 'block';
                    }  
                    else {

                        preds[j].style.display = 'block';
                        graphs[j].style.display = 'none';
                    }

                }
                
                show_graph = !show_graph;
            }
            </script>
    '''
    html_str = '<html><head><link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" rel="nofollow" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous"><head><body>'
    html_str += script_js
    html_str += '<button onclick=switchView()> Switch Preds/Graph </button>'
    header_names = ['Action', 'Score', 'Input', 'GT'] + [f'Pred {i}' for i in range(5)]
    header = ''.join(['<th>{}</th>'.format(name) for name in header_names])
    table_resp = f'<table class="table"><tr>{header}</tr>'
    numtsteps = len(program_gt) - 1

    rows = []
    # ipdb.set_trace()
    for tstep in range(numtsteps):
        columns = []
        # ipdb.set_trace()
        
        # score_str = ''
        # ipdb.set_t÷race()
        #
        score_str = '<br>'.join(['{}: {:03f}'.format(name, value[index][tstep]) for name, value in scores_step.items()])
        columns = [program_gt[tstep].replace('<', '').replace('>', ''), score_str]
        # ipdb.set_trace()
        

        gt_task_str = get_pred_task_str(gt_task[tstep]['output_task'], gt_task[tstep]['mask'])
        gt_task_str_filter = get_pred_task_str(gt_task[tstep]['output_task'], gt_task[tstep]['mask'], remove=True)
        input_task_str = get_pred_task_str(gt_task[tstep]['input_task'])

        columns += [(input_task_str, input_task_str), (gt_task_str, gt_task_str_filter)]

        gt_task_tensor_curr = gt_graph_tensor[tstep]
        for gind in range(min(5, len(pred_task_tensor_list))):
            pred_task_tensor_curr = pred_task_tensor_list[gind][tstep]
            # ipdb.set_trace()

            correct = list((pred_task_tensor_curr ==  gt_task_tensor_curr)[pred_task_tensor_curr > 0])
            # ipdb.set_trace()
            predicates_str = get_pred_task_str(pred_task[gind][tstep]['output_task'],pred_task[gind][tstep]['mask'], correct=correct)
            predicates_str_filtered = get_pred_task_str(pred_task[gind][tstep]['output_task'],pred_task[gind][tstep]['mask'], correct=correct, remove=True)
            
            # ipdb.set_trace()
            columns.append((predicates_str, predicates_str_filtered))
        # ipdb.set_trace()
        # ipdb.set_trace()
        style_str = 'overflow: auto; width: 500px'
        style_str2 = 'overflow: auto; width: 150px'
        column_str = ''.join(['<td><div style="{}">{}</div></td>'.format(style_str2, col) for col in columns[:2]])
        column_str += ''.join(['<td><div class="preds" style="{style}">{content_pred}</div><div class="graph" style="{style}; display: none">{content_graph}</div></td>'.format(style=style_str, content_graph=col[0], content_pred=col[1]) for col in columns[2:]])
        rows.append(column_str)
    
    table_resp += ''.join(['<tr>{}</tr>'.format(row) for row in rows])
    
    table_resp += '</table>'
    html_str += table_resp 
    html_str += '</body></html>'
    return html_str


def get_html(results, graph_helper):
    other_info = results['other_info']
    scores_step = other_info['metrics_tstep'] 
    gt_graph = results['gt_graph']
    pred_graph = results['pred_graph']
    program_gt = other_info['prog_gt']
    index = other_info['index']
    ipdb.set_trace()
    script_js = '''
            <script>
            var show_graph = false;
            function switchView(){
                var preds = document.getElementsByClassName('preds');
                var graphs = document.getElementsByClassName('graph'); 
                for (var j = 0; j < preds.length; j++){
                    if (show_graph){

                        preds[j].style.display = 'none';
                        graphs[j].style.display = 'block';
                    }  
                    else {

                        preds[j].style.display = 'block';
                        graphs[j].style.display = 'none';
                    }

                }
                
                show_graph = !show_graph;
            }
            </script>
    '''
    html_str = '<html><head><link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" rel="nofollow" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous"><head><body>'
    html_str += script_js
    html_str += '<button onclick=switchView()> Switch Preds/Graph </button>'
    header_names = ['Action', 'Score', 'Input', 'GT'] + [f'Pred {i}' for i in range(5)]
    header = ''.join(['<th>{}</th>'.format(name) for name in header_names])
    table_resp = f'<table class="table"><tr>{header}</tr>'
    numtsteps = len(program_gt) - 1
    edge_names = gt_graph['edge_names']

    rows = []
    for tstep in range(numtsteps):
        columns = []
        # ipdb.set_trace()
        score_str = '<br>'.join(['{}: {:03f}'.format(name, value[index][tstep]) for name, value in scores_step.items()])
        # ipdb.set_trace()
        columns = [program_gt[tstep].replace('<', '').replace('>', ''), score_str]
        graph_input = build_graph_str(
            gt_graph['from_id_input'], gt_graph['to_id_input'], gt_graph['edge_input'], gt_graph['nodes'], tstep, graph_helper) 
        graph_gt = build_graph_str(
            gt_graph['from_id'], gt_graph['to_id'], gt_graph['edge_pred'], gt_graph['nodes'], tstep, graph_helper, new_nodes=gt_graph['new_marker'][tstep])
        
        input_gt_pred = get_predicates(gt_graph, tstep, 'input')
        pred_gt = get_predicates(gt_graph, tstep)
        diff_gt_str = get_pred_str(get_difference_pred(pred_gt, input_gt_pred))
        input_gt_str = get_pred_str(input_gt_pred)

        columns += [(graph_input, input_gt_str), (graph_gt, diff_gt_str)]
        for gind in range(5):
            c_pred_graph = pred_graph[gind]
            # ipdb.set_trace()
            pred_graph_str = build_graph_str(
                c_pred_graph['from_id'], c_pred_graph['to_id'], c_pred_graph['edge_pred'], c_pred_graph['nodes'], tstep, graph_helper, new_nodes=c_pred_graph['new_marker'][tstep])
            # ipdb.set_trace()
            predicates_pred = get_predicates(c_pred_graph, tstep)
            predicates_str = get_pred_str(get_difference_pred(predicates_pred, input_gt_pred))
            columns.append((pred_graph_str, predicates_str))
        # ipdb.set_trace()
        style_str = 'overflow: auto; width: 500px'
        style_str2 = 'overflow: auto; width: 150px'
        column_str = ''.join(['<td><div style="{}">{}</div></td>'.format(style_str2, col) for col in columns[:2]])
        column_str += ''.join(['<td><div class="preds" style="{style}">{content_pred}</div><div class="graph" style="{style}; display: none">{content_graph}</div></td>'.format(style=style_str, content_graph=col[0], content_pred=col[1]) for col in columns[2:]])
        rows.append(column_str)
    
    table_resp += ''.join(['<tr>{}</tr>'.format(row) for row in rows])
    
    table_resp += '</table>'
    html_str += table_resp 
    html_str += '</body></html>'
    return html_str


def obtain_task_graph(
    graph_helper,
    graph_info,
    task_graph,
    task_mask,
    input_task,
    index,
    len_mask
):
    num_tsteps = int(len_mask[index].sum())
    task_info = []
    # ipdb.set_trace()
    for i in range(num_tsteps):
        # print(input_task.sha÷pe, task_graph.shape, index, i)
        cinput_task = graph_helper.get_task_graph(input_task[index, i])  
        cmask = task_mask[index, i][task_graph[index, i] != 0]
        try:
            coutput_task = graph_helper.get_task_graph(task_graph[index, i]) 
        except:
            ipdb.set_trace()
        task_info.append({
            'input_task': cinput_task,
            'output_task': coutput_task,
            'mask': list(cmask)
        })
    # print('--')
    return task_info

# def obtain_task_graph(
#     graph_helper,
#     graph_info,
#     task_graph,
#     task_mask,
#     input_task,
#     index,
#     step
# ):
#     task = task_graph[index, step]
#     mask = task_mask[index, step]
#     return graph_helper.get_task_graph(task, mask)

def print_task_graph(
    graph_helper,
    graph_info,
    task_graph,
    task_mask,
    input_task,
    index,
    step
):
    task = task_graph[index, step]
    mask = task_mask[index, step]
    graph_helper.print_task_graph(task, mask)

def print_graph_3_2(
    graph_helper,
    graph,
    edge_info,
    mask_edge,
    state_info,
    changed_edges,
    batch_item,
    step,
    changed_nodes=None,
):


    # If we are only predicitng edge change, the edge is a combination of previous edge and new, modulagted by prediction
    if len(changed_edges) > 0:

        if changed_edges[0].shape[2] != changed_edges[1].shape[2]:
            # Changes as nodes
            num_nodes = changed_edges[0].shape[2]

            if torch.is_tensor(changed_edges[0]):
                changed_edges_build = changed_edges[0].repeat_interleave(num_nodes, dim=2)
            else:
                changed_edges_build = changed_edges[0].repeat(num_nodes, axis=2)
            changed_edges = [changed_edges_build, changed_edges[1]]


        # The changed edges should be boolean at this point
        # ipdb.set_trace()
        assert (changed_edges[0] == 1).sum() == (changed_edges[0] != 0).sum()

        # ipdb.set_trace()
        try:
            edge_info = (
                changed_edges[0] * edge_info + (1 - changed_edges[0]) * changed_edges[1]
            )
        except:
            ipdb.set_trace()


    # We are predicting the next graph, so we sum 1
    offset = 1
    mask_object = int(graph['mask_object'][batch_item, step + offset].sum())
    object_names = graph['class_objects'][batch_item, step + offset]
    object_states = state_info[batch_item, step]
    num_nodes = graph['mask_object'].shape[-1]

    # object_coords = graph['object_coords'][batch_item, step+offset]

    # ipdb.set_trace()
    node_ids = graph['node_ids'][batch_item, step + offset]

    # ipdb.set_trace()
    print_node = False
    print("Graph")
    print("==========")
    if print_node:
        print("Nodes:")
    obj_names = []
    for nid in range(mask_object):

        state_names = [
            graph_helper.states[it]
            for it in range(4)
            if int(object_states[nid][it]) == 1
        ]
        state_names = ' '.join(state_names)
        class_name = graph_helper.object_dict.get_el(int(object_names[nid]))
        idi = int(node_ids[nid])
        # coords = list(object_coords[nid][:3])
        # coords_str = '{:.2f}, {:.2f}, {:.2f}'.format(coords[0], coords[1], coords[2])
        obj_name_complete = f"{class_name}.{idi}"
        obj_name_complete += ' ' * (20 - len(obj_name_complete))
        obj_names.append(obj_name_complete)
        if print_node:
            print(f"{obj_name_complete}. {state_names}")

    if print_node:
        print('\n')

    print('Edges')
    current_mask_edge = mask_edge[batch_item, step]
    current_edge = edge_info[batch_item, step]
    # Only store on and inside edges, and hold



    num_edge =  (current_mask_edge > 0).sum()
    on_from = np.arange(num_edge)
    on_to = current_edge[:num_edge]
    from_edge = {}

    for elem_from, elem_to in zip(on_from.tolist(), on_to.tolist()):
        if int(elem_to) not in from_edge:
            from_edge[int(elem_to)] = []
        from_edge[int(elem_to)].append(int(elem_from))

    all_elems = sorted(list(set(list(from_edge.keys()))))

    # ipdb.set_trace()
    # print("HOLDING:", list(zip(hold_from, hold_to)))
    for elem in all_elems:
        on_curr = []
        if elem in from_edge:
            on_curr = from_edge[elem]
        # ipdb.set_trace()
        on_str = ' '.join([obj_names[itt].strip() for itt in on_curr])
        elem2 = obj_names[elem]
        print(f'{elem2}: relation: [{on_str}]')
    # print("==========")


def obtain_graph_3_2(
    graph_helper,
    graph,
    edge_prob,
    state_prob,
    mask_edge,
    changed_edges,
    len_mask,
    batch_item=0,
    changed_nodes=None,
    samples=None,
    include_last=True
):
    
    # TODO: modify this fro exclusive edge perd
    all_samples = []
    if changed_edges[0] is not None:
        prev_changed_edges = [changed_edges[0], changed_edges[1]]
        changed_edges_new = [changed_edges[0], changed_edges[1]]
        
    prev_step_edges = changed_edges[1].argmax(-1)

    do_sample = True

    if samples is None:
        samples = 1
        do_sample = False
        # edge_prob = edge_prob.cpu().numpy(
    else:
        pass
        # edge_prob = nn.functional.softmax(edge_prob, dim=-1).cpu().numpy()
    # ipdb.set_trace()
    for sample in range(samples):
        # Sample edge_prob
        if do_sample:
            edge_pred = vectorized(edge_prob)
        else:
            edge_pred = edge_prob.argmax(-1)
        if changed_edges[0] is not None:
            changed_edges = prev_changed_edges    
            

            # Sample changed edges
            if do_sample:
                changed_edges_new[0] = vectorized(changed_edges[0])
            else:
                changed_edges_new[0] = changed_edges[0].argmax(-1)
            # The changed edges should be boolean at this point
            # ipdb.set_trace()
            try:
                assert (changed_edges_new[0] == 1).sum() == (changed_edges_new[0] != 0).sum()
            except:
                ipdb.set_trace()
            # ipdb.set_trace()
            try:
                edge_pred = (
                    changed_edges_new[0][...] * edge_pred + (1 - changed_edges_new[0][...]) * prev_step_edges
                )
            except:
                ipdb.set_trace()

        # We are predicting the next graph, so we sum
        num_tsteps = int(len_mask[batch_item].sum())
        if include_last:
            num_tsteps -= 1
        offset = 0
        nedges = len(graph_helper.relation_dict)
        state_names = [(graph_helper.states[it],) for it in range(4)]
        edge_names = [graph_helper.relation_dict.get_el(it) for it in range(nedges)]

        info = {'results': [], 'state_names': state_names, 'edge_names': edge_names}

        all_edges, all_from, all_to, all_edges_input, all_from_input, all_to_input = [], [], [], [], [], []
        object_states = state_prob[batch_item, :num_tsteps].numpy()
        # print(num_tsteps)
        for step in range(num_tsteps):
            result = {}
            mask_object = int(graph['mask_object'][batch_item, step + offset].sum())
            object_names = graph['class_objects'][batch_item, step + offset]
            num_nodes = graph['mask_object'].shape[-1]

            node_ids = graph['node_ids'][batch_item, step + offset]

            print_node = False
            obj_names = []
            for nid in range(mask_object):

                class_name = graph_helper.object_dict.get_el(int(object_names[nid]))
                idi = int(node_ids[nid])
                obj_name_complete = f"{class_name}.{idi}"
                obj_names.append(obj_name_complete)

            # ipdb.set_trace()
            current_mask_edge = mask_edge[batch_item, step]
            # current_edge = edge_info[batch_item, step]
            indices_valid = np.where(current_mask_edge == 1)[0]

            edge_pred_step = edge_pred[batch_item, step + offset, indices_valid]
            edge_input_step = prev_step_edges[batch_item, step + offset, indices_valid]
            # edge_probs = edge_prob[batch_item, step + offset, indices_valid]

            from_id = indices_valid
            to_id = edge_pred_step

            from_id_input = indices_valid
            to_id_input = edge_input_step

            curr_res = {}
            # all_edges.append(edge_prob[None, :].numpy())

            # obtain class
            edge_pred_step_class = obtain_class_edge(from_id, to_id, obj_names, graph_helper.object_dict, graph_helper.relation_dict)
            edge_input_step_class = obtain_class_edge(from_id_input, to_id_input, obj_names, graph_helper.object_dict, graph_helper.relation_dict)
            # ipdb.set_trace()

            all_edges.append(edge_pred_step_class[None, :])
            all_edges_input.append(edge_input_step_class[None, :])
            all_from.append(from_id[None, :])
            all_to.append(to_id[None, :]
                    )
            all_from_input.append(from_id_input[None, :])
            all_to_input.append(to_id_input[None, :])

        all_edges = np.concatenate(all_edges, 0)
        all_edges_input = np.concatenate(all_edges_input, 0)
        all_from = np.concatenate(all_from, 0)
        all_to = np.concatenate(all_to, 0)
        all_from_input = np.concatenate(all_from_input, 0)
        all_to_input = np.concatenate(all_to_input, 0)
        
        info['edge_pred'] = all_edges
        info['edge_input'] = all_edges_input
        info['from_id'] = all_from
        info['to_id'] = all_to

        info['from_id_input'] = all_from_input
        info['to_id_input'] = all_to_input
        info['states'] = object_states

        info['nodes'] = obj_names
        # ipdb.set_trace()
        if len(changed_edges) > 0:
            info['changed_edges'] = changed_edges[0]
        all_samples.append(info)
    return all_samples

# Convert adjacency list to adjacency matrix
# Convert adjacency list to adjacency matrix
def build_gt_edge(graph_info, graph_helper, exclusive_edge=False):
    batch, time, num_nodes = graph_info['mask_object'].shape
    
    
    
    # most edges have relation with nothing
    # last_node = 

    # num_edges = gt_edges.shape[-1]
    edge_tuples = graph_info['edge_tuples']
    
    
    
    if exclusive_edge:
        gt_edges = torch.zeros([batch, time, num_nodes])
    
        # Get the index of the -1 object for every element in the batch
        node_ids_none = ((graph_info['node_ids'][:, 0, :] > 0).sum(-1))[:, None, None]
        gt_edges = node_ids_none.repeat(1, time, num_nodes).float()

        edge_to = edge_tuples[..., 1]
        edge_from = edge_tuples[..., 0]
        # All the edges that have type 0, we will put them as going from node none to node none
        mask_edge = (graph_info['edge_classes'] > 0).float()
        edge_from = edge_from * mask_edge + (1-mask_edge) * node_ids_none
        edge_to = edge_to * mask_edge + (1-mask_edge) * node_ids_none
        gt_edges = gt_edges.scatter(2, edge_from.long(), edge_to)
    
    else:
        gt_edges = torch.zeros([batch, time, num_nodes ** 2])
        index_edges = edge_tuples[..., 0] * num_nodes + edge_tuples[..., 1]
        edge_types = graph_info['edge_classes']  # - 1
        gt_edges = gt_edges.scatter(2, index_edges.long(), edge_types)
    
    gt_edges = gt_edges.long()
    class_names = ['cupcake', 'apple', 'plate', 'waterglass']
    ids_interest = [graph_helper.object_dict.get_id(name) for name in class_names]
    assert len([idi for idi in ids_interest if idi == 0]) == 0, 'Object of interest not recognized {}'.format(str(ids_interest))
    
    # Mask of objects that we care about
    mask_obj_interest = torch.zeros(graph_info['mask_object'].shape)
    
    for id_interest in ids_interest:
        mask_obj_interest[graph_info['class_objects'] == id_interest] = 1.


    edge_dict = {}
    if not exclusive_edge:
        mask_obj_interest_2 = torch.zeros(graph_info['mask_object'].shape)
        mask_obj_interest_2[graph_info['class_objects'] == graph_helper.object_dict.get_id('kitchentable')] = 1.
    
        # We only care about edges that from is in mask_obj_interest
        edge_interest_from = mask_obj_interest.repeat_interleave(num_nodes, dim=2)
        edge_interest_to = mask_obj_interest_2.repeat(1, 1, num_nodes)
        edge_interest = edge_interest_from * edge_interest_to
    else:
        edge_interest = mask_obj_interest
        edge_dict['id_nothing'] = node_ids_none[:, 0, 0]
    
    edge_dict['gt_edges'] = gt_edges
    edge_dict['edge_interest'] = edge_interest
    return edge_dict



# Nice vectorized sampling function 
# https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix/34190035
def vectorized(prob_matrix):
    s = prob_matrix.cumsum(axis=-1)
    r = np.random.rand(*prob_matrix.shape[:-1])[..., None]
    # ipdb.set_trace()
    k = (s < r).sum(axis=-1)
    if k.max() > prob_matrix.shape[-1]:
        ipdb.set_trace()
    k[k == prob_matrix.shape[-1]] = prob_matrix.shape[-1] - 1
    return k


def obtain_graph_from_graph_dict(graph_helper, graphs):
    state_names = [(graph_helper.states[it],) for it in range(4)]
    edge_names = [graph_helper.relation_dict.get_el(it) for it in range(nedges)]
    info = {'results': [], 'state_names': state_names, 'edge_names': edge_names}


    all_from, all_to, all_edges_input, object_states = [], [], [], []


    for graph in graphs:
        output = utils_rl_agent.obtain_graph(graph, 1)
        object_states.append(output['states_objects'][None, :])
        all_edges_input.append(output['edge_classes'][None, :])
        all_from.append(output['edge_tuples'][:, 0][None, :])
        all_to.append(output['edge_tuples'][:, 1][None, :])

    object_states = np.concatenate(object_states, 0)
    all_edges_input = np.concatenate(all_edges_input, 0)
    all_from = np.concatenate(all_from, 0)
    all_to = np.concatenate(all_to, 0)


    info['nodes'] = output['class_objects']
    info['edge_input'] = all_edges_input
    info['from_id'] = all_from
    info['to_id'] = all_to
    info['states'] = object_states
    return [info]            

def prepare_graph_for_model(graphs, observations, program_hist, args_config, graph_helper):

    max_tsteps = args_config['model']['max_tsteps']
    obs_ids = None
    attributes_include = ['class_objects', 'states_objects', 'object_coords', 'mask_object', 'node_ids', 'mask_obs_node']
    attributes_include += ['edge_tuples', 'edge_classes', 'mask_edge']
    time_graph = {attr_name: [] for attr_name in attributes_include}
    time_graph['mask_close'] = []

    ##################
    # Build graph
    it = 0
    for graph in graphs:
        graph_info, _ = graph_helper.build_graph(
                        graph, character_id=1, include_edges=True, 
                        obs_ids=observations[it], relative_coords=args_config.model.relative_coords,
                        unique_from=args_config.model.exclusive_edge)
        it += 1
        for attribute_name in attributes_include:
            time_graph[attribute_name].append(torch.tensor(graph_info[attribute_name]))
        
        # Build closeness and goal mask
        close_rel_id = graph_helper.relation_dict.get_id('CLOSE')
        close_nodes = list(graph_info['edge_tuples'][graph_info['edge_classes'] == close_rel_id])
        
        mask_close = np.zeros(graph_info['class_objects'].shape)
        mask_goal = np.zeros(graph_info['class_objects'].shape) 

        # fill up the closeness mask
        if len(close_nodes) > 0:
            indexe = [int(edge[1]) for edge in close_nodes if edge[0] == 0]
            if len(indexe) > 0:
                mask_close[np.array(indexe)] = 1.0

        time_graph['mask_close'].append(torch.tensor(mask_close))

    # batch graph

    for attribute_name in time_graph.keys():
        unpadded_tensor = torch.cat([item[None, :] for item in time_graph[attribute_name]]).float()
        time_graph[attribute_name] = unpadded_tensor[None, :]
        # print(attribute_name, unpadded_tensor.shape)
    # ipdb.set_trace()
    ####################

    node_ids = graph_info['node_ids']
    indexgraph2ind = {node_id: idi for idi, node_id in enumerate(node_ids)}

    ##################
    # Build program
    # We will start with a No-OP action
    program_batch = {
        'action': [],
        'obj1': [],
        'obj2': [],
        'indobj1': [],
        'indobj2': [],
    }

    for it, instr in enumerate(program_hist):
        
        # we want to add an ending action
        # if it >= max_tsteps - 1:
        #     break
        instr_item = graph_helper.actionstr2index(instr)
        program_batch['action'].append(instr_item[0])
        program_batch['obj1'].append(instr_item[1])
        program_batch['obj2'].append(instr_item[2])
        try:
            program_batch['indobj1'].append(indexgraph2ind[instr_item[1]])
            program_batch['indobj2'].append(indexgraph2ind[instr_item[2]])
        except:
            #print("Index", index, program, it)
            raise Exception
    
    if False:
        program_batch['action'].append(args_config.model.max_actions  - 1)
        program_batch['obj1'].append(-1)
        program_batch['obj2'].append(-1)
        program_batch['indobj1'].append(indexgraph2ind[-1])
        program_batch['indobj2'].append(indexgraph2ind[-1])

    # batch program
    for prog_key, prog_val in program_batch.items():
        program_batch[prog_key] = torch.tensor(prog_val)[None, :]
    ######################


    num_tsteps = program_batch['action'].shape[-1]

    length_mask = torch.ones(num_tsteps)[None, :]
    # ipdb.set_trace()

    inputs = {
        'program': program_batch,
        'graph': time_graph,
        'mask_len': length_mask
    }
    return inputs

def obtain_graph_3(
    graph_helper,
    graph,
    edge_pred,
    state_pred,
    change_pred,
    mask_edge,
    input_edges,
    len_mask,
    batch_item=0,
    changed_nodes=None,
    include_last=True
):
    
    # TODO: modify this fro exclusive edge perd
    # if changed_edges[0] is not None:
    #     prev_changed_edges = [changed_edges[0], changed_edges[1]]
    #     changed_edges_new = [changed_edges[0], changed_edges[1]]
        
    # prev_step_edges = changed_edges[1].argmax(-1)


    # Sample edge_prob
    # if do_sample:
    #     edge_pred = vectorized(edge_prob)
    # else:
    #     edge_pred = edge_prob.argmax(-1)


    try:
        assert (change_pred == 1).sum() == (change_pred != 0).sum()
    except:
        ipdb.set_trace()
    # ipdb.set_trace()
    try:
        edge_pred = (
            change_pred[...] * edge_pred + (1 - change_pred[...]) * input_edges
        )
    except:
        ipdb.set_trace()

    # We are predicting the next graph, so we sum
    num_tsteps = int(len_mask[batch_item].sum())
    if include_last:
        num_tsteps -= 1
    offset = 0
    nedges = len(graph_helper.relation_dict)
    state_names = [(graph_helper.states[it],) for it in range(4)]
    edge_names = [graph_helper.relation_dict.get_el(it) for it in range(nedges)]

    info = {'results': [], 'state_names': state_names, 'edge_names': edge_names}

    all_edges, all_from, all_to, all_edges_input, all_from_input, all_to_input = [], [], [], [], [], []
    object_states = state_pred[batch_item, :num_tsteps]
    new_mark = []
    # print(num_tsteps)
    for step in range(num_tsteps):
        result = {}
        mask_object = int(graph['mask_object'][batch_item, step + offset].sum())
        object_names = graph['class_objects'][batch_item, step + offset]
        num_nodes = graph['mask_object'].shape[-1]

        node_ids = graph['node_ids'][batch_item, step + offset]

        print_node = False
        obj_names = []
        for nid in range(mask_object):

            class_name = graph_helper.object_dict.get_el(int(object_names[nid]))
            idi = int(node_ids[nid])
            obj_name_complete = f"{class_name}.{idi}"
            obj_names.append(obj_name_complete)

        # ipdb.set_trace()
        current_mask_edge = mask_edge[batch_item, step]
        # current_edge = edge_info[batch_item, step]
        indices_valid = np.where(current_mask_edge == 1)[0]
        new_edge = change_pred[batch_item, step + offset, indices_valid]

        edge_pred_step = edge_pred[batch_item, step + offset, indices_valid]
        edge_input_step = input_edges[batch_item, step + offset, indices_valid]
        # edge_probs = edge_prob[batch_item, step + offset, indices_valid]

        from_id = indices_valid
        to_id = edge_pred_step

        from_id_input = indices_valid
        to_id_input = edge_input_step

        curr_res = {}
        # all_edges.append(edge_prob[None, :].numpy())

        # obtain class
        edge_pred_step_class = obtain_class_edge(from_id, to_id, obj_names, graph_helper.object_dict, graph_helper.relation_dict)
        edge_input_step_class = obtain_class_edge(from_id_input, to_id_input, obj_names, graph_helper.object_dict, graph_helper.relation_dict)
        # ipdb.set_trace()

        all_edges.append(edge_pred_step_class[None, :])
        all_edges_input.append(edge_input_step_class[None, :])
        all_from.append(from_id[None, :])
        all_to.append(to_id[None, :])
        new_mark.append(new_edge[None, :])


        all_from_input.append(from_id_input[None, :])
        all_to_input.append(to_id_input[None, :])

    all_edges = np.concatenate(all_edges, 0)
    all_edges_input = np.concatenate(all_edges_input, 0)
    all_from = np.concatenate(all_from, 0)
    all_to = np.concatenate(all_to, 0)
    all_from_input = np.concatenate(all_from_input, 0)
    all_to_input = np.concatenate(all_to_input, 0)
    new_mark = np.concatenate(new_mark, 0)
    

    info['edge_pred'] = all_edges
    info['edge_input'] = all_edges_input
    info['from_id'] = all_from
    info['to_id'] = all_to
    info['new_marker'] = new_mark

    info['from_id_input'] = all_from_input
    info['to_id_input'] = all_to_input
    info['states'] = object_states

    info['nodes'] = obj_names
    # ipdb.set_trace()
    if change_pred is not None:
        info['changed_edges'] = change_pred
    return info

def obtain_class_edge(from_id, to_id, obj_names, obj_dict, relation_dict):
    info_objects = {
        "objects_inside": [
          "bathroomcabinet",
          "kitchencabinet",
          "cabinet",
          "fridge",
          "stove",
          "dishwasher",
          "microwave"],
        "objects_surface": ["bench",
                             "cabinet",
                             "chair",
                             "coffeetable",
                             "desk",
                             "kitchencounter",
                             "kitchentable",
                             "nightstand",
                             "sofa"],
        "objects_grab": [
                         "apple",
                         "book",
                         "coffeepot",
                         "cupcake",
                             "cutleryfork",
                         "juice",
                         "pancake",
                         "plate",
                         "poundcake",
                         "pudding",
                         "remotecontrol",
                         "waterglass",
                         "whippedcream",
                         "wine",
                         "wineglass"],
        "others": ["character"]

    }
    relations = []
    for index in list(to_id):
        obj_name = obj_names[index].split('.')[0]

        obj_class_name = obj_name.split('.')[0]
        if obj_name in info_objects['objects_inside']:
            relation =  relation_dict.get_id('inside')
        elif obj_name == 'character':
            relation = relation_dict.get_id('hold')
        elif obj_name in info_objects['objects_surface']:
            relation = relation_dict.get_id('on')
        elif obj_name in info_objects['objects_grab']:
            relation = relation_dict.get_id('on')
        else:
            relation = relation_dict.get_id('inside')

        relations.append(relation)
    return np.array(relations)


def obtain_graph(
    graph_helper,
    graph,
    edge_prob,
    state_prob,
    mask_edge,
    changed_edges,
    batch_item,
    len_mask,
    changed_nodes=None,
    samples=None
):
    
    all_samples = []
    if changed_edges[0] is not None:
        prev_changed_edges = [changed_edges[0], changed_edges[1]]
        changed_edges_new = [changed_edges[0], changed_edges[1]]
        
    prev_step_edges = changed_edges[1].argmax(-1)

    do_sample = True

    if samples is None:
        samples = 1
        do_sample = False
        # edge_prob = edge_prob.cpu().numpy(
    else:
        pass
        # edge_prob = nn.functional.softmax(edge_prob, dim=-1).cpu().numpy()
    
    for sample in range(samples):
        # Sample edge_prob
        if do_sample:
            edge_pred = vectorized(edge_prob)
        else:
            edge_pred = edge_prob.argmax(-1)
        if changed_edges[0] is not None:
            changed_edges = prev_changed_edges    
            if changed_edges[0].shape[2] != changed_edges[1].shape[2]:
                # Changes as nodes
                num_nodes = changed_edges[0].shape[2]
                changed_edges_build = changed_edges[0].repeat(num_nodes, axis=2)
                changed_edges = [changed_edges_build, changed_edges[1]]


            # Sample changed edges
            if do_sample:
                changed_edges_new[0] = vectorized(changed_edges[0])
            else:
                changed_edges_new[0] = changed_edges[0].argmax(-1)
            # The changed edges should be boolean at this point
            # ipdb.set_trace()


            try:
                assert (changed_edges_new[0] == 1).sum() == (changed_edges_new[0] != 0).sum()
            except:
                ipdb.set_trace()
            # ipdb.set_trace()
            try:
                edge_pred = (
                    changed_edges_new[0][...] * edge_pred + (1 - changed_edges_new[0][...]) * prev_step_edges
                )
            except:
                ipdb.set_trace()


        # We are predicting the next graph, so we sum
        num_tsteps = int(len_mask[batch_item].sum()) - 1
        offset = 0
        nedges = len(graph_helper.relation_dict)
        state_names = [(graph_helper.states[it],) for it in range(4)]
        edge_names = [graph_helper.relation_dict.get_el(it) for it in range(nedges)]
        info = {'results': [], 'state_names': state_names, 'edge_names': edge_names}
        all_edges, all_from, all_to, all_edges_input = [], [], [], []
        object_states = state_prob[batch_item, :num_tsteps].numpy()
        for step in range(num_tsteps):
            result = {}
            mask_object = int(graph['mask_object'][batch_item, step + offset].sum())
            object_names = graph['class_objects'][batch_item, step + offset]
            num_nodes = graph['mask_object'].shape[-1]

            node_ids = graph['node_ids'][batch_item, step + offset]

            print_node = False
            obj_names = []
            for nid in range(mask_object):

                class_name = graph_helper.object_dict.get_el(int(object_names[nid]))
                idi = int(node_ids[nid])
                obj_name_complete = f"{class_name}.{idi}"
                obj_names.append(obj_name_complete)

            current_mask_edge = mask_edge[batch_item, step]
            # current_edge = edge_info[batch_item, step]
            indices_valid = np.where(current_mask_edge == 1)[0]

            edge_pred_step = edge_pred[batch_item, step + offset, indices_valid]
            edge_input_step = prev_step_edges[batch_item, step + offset, indices_valid]
            # edge_probs = edge_prob[batch_item, step + offset, indices_valid]

            from_id = indices_valid // num_nodes
            to_id = indices_valid % num_nodes

            curr_res = {}
            # all_edges.append(edge_prob[None, :].numpy())
            all_edges.append(edge_pred_step[None, :])
            all_edges_input.append(edge_input_step[None, :])
            all_from.append(from_id[None, :])
            all_to.append(to_id[None, :])

        all_edges = np.concatenate(all_edges, 0)
        all_edges_input = np.concatenate(all_edges_input, 0)
        all_from = np.concatenate(all_from, 0)
        all_to = np.concatenate(all_to, 0)
        info['edge_pred'] = all_edges
        info['edge_input'] = all_edges_input
        info['from_id'] = all_from
        info['to_id'] = all_to
        info['states'] = object_states

        info['nodes'] = obj_names
        # ipdb.set_trace()
        if len(changed_edges) > 0:
            info['changed_edges'] = changed_edges[0]
        all_samples.append(info)
    return all_samples


def print_graph_3(
    graph_helper,
    graph,
    edge_info,
    mask_edge,
    state_info,
    input_edge,
    change_pred,
    batch_item,
    step,
    changed_nodes=None,
):


    # If we are only predicitng edge change, the edge is a combination of previous edge and new, modulagted by prediction
    if not change_pred is None:
        if change_pred.shape[2] != input_edge.shape[2]:
            # Changes as nodes
            num_nodes = change_pred.shape[2]

            if torch.is_tensor(change_pred):
                changed_edges_build = change_pred.repeat_interleave(num_nodes, dim=2)
            else:
                changed_edges_build = change_pred.repeat(num_nodes, axis=2)
            changed_edges = changed_edges_build

        else:

            changed_edges = change_pred
        # The changed edges should be boolean at this point
        # ipdb.set_trace()
        try:
            assert (changed_edges == 1).sum() == (changed_edges != 0).sum()
        except:
            ipdb.set_trace()
        # ipdb.set_trace()
        # try:
        edge_info = (
            changed_edges * edge_info + (1 - changed_edges) * input_edge
        )
        # except:
        #     ipdb.set_trace()


    # We are predicting the next graph, so we sum 1
    offset = 1
    mask_object = int(graph['mask_object'][batch_item, step + offset].sum())
    object_names = graph['class_objects'][batch_item, step + offset]
    object_states = state_info[batch_item, step]
    num_nodes = graph['mask_object'].shape[-1]

    # object_coords = graph['object_coords'][batch_item, step+offset]

    # ipdb.set_trace()
    node_ids = graph['node_ids'][batch_item, step + offset]

    # ipdb.set_trace()
    print_node = False
    print("Graph")
    print("==========")
    if print_node:
        print("Nodes:")
    obj_names = []
    for nid in range(mask_object):

        state_names = [
            graph_helper.states[it]
            for it in range(4)
            if int(object_states[nid][it]) == 1
        ]
        state_names = ' '.join(state_names)
        class_name = graph_helper.object_dict.get_el(int(object_names[nid]))
        idi = int(node_ids[nid])
        # coords = list(object_coords[nid][:3])
        # coords_str = '{:.2f}, {:.2f}, {:.2f}'.format(coords[0], coords[1], coords[2])
        obj_name_complete = f"{class_name}.{idi}"
        obj_name_complete += ' ' * (20 - len(obj_name_complete))
        obj_names.append(obj_name_complete)
        if print_node:
            print(f"{obj_name_complete}. {state_names}")

    if print_node:
        print('\n')

    print('Edges')
    current_mask_edge = mask_edge[batch_item, step]
    current_edge = edge_info[batch_item, step]
    # Only store on and inside edges, and hold



    num_edge =  (current_mask_edge > 0).sum()
    on_from = np.arange(num_edge)
    on_to = current_edge[:num_edge]
    from_edge = {}

    for elem_from, elem_to in zip(on_from.tolist(), on_to.tolist()):
        if int(elem_to) not in from_edge:
            from_edge[int(elem_to)] = []
        from_edge[int(elem_to)].append(int(elem_from))

    all_elems = sorted(list(set(list(from_edge.keys()))))

    # ipdb.set_trace()
    # print("HOLDING:", list(zip(hold_from, hold_to)))
    for elem in all_elems:
        on_curr = []
        if elem in from_edge:
            on_curr = from_edge[elem]
        # ipdb.set_trace()
        on_str = ' '.join([obj_names[itt].strip() for itt in on_curr])
        elem2 = obj_names[elem]
        print(f'{elem2}: relation: [{on_str}]')
    # print("==========")


def print_graph_2(
    graph_helper,
    graph,
    edge_info,
    mask_edge,
    state_info,
    changed_edges,
    batch_item,
    step,
    changed_nodes=None,
):

    # If we are only predicitng edge change, the edge is a combination of previous edge and new, modulagted by prediction
    if len(changed_edges) > 0:

        if changed_edges[0].shape[2] != changed_edges[1].shape[2]:
            # Changes as nodes
            num_nodes = changed_edges[0].shape[2]

            if torch.is_tensor(changed_edges[0]):
                changed_edges_build = changed_edges[0].repeat_interleave(num_nodes, dim=2)
            else:
                changed_edges_build = changed_edges[0].repeat(num_nodes, axis=2)
            changed_edges = [changed_edges_build, changed_edges[1]]


        # The changed edges should be boolean at this point
        # ipdb.set_trace()
        assert (changed_edges[0] == 1).sum() == (changed_edges[0] != 0).sum()

        # ipdb.set_trace()
        try:
            edge_info = (
                changed_edges[0] * edge_info + (1 - changed_edges[0]) * changed_edges[1]
            )
        except:
            ipdb.set_trace()
    # We are predicting the next graph, so we sum 1
    offset = 1
    mask_object = int(graph['mask_object'][batch_item, step + offset].sum())
    object_names = graph['class_objects'][batch_item, step + offset]
    object_states = state_info[batch_item, step]
    num_nodes = graph['mask_object'].shape[-1]

    # object_coords = graph['object_coords'][batch_item, step+offset]

    # ipdb.set_trace()
    node_ids = graph['node_ids'][batch_item, step + offset]

    # ipdb.set_trace()
    print_node = False
    print("Graph")
    print("==========")
    if print_node:
        print("Nodes:")
    obj_names = []
    for nid in range(mask_object):

        state_names = [
            graph_helper.states[it]
            for it in range(4)
            if int(object_states[nid][it]) == 1
        ]
        state_names = ' '.join(state_names)
        class_name = graph_helper.object_dict.get_el(int(object_names[nid]))
        idi = int(node_ids[nid])
        # coords = list(object_coords[nid][:3])
        # coords_str = '{:.2f}, {:.2f}, {:.2f}'.format(coords[0], coords[1], coords[2])
        obj_name_complete = f"{class_name}.{idi}"
        obj_name_complete += ' ' * (20 - len(obj_name_complete))
        obj_names.append(obj_name_complete)
        if print_node:
            print(f"{obj_name_complete}. {state_names}")

    if print_node:
        print('\n')

    print('Edges')
    current_mask_edge = mask_edge[batch_item, step]
    current_edge = edge_info[batch_item, step]
    # Only store on and inside edges, and hold
    inside_of = {}
    id_inside = graph_helper.relation_dict.get_id('inside')
    id_on = graph_helper.relation_dict.get_id('on')
    id_hold = graph_helper.relation_dict.get_id('hold')

    inside = np.where(
        np.logical_and(current_edge == id_inside, current_mask_edge == 1)
    )[0]
    on = np.where(np.logical_and(current_edge == id_on, current_mask_edge == 1))[0]
    hold = np.where(np.logical_and(current_edge == id_hold, current_mask_edge == 1))[0]

    inside_from, inside_to = (inside // num_nodes), inside % num_nodes
    on_from, on_to = (on // num_nodes), on % num_nodes
    hold_from, hold_to = (hold // num_nodes), hold % num_nodes

    on = {}
    inside_of = {}
    for elem_from, elem_to in zip(inside_from.tolist(), inside_to.tolist()):
        if int(elem_to) not in inside_of:
            inside_of[int(elem_to)] = []
        inside_of[int(elem_to)].append(int(elem_from))

    for elem_from, elem_to in zip(on_from.tolist(), on_to.tolist()):
        if int(elem_to) not in on:
            on[int(elem_to)] = []
        on[int(elem_to)].append(int(elem_from))

    all_elems = sorted(list(set(list(on.keys()) + list(inside_of.keys()))))

    print("HOLDING:", list(zip(hold_from, hold_to)))
    for elem in all_elems:
        inside_curr, on_curr = [], []
        if elem in inside_of:
            inside_curr = inside_of[elem]
        if elem in on:
            on_curr = on[elem]
        # ipdb.set_trace()
        on_str = ' '.join([obj_names[itt].strip() for itt in on_curr])
        inside_str = ' '.join([obj_names[itt].strip() for itt in inside_curr])
        elem2 = obj_names[elem]
        print(f'{elem2}: ON: [{on_str}]   INSIDE: [{inside_str}]')
    print("==========")


def print_graph(graph_helper, graph, batch_item, step):
    mask_object = int(graph['mask_object'][batch_item, step].sum())
    object_names = graph['class_objects'][batch_item, step]
    object_states = graph['states_objects'][batch_item, step]
    object_coords = graph['object_coords'][batch_item, step]
    # ipdb.set_trace()
    node_ids = graph['node_ids'][batch_item, step]
    # ipdb.set_trace()
    print("Graph")
    print("==========")
    print("Nodes:")
    obj_names = []
    for nid in range(mask_object):

        state_names = [
            graph_helper.states[it]
            for it in range(4)
            if int(object_states[nid][it]) == 1
        ]
        state_names = ' '.join(state_names)
        class_name = graph_helper.object_dict.get_el(int(object_names[nid]))
        idi = int(node_ids[nid])
        coords = list(object_coords[nid][:3])
        coords_str = '{:.2f}, {:.2f}, {:.2f}'.format(coords[0], coords[1], coords[2])
        obj_name_complete = f"{class_name}.{idi}"
        obj_name_complete += ' ' * (20 - len(obj_name_complete))
        obj_names.append(obj_name_complete)

        print(f"{obj_name_complete} ({coords_str}). {state_names}")
    print('\nEdges')
    # Only store on and inside edges
    inside_of = {}
    id_inside = graph_helper.relation_dict.get_id('inside')
    id_on = graph_helper.relation_dict.get_id('on')
    inside = graph['edge_tuples'][batch_item, step][
        graph['edge_classes'][batch_item, step] == id_inside
    ]
    inside_from, inside_to = inside[:, 0], inside[:, 1]
    on = graph['edge_tuples'][batch_item, step][
        graph['edge_classes'][batch_item, step] == id_on
    ]
    on_from, on_to = on[:, 0], on[:, 1]

    on = {}
    inside_of = {}
    for elem_from, elem_to in zip(inside_from.tolist(), inside_to.tolist()):
        if int(elem_to) not in inside_of:
            inside_of[int(elem_to)] = []
        inside_of[int(elem_to)].append(int(elem_from))

    for elem_from, elem_to in zip(on_from.tolist(), on_to.tolist()):
        if int(elem_to) not in on:
            on[int(elem_to)] = []
        on[int(elem_to)].append(int(elem_from))

    all_elems = sorted(list(set(list(on.keys()) + list(inside_of.keys()))))
    for elem in all_elems:
        inside_curr, on_curr = [], []
        if elem in inside_of:
            inside_curr = inside_of[elem]
        if elem in on:
            on_curr = on[elem]
        # ipdb.set_trace()
        on_str = ' '.join([obj_names[itt].strip() for itt in on_curr])
        inside_str = ' '.join([obj_names[itt].strip() for itt in inside_curr])
        elem2 = obj_names[elem]
        print(f'{elem2}: ON: [{on_str}]   INSIDE: [{inside_str}]')
    print("==========")


def print_script(graph_helper, program):
    program_str = decode_program(graph_helper, program)
    ipdb.set_trace()


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_pred_results_str(graph_helper, prog_gt, prog_pred):
    program_gt = decode_program(graph_helper, prog_gt)
    program_pred = decode_program(graph_helper, prog_pred)
    res_str = ''
    res_str += "GT{}|Pred\n".format(' ' * 43)
    if len(program_pred) < len(program_gt):
        program_pred += [' '] * (len(program_gt) - len(program_pred))
    else:
        program_pred[: len(program_gt)]

    for instr_gt, instr_pred in zip(program_gt, program_pred):
        res_str += '{: <45}| {}\n'.format(instr_gt, instr_pred)

    res_str += '{}\n'.format('-' * 45)
    return res_str


def decode_program(graph_helper, program_info, index=0):
    # ipdb.set_trace()
    program_info_new = {}

    program_info['class_objects'] = program_info['graph']['class_objects']
    program_info['node_ids'] = program_info['graph']['node_ids']
    for key, val in program_info.items():
        if key != 'graph':

            program_info_new[key] = val[index, :]

    length = int(program_info_new['mask_len'].sum())

    action_ind = list(program_info_new['action'][:length])
    o1_ind = list(program_info_new['o1'][:length])
    o2_ind = list(program_info_new['o2'][:length])

    class_obj = list(program_info_new['class_objects'][index])
    node_ids = list(program_info_new['node_ids'][index])

    program_str = []
    # ipdb.set_trace()
    # print(length)
    for it in range(length):
        if action_ind[it] == len(graph_helper.action_dict):
            action_str = '[None]'
        else:
            action_str = '[{}]'.format(graph_helper.action_dict.get_el(action_ind[it]))
        o1_id = int(node_ids[o1_ind[it]])
        o2_id = int(node_ids[o2_ind[it]])
        # ipdb.set_trace()
        if o1_id != -1:
            action_str += ' <{}> ({})'.format(
                graph_helper.object_dict.get_el(int(class_obj[o1_ind[it]])), o1_id
            )
        if o2_id != -1:
            action_str += ' <{}> ({})'.format(
                graph_helper.object_dict.get_el(int(class_obj[o2_ind[it]])), o2_id
            )

        program_str.append(action_str)
    return program_str


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, do_print=True):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]

        if do_print:
            print('\t'.join(entries))
        else:
            return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def get_epsilon(init_eps, end_eps, num_steps, episode):
    return (init_eps - end_eps) * (
        1.0 - min(1.0, float(episode) / float(num_steps))
    ) + end_eps


class AggregatedStats:
    def __init__(self):
        self.success_per_apt = {}
        self.success_per_goal = {}

    def reset(self):
        success_per_apt = {}
        success_per_goal = {}

    def add(self, success, goal, apt):
        if goal not in self.success_per_goal:
            sg_count, g_count = 0, 0
        else:
            sg_count, g_count = self.success_per_goal[goal]
        if apt not in self.success_per_apt:
            sa_count, a_count = 0, 0
        else:
            sa_count, a_count = self.success_per_apt[apt]
        r = 1 if success else 0
        self.success_per_goal[goal] = [sg_count + r, g_count + 1]
        self.success_per_apt[apt] = [sa_count + r, a_count + 1]

    def add_list(self, info_list):
        for item_info in info_list:
            self.add(item_info['success'], item_info['goal'], item_info['apt'])

    def barplot(self, values, names=None, special_ids=None):
        fig, ax = plt.subplots()
        index = range(values.shape[0])

        a1 = plt.bar(index, values)

        if names is not None:
            ax.set_xticks(np.asarray([i for i in range(len(names))]))
            ax.set_xticklabels(names, rotation=65)
        if special_ids is not None:
            a1[special_ids].set_color('r')
        plt.tight_layout()
        return fig

    def create_histogram(self, cont_dict):
        keys = sorted(list(cont_dict.keys()))
        values = [cont_dict[x] for x in keys]
        success = np.array([x[0] for x in values])
        cont = np.array([x[1] for x in values])
        return self.barplot(success, keys), self.barplot(cont, keys)

    def print_hist(self, tb_writer):
        img_goal, cnt_goal = self.create_histogram(self.success_per_goal)
        img_apt, cnt_apt = self.create_histogram(self.success_per_apt)

        tb_writer.add_figure("histogram/success_per_goal", img_goal)
        tb_writer.add_figure("histogram/success_per_apt", img_apt)
        tb_writer.add_figure("histogram/count_per_goal", cnt_goal)
        tb_writer.add_figure("histogram/count_per_apt", cnt_apt)


def dict2md(content, tabs=0):
    mdown_str = ''
    for key, val in content.items():
        if type(val) == dict:
            actual_str = '  <br />' + dict2md(val, tabs=tabs + 2)
        else:
            actual_str = str(val)

        mdown_str += '{}**{}**: {}  <br />'.format('..' * tabs, key, actual_str)
    # print(tabs, mdown_str)
    return mdown_str


class LoggerSteps:
    def __init__(self, args, log_steps=True):
        self.args = args
        self.log_steps = log_steps
        self.experiment_name = self.get_experiment_name()
        self.wandb = None
        self.save_dir = os.path.dirname(get_original_cwd())

        self.name_log = None if len(args.name_log) == 0 else args.name_log+str(random.randint(0,100))

        self.ckpt_save_dir = os.path.join(self.save_dir, 'ckpts', self.experiment_name)
        self.results_path = os.path.join(self.save_dir, 'results', self.experiment_name)
        self.logs_logdir = os.path.join(
            self.save_dir, 'logs_model', self.experiment_name
        )

        now = datetime.datetime.now()
        self.tstmp = now.strftime('%Y-%m-%d_%H-%M-%S')
        self.set_tensorboard()
        self.first_log = False
        # self.stats = AggregatedStats()

        # self.plot = Plotter2(self.experiment_name, root_dir=self.logs_logdir)
        self.info_episodes = []
        self.file_name_log = '{}/{}/log.json'.format(
            self.logs_logdir, self.experiment_name
        )
        print("Saving to: {}".format(self.experiment_name))

        # f.writelines(json.dumps(dict_args, indent=4))

    def set_tensorboard(self):
        if self.log_steps:
            self.wandb = wandb.init(
                project="graph-prediction",
                name=self.name_log,
                entity='virtualhome',
                config=OmegaConf.to_container(self.args),
            )

    def get_experiment_name(self):
        args = self.args
        pred_change = 'none'

        if args['model']['predict_edge_change']:
            pred_change = 'edge'
        if args['model']['predict_node_change']:
            pred_change = 'node'
        experiment_name = (
            'predict_graph/train_data.{}-agents{}/'
            'lr{}-bs.{}-'
            'goalenc.{}_extended._costclose.{}_costgoal.{}_agentembed.{}_predchange.{}_inputgoal.{}_excledge.{}_preddiff.{}_goodaction'
            '{}_logname.{}'
        ).format(
            args['train']['lr'],
            args['train']['batch_size'],
            args['model']['goal_inp'],
            args['train']['loss_close'],
            args['train']['loss_goal'],
            args['model']['agent_embed'],
            pred_change,
            args['model']['input_goal'],
            args['model']['exclusive_edge'],
            args['model']['predict_diff'],
            'reduced_walk' if args['model']['condense_walking'] else '',
            args['name_log']
        )
        if args['model']['gated']:
            experiment_name += '_gated'
        if 'VAE' in args['model']['time_aggregate']:
            experiment_name += '_condprior.{}_zvec.{}'.format(args['model']['cond_prior'], args['model']['zvec'])
        if 'exp_name' in self.args and self.args.exp_name != '':
            experiment_name += '_{}'.format(args['exp_name'])
        if 'debug' in args:
            experiment_name += 'debug'
        if args['train']['overfit']:
            experiment_name += 'overfit'
        return experiment_name

    def log_embeds(self, total_num_steps, embed_info):
        embeddings = embed_info.weight
        embedding_labels = []
        for i in range(embeddings.shape[0]):
            agentn = int(i / 5)
            seedn = i % 5
            embedding_labels.append("agent.{}_seed.{}".format(agentn, seedn))

        self.wandb.add_embedding(embeddings, metadata=embedding_labels)

    def log_data2(self, total_num_steps, info):
        if self.first_log:
            self.first_log = False
            if self.tensorboard_logdir is not None:
                self.set_tensorboard()

        if self.wandb is not None:
            for agent_id in info.keys():
                for acc_name, acc_item in info[agent_id]['accuracy'].items():
                    self.wandb.log(
                        "agents/accuracy/agent_{}/{}".format(agent_id, acc_name),
                        acc_item,
                        total_num_steps,
                    )

    def log_data(self, total_num_steps, info):
        if self.first_log:
            self.first_log = False
            if self.tensorboard_logdir is not None:
                self.set_tensorboard()

        res_dict = {}
        res_dict['total_num_steps'] = total_num_steps
        res_dict['epoch'] = info['misc']['epoch']

        if self.wandb is not None:
            for loss_name, loss_item in info['losses'].items():
                res_dict.update({"losses/{}".format(loss_name): loss_item})

            for acc_name, acc_item in info['accuracy'].items():

                res_dict.update({"accuracy/{}".format(acc_name): acc_item})

            if 'misc' in info.keys():
                for acc_name, acc_item in info['misc'].items():
                    res_dict.update({"misc/{}".format(acc_name): acc_item})
            
            # if 'misc_hist' in info.keys():
            #     ipdb.set_trace()

            # ipdb.set_trace()
            self.wandb.log(res_dict)


    def log_info(self, info_ep):
        try:
            os.makedirs(self.logs_logdir)

        except:
            pass

        log_txt_file = '{}/logs.txt'.format(self.logs_logdir)
        with open(log_txt_file, 'a+') as f:
            f.write(info_ep['str'])

    def save_model(self, j, model, optimizer):

        save_path = os.path.join(self.ckpt_save_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.wandb.log({'misc/ckpt': wandb.Html('<a>{}</a>'.format(str(save_path)))})
        with open('{}/config.yaml'.format(self.ckpt_save_dir), 'w+') as f:
            f.write(OmegaConf.to_yaml(self.args))
        # ipdb.set_trace()
        torch.save(
            {'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
            os.path.join(save_path, "{}.pt".format(j)),
        )
