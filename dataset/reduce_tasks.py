import pickle as pkl
import ipdb
import numpy as np
import random

def reduce_file(file_cont):
    tg = file_cont['task_goal'][0]
    for tgkey, tgval in tg.items():
        if 'sit' in tgkey:
            tg[tgkey] = 0
    pred_counts = list(tg.values())
    pred_keys = list(tg.keys())
    num_preds_curr = sum(pred_counts)

    # only between 2 and 4 goals
    target_preds = min(random.randint(2, 4), num_preds_curr)
    target_preds = 1

    # How many goals we should remove
    preds_to_remove = num_preds_curr - target_preds
    elements_remove = random.sample(list(range(num_preds_curr)), preds_to_remove)

    # Figure out how many preds to remove from each bin
    preds_acum = list(np.cumsum(pred_counts))
    for curr_pred_remove in elements_remove:
        bin_remove = [ind for ind, elem in enumerate(preds_acum) if curr_pred_remove < elem][0]
        tg[pred_keys[bin_remove]] -= 1
        if tg[pred_keys[bin_remove]] < 0:
            ipdb.set_trace()
    updated_tg = {0: tg, 1: tg}
    print(sum(tg.values())) 
    file_cont['task_goal'] = updated_tg
    return file_cont

file_names = ['test_env_task_set_10_full.pik', 'train_env_task_set_20_full.pik']
for file_name in file_names:
    new_name = file_name.replace('full.pik', 'full_reduced_tasks_single.pik')
    with open(file_name, 'rb') as f:
        content = pkl.load(f)

    new_content = [reduce_file(ct) for ct in content]
    with open(new_name, 'wb+') as f:
        pkl.dump(new_content, f)
