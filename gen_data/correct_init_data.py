import json
import glob
import ipdb

file_all = 'data/real_object_placing.json'
file_name_size = 'data/class_name_size.json'
files_correct = []
names = []



def convert(list_containers):
    return ['{}.{}'.format(x[0], x[1]) for x in list_containers]

def correct(dest_name):
    dest_name = dest_name.replace('_', '')
    map_dest = {
        'freezer': 'fridge',
        'bathroom_cabinet': 'bathroomcabinet',
        'mini-fridge': 'fridge',
        'filingcabinet': 'cabinet',
        'tablecloth': 'kitchentable',
        'placemat': 'kitchentable',
        'dishrack': 'dishwasher',
        'oven': 'stove',
        'loveseat': 'sofa',
        'couch': 'sofa',
        'saucepan': 'fryingpan',
        'painobench': 'bench'

    }
    if dest_name not in map_dest:
        return dest_name
    return map_dest[dest_name]

def correct2(dest_name):
    map_dest = {
        'barsoap': 'soap',
        'washingsponge': 'sponge',
        'dishbowl': 'bowl',
        'cutleryknife': 'knife',
        'cutleryfork': 'fork',
        'glasses': 'spectacles',
        'pancake': 'food_dessert',
        'milkshake': 'food_dessert',
        'carrot': 'food_carrot',
        'salmon': 'food_fish',
        'sundae': 'food_dessert',
        'cupcake': 'food_dessert',
        'chicken': 'food_chicken',
        'bananas': 'food_fruit',
        'bellpepper': 'food_fruit',
        'poundcake': 'food_cake',
        'plum': 'food_fruit',
        'pear': 'food_fruit',
        'apple': 'food_fruit',
        'lime': 'food_fruit',
        'boardgame': 'board_game',
        'wineglass': 'wine_glass',
        'waterglass': 'water_glass',
        'toiletpaper': 'toilet_paper',
        'pie': 'food_cake',
        'cuttingboard': 'cutting_board',
        'toothpaste': 'tooth_paste',
        'remotecontrol': 'remote_control',
        'coffeepot': 'coffee_pot',
        'cereal': 'food_cereal',
        'condimentbottle': 'food_salt',
        'condimentshaker': 'food_salt',
        'oventray': 'tray'
    }
    if dest_name not in map_dest:
        return dest_name
    return map_dest[dest_name]

with open(file_all, 'r') as f:
    cont = json.load(f)


with open(file_name_size, 'r') as f:
    name_size = json.load(f)

all_names = []
for i in range(7):
    file_name = f'data/object_info{i+1}.json'
    with open(file_name, 'r') as f:
        map_content = json.load(f)
    files_correct.append(map_content)
    names.append(list(map_content.keys()))
    all_names += names[-1]
all_names = set(all_names)

reported = []
object_info_final = {}
ignore_dest = ['washing_machine', 'pantry']
for name in all_names:
    used_rel = []
    object_info_final[name] = []
    newname = correct2(name)
    if newname not in cont and newname not in reported:
        # print("Missing name", name)
        reported.append(name)
    else:
        for containers in cont[newname]:
            dest = correct(containers['destination'])
            rel = containers['relation']

            if dest not in name_size and dest not in reported:
                # print("Missing dest", dest)
                reported.append(dest)
            else:
                if rel == 'IN':
                    rel = 'INSIDE'
                if rel in ['ON', 'INSIDE']:
                    if (rel, dest) not in used_rel:
                        object_info_final[name].append([rel, dest])
                        used_rel.append((rel, dest))
with open('data/object_info_final.json', 'w+') as f:
    f.write(json.dumps(object_info_final, indent=4))