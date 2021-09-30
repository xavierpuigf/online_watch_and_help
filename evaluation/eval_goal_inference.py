def compute_prec


for i in range(5, 7):
	comm.reset(i)
	s, g = comm.environment_graph()
	door_ids = [node for node in g['nodes'] if node['category'] == "Doors"]
	room_ids = [node['id'] for node in g['nodes'] if node['category'] == "Rooms"]
	print(door_ids)
	for door_id in door_ids:
		edges = [edge['to_id'] for edge in g['edges'] if (edge['from_id'] == door_id) and edge['relation_type'] == "BETWEEN"]
		print(door_id, edges)
	print(edges)
	print('****')