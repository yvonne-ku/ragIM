import pickle
import networkx as nx

# 加载社区图
with open('/data/outputs/community_graph.pkl', 'rb') as f:
    G = pickle.load(f)

# 按社区分组实体和chunks
community_data = {}
# 先收集每个社区的chunks
for node in G.nodes():
    if G.nodes[node].get('type') == 'chunk':
        comm = G.nodes[node].get('community', -1)
        if comm not in community_data:
            community_data[comm] = {
                "chunks": [],
                "entities": [],
                "neighbors": {}
            }
        if node not in community_data[comm]["chunks"]:
            community_data[comm]["chunks"].append(node)
# 再收集每个社区的实体
for node in G.nodes():
    if G.nodes[node].get('type') == 'entity':
        comm = G.nodes[node].get('community', -1)
        if comm not in community_data:
            community_data[comm] = {
                "chunks": [],
                "entities": [],
                "neighbors": {}
            }
        entity_name = G.nodes[node].get('name', node)
        if entity_name not in community_data[comm]["entities"]:
            community_data[comm]["entities"].append(entity_name)

# 收集每个实体的邻居
for comm, data in community_data.items():
    for entity_name in data["entities"]:
        entity_id = f'entity::{entity_name}'
        neighbors = list(G.neighbors(entity_id))
        neighbor_list = []
        for neighbor in neighbors:
            neighbor_type = G.nodes[neighbor].get('type', 'unknown')
            if neighbor_type == 'entity':
                neighbor_name = G.nodes[neighbor].get('name', neighbor)
                neighbor_list.append(neighbor_name)
            elif neighbor_type == 'chunk':
                neighbor_list.append(neighbor)
        data["neighbors"][entity_name] = neighbor_list

# 输出符合要求的格式
for comm, data in community_data.items():
    print('{')
    print(f'  "community": {comm},')
    print('  "chunks": [')
    for i, chunk in enumerate(data["chunks"]):
        if i == len(data["chunks"]) - 1:
            print(f'    "{chunk}"')
        else:
            print(f'    "{chunk}",')
    print('  ],')
    print('  "entities": [')
    for i, entity in enumerate(data["entities"]):
        if i == len(data["entities"]) - 1:
            print(f'    "{entity}"')
        else:
            print(f'    "{entity}",')
    print('  ],')
    print('  "neighbors": [')
    for i, entity in enumerate(data["entities"]):
        neighbors_str = '[' + ', '.join(f'"{neighbor}"' for neighbor in data["neighbors"][entity]) + ']'
        if i == len(data["entities"]) - 1:
            print(f'    {{"{entity}": {neighbors_str}}}')
        else:
            print(f'    {{"{entity}": {neighbors_str}}},')
    print('  ]')
    print('}')
    print()