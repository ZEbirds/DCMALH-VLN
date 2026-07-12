import open3d as o3d
import numpy as np
import json
import os
import matplotlib.pyplot as plt

def create_hierarchical_visualization(ply_path, json_path, z_offsets, node_radii, layer_colors):
    """
    生成用于论文的分层 DSG 3D 可视化 (纯净几何版，无文字干扰)
    """
    print("1. 加载底层点云/Mesh...")
    pcd = o3d.io.read_point_cloud(ply_path)
    
    # 可选：如果点云太暗，可以人为调亮一点，方便论文展示
    colors = np.asarray(pcd.colors)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors * 1.2, 0, 1))

    print("2. 解析 DSG JSON...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # 兼容处理 nodes 和 edges
    nodes_data = data.get('nodes', [])
    if isinstance(nodes_data, dict):
        nodes_data = list(nodes_data.values())
        
    edges_data = data.get('edges', [])
    if isinstance(edges_data, dict):
        edges_data = list(edges_data.values())

    rendered_nodes = {}
    geometries = [pcd]  

    print(f"3. 生成各层级节点 (共 {len(nodes_data)} 个)...")
    for node in nodes_data:
        node_id = node.get('id')
        layer = node.get('layer', 2) 
        attrs = node.get('attributes', {})
        pos = attrs.get('position')

        if pos is None:
            continue

        z_offset = z_offsets.get(layer, 0.0)
        display_pos = [pos[0], pos[1], pos[2] + z_offset]
        rendered_nodes[node_id] = {'pos': display_pos, 'layer': layer}

        radius = node_radii.get(layer, 0.1)
        color = layer_colors.get(layer, [0.5, 0.5, 0.5])

        if layer >= 2: 
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.translate(display_pos)
            sphere.paint_uniform_color(color)
            sphere.compute_vertex_normals()
            geometries.append(sphere)

    print(f"4. 生成层级连线 (共 {len(edges_data)} 条)...")
    line_points = []
    lines = []
    line_colors = []
    
    point_idx_map = {} 
    
    for edge in edges_data:
        source_id = edge.get('source')
        target_id = edge.get('target')

        if source_id in rendered_nodes and target_id in rendered_nodes:
            s_node = rendered_nodes[source_id]
            t_node = rendered_nodes[target_id]
            
            if source_id not in point_idx_map:
                point_idx_map[source_id] = len(line_points)
                line_points.append(s_node['pos'])
            if target_id not in point_idx_map:
                point_idx_map[target_id] = len(line_points)
                line_points.append(t_node['pos'])
                
            lines.append([point_idx_map[source_id], point_idx_map[target_id]])
            
            higher_layer = max(s_node['layer'], t_node['layer'])
            line_colors.append(layer_colors.get(higher_layer, [0.7, 0.7, 0.7]))

    if len(lines) > 0:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        geometries.append(line_set)

    print("5. 启动可视化器...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Paper DSG Visualization", width=1920, height=1080)
    
    for geom in geometries:
        vis.add_geometry(geom)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0]) 
    opt.point_size = 2.0
    opt.line_width = 2.0 

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    base_dir = "../outputs/lhpr_vln_verification_old/5_00557-fRZhp6vWGw7_/backend/"
    ply_path = os.path.join(base_dir, "mesh.ply")
    json_path = os.path.join(base_dir, "dsg_with_mesh.json") 
    
    Z_OFFSETS = {
        2: 0,      # 物体层：保持在原始物理位置
        3: 3.5,    # 地点层：悬浮在 3.5 米
        4: 7.5,    # 房间层：悬浮在 7.5 米
        5: 11.5    # 建筑层：在最顶端
    }
    
    NODE_RADII = {2: 0.15, 3: 0.15, 4: 0.40, 5: 0.60}
    LAYER_COLORS = {
        2: [0.8, 0.2, 0.2], 3: [0.2, 0.8, 0.2], 4: [0.2, 0.2, 0.8], 5: [0.6, 0.2, 0.8]
    }

    create_hierarchical_visualization(ply_path, json_path, Z_OFFSETS, NODE_RADII, LAYER_COLORS)