import open3d as o3d
import numpy as np
import json
import matplotlib.pyplot as plt
import os

def load_json_layers(json_path, num_points):
    """
    解析 dsg.json，根据节点的 position 构建各层级的颜色划分
    """
    if not os.path.exists(json_path):
        print(f"⚠️ 找不到 JSON 文件: {json_path}")
        return {}

    print(f"正在解析 JSON 文件: {json_path} ...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取 JSON 失败: {e}")
        return {}

    # 兼容 Hydra 的 JSON 结构 (节点通常在一个名为 'nodes' 的列表或字典里)
    nodes = []
    if isinstance(data, list):
        nodes = data
    elif isinstance(data, dict):
        if 'nodes' in data:
            # 如果 'nodes' 是字典，提取它的 values，否则直接用
            nodes = list(data['nodes'].values()) if isinstance(data['nodes'], dict) else data['nodes']
        else:
            nodes = [v for k, v in data.items() if isinstance(v, dict) and 'layer' in v]

    print(f"✅ 成功读取了 {len(nodes)} 个节点数据，正在进行层级归类...")

    # 按 layer 分组存放节点坐标和 ID
    layer_data = {}
    for node in nodes:
        layer_id = node.get('layer')
        attrs = node.get('attributes', {})
        pos = attrs.get('position')
        node_id = node.get('id')

        if layer_id is not None and pos is not None:
            if layer_id not in layer_data:
                layer_data[layer_id] = {'positions': [], 'ids': []}
            layer_data[layer_id]['positions'].append(pos)
            layer_data[layer_id]['ids'].append(node_id)

    # 为每一个提取出的层级，计算点云的颜色
    layer_colors = {}
    cmap = plt.get_cmap("tab20")

    for layer_id, l_data in layer_data.items():
        positions = np.array(l_data['positions'])
        ids = np.array(l_data['ids'])
        unique_ids = np.unique(ids)
        
        print(f" - Layer {layer_id} 包含 {len(positions)} 个空间节点，正在划分点云空间...")

        # 给这个层级的每个唯一节点分配一个颜色
        id_to_color = {uid: cmap(i % 20)[:3] for i, uid in enumerate(unique_ids)}

        # 使用 KDTree 进行空间划分 (Voronoi 镶嵌)
        centers_pcd = o3d.geometry.PointCloud()
        centers_pcd.points = o3d.utility.Vector3dVector(positions)
        kdtree = o3d.geometry.KDTreeFlann(centers_pcd)

        colors = np.zeros((num_points, 3))
        # （由于这个预计算在 Python 里跑一个大循环可能会耗时几秒钟，请耐心等待）
        for i in range(num_points):
            # 因为不知道点云坐标，这里依赖于外部传入一个点云对象的引用，稍作修改
            pass 
            
    return layer_data

def calculate_colors_for_layer(ply_points, positions, ids):
    """使用 KDTree 快速为点云上色"""
    cmap = plt.get_cmap("tab20")
    unique_ids = np.unique(ids)
    
    # 打乱颜色顺序，防止相邻 ID 颜色相近
    colors_palette = [cmap(i % 20)[:3] for i in range(len(unique_ids))]
    np.random.seed(42)
    np.random.shuffle(colors_palette)
    id_to_color = dict(zip(unique_ids, colors_palette))

    centers_pcd = o3d.geometry.PointCloud()
    centers_pcd.points = o3d.utility.Vector3dVector(positions)
    kdtree = o3d.geometry.KDTreeFlann(centers_pcd)

    colors = np.zeros_like(ply_points)
    for i in range(len(ply_points)):
        [_, idx, _] = kdtree.search_knn_vector_3d(ply_points[i], 1)
        colors[i] = id_to_color[ids[idx[0]]]
        
    return colors

def main():
    base_dir = "../outputs/lhpr_vln_verification/15_00269-JNiWU5TZLtt_/backend/"
    ply_path = os.path.join(base_dir, "mesh.ply")
    json_path = os.path.join(base_dir, "dsg_with_mesh.json") 
    
    if not os.path.exists(ply_path):
        print(f"⚠️ 找不到点云文件: {ply_path}")
        return

    print("正在加载基础点云...")
    original_pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(original_pcd.points)
    original_colors = np.asarray(original_pcd.colors)

    # 预计算屋顶索引 (隐藏最高的 15%)
    z_values = points[:, 2] 
    height_threshold = np.max(z_values) - (np.max(z_values) - np.min(z_values)) * 0.15 
    keep_indices = np.where(z_values < height_threshold)[0]

    # 解析 JSON 数据
    layer_data = load_json_layers(json_path, len(points))
    
    # 存储不同图层的颜色
    print("\n📦 正在为点云上色 (可能需要 5-10 秒，请耐心等待)...")
    layer_colors_map = {'0': original_colors}
    
    available_layers = list(layer_data.keys())
    available_layers.sort()

    for idx, layer_id in enumerate(available_layers):
        positions = np.array(layer_data[layer_id]['positions'])
        ids = np.array(layer_data[layer_id]['ids'])
        
        # 将层级 ID 映射为键盘上的数字键 (1, 2, 3...)
        key_str = str(idx + 1)
        if int(key_str) <= 9: 
            layer_colors_map[key_str] = calculate_colors_for_layer(points, positions, ids)

    # 状态变量
    state = {'roof_visible': True, 'current_layer': '0'}

    display_pcd = o3d.geometry.PointCloud()
    display_pcd.points = o3d.utility.Vector3dVector(points)
    display_pcd.colors = o3d.utility.Vector3dVector(original_colors)

    def update_rendering(vis):
        current_color_array = layer_colors_map.get(state['current_layer'], original_colors)
            
        if state['roof_visible']:
            display_pcd.points = o3d.utility.Vector3dVector(points)
            display_pcd.colors = o3d.utility.Vector3dVector(current_color_array)
        else:
            display_pcd.points = o3d.utility.Vector3dVector(points[keep_indices])
            display_pcd.colors = o3d.utility.Vector3dVector(current_color_array[keep_indices])
            
        vis.update_geometry(display_pcd)
        return False

    def toggle_roof(vis):
        state['roof_visible'] = not state['roof_visible']
        print("🏠 屋顶已" + ("显示" if state['roof_visible'] else "隐藏"))
        return update_rendering(vis)

    def set_layer(layer_key, layer_name):
        def callback(vis):
            if layer_key in layer_colors_map:
                state['current_layer'] = layer_key
                print(f"🎨 已切换至: {layer_name}")
                return update_rendering(vis)
            else:
                print(f"⚠️ 当前数据集没有这一层。")
        return callback

    # 创建可视化窗口
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Ultimate DSG JSON Viewer", width=1280, height=720)
    
    opt = vis.get_render_option()
    opt.point_size = 4.0 

    vis.add_geometry(display_pcd)

    # 绑定快捷键
    vis.register_key_callback(ord('R'), toggle_roof)
    vis.register_key_callback(ord('0'), set_layer('0', "原始彩色 (Original RGB)"))
    
    # 动态注册键盘快捷键
    print("\n" + "="*50)
    print("🚀 交互式 DSG 图谱查看器已就绪！")
    print("操作指南 (请在弹出的 3D 窗口中按键):")
    print(" [ 0 ] - 恢复原始彩色点云")
    
    for idx, layer_id in enumerate(available_layers):
        key_char = str(idx + 1)
        if int(key_char) <= 9:
            layer_name = f"Layer {layer_id} 染色 (基于 JSON 数据)"
            print(f" [ {key_char} ] - 切换到 {layer_name}")
            vis.register_key_callback(ord(key_char), set_layer(key_char, layer_name))
            
    print(" [ R ] - 隐藏 / 显示 屋顶")
    print("="*50 + "\n")

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()