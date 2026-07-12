import json
import os
import networkx as nx
import matplotlib.pyplot as plt

def decode_hydra_id(node_id):
    """解密底层的 64位长数字 ID"""
    try:
        node_id_int = int(node_id)
        category_char = chr(node_id_int >> 56)
        index = node_id_int & ((1 << 56) - 1)
        prefix_map = {'o': 'Obj', 'p': 'Place', 'r': 'Room', 'b': 'Bldg', 'f': 'Frontier', 'a': 'Agent'}
        return f"{prefix_map.get(category_char, category_char)}_{index}"
    except Exception:
        return str(node_id)

def draw_topology_graph(json_path, skip_regions=True):
    """
    生成 2D 语义拓扑图 (跨层级缝合版)
    """
    print(f"正在解析 {json_path} ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes_data = data.get('nodes', [])
    if isinstance(nodes_data, dict):
        nodes_data = list(nodes_data.values())
        
    edges_data = data.get('edges', [])
    if isinstance(edges_data, dict):
        edges_data = list(edges_data.values())

    G = nx.Graph()
    G_full = nx.Graph() # 辅助图：用来记录它原本完整的连通性

    # 1. 记录所有的原始连线
    for edge in edges_data:
        G_full.add_edge(str(edge.get('source')), str(edge.get('target')))
        
    layer_colors_map = {
        2: '#CC3333', 
        3: '#33CC33', 
        4: '#3333CC', 
        5: '#9933CC'  
    }

    valid_nodes = set()
    for node in nodes_data:
        layer = node.get('layer', 2)
        node_id = str(node.get('id'))
        
        # ==========================================
        # 🛡️ 过滤一：屏蔽所有的历史轨迹点 (Agent)
        # ==========================================
        if 'agent' in node_id.lower():
            continue

        # ==========================================
        # 🛡️ 过滤二：是否屏蔽区域 (Region/Place)
        # ==========================================
        if skip_regions and layer == 3:
            continue
            
        attrs = node.get('attributes', {})
        name_candidates = [attrs.get('name'), attrs.get('semantic_label'), node.get('name')]
        valid_names = [str(n) for n in name_candidates if n is not None and str(n).strip() != ""]
        
        if valid_names and not valid_names[0].isdigit():
            node_name = valid_names[0]
        else:
            node_name = decode_hydra_id(node_id)

        valid_nodes.add(node_id)
        color = layer_colors_map.get(layer, '#888888')
        
        G.add_node(node_id, label=node_name, layer=layer, color=color)

    # 2. 先把存活节点的直接连线画上
    for edge in edges_data:
        u, v = str(edge.get('source')), str(edge.get('target'))
        if u in valid_nodes and v in valid_nodes:
            G.add_edge(u, v)

    # ==========================================
    # 🎯 核心黑科技：精准缝合断层 (Star-like Edge Contraction)
    # 严格让失去 Region 的 Object 只挂载到 Room 上，形成清晰的从属关系
    # ==========================================
    if skip_regions:
        for node in nodes_data:
            if node.get('layer') == 3: # 找到被隐身的 Region 节点
                region_id = str(node.get('id'))
                if region_id in G_full:
                    # 找出这个 Region 连着的所有存活节点
                    neighbors = [n for n in G_full.neighbors(region_id) if n in valid_nodes]
                    
                    # 分类：找出里面的 Room 和 Object
                    rooms = [n for n in neighbors if G.nodes[n]['layer'] >= 4] 
                    objects = [n for n in neighbors if G.nodes[n]['layer'] == 2] 
                    
                    # 让 Object 直接连接到 Room，防止 Object 之间互相乱连
                    for r in rooms:
                        for o in objects:
                            G.add_edge(r, o)

    if len(G.nodes) == 0:
        print("没有符合条件的节点可画！")
        return

    # ==========================================
    # 🎨 矩形文本框风格 - 高清晰度论文排版版
    # ==========================================
    # 缩小画布尺寸并提高 DPI，这会强迫文字在视觉上变大
    plt.figure(figsize=(16, 10), facecolor='white')
    
    # k=0.15 缩小排斥力，让节点靠得更近（不再是稀疏的星空）
    pos = nx.spring_layout(G, k=0.15, iterations=200, seed=42)

    # 提取节点标签和颜色
    labels = {n: G.nodes[n]['label'] for n in G.nodes}
    
    # 1. 先画连线：置于底层，颜色深一点保证清晰
    nx.draw_networkx_edges(G, pos, edge_color='#888888', width=1.0, alpha=0.5)

    # 2. 核心：不直接画圆球，而是遍历节点画“带背景色的文字框”
    ax = plt.gca()
    for node, (x, y) in pos.items():
        layer = G.nodes[node]['layer']
        label = labels[node]
        
        # 根据层级设定不同的颜色
        bg_color = G.nodes[node]['color']
        
        # 设定字号：Room/Building 用超大字，Object 用大字
        f_size = 12 if layer >= 4 else 9
        
        # 使用文本框代替球体
        ax.text(x, y, label,
                size=f_size,
                fontweight='bold',
                color='white',
                ha='center', va='center',
                bbox=dict(
                    boxstyle=f"round,pad=0.4", # 矩形框，带圆角和衬距
                    facecolor=bg_color, 
                    edgecolor='white', 
                    linewidth=1,
                    alpha=0.9
                ))

    # 标题
    plt.title("Semantic Topology (Room & Objects)", fontsize=20, fontweight='bold', pad=20)
    plt.axis('off')
    
    # 自动裁剪多余留白，让图占满整张纸
    plt.tight_layout()
    
    output_img = os.path.join(os.path.dirname(json_path), "2D_Text_Topology.png")
    # 保持超高分辨率
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"✅ 矩形文本版拓扑图已生成: {output_img}")
    plt.show()

if __name__ == "__main__":
    base_dir = "../outputs/lhpr_vln_verification/5_00557-fRZhp6vWGw7_/"
    json_path = os.path.join(base_dir, "enhanced_dsg.json") 
    
    # skip_regions=True: 屏蔽掉绿色的地点节点，直接连出 房间->物体 的结构
    draw_topology_graph(json_path, skip_regions=True)