import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import plotly.graph_objects as go
from pathlib import Path
import pickle
import os
from typing import Dict, List, Tuple, Optional

class SceneGraphVisualizer:
    """可视化场景图工具类"""
    
    def __init__(self, sg_path: Path, output_dir: Optional[Path] = None):
        """
        初始化可视化工具
        
        Args:
            sg_path: 场景图JSON文件路径
            output_dir: 输出目录
        """
        self.sg_path = Path(sg_path)
        self.output_dir = output_dir or self.sg_path.parent / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载场景图
        self.load_scene_graph()
        
        # 颜色映射
        self.node_type_colors = {
            'agent': 'red',
            'room': 'blue',
            'object': 'green',
            'region': 'yellow',
            'frontier': 'purple',
            'building': 'brown',
            'unknown': 'gray'
        }
        
        self.edge_type_colors = {
            'room-to-object': 'blue',
            'room-to-agent': 'red',
            'room-to-region': 'yellow',
            'region-to-object': 'green',
            'agent-to-object': 'orange',
            'object-to-object': 'lightgreen',
            'frontier-to-object': 'purple',
            'unknown': 'gray'
        }
    
    def load_scene_graph(self):
        """加载场景图"""
        with open(self.sg_path, 'r') as f:
            self.scene_graph = json.load(f)
        
        # 转换为NetworkX图
        self.G = nx.node_link_graph(self.scene_graph)
        print(f"场景图加载成功: {len(self.G.nodes())} 个节点, {len(self.G.edges())} 条边")
    
    def analyze_graph(self):
        """分析场景图结构"""
        print("\n=== 场景图分析 ===")
        
        # 统计节点类型
        node_types = {}
        for node, data in self.G.nodes(data=True):
            node_type = self._get_node_type(node, data)
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        print("节点类型统计:")
        for node_type, count in node_types.items():
            print(f"  {node_type}: {count}")
        
        # 统计边类型
        edge_types = {}
        for u, v, data in self.G.edges(data=True):
            edge_type = data.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        print("\n边类型统计:")
        for edge_type, count in edge_types.items():
            print(f"  {edge_type}: {count}")
        
        # 找到中心节点（度最高）
        if len(self.G.nodes()) > 0:
            degrees = dict(self.G.degree())
            max_degree_node = max(degrees.items(), key=lambda x: x[1])
            print(f"\n中心节点: {max_degree_node[0]} (度: {max_degree_node[1]})")
            
            # 找到可能的agent节点
            agent_nodes = [n for n, d in self.G.nodes(data=True) 
                          if 'agent' in n.lower() or 'agent' in str(d.get('name', '')).lower()]
            if agent_nodes:
                print(f"Agent节点: {agent_nodes}")
    
    def _get_node_type(self, node_id: str, node_data: Dict) -> str:
        """根据节点ID和数据判断节点类型"""
        node_str = str(node_id).lower()
        
        if 'agent' in node_str:
            return 'agent'
        elif 'room' in node_str:
            return 'room'
        elif 'object' in node_str:
            return 'object'
        elif 'region' in node_str:
            return 'region'
        elif 'frontier' in node_str:
            return 'frontier'
        elif 'building' in node_str:
            return 'building'
        else:
            return 'unknown'
    
    def get_node_positions(self) -> Dict:
        """提取所有节点的3D位置"""
        positions = {}
        
        for node, data in self.G.nodes(data=True):
            if 'position' in data:
                pos = data['position']
                if isinstance(pos, list) and len(pos) >= 3:
                    positions[node] = np.array([pos[0], pos[1], pos[2]])
            elif 'pos' in data:
                pos = data['pos']
                if isinstance(pos, list) and len(pos) >= 3:
                    positions[node] = np.array([pos[0], pos[1], pos[2]])
        
        print(f"提取到 {len(positions)} 个节点的位置信息")
        return positions
    
    def get_trajectory_from_agent(self) -> List[np.ndarray]:
        """从agent节点提取轨迹"""
        trajectory = []
        
        # 查找所有agent节点
        agent_nodes = []
        for node, data in self.G.nodes(data=True):
            if 'agent' in str(node).lower():
                agent_nodes.append((node, data))
        
        # 按时间戳排序（如果有的话）
        if all('timestamp' in data for _, data in agent_nodes):
            agent_nodes.sort(key=lambda x: x[1].get('timestamp', 0))
        
        # 提取位置
        for node, data in agent_nodes:
            if 'position' in data:
                pos = data['position']
                if isinstance(pos, list) and len(pos) >= 3:
                    trajectory.append(np.array([pos[0], pos[1], pos[2]]))
        
        return trajectory
    
    def visualize_2d(self, show_labels: bool = True, figsize: Tuple[int, int] = (12, 10)):
        """2D可视化场景图"""
        positions = self.get_node_positions()
        
        if not positions:
            print("没有找到位置信息，无法进行2D可视化")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制节点
        for node, pos in positions.items():
            node_type = self._get_node_type(node, self.G.nodes[node])
            color = self.node_type_colors.get(node_type, 'gray')
            
            # 只使用x和z坐标（假设y是高度）
            ax.scatter(pos[0], pos[2], color=color, s=100, alpha=0.7, 
                      label=node_type if node_type not in ax.get_legend_handles_labels()[1] else "")
            
            if show_labels:
                # 显示节点标签
                label = self.G.nodes[node].get('name', node)
                ax.annotate(str(label), (pos[0], pos[2]), fontsize=8)
        
        # 绘制边
        for u, v, data in self.G.edges(data=True):
            if u in positions and v in positions:
                pos_u = positions[u]
                pos_v = positions[v]
                
                edge_type = data.get('type', 'unknown')
                color = self.edge_type_colors.get(edge_type, 'gray')
                
                ax.plot([pos_u[0], pos_v[0]], [pos_u[2], pos_v[2]], 
                       color=color, alpha=0.3, linewidth=1)
        
        # 绘制轨迹（如果有）
        trajectory = self.get_trajectory_from_agent()
        if len(trajectory) > 1:
            traj_x = [p[0] for p in trajectory]
            traj_z = [p[2] for p in trajectory]
            ax.plot(traj_x, traj_z, 'r-', linewidth=2, alpha=0.7, label='Agent Trajectory')
            
            # 标记起点和终点
            ax.scatter(traj_x[0], traj_z[0], color='green', s=200, marker='o', label='Start')
            ax.scatter(traj_x[-1], traj_z[-1], color='red', s=200, marker='s', label='End')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title(f'Scene Graph Visualization - {self.sg_path.name}')
        ax.grid(True, alpha=0.3)
        
        # 添加图例
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        
        if unique_handles:
            ax.legend(unique_handles, unique_labels, loc='upper right')
        
        plt.tight_layout()
        
        # 保存图像
        output_path = self.output_dir / f"{self.sg_path.stem}_2d.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"2D可视化已保存到: {output_path}")
        
        plt.show()
    
    def visualize_3d_matplotlib(self, show_labels: bool = False, figsize: Tuple[int, int] = (12, 10)):
        """使用Matplotlib进行3D可视化"""
        positions = self.get_node_positions()
        
        if not positions:
            print("没有找到位置信息，无法进行3D可视化")
            return
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制节点
        for node, pos in positions.items():
            node_type = self._get_node_type(node, self.G.nodes[node])
            color = self.node_type_colors.get(node_type, 'gray')
            
            ax.scatter(pos[0], pos[1], pos[2], color=color, s=50, alpha=0.7, 
                      label=node_type if node_type not in ax.get_legend_handles_labels()[1] else "")
        
        # 绘制边
        for u, v, data in self.G.edges(data=True):
            if u in positions and v in positions:
                pos_u = positions[u]
                pos_v = positions[v]
                
                edge_type = data.get('type', 'unknown')
                color = self.edge_type_colors.get(edge_type, 'gray')
                
                ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]],
                       color=color, alpha=0.3, linewidth=1)
        
        # 绘制轨迹（如果有）
        trajectory = self.get_trajectory_from_agent()
        if len(trajectory) > 1:
            traj_x = [p[0] for p in trajectory]
            traj_y = [p[1] for p in trajectory]
            traj_z = [p[2] for p in trajectory]
            ax.plot(traj_x, traj_y, traj_z, 'r-', linewidth=3, alpha=0.7, label='Agent Trajectory')
            
            # 标记起点和终点
            ax.scatter(traj_x[0], traj_y[0], traj_z[0], color='green', s=200, marker='o', label='Start')
            ax.scatter(traj_x[-1], traj_y[-1], traj_z[-1], color='red', s=200, marker='s', label='End')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Scene Graph - {self.sg_path.name}')
        
        # 添加图例
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        
        if unique_handles:
            ax.legend(unique_handles, unique_labels, loc='upper right')
        
        # 保存图像
        output_path = self.output_dir / f"{self.sg_path.stem}_3d_matplotlib.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"3D Matplotlib可视化已保存到: {output_path}")
        
        plt.show()
    
    def visualize_3d_plotly(self, interactive: bool = True):
        """使用Plotly进行交互式3D可视化"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("请安装plotly: pip install plotly")
            return
        
        positions = self.get_node_positions()
        
        if not positions:
            print("没有找到位置信息，无法进行3D可视化")
            return
        
        fig = go.Figure()
        
        # 按节点类型分组
        node_traces = {}
        for node_type in self.node_type_colors.keys():
            node_traces[node_type] = {
                'x': [], 'y': [], 'z': [],
                'names': [], 'ids': []
            }
        
        # 收集节点数据
        for node, pos in positions.items():
            node_type = self._get_node_type(node, self.G.nodes[node])
            if node_type not in node_traces:
                node_type = 'unknown'
            
            node_traces[node_type]['x'].append(pos[0])
            node_traces[node_type]['y'].append(pos[1])
            node_traces[node_type]['z'].append(pos[2])
            node_traces[node_type]['names'].append(self.G.nodes[node].get('name', node))
            node_traces[node_type]['ids'].append(node)
        
        # 添加节点轨迹
        for node_type, data in node_traces.items():
            if data['x']:  # 如果该类型有节点
                fig.add_trace(go.Scatter3d(
                    x=data['x'],
                    y=data['y'],
                    z=data['z'],
                    mode='markers',
                    name=node_type,
                    marker=dict(
                        size=8,
                        color=self.node_type_colors.get(node_type, 'gray'),
                        opacity=0.8
                    ),
                    text=data['names'],
                    hovertemplate='<b>%{text}</b><br>ID: %{customdata}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
                    customdata=data['ids']
                ))
        
        # 添加边
        edge_x, edge_y, edge_z = [], [], []
        for u, v, data in self.G.edges(data=True):
            if u in positions and v in positions:
                pos_u = positions[u]
                pos_v = positions[v]
                
                # 添加边的起点和终点（中间添加None以断开线条）
                edge_x.extend([pos_u[0], pos_v[0], None])
                edge_y.extend([pos_u[1], pos_v[1], None])
                edge_z.extend([pos_u[2], pos_v[2], None])
        
        if edge_x:
            fig.add_trace(go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode='lines',
                name='Edges',
                line=dict(color='gray', width=2, dash='dash'),
                hoverinfo='skip'
            ))
        
        # 添加轨迹（如果有）
        trajectory = self.get_trajectory_from_agent()
        if len(trajectory) > 1:
            traj_x = [p[0] for p in trajectory]
            traj_y = [p[1] for p in trajectory]
            traj_z = [p[2] for p in trajectory]
            
            fig.add_trace(go.Scatter3d(
                x=traj_x,
                y=traj_y,
                z=traj_z,
                mode='lines+markers',
                name='Agent Trajectory',
                line=dict(color='red', width=4),
                marker=dict(size=4, color='red'),
                hovertemplate='Trajectory Point<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            ))
            
            # 标记起点和终点
            fig.add_trace(go.Scatter3d(
                x=[traj_x[0]],
                y=[traj_y[0]],
                z=[traj_z[0]],
                mode='markers',
                name='Start',
                marker=dict(size=10, color='green', symbol='circle'),
                hovertemplate='Start Point<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[traj_x[-1]],
                y=[traj_y[-1]],
                z=[traj_z[-1]],
                mode='markers',
                name='End',
                marker=dict(size=10, color='red', symbol='square'),
                hovertemplate='End Point<extra></extra>'
            ))
        
        # 更新布局
        fig.update_layout(
            title=f'Interactive 3D Scene Graph - {self.sg_path.name}',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=1000,
            height=800,
            showlegend=True
        )
        
        # 保存为HTML
        output_path = self.output_dir / f"{self.sg_path.stem}_3d_interactive.html"
        fig.write_html(output_path)
        print(f"交互式3D可视化已保存到: {output_path}")
        
        if interactive:
            fig.show()
        
        return fig
    
    def create_summary_report(self):
        """创建场景图总结报告"""
        report = []
        report.append(f"# 场景图分析报告: {self.sg_path.name}")
        report.append(f"生成时间: {self.sg_path.stat().st_mtime}")
        report.append("")
        
        # 基本统计
        report.append("## 基本统计")
        report.append(f"- 节点总数: {len(self.G.nodes())}")
        report.append(f"- 边总数: {len(self.G.edges())}")
        report.append(f"- 图密度: {nx.density(self.G):.4f}")
        
        if nx.is_connected(self.G.to_undirected()):
            report.append(f"- 图直径: {nx.diameter(self.G.to_undirected())}")
        else:
            report.append("- 图包含多个连通分量")
        
        # 节点类型统计
        report.append("\n## 节点类型统计")
        node_types = {}
        for node, data in self.G.nodes(data=True):
            node_type = self._get_node_type(node, data)
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        for node_type, count in sorted(node_types.items()):
            report.append(f"- {node_type}: {count}")
        
        # 边类型统计
        report.append("\n## 边类型统计")
        edge_types = {}
        for u, v, data in self.G.edges(data=True):
            edge_type = data.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        for edge_type, count in sorted(edge_types.items()):
            report.append(f"- {edge_type}: {count}")
        
        # 节点详细信息
        report.append("\n## 重要节点")
        
        # 按度排序
        degrees = dict(self.G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for node, degree in top_nodes:
            node_data = self.G.nodes[node]
            node_type = self._get_node_type(node, node_data)
            name = node_data.get('name', 'N/A')
            report.append(f"- {node} ({node_type}, 度: {degree}): {name}")
        
        # 保存报告
        report_path = self.output_dir / f"{self.sg_path.stem}_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"总结报告已保存到: {report_path}")
        
        # 打印部分报告
        print("\n=== 场景图总结报告 ===")
        for line in report[:20]:  # 只打印前20行
            print(line)
        
        return report
    
    def visualize_all(self):
        """执行所有可视化"""
        print("开始场景图可视化...")
        
        # 分析图结构
        self.analyze_graph()
        
        # 创建总结报告
        self.create_summary_report()
        
        # 执行可视化
        print("\n生成2D可视化...")
        self.visualize_2d(show_labels=False)
        
        print("\n生成3D Matplotlib可视化...")
        self.visualize_3d_matplotlib()
        
        print("\n生成交互式3D可视化...")
        self.visualize_3d_plotly(interactive=False)
        
        print("\n所有可视化已完成！")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化场景图')
    parser.add_argument('--sg_path', type=str, required=True, 
                       help='场景图JSON文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认：场景图同目录下的visualizations文件夹）')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['2d', '3d', 'interactive', 'report', 'all'],
                       help='可视化模式')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = SceneGraphVisualizer(
        sg_path=Path(args.sg_path),
        output_dir=Path(args.output_dir) if args.output_dir else None
    )
    
    # 根据模式执行可视化
    if args.mode == '2d':
        visualizer.visualize_2d()
    elif args.mode == '3d':
        visualizer.visualize_3d_matplotlib()
    elif args.mode == 'interactive':
        visualizer.visualize_3d_plotly(interactive=True)
    elif args.mode == 'report':
        visualizer.create_summary_report()
    else:  # 'all'
        visualizer.visualize_all()


if __name__ == "__main__":
    main()