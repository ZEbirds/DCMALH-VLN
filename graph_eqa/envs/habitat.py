import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation as R
from graph_eqa.envs.utils import hydra_get_mesh, get_cam_pose_tsdf, pos_habitat_to_normal

def hydra_output_callback(pipeline, visualizer):
    """Show graph."""
    if visualizer:
        visualizer.update_graph(pipeline.graph)

def _take_step(pipeline, data, pose, labels, image_viz, is_eqa=False, segmenter=None):
    timestamp, world_t_body, q_wxyz = pose
    q_xyzw = np.roll(q_wxyz, -1) #changing to xyzw format

    world_T_body = np.eye(4)
    world_T_body[:3, 3] = world_t_body
    world_T_body[:3, :3] = R.from_quat(q_xyzw).as_matrix()
    data.set_pose(timestamp, world_T_body, is_eqa=is_eqa)

    if data.rgb is not None:
        labels = segmenter(data.rgb) if segmenter else data.labels

    if is_eqa:
        pose_cam = get_cam_pose_tsdf(data.get_depth_sensor_state())
        world_t_body = pose_cam[:3, 3]
        q_xyzw = R.from_matrix(pose_cam[:3, :3]).as_quat()
        q_wxyz = np.roll(q_xyzw, 1)

    pipeline.step(timestamp, world_t_body, q_wxyz, data.depth, labels, data.rgb)

def run(
    pipeline,
    habitat_data,
    pose_source,
    segmenter=None,
    step_callback=hydra_output_callback,
    output_path=None,
    rr_logger=None,
    sg_sim=None,
    tsdf_planner=None,
    save_image=False,
    save_each_step=False,  # 新增：是否每步都保存图片
    step_logger=None,  # 新增：用于保存图片的logger
    step_number=0,  # 新增：当前主步骤编号
    subtask_info=None  # 新增：子任务信息
):
    """
    执行一系列位姿，更新TSDF，重建场景图
    新增参数：
    - save_each_step: 是否在每一步都保存图片
    - step_logger: 用于保存图片的logger
    - step_number: 当前主步骤编号
    - subtask_info: 子任务信息
    """
    agent_positions, agent_quats_wxyz = [], []
    imgs_rgb, imgs_depth, extrinsics = [], [], []
    
    # 计算子步骤总数
    total_substeps = len(pose_source)
    
    # 使用enumerate获取子步骤索引
    for i, pose in enumerate(tqdm(pose_source, desc='Executing traj')):
        pipeline.graph.save(output_path / "dsg.json", False)
        pipeline.graph.save_filtered(output_path / "filtered_dsg.json", False)
        
        if habitat_data.rgb is not None:
            labels = segmenter(habitat_data.rgb) if segmenter else habitat_data.labels
        else:
            labels = np.zeros((640, 480)).astype(int)
            
        _take_step(pipeline, habitat_data, pose, labels, image_viz=None, is_eqa=True, segmenter=segmenter)
        imgs_rgb.append(habitat_data.rgb)
        imgs_depth.append(habitat_data.depth)

        agent_pos, agent_quat_wxyz = habitat_data.get_state(is_eqa=True)
        agent_positions.append(agent_pos)
        agent_quats_wxyz.append(agent_quat_wxyz)
        camera_pos, camera_quat_wxyz = habitat_data.get_camera_pos(is_eqa=True)
        mesh_vertices, mesh_colors, mesh_triangles = hydra_get_mesh(pipeline)

        cam_pose_tsdf = get_cam_pose_tsdf(habitat_data.get_depth_sensor_state())
        extrinsics.append(cam_pose_tsdf)
        pts_normal = pos_habitat_to_normal(pose[1])

        if tsdf_planner:
            tsdf_planner.update(
                habitat_data.rgb,
                habitat_data.depth,
                pts_normal,
                cam_pose_tsdf,
            )
            frontier_nodes = tsdf_planner.frontier_to_sample_normal
            
        if rr_logger:
            rr_logger.log_mesh_data(mesh_vertices, mesh_colors, mesh_triangles)
            rr_logger.log_agent_data(agent_positions)
            rr_logger.log_agent_tf(agent_pos, agent_quat_wxyz)
            rr_logger.log_camera_tf(camera_pos, camera_quat_wxyz)
            rr_logger.log_img_data(habitat_data.rgb, labels)
            rr_logger.step()

        if step_callback:
            step_callback(pipeline, None)
        
        # 新增：如果启用了每步保存且step_logger存在，则保存当前子步骤的图片
        if save_each_step and step_logger is not None:
            try:
                # 创建子步骤信息
                substep_info = {
                    'main_step': step_number,
                    'substep': i,
                    'total_substeps': total_substeps,
                    'pose_index': i,
                    'pose_count': total_substeps,
                    'pose_position': pose[1].tolist() if hasattr(pose[1], 'tolist') else str(pose[1]),
                    'timestamp': pose[0]
                }
                
                # 合并传入的子任务信息
                if subtask_info:
                    # 确保不覆盖子步骤信息
                    for key, value in subtask_info.items():
                        if key not in substep_info:
                            substep_info[key] = value
                
                # 准备动作信息
                action_info = {
                    'type': 'navigation_substep',
                    'description': f'Moving to pose {i+1}/{total_substeps}',
                    'main_step_number': step_number,
                    'substep_number': i
                }
                
                # 保存当前视图
                # 注意：这里直接调用step_logger的save_current_view方法，不调用log_step
                # 因为log_step可能会记录额外数据，而我们只需要保存图片
                current_image_path = step_logger.save_current_view(
                    habitat_data,
                    step_number=f"{step_number}_{i:03d}",  # 例如: "5_001", "5_002"
                    subtask_info=substep_info,
                    action_info=action_info,
                    confidence=None
                )
                
                # if current_image_path:
                #     print(f"Saved substep image: {current_image_path}")
                    
            except Exception as e:
                print(f"Warning: Could not save substep image: {e}")

    # 原有的保存最终图像的逻辑
    if save_image:
        curr_img = Image.fromarray(habitat_data.rgb)
        curr_img.save(output_path / "current_img.png")

    if sg_sim:
        # Should be done after saving default image cos this update overwrites it
        sg_sim.update(
            imgs_rgb=imgs_rgb, 
            imgs_depth=imgs_depth, 
            intrinsics=habitat_data.intrinsics, 
            extrinsics=extrinsics, 
            frontier_nodes=frontier_nodes)
    
    # 返回执行过的位姿
    return pose_source