# enhanced_lhpr_vln_runner.py
from tqdm import tqdm
from omegaconf import OmegaConf
import click
import os, time, shutil, importlib, math, sys
from pathlib import Path
import numpy as np
import torch
import json
import habitat_sim
import hydra_python
import re
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from datetime import datetime

# ==========================================
# [新增] 终端日志拦截器类
# ==========================================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.__stdout__ # 永远保留原始控制台输出
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

# visualization & GUI libs
import matplotlib.pyplot as plt
plt.ion()
import imageio
import PyQt5
# ensure Qt platform plugins found (user-provided snippet)
dirname = os.path.dirname(PyQt5.__file__)
qt_dir = os.path.join(dirname, 'Qt5', 'plugins', 'platforms')
os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', qt_dir)

# optional magnum (used if path conversion needed)
try:
    import magnum as mn
except Exception:
    mn = None

# Graph EQA imports
from graph_eqa.logging.utils import should_skip_experiment, log_experiment_status
from graph_eqa.envs.utils import pos_habitat_to_normal, pos_normal_to_habitat
from graph_eqa.occupancy_mapping.geom import get_scene_bnds, get_cam_intr
from graph_eqa.envs.habitat import run
from graph_eqa.logging.rr_logger import RRLogger
from graph_eqa.occupancy_mapping.tsdf import TSDFPlanner
from graph_eqa.utils.data_utils import load_eqa_data, get_traj_len_from_poses
from graph_eqa.utils.hydra_utils import initialize_hydra_pipeline
from graph_eqa.scene_graph.scene_graph_sim import SceneGraphSim
from graph_eqa.envs.habitat_interface import HabitatInterface
from graph_eqa.utils.dataloading_utils import load_lhpr_vln_dataset_simple

# Import the new Planner
try:
    from graph_eqa.planners.Ablation_study_planner import VLMPlannerLHPRVLNGPT
except ImportError:
    from Ablation_study_planner import VLMPlannerLHPRVLNGPT

# Try to import Habitat-Lab maps visualizations if available (for nicer topdown)
maps = None
try:
    from habitat.utils.visualizations import maps as maps_module  # habitat-lab v1.0+ path
    maps = maps_module
except Exception:
    try:
        module_path = os.path.abspath('habitat-lab/habitat-lab/habitat/utils/visualizations/maps.py')
        if os.path.exists(module_path):
            spec = importlib.util.spec_from_file_location("maps", module_path)
            maps = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(maps)
    except Exception:
        maps = None

# ----------------------------
# Display helper functions
# ----------------------------
def display_env(observations, action, save_path, step, obj_target):
    """
    Build and optionally save left/front/right RGB and depth images.
    """
    try:
        rgb_r = observations.get("color_sensor_r")
        rgb_l = observations.get("color_sensor_l")
        rgb_f = observations.get("color_sensor_f")
        if rgb_f is None and "color_sensor" in observations:
            rgb_f = observations["color_sensor"]

        def to_pil_rgb(arr):
            if arr is None:
                return Image.new('RGBA', (366, 366), (0,0,0,255))
            a = np.asarray(arr)
            if a.dtype == np.float32 or a.dtype == np.float64:
                a = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                a = a.astype(np.uint8)
            if a.shape[2] > 3:
                a = a[:, :, :3]
            return Image.fromarray(a).convert('RGBA')

        rgb_img_r = to_pil_rgb(rgb_r)
        rgb_img_l = to_pil_rgb(rgb_l)
        rgb_img_f = to_pil_rgb(rgb_f)

        depth_r = observations.get("depth_sensor_r")
        depth_l = observations.get("depth_sensor_l")
        depth_f = observations.get("depth_sensor_f")
        if depth_f is None and "depth_sensor" in observations:
            depth_f = observations["depth_sensor"]

        def to_pil_depth(arr):
            if arr is None:
                return Image.new('L', (366, 366), 0)
            d = np.asarray(arr)
            dnorm = np.clip(d / 10.0, 0.0, 1.0)
            img = (dnorm * 255).astype(np.uint8)
            return Image.fromarray(img, mode='L')

        depth_img_r = to_pil_depth(depth_r)
        depth_img_l = to_pil_depth(depth_l)
        depth_img_f = to_pil_depth(depth_f)

        arr = [rgb_img_l, rgb_img_f, rgb_img_r, depth_img_l, depth_img_f, depth_img_r]
        arr_new = []
        for img in arr:
            try:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.5)
            except: pass
            try: img = img.resize((366, 366))
            except: pass
            arr_new.append(img)
        titles = ["left", "front", "right", "depth_left", "depth_front", "depth_right"]
        arr = arr_new

        if not save_path:
            return arr[:3]

        base_save_dir = os.path.join(save_path, 'temp')
        obj_target_sanitized = obj_target.replace('/', '') if obj_target and '/' in obj_target else (obj_target or '')
        if not os.path.isdir(base_save_dir):
            os.makedirs(base_save_dir, exist_ok=True)
        elif os.path.isdir(base_save_dir) and step == -1:
            shutil.rmtree(base_save_dir)
            os.makedirs(base_save_dir, exist_ok=True)

        current_image_folder = os.path.join(base_save_dir, f"{step}_{action}_for_{obj_target_sanitized}")
        os.makedirs(current_image_folder, exist_ok=True)

        for i, data_img in enumerate(arr):
            image_filename = titles[i] + ".png"
            full_image_path = os.path.join(current_image_folder, image_filename)
            try:
                if data_img.mode == "RGBA":
                    data_img.convert("RGB").save(full_image_path)
                else:
                    data_img.save(full_image_path)
            except Exception as e:
                print(f"Error saving image {full_image_path}: {e}")
        return arr[:3]
    except Exception as e:
        print(f"display_env error: {e}")
        return []

def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([]), key_points=None, save_path=None):
    """
    Display an RGB/semantic/depth triplet (non-blocking) and optionally save to file.
    """
    try:
        d3_40_colors_rgb = None
        try:
            from habitat_sim.utils.common import d3_40_colors_rgb
        except Exception:
            d3_40_colors_rgb = np.vstack([np.random.randint(0,255,(40,3))]).astype(np.uint8)

        rgb_img = Image.fromarray((np.clip(rgb_obs,0,1)*255).astype(np.uint8)) if rgb_obs.dtype in [np.float32, np.float64] else Image.fromarray(rgb_obs)
        rgb_img = rgb_img.convert("RGBA")

        arr = [rgb_img]
        titles = ["rgb"]
        if semantic_obs.size != 0:
            semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
            if d3_40_colors_rgb is None:
                from habitat_sim.utils.common import d3_40_colors_rgb
            palette = d3_40_colors_rgb.flatten()
            semantic_img.putpalette(palette.tolist())
            semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
            semantic_img = semantic_img.convert("RGBA")
            arr.append(semantic_img)
            titles.append("semantic")

        if depth_obs.size != 0:
            depth_img = Image.fromarray((np.clip(depth_obs / 10.0,0,1) * 255).astype(np.uint8), mode="L")
            arr.append(depth_img)
            titles.append("depth")

        n = len(arr)
        fig = plt.figure(figsize=(4*n, 4))
        for i, data in enumerate(arr):
            ax = plt.subplot(1, n, i+1)
            ax.axis("off")
            ax.set_title(titles[i])
            if key_points is not None:
                for point in key_points:
                    plt.plot(point[0], point[1], marker="o", markersize=6, alpha=0.8)
            if isinstance(data, Image.Image):
                ax.imshow(np.asarray(data))
            else:
                ax.imshow(data)
        plt.pause(0.001)

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            out_name = os.path.join(save_path, "display_sample.png")
            fig.savefig(out_name)
        return arr
    except Exception as e:
        print(f"display_sample error: {e}")
        return []

def convert_to_serializable(obj):
    """递归地将 NumPy 类型转换为 Python 原生类型，以便 JSON 序列化"""
    if isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.bool_): return bool(obj)
    elif isinstance(obj, dict): return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)): return [convert_to_serializable(item) for item in obj]
    else: return obj

# ----------------------------
# PurePythonHabitatVisualizer
# ----------------------------
class PurePythonHabitatVisualizer:
    def __init__(self, habitat_data, config=None, realtime=True):
        self.habitat_data = habitat_data
        self.config = config or {}
        self.save_video = bool(self.config.get('save_video', True))
        self.fps = int(self.config.get('fps', 5))
        self.video_frames = []
        self.video_path = self.config.get('video_path', None)
        self.recording = False
        self.realtime = realtime

        self._fig, self._ax = None, None
        if self.realtime:
            try:
                self._fig, self._ax = plt.subplots(figsize=(12,8))
                self._im = None
            except:
                self._fig, self._ax = None, None

        self.trajectory_history = []
        print("📊 Using pure Python visualizer (no OpenCV)")

    def get_current_view(self):
        try:
            obs = self.habitat_data._sim.get_sensor_observations()
            rgb_obs = None
            for key in ["color_sensor", "color_sensor_0", "color_sensor_1", "rgb", "color"]:
                if key in obs and obs.get(key) is not None:
                    rgb_obs = obs.get(key)
                    break
            if rgb_obs is None:
                for k, v in obs.items():
                    try:
                        arr = np.asarray(v)
                        if arr.ndim == 3 and arr.shape[2] >= 3:
                            rgb_obs = arr
                            break
                    except: continue
            if rgb_obs is not None:
                arr = np.array(rgb_obs)
                if arr.dtype in [np.float32, np.float64]:
                    arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
                if arr.shape[2] > 3: arr = arr[:, :, :3]
                return {'rgb': arr, 'observations': obs}
        except Exception as e:
            print(f"Error getting current view: {e}")
        return None

    def _get_topdown_map(self, meters_per_pixel=0.1, height=None, target_pos=None):
        try:
            pathfinder = self.habitat_data.pathfinder
            try:
                agent = self.habitat_data._sim.get_agent(0)
                agent_pos = agent.get_state().position
                curr_x, curr_z = agent_pos[0], agent_pos[2]
                
                if len(self.trajectory_history) > 0:
                    last_x, last_z = self.trajectory_history[-1]
                    if ((curr_x - last_x)**2 + (curr_z - last_z)**2)**0.5 > 2.0:
                        self.trajectory_history = []
                
                if not self.trajectory_history or self.trajectory_history[-1] != (curr_x, curr_z):
                    self.trajectory_history.append((curr_x, curr_z))
                
                if height is None: height = float(agent_pos[1]) + 0.5
            except:
                agent_pos = None
                if height is None:
                    try: bounds = pathfinder.get_bounds(); height = float(bounds[0][1]) + 0.5
                    except: height = 1.0

            try: bounds = pathfinder.get_bounds()
            except: bounds = None

            def draw_trajectory_and_agent(img_obj):
                if agent_pos is None or bounds is None: return img_obj
                try:
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(img_obj)
                    def world_to_pixel(world_x, world_z):
                        px = int((world_x - bounds[0][0]) / meters_per_pixel)
                        py = int((world_z - bounds[0][2]) / meters_per_pixel)
                        return px, py
                    
                    if len(self.trajectory_history) > 1:
                        line_points = [world_to_pixel(x, z) for x, z in self.trajectory_history]
                        draw.line(line_points, fill=(0, 0, 255), width=1, joint="curve")
                        
                    curr_px, curr_py = world_to_pixel(agent_pos[0], agent_pos[2])
                    draw.ellipse((curr_px-1, curr_py-1, curr_px+1, curr_py+1), fill=(255, 0, 0))
                    
                    if target_pos is not None:
                        target_px, target_py = world_to_pixel(target_pos[0], target_pos[2])
                        draw.ellipse((target_px-1, target_py-1, target_px+1, target_py+1), fill=(0, 255, 0))
                        draw.line([(curr_px, curr_py), (target_px, target_py)], fill=(0, 255, 0, 128), width=1)
                except Exception as e:
                    print(f"Error drawing topdown trajectory: {e}")
                return img_obj

            if maps is not None:
                try:
                    hablab_topdown_map = maps.get_topdown_map(pathfinder, height, meters_per_pixel=meters_per_pixel)
                    recolor_map = np.array([[255,255,255], [200,200,200], [50,50,50]], dtype=np.uint8)
                    rgb = recolor_map[np.clip(hablab_topdown_map, 0, 2)] if hablab_topdown_map.ndim == 2 else hablab_topdown_map
                    return draw_trajectory_and_agent(Image.fromarray(rgb.astype(np.uint8)))
                except: pass

            if hasattr(pathfinder, "get_topdown_view"):
                sim_topdown_map = pathfinder.get_topdown_view(meters_per_pixel, height)
                arr = np.asarray(sim_topdown_map)
                if arr.dtype == np.bool_:
                    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
                    rgb[arr], rgb[~arr] = [200, 200, 200], [255, 255, 255]
                else:
                    recolor_map = np.array([[255,255,255],[200,200,200],[50,50,50]], dtype=np.uint8)
                    rgb = recolor_map[np.clip(arr, 0, 2)]
                return draw_trajectory_and_agent(Image.fromarray(rgb.astype(np.uint8)))
        except Exception as e:
            print(f"Error creating top-down map: {e}")
        return Image.new('RGB', (400, 400), (220, 220, 220))

    def create_info_overlay(self, step_info):
        import textwrap
        info_img = Image.new('RGB', (800, 200), (255, 255, 255))
        draw = ImageDraw.Draw(info_img)
        try: font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except:
            try: font = ImageFont.truetype("arial.ttf", 16)
            except: font = ImageFont.load_default()

        y_offset, line_height = 15, 22
        info_lines = [
            f"Step: {step_info.get('step', 0)}",
            f"Subtask: {step_info.get('subtask', 'N/A')}",
            f"Action: {step_info.get('action', 'N/A')}",
            f"Position: {step_info.get('position', 'N/A')}",
            f"Carrying: {step_info.get('carrying', 'Nothing')}"
        ]
        current_y = y_offset
        for line in info_lines:
            wrapped_lines = textwrap.wrap(line, width=55)
            for w_line in wrapped_lines:
                draw.text((10, current_y), w_line, fill=(0,0,0), font=font)
                current_y += line_height
        return info_img

    def combine_images(self, view_img, map_img, info_img):
        try: view_img = view_img.resize((640,480))
        except: pass
        try: map_img = map_img.resize((320,320))
        except: pass
        try: info_img = info_img.resize((800,200))
        except: pass

        combined_height = max(view_img.height, map_img.height + info_img.height)
        combined_width = view_img.width + map_img.width
        combined_img = Image.new('RGB', (combined_width, combined_height), (50,50,50))
        combined_img.paste(view_img, (0,0))
        combined_img.paste(map_img, (view_img.width, 0))
        combined_img.paste(info_img, (view_img.width, map_img.height))
        draw = ImageDraw.Draw(combined_img)
        draw.rectangle([(0,0),(view_img.width-1, view_img.height-1)], outline=(0,255,0), width=2)
        draw.text((10,10), "First Person View", fill=(0,255,0))
        draw.rectangle([(view_img.width,0),(combined_img.width-1,map_img.height-1)], outline=(255,0,0), width=2)
        draw.text((view_img.width+10,10), "Top Down Map", fill=(255,0,0))
        return combined_img

    def update_display(self, step_info=None):
        view_data = self.get_current_view()
        if view_data is None: return None
        view_img = Image.fromarray(view_data['rgb'])
        target_pos = step_info.get('target_pos', None) if step_info else None
        map_img = self._get_topdown_map(meters_per_pixel=self.config.get('meters_per_pixel', 0.1), target_pos=target_pos)
        info_img = self.create_info_overlay(step_info or {})
        combined_img = self.combine_images(view_img, map_img, info_img)

        arr = np.array(combined_img).astype(np.uint8)
        if self.save_video: self.video_frames.append(arr)

        if self.realtime and self._ax is not None:
            try:
                self._ax.clear()
                self._ax.imshow(arr)
                self._ax.axis('off')
                self._fig.canvas.draw()
                self._fig.canvas.flush_events()
            except: pass
        return combined_img

    def start_recording(self, video_path=None):
        if video_path: self.video_path = str(video_path)
        if self.video_path: os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
        self.video_frames, self.recording = [], True
        print(f"📹 Visualizer recording to: {self.video_path}")

    def stop_recording(self):
        if not self.video_frames or not self.video_path:
            self.recording = False
            return
        try:
            with imageio.get_writer(self.video_path, fps=self.fps) as writer:
                for frame in self.video_frames: writer.append_data(frame)
            print(f"📹 Video saved to {self.video_path}")
        except:
            try:
                gif_path = os.path.splitext(self.video_path)[0] + ".gif"
                imageio.mimsave(gif_path, self.video_frames, fps=self.fps)
                print(f"📹 GIF saved to {gif_path}")
            except: pass
        self.video_frames, self.recording = [], False

    def close(self):
        try: self.stop_recording()
        except: pass
        try:
            if self._fig is not None: plt.close(self._fig)
        except: pass

# ----------------------------
# StepImageLogger
# ----------------------------
class StepImageLogger:
    def __init__(self, task_path, experiment_id, habitat_data=None, visualizer_config=None, step_info=None, realtime=True):
        self.task_path = Path(task_path)
        self.experiment_id = experiment_id
        self.habitat_data = habitat_data
        self.step_info = step_info or {}

        self.images_dir = self.task_path / "step_images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir = self.task_path / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self.steps_data = []
        self.current_step = 0

        self.global_action_str = "Initializing..."
        self.global_subtask_str = "Loading tasks..."
        self.global_target_pos = None

        self.visualizer = None
        if habitat_data:
            video_path = self.video_dir / f"{experiment_id}_execution.mp4"
            visualizer_config = visualizer_config or {}
            visualizer_config['video_path'] = str(video_path)
            self.visualizer = PurePythonHabitatVisualizer(habitat_data, visualizer_config, realtime=realtime)
            self.visualizer.start_recording(str(video_path))

    def log_step(self, habitat_data, step_number, step_type, subtask_info=None, action_info=None, confidence=None):
        self.current_step = step_number
        agent_state = self.get_agent_state(habitat_data)
        current_image_path = self.save_current_view(habitat_data, step_number, subtask_info, action_info, confidence)

        step_data = {
            'step_number': step_number,
            'step_type': step_type,
            'timestamp': datetime.now().isoformat(),
            'subtask_info': subtask_info or {},
            'action_info': action_info or {},
            'confidence': confidence,
            'current_image': current_image_path,
            'agent_position': agent_state.get('position'),
            'agent_heading': agent_state.get('heading')
        }
        self.steps_data.append(step_data)
        return step_data

    def save_current_view(self, habitat_data, step_number, subtask_info=None, action_info=None, confidence=None):
        try:
            agent_state = self.get_agent_state(habitat_data)
            if action_info and 'target_pose' in action_info:
                tp = action_info['target_pose']
                if tp is not None:
                    try:
                        tp_array = np.array(tp)
                        tp_hab = pos_normal_to_habitat(tp_array)
                        self.global_target_pos = [float(tp_hab[0]), float(tp_hab[1]), float(tp_hab[2])]
                    except Exception as e: print(f"Error transforming target pose: {e}")
                else: self.global_target_pos = None

            viz_info = {
                'step': step_number,
                'subtask': subtask_info,
                'action': action_info.get('action_output', action_info.get('type', 'N/A')) if action_info else 'N/A',
                'confidence': confidence or 0.0,
                'position': f"({agent_state.get('position', {}).get('x',0):.2f}, {agent_state.get('position', {}).get('z',0):.2f})" if agent_state.get('position') else 'N/A',
                'carrying': subtask_info.get('carried_object', 'Nothing') if subtask_info else 'Nothing',
                'target_pos': self.global_target_pos 
            }

            if self.visualizer:
                combined_img = self._get_combined_image_with_memory(viz_info)
                if combined_img is not None:
                    try: img_path = self.images_dir / f"step_{int(step_number):03d}.png"
                    except: img_path = self.images_dir / f"step_{step_number}.png"
                    combined_img.save(img_path)
                    return str(img_path)
            return self.save_raw_view(habitat_data, step_number)
        except Exception as e: print(f"Error saving view: {e}")
        return None

    def _get_combined_image_with_memory(self, viz_info):
        action_str = viz_info.get('action', 'N/A')
        conf = viz_info.get('confidence', 0.0)
        
        if action_str in ['N/A', '', 'navigation_substep', 'navigation']: action_str = self.global_action_str
        elif conf > 0: action_str = f"{action_str} (Conf: {conf:.2f})"; self.global_action_str = action_str 
        else: self.global_action_str = action_str
            
        subtask_info = viz_info.get('subtask')
        subtask_str = "N/A"
        if isinstance(subtask_info, dict):
            curr_idx = subtask_info.get('current_index', 0)
            tot = subtask_info.get('total', 1)
            desc = ""
            if 'current_subtask' in subtask_info and subtask_info['current_subtask']:
                tsk = subtask_info['current_subtask']
                desc = f" - {tsk.get('type','')} -> {tsk.get('target','')}"
            subtask_str = f"{curr_idx+1}/{tot}{desc}"
        elif isinstance(subtask_info, str): subtask_str = subtask_info

        if subtask_str == 'N/A' or '-' not in subtask_str: subtask_str = self.global_subtask_str
        else: self.global_subtask_str = subtask_str 

        final_viz_info = {
            'step': viz_info.get('step', 0), 'subtask': subtask_str, 'action': action_str,
            'position': viz_info.get('position', 'N/A'), 'carrying': viz_info.get('carrying', 'Nothing'),
            'target_pos': viz_info.get('target_pos')
        }
        return self.visualizer.update_display(final_viz_info)

    def get_agent_state(self, habitat_data):
        try:
            agent = habitat_data._sim.get_agent(0)
            state = agent.get_state()
            return {
                'position': {'x': float(state.position[0]), 'y': float(state.position[1]), 'z': float(state.position[2])},
                'heading': float(habitat_data.get_heading_angle()) if hasattr(habitat_data, 'get_heading_angle') else 0.0
            }
        except: return {}

    def save_raw_view(self, habitat_data, step_number):
        try:
            obs = habitat_data._sim.get_sensor_observations()
            rgb_obs = None
            for key in ["color_sensor", "color_sensor_0", "color_sensor_1", "rgb", "color"]:
                if key in obs and obs.get(key) is not None: rgb_obs = obs.get(key); break
            if rgb_obs is not None:
                arr = np.array(rgb_obs)
                arr = (np.clip(arr,0,1)*255).astype(np.uint8) if arr.dtype in [np.float32, np.float64] else arr.astype(np.uint8)
                if arr.shape[2] > 3: arr = arr[:, :, :3]
                img = Image.fromarray(arr)
                try: img_path = self.images_dir / f"step_{int(step_number):03d}.png"
                except: img_path = self.images_dir / f"step_{step_number}.png"
                img.save(img_path)
                return str(img_path)
        except Exception as e: print(f"Error saving raw view: {e}")
        return None

    def close(self):
        if self.visualizer: self.visualizer.close()
        print(f"📁 Images saved to: {self.images_dir}")

# ----------------------------
# LHPRVLNTaskManager
# ----------------------------
class LHPRVLNTaskManager:
    def __init__(self, subtasks):
        self.subtasks = subtasks
        self.current_index = 0
        self.task_state = {'current_subtask': None, 'status': 'not_started', 'carried_object': None}
        self.history = []
        self._initialize()
        
    def mark_current_failed(self, reason, final_pos=None):
        if self.current_index < len(self.subtasks):
            self.history.append({
                "subtask_index": self.current_index + 1,
                "subtask_type": self.subtasks[self.current_index]['type'],
                "target": self.subtasks[self.current_index].get('target', 'N/A'),
                "success": False, "reason": reason, "final_position": final_pos
            })

    def complete_current_subtask(self, final_pos=None):
        if self.current_index < len(self.subtasks):
            current_subtask = self.subtasks[self.current_index]
            self.history.append({
                "subtask_index": self.current_index + 1,
                "subtask_type": current_subtask['type'],
                "target": current_subtask.get('target', 'N/A'),
                "success": True, "reason": "completed", "final_position": final_pos
            })
            if current_subtask['type'] == 'grab': self.task_state['carried_object'] = current_subtask['target']
            elif current_subtask['type'] == 'release': self.task_state['carried_object'] = None
            
            self.current_index += 1
            if self.current_index < len(self.subtasks):
                self.task_state['current_subtask'] = self.subtasks[self.current_index]
                return False
            else:
                self.task_state['status'] = 'completed'
                return True
        return False
    
    def _initialize(self):
        if self.subtasks and len(self.subtasks) > 0:
            self.task_state['current_subtask'] = self.subtasks[0]
            self.task_state['status'] = 'in_progress'
            
    def get_current_subtask(self):
        return self.subtasks[self.current_index] if self.current_index < len(self.subtasks) else None

    def get_task_progress(self):
        return {'current_index': self.current_index, 'total': len(self.subtasks), 'status': self.task_state['status'], 'carried_object': self.task_state['carried_object']}
    
    def is_task_completed(self):
        return self.current_index >= len(self.subtasks)

# ----------------------------
# Scene File Finder
# ----------------------------
def find_scene_file(scene_id, scene_data_path):
    scene_id_str = str(scene_id)
    possible_filenames = [scene_id_str]
    match = re.match(r'^\d+-(.+)$', scene_id_str)
    if match: possible_filenames.append(match.group(1))
    possible_filenames.append(scene_id_str.split('-')[-1])
    possible_extensions = ['.basis.glb', '.glb', '.basis_v2.glb']
    base_scene_path = Path(scene_data_path)
    scene_dir = base_scene_path / scene_id_str
    if scene_dir.exists() and scene_dir.is_dir():
        for filename in possible_filenames:
            for ext in possible_extensions:
                file_path = scene_dir / (filename + ext)
                if file_path.exists(): return str(file_path)
        for file_path in scene_dir.glob("*.glb"): return str(file_path)
    else:
        for filename in possible_filenames:
            for ext in possible_extensions:
                file_path = base_scene_path / (filename + ext)
                if file_path.exists(): return str(file_path)
    return None

# ----------------------------
# Main Execution Pipeline
# ----------------------------
def main(cfg):
    click.secho("\n" + "="*80, fg="green", bold=True)
    click.secho("STARTING LHPR-VLN DATA LOADING", fg="green", bold=True)
    click.secho("="*80, fg="green")
    lhpr_vln_tasks = load_lhpr_vln_dataset_simple(cfg)
    if not lhpr_vln_tasks:
        click.secho("\n✗ Failed to load LHPR-VLN tasks. Exiting.", fg="red", bold=True)
        return
    click.secho(f"✓ Successfully loaded {len(lhpr_vln_tasks)} LHPR-VLN tasks.", fg="blue")
    output_path = Path(__file__).resolve().parent.parent / cfg.output_path
    os.makedirs(str(output_path), exist_ok=True)
    results_filename = output_path / f'{cfg.results_filename}.json'
    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"
    visualizer_config = {'save_video': True, 'fps': 5, 'meters_per_pixel': cfg.get('meters_per_pixel', 0.1)}
    segmenter = None if cfg.data.use_semantic_data else DeticSegmenter(cfg)
    successes, skipped_tasks, total_tasks = 0, 0, len(lhpr_vln_tasks)

    for task_ind in tqdm(range(total_tasks)):
        task_data = lhpr_vln_tasks[task_ind]
        scene, floor = task_data.get("Scene", "Unknown"), task_data.get("floor", "")
        experiment_id = f'{task_ind}_{scene}_{floor}'
        if should_skip_experiment(experiment_id, filename=results_filename):
            click.secho(f'\nSkipping existing==Index: {task_ind} Scene: {scene}', fg="yellow")
            skipped_tasks += 1; continue

        vlm_instruction = task_data.get("Task instruction", "")
        subtasks = task_data.get("parsed_subtasks", [])
        task_manager = LHPRVLNTaskManager(subtasks)
        click.secho(f'\nExecuting=========Index: {task_ind} Scene: {scene}=======', fg="green")
        click.secho(f"Instruction: {vlm_instruction}", fg="cyan")
        for i, subtask in enumerate(subtasks):
            click.secho(f"  {i+1}. {subtask['type']} -> {subtask.get('target', 'N/A')}", fg="cyan")

        scene_file_path = find_scene_file(scene, cfg.data.scene_data_path)
        if not scene_file_path:
            click.secho(f"✗ Scene file not found for {scene}. Skipping...", fg="red")
            skipped_tasks += 1
            log_experiment_status(experiment_id, False, metrics={'error': 'scene_not_found', 'completed_subtasks': 0}, filename=results_filename)
            continue

        task_path = hydra_python.resolve_output_path(output_path / experiment_id)
        if isinstance(sys.stdout, Logger): sys.stdout.close()
        os.makedirs(str(task_path), exist_ok=True)
        sys.stdout = Logger(str(task_path / "terminal_run_log.txt"))
        click.secho(f"📄 正在记录当前场景专属日志: {task_path / 'terminal_run_log.txt'}", fg="magenta")

        try: habitat_data = HabitatInterface(scene_file_path, cfg=cfg.habitat, device=device)
        except Exception as e:
            click.secho(f"✗ Error initializing Habitat for scene {scene}: {e}", fg="red")
            skipped_tasks += 1; continue

        pipeline = initialize_hydra_pipeline(cfg.hydra, habitat_data, task_path)
        rr_logger = RRLogger(task_path)
        step_logger = StepImageLogger(task_path, experiment_id, habitat_data=habitat_data, visualizer_config=visualizer_config, step_info={'instruction': vlm_instruction, 'subtasks': subtasks, 'scene_file': scene_file_path}, realtime=True)

        init_pts = np.array(task_data["Start pos"])
        episode_trajectory = [init_pts.tolist()]
        pts_normal = pos_habitat_to_normal(init_pts)
        tsdf_bnds, scene_size = get_scene_bnds(habitat_data.pathfinder, pts_normal[-1])
        cam_intr = get_cam_intr(cfg.habitat.hfov, cfg.habitat.img_height, cfg.habitat.img_width)
        tsdf_planner = TSDFPlanner(cfg=cfg.frontier_mapping, vol_bnds=tsdf_bnds, cam_intr=cam_intr, floor_height_offset=0, pts_init=pts_normal, rr_logger=rr_logger)
        sg_sim = SceneGraphSim(cfg, task_path, pipeline, rr_logger, device=device, clean_ques_ans=vlm_instruction, enrich_object_labels=' ')
        poses = habitat_data.get_init_poses_eqa(init_pts, 0.0, cfg.habitat.camera_tilt_deg)

        step_logger.log_step(habitat_data, step_number=0, step_type="initialization", subtask_info={'current_index':0,'total':len(subtasks),'current_subtask':subtasks[0] if subtasks else None}, action_info={'type':'init','description':'Initial position'})
        run(pipeline, habitat_data, poses, output_path=task_path, rr_logger=rr_logger, tsdf_planner=tsdf_planner, sg_sim=sg_sim, save_image=cfg.vlm.use_image, segmenter=segmenter, save_each_step=True, step_logger=step_logger, step_number=0, subtask_info={'current_index': task_manager.current_index, 'total': len(subtasks), 'current_subtask': task_manager.get_current_subtask(), 'carried_object': task_manager.get_task_progress()['carried_object'], 'main_step': 0, 'subtask_progress': f"{task_manager.current_index+1}/{len(subtasks)}"})

        vlm_planner = VLMPlannerLHPRVLNGPT(cfg.vlm, sg_sim, vlm_instruction, subtasks, task_path, task_manager)
        num_steps, planning_steps, traj_length, task_completed, max_consecutive_failures, consecutive_failures, cnt = 30, 0, 0.0, False, 10, 0, 0
         
        for cnt_step in range(num_steps):
            if task_manager.is_task_completed():
                task_completed = True; successes += 1
                step_logger.log_step(habitat_data, step_number=cnt_step+1, step_type="task_completed", subtask_info={'current_index':task_manager.current_index,'total':len(subtasks),'status':'completed'}, action_info={'type':'completion','description':'All subtasks completed'})
                click.secho(f"\n✓ Task completed! All {len(subtasks)} subtasks finished.", fg="green", bold=True)
                break

            current_subtask = task_manager.get_current_subtask()
            progress = task_manager.get_task_progress()
            click.secho(f"\n[Step {cnt_step+1}/{num_steps}] Subtask {progress['current_index']+1}/{progress['total']}: {current_subtask['type']} -> {current_subtask.get('target','N/A')}", fg="yellow")
            observations = habitat_data._sim.get_sensor_observations()
            
            imgs_rgb = [observations.get("color_sensor_l"), observations.get("color_sensor_f"), observations.get("color_sensor_r")]
            imgs_rgb = [img for img in imgs_rgb if img is not None]
            imgs_depth = [observations.get("depth_sensor_l"), observations.get("depth_sensor_f"), observations.get("depth_sensor_r")]
            imgs_depth = [depth for depth in imgs_depth if depth is not None]
            
            try:
                frontier_nodes = tsdf_planner.get_frontiers()
                if len(frontier_nodes) > 0: print("Unexplored frontiers:", frontier_nodes); cnt = 0
                else:
                    print("No unexplored frontiers found!!!!!!!!!!!!!!!!!!!!!!!!!!!.")
                    cnt += 1; sb = habitat_data.pathfinder.get_bounds()
                    sp = np.random.uniform(sb[0], sb[1], (100, 3))
                    navigable = [pt for pt in sp if habitat_data.pathfinder.is_navigable(pt)]
                    frontier_nodes = np.array(navigable) if navigable else np.array([])
            except AttributeError:
                sb = habitat_data.pathfinder.get_bounds()
                sp = np.random.uniform(sb[0], sb[1], (100, 3))
                navigable = [pt for pt in sp if habitat_data.pathfinder.is_navigable(pt)]
                frontier_nodes = np.array(navigable) if navigable else np.array([])

            start = time.time()
            # ─── 🌟 [新增：多解包接收] 解包捕获消融试验控制位 `disable_safety_filter` ───
            target_pose, target_id, is_confident, confidence_level, action_output, disable_safety_filter = vlm_planner.get_next_action(None, imgs_rgb=imgs_rgb, imgs_depth=imgs_depth, intrinsics=cam_intr, extrinsics=None, frontier_nodes=frontier_nodes)
            original_target_pose = target_pose.copy() if target_pose is not None else None
            pose_str = f"[{target_pose[0]:.2f}, {target_pose[1]:.2f}, {target_pose[2]:.2f}]" if target_pose is not None else "None"
            click.secho(f"VLM planning time: {time.time()-start:.2f}s | Action: {action_output} | Target: {pose_str} | Conf: {confidence_level:.2f}", fg="green")

            try:
                agent_curr = habitat_data._sim.get_agent(0)
                agent_pos_tuple = agent_curr.get_state().position
                agent_pos_list = [float(agent_pos_tuple[0]), float(agent_pos_tuple[1]), float(agent_pos_tuple[2])]
            except: agent_pos_list = None

            if action_output == 'give_up':
                target_name = current_subtask.get('target', 'N/A')
                click.secho(f"\n❌ [任务失败] 探索率已达极限，仍未找到合格的 '{target_name}'！直接宣告当前子任务失败并跳过！", fg="red", bold=True)
                step_logger.log_step(habitat_data, step_number=cnt_step+1, step_type="subtask_failed_exhausted", subtask_info={'failed_index': progress['current_index'], 'total': progress['total'], 'failed_subtask': current_subtask, 'reason': 'exploration_exhausted', 'carried_object': progress['carried_object']}, action_info={'type': 'give_up', 'reason': 'Explored all frontiers but no target score > 0.6', 'target': current_subtask.get('target', 'N/A')})
                task_manager.mark_current_failed("exploration_exhausted", final_pos=original_target_pose)

                if task_manager.current_index < len(task_manager.subtasks):
                    task_manager.current_index += 1
                    if task_manager.current_index < len(task_manager.subtasks):
                        task_manager.task_state['current_subtask'] = task_manager.subtasks[task_manager.current_index]
                    else: task_manager.task_state['status'] = 'completed'
                consecutive_failures = 0
                
                if task_manager.current_index >= len(task_manager.subtasks):
                    task_completed = True
                    step_logger.log_step(habitat_data, step_number=cnt_step+1, step_type="task_finished_with_failures", subtask_info={'status': 'finished_with_failures'}, action_info={'type': 'completion'})
                    click.secho(f"\n🏁 任务序列结束！但部分子任务因未能找到目标而失败。", fg="yellow", bold=True); break
                else:
                    click.secho(f"➡️ 继续尝试下一个子任务: {task_manager.current_index+1}/{len(subtasks)}", fg="yellow"); continue

            step_logger.log_step(habitat_data, step_number=cnt_step+1, step_type="planning", subtask_info={'current_index':progress['current_index'],'total':progress['total'],'current_subtask':current_subtask,'subtask_type':current_subtask['type'],'target':current_subtask.get('target','N/A'),'carried_object':progress['carried_object']}, action_info={'action_output':action_output,'target_pose':target_pose.tolist() if target_pose is not None else None,'target_id':target_id,'is_confident':is_confident}, confidence=confidence_level)
            time.sleep(0.1)

            current_subtask_completed = False
            if is_confident and confidence_level > 0.6: current_subtask_completed = True
            elif action_output in ["grab", "release"]: current_subtask_completed = True

            if target_pose is not None:
                current_heading = habitat_data.get_heading_angle()
                agent_loop = habitat_data._sim.get_agent(0)
                current_pos = np.array(agent_loop.get_state().position)
                frontier_habitat = pos_normal_to_habitat(target_pose)
                frontier_habitat[1] = current_pos[1]

                # ─── 🌟 [新增：消融实验控制分支] ───
                if disable_safety_filter:
                    click.secho("🚨 [消融激活] w/o Safety Filter 开关开启: 彻底跳过边缘吸附、30cm退让及回退逼近，下发原始坐标！", fg="red", bold=True)
                    frontier_habitat = frontier_habitat.tolist() if not isinstance(frontier_habitat, list) else frontier_habitat
                else:
                    # Full Framework 完备的几何约束过滤器
                    if not habitat_data.pathfinder.is_navigable(frontier_habitat):
                        click.secho(f"⚠️ 目标处于破损区域，尝试吸附...", fg="yellow")
                        snapped_pos = habitat_data.pathfinder.snap_point(frontier_habitat)
                        if not np.isnan(snapped_pos[0]):
                            direction = snapped_pos - current_pos
                            dist = np.linalg.norm(direction)
                            safe_margin = 0.3
                            if dist > safe_margin:
                                safe_pos = snapped_pos - (direction / dist) * safe_margin
                                final_pos = habitat_data.pathfinder.snap_point(safe_pos)
                                frontier_habitat = [float(final_pos[0]), float(final_pos[1]), float(final_pos[2])] if not np.isnan(final_pos[0]) else [float(snapped_pos[0]), float(snapped_pos[1]), float(snapped_pos[2])]
                            else: frontier_habitat = [float(snapped_pos[0]), float(snapped_pos[1]), float(snapped_pos[2])]
                        else:
                            click.secho(f"❌ 修正失败，该区域完全无法落脚！", fg="red")
                            consecutive_failures += 1; continue

                path = habitat_sim.nav.ShortestPath()
                path.requested_start = current_pos.tolist()
                path.requested_end = frontier_habitat if isinstance(frontier_habitat, list) else frontier_habitat.tolist()
                found_path = habitat_data.pathfinder.find_path(path)
                
                # 寻路失败且安全机制未消融时激活 0.2m 线性回退逼近算法
                if not found_path and not disable_safety_filter:
                    click.secho(f"⚠️ [寻路断层] 目标点无法直接到达，启动线性逼近算法...", fg="yellow")
                    start_np, end_np = np.array(path.requested_start), np.array(path.requested_end)
                    vec = end_np - start_np
                    total_dist = np.linalg.norm(vec)
                    if total_dist > 0.1:
                        unit_vec = vec / total_dist
                        for step_dist in np.arange(total_dist, 0.0, -0.2):
                            test_pos = start_np + unit_vec * step_dist
                            test_pos[1] = start_np[1]
                            snapped_test = habitat_data.pathfinder.snap_point(test_pos)
                            if not np.isnan(snapped_test[0]):
                                snapped_test_list = [float(snapped_test[0]), float(snapped_test[1]), float(snapped_test[2])]
                                path.requested_end = snapped_test_list
                                if habitat_data.pathfinder.find_path(path):
                                    found_path = True
                                    frontier_habitat = snapped_test_list
                                    click.secho(f"🎯 逼近成功！在距离起点 {step_dist:.2f}m 处找到合法连通通路！", fg="green"); break

                if found_path:
                    desired_path = pos_habitat_to_normal(np.array(path.points))
                    rr_logger.log_traj_data(desired_path)
                    rr_logger.log_target_poses(target_pose)
                    for pt in path.points:
                        pt_list = [float(pt[0]), float(pt[1]), float(pt[2])]
                        if not episode_trajectory or episode_trajectory[-1] != pt_list: episode_trajectory.append(pt_list)

                    poses = habitat_data.get_trajectory_from_path_habitat_frame(target_pose, desired_path, current_heading, cfg.habitat.camera_tilt_deg)
                    if poses is not None:
                        click.secho(f"Executing nav trajectory (Steps: {len(poses)})", fg="yellow")
                        run(pipeline, habitat_data, poses, output_path=task_path, rr_logger=rr_logger, tsdf_planner=tsdf_planner, sg_sim=sg_sim, save_image=cfg.vlm.use_image, segmenter=segmenter, save_each_step=True, step_logger=step_logger, step_number=cnt_step+1, subtask_info={'current_index': task_manager.current_index, 'total': len(subtasks), 'current_subtask': task_manager.get_current_subtask(), 'carried_object': task_manager.get_task_progress()['carried_object'], 'main_step': cnt_step+1, 'subtask_progress': f"{task_manager.current_index+1}/{len(subtasks)}"})
                        traj_length += get_traj_len_from_poses(poses)
                        rr_logger.log_text_data(vlm_planner.full_plan)
                        planning_steps += 1

                        if action_output == 'explore_frontier':
                            click.secho("👀 抵达前沿点，执行 360 度环视扫描以彻底消除盲区...", fg="cyan")
                            agent_sweep = habitat_data._sim.get_agent(0)
                            sweep_poses = habitat_data.get_init_poses_eqa(agent_sweep.get_state().position, habitat_data.get_heading_angle(), cfg.habitat.camera_tilt_deg)
                            run(pipeline, habitat_data, sweep_poses, output_path=task_path, rr_logger=rr_logger, tsdf_planner=tsdf_planner, sg_sim=sg_sim, save_image=False, segmenter=segmenter, save_each_step=False, step_logger=step_logger, step_number=cnt_step+1, subtask_info={'current_index': task_manager.current_index, 'total': len(subtasks), 'current_subtask': task_manager.get_current_subtask(), 'carried_object': task_manager.get_task_progress()['carried_object'], 'main_step': cnt_step+1, 'subtask_progress': f"{task_manager.current_index+1}/{len(subtasks)}"})
                        consecutive_failures = 0
                        final_intended_pos = [float(frontier_habitat[0]), float(frontier_habitat[1]), float(frontier_habitat[2])]
                        step_logger.log_step(habitat_data, step_number=cnt_step+1, step_type="navigation_complete", subtask_info={'current_index':progress['current_index'],'total':progress['total'],'carried_object':progress['carried_object']}, action_info={'type':'navigation_complete','trajectory_length':traj_length})
                    else: click.secho("Trajectory generation failed.", fg="red"); consecutive_failures += 1
                else: click.secho(f"Cannot find navigable path to target.", fg="red"); consecutive_failures += 1
            else: click.secho("No target position available. VLM may need more exploration.", fg="yellow"); consecutive_failures += 1

            if current_subtask_completed:
                real_final_pos = final_intended_pos if ('final_intended_pos' in locals() and final_intended_pos is not None) else agent_pos_list
                step_logger.log_step(habitat_data, step_number=cnt_step+1, step_type=f"subtask_completed_{action_output}", subtask_info={'completed_index':progress['current_index'],'total':progress['total'],'completed_subtask':current_subtask,'carried_object':progress['carried_object']}, action_info={'type': action_output,'target': current_subtask.get('target','N/A')})
                task_manager.complete_current_subtask(final_pos=original_target_pose)
                click.secho(f"✓ Completed subtask {progress['current_index']+1}/{progress['total']}", fg="green")
                planning_steps += 1; rr_logger.log_text_data(vlm_planner.full_plan); continue
                
            if consecutive_failures >= max_consecutive_failures or cnt >= 10:
                click.secho(f"\n✗ Too many consecutive failures ({consecutive_failures}). Skipping current subtask.", fg="red")
                step_logger.log_step(habitat_data, step_number=cnt_step+1, step_type="subtask_failed_skip", subtask_info={'failed_index': progress['current_index'], 'total': progress['total'], 'failed_subtask': current_subtask, 'reason': 'max_consecutive_failures', 'carried_object': progress['carried_object']}, action_info={'type': 'skip_failure', 'reason': f'Consecutive failures: {consecutive_failures}', 'target': current_subtask.get('target', 'N/A')})
                task_manager.mark_current_failed("max_consecutive_failures", final_pos=original_target_pose)

                if task_manager.current_index < len(task_manager.subtasks):
                    task_manager.current_index += 1
                    if task_manager.current_index < len(task_manager.subtasks):
                        task_manager.task_state['current_subtask'] = task_manager.subtasks[task_manager.current_index]
                    else: task_manager.task_state['status'] = 'completed'
                consecutive_failures = 0
                
                if task_manager.current_index >= len(task_manager.subtasks):
                    task_completed = True; successes += 1
                    step_logger.log_step(habitat_data, step_number=cnt_step+1, step_type="task_completed_after_skip", subtask_info={'current_index': task_manager.current_index, 'total': len(subtasks), 'status': 'completed_after_skip'}, action_info={'type': 'completion', 'description': 'All subtasks completed (some skipped)'})
                    click.secho(f"\n✓ Task completed! All {len(subtasks)} subtasks finished (some may have been skipped).", fg="green", bold=True); break
                else:
                    click.secho(f"✓ Moving to next subtask: {task_manager.current_index+1}/{len(subtasks)}", fg="yellow"); continue

        try:
            if hasattr(tsdf_planner, 'visualize_with_open3d'):
                click.secho("📊 调用TSDF 3D可视化...", fg="cyan")
                tsdf_planner.visualize_with_open3d(extract_mesh=True)
                click.secho("✅ TSDF 3D可视化完成", fg="green")
        except Exception as e: click.secho(f"❌ TSDF可视化失败: {e}", fg="red")

        if not task_completed:
            click.secho(f"\n✗ Task incomplete after {num_steps} steps.", fg="red")
            progress = task_manager.get_task_progress()
            click.secho(f"Completed {progress['current_index']} of {progress['total']} subtasks.", fg="yellow")

        step_logger.close()
        metrics = {
            'vlm_steps': planning_steps, 'overall_steps': cnt_step+1 if 'cnt_step' in locals() else 0,
            'is_confident': is_confident if 'is_confident' in locals() else False, 'confidence_level': confidence_level if 'confidence_level' in locals() else 0.0,
            'traj_length': traj_length, 'completed_subtasks': task_manager.current_index, 'total_subtasks': len(subtasks), 'success': task_completed, 'steps_logged': len(step_logger.steps_data)
        }
        log_experiment_status(experiment_id, task_completed, metrics=metrics, filename=results_filename)
        
        log_data = {
            'instruction': vlm_instruction, 'scene': scene, 'scene_file': scene_file_path, 'subtasks': subtasks, 'execution_metrics': metrics,
            'progress': task_manager.get_task_progress(), 'subtask_results': task_manager.history, 'final_state': 'completed' if task_completed else 'incomplete',
            'detailed_trajectory': step_logger.steps_data, 'full_trajectory': episode_trajectory,
            'steps_summary': {'total_steps': len(step_logger.steps_data), 'image_dir': str(step_logger.images_dir), 'video_dir': str(step_logger.video_dir)}
        }
        with open(task_path / "task_execution_details.json", 'w') as f: json.dump(convert_to_serializable(log_data), f, indent=2)
        
        click.secho("🔄 正在进行场景图终极同步 (提取完整的真实多房间结构)...", fg="cyan")
        try:
            sg_sim._build_sg_from_hydra_graph()
            if sg_sim.enrich_rooms: sg_sim.add_room_labels_to_sg()
            sg_sim.save_enhanced_json()
            click.secho("✅ 终极版 enhanced_dsg.json 同步保存成功！", fg="green")
        except Exception as e: click.secho(f"⚠️ 终极场景图同步失败: {e}", fg="yellow")

        habitat_data._sim.close(destroy=True)
        pipeline.save()

    click.secho("\n" + "="*80, fg="green", bold=True)
    click.secho("EXECUTION SUMMARY", fg="green", bold=True)
    print(f"Total tasks: {total_tasks}\nSuccessful tasks: {successes}\nSkipped tasks: {skipped_tasks}")
    if total_tasks - skipped_tasks > 0:
        sr = successes / (total_tasks - skipped_tasks)
        click.secho(f"Success rate (excluding skipped): {sr*100:.1f}%", fg="green" if sr > 0.5 else "yellow")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file name", default="", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(Path(__file__).resolve().parent.parent / 'cfg' / f'{args.cfg_file}.yaml')
    OmegaConf.resolve(cfg)
    main(cfg)