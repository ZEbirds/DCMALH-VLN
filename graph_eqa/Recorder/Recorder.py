from PIL import Image
import json
from datetime import datetime
from pathlib import Path

class StepImageLogger:
    """记录每步的照片，便于调试"""
    
    def __init__(self, task_path, experiment_id, step_info=None):
        self.task_path = Path(task_path)
        self.experiment_id = experiment_id
        self.step_info = step_info or {}
        
        # 创建图片保存目录
        self.images_dir = self.task_path / "step_images"
        self.images_dir.mkdir(exist_ok=True)
        
        # 创建全景图目录
        self.panorama_dir = self.images_dir / "panoramas"
        self.panorama_dir.mkdir(exist_ok=True)
        
        # 记录步骤信息
        self.steps_data = []
        self.current_step = 0
        
    def log_step(self, habitat_data, step_number, step_type, 
                 subtask_info=None, action_info=None, confidence=None):
        """
        记录当前步骤的图片和信息
        
        Args:
            habitat_data: HabitatInterface实例
            step_number: 步骤编号
            step_type: 步骤类型（规划、导航、grab等）
            subtask_info: 子任务信息
            action_info: 动作信息
            confidence: 置信度
        """
        self.current_step = step_number
        
        # 保存当前视角图片
        current_images = self.save_current_view(habitat_data, step_number)
        
        # 保存多视角图片
        multi_view_images = self.save_multi_view_images(habitat_data, step_number)
        
        # 保存全景图
        panorama_path = self.create_panorama(multi_view_images, step_number)
        
        # 记录步骤数据
        step_data = {
            'step_number': step_number,
            'step_type': step_type,
            'timestamp': datetime.now().isoformat(),
            'subtask_info': subtask_info or {},
            'action_info': action_info or {},
            'confidence': confidence,
            'current_image': current_images.get('center') if current_images else None,
            'left_image': multi_view_images.get('left'),
            'center_image': multi_view_images.get('center'),
            'right_image': multi_view_images.get('right'),
            'panorama_image': panorama_path,
            'agent_position': self.get_agent_position(habitat_data),
            'agent_heading': self.get_agent_heading(habitat_data)
        }
        
        self.steps_data.append(step_data)
        
        # 实时保存到文件
        self.save_steps_json()
        
        return step_data
    
    def save_current_view(self, habitat_data, step_number):
        """保存当前视角图片"""
        try:
            # 获取当前观测
            obs = habitat_data._sim.get_sensor_observations()
            rgb_obs = obs.get("color_sensor", None)
            
            if rgb_obs is not None:
                # 转换为PIL Image
                rgb_img = Image.fromarray(rgb_obs[:, :, :3])
                
                # 保存图片
                img_path = self.images_dir / f"step_{step_number:03d}_current.png"
                rgb_img.save(img_path)
                
                return {
                    'center': str(img_path),
                    'image_size': rgb_img.size
                }
        except Exception as e:
            print(f"Error saving current view: {e}")
        
        return None
    
    def save_multi_view_images(self, habitat_data, step_number):
        """保存左、中、右三个视角的图片"""
        try:
            agent = habitat_data._sim.get_agent(0)
            original_state = agent.get_state()
            
            # 定义三个角度：左(-90度)，中(0度)，右(90度)
            angles = [-90, 0, 90]
            directions = ['left', 'center', 'right']
            
            images = {}
            
            for angle, direction in zip(angles, directions):
                # 旋转智能体到指定角度
                new_rotation = original_state.rotation
                new_rotation = habitat_sim.utils.common.quat_from_angle_axis(
                    np.radians(angle), habitat_sim.geo.UP
                ) * new_rotation
                
                agent.set_state(habitat_sim.AgentState(
                    position=original_state.position,
                    rotation=new_rotation
                ))
                
                # 获取观测
                obs = habitat_data._sim.get_sensor_observations()
                rgb_obs = obs.get("color_sensor", None)
                
                if rgb_obs is not None:
                    # 转换为PIL Image
                    rgb_img = Image.fromarray(rgb_obs[:, :, :3])
                    
                    # 保存图片
                    img_path = self.images_dir / f"step_{step_number:03d}_{direction}.png"
                    rgb_img.save(img_path)
                    
                    images[direction] = str(img_path)
            
            # 恢复原始状态
            agent.set_state(original_state)
            
            return images
            
        except Exception as e:
            print(f"Error saving multi-view images: {e}")
            return {}
    
    def create_panorama(self, multi_view_images, step_number):
        """创建全景拼接图"""
        if not multi_view_images or len(multi_view_images) < 3:
            return None
        
        try:
            pil_images = []
            for direction in ['left', 'center', 'right']:
                if direction in multi_view_images:
                    img = Image.open(multi_view_images[direction])
                    pil_images.append(img)
                else:
                    # 如果缺少某个视角，用黑色图片替代
                    black_img = Image.new('RGB', (640, 480), (0, 0, 0))
                    pil_images.append(black_img)
            
            # 横向拼接
            widths, heights = zip(*(i.size for i in pil_images))
            total_width = sum(widths)
            max_height = max(heights)
            
            panorama = Image.new('RGB', (total_width, max_height))
            
            x_offset = 0
            for img in pil_images:
                panorama.paste(img, (x_offset, 0))
                x_offset += img.size[0]
            
            # 添加步骤信息水印
            self.add_watermark(panorama, step_number)
            
            panorama_path = self.panorama_dir / f"step_{step_number:03d}_panorama.png"
            panorama.save(panorama_path)
            
            return str(panorama_path)
            
        except Exception as e:
            print(f"Error creating panorama: {e}")
            return None
    
    def add_watermark(self, image, step_number):
        """在图片上添加水印信息"""
        try:
            from PIL import ImageDraw, ImageFont
            
            draw = ImageDraw.Draw(image)
            
            # 使用默认字体
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # 添加水印文本
            watermark_text = f"Step: {step_number} | Experiment: {self.experiment_id} | Time: {datetime.now().strftime('%H:%M:%S')}"
            text_width, text_height = draw.textsize(watermark_text, font=font)
            
            # 在图片底部添加半透明背景和水印
            padding = 5
            background_position = [10, image.height - text_height - padding*2]
            background_size = [text_width + padding*2, text_height + padding*2]
            
            # 绘制半透明背景
            background = Image.new('RGBA', (background_size[0], background_size[1]), (0, 0, 0, 128))
            image.paste(background, (background_position[0], background_position[1]), background)
            
            # 绘制文本
            draw.text(
                (background_position[0] + padding, background_position[1] + padding),
                watermark_text,
                fill=(255, 255, 255, 255),
                font=font
            )
            
        except Exception as e:
            print(f"Error adding watermark: {e}")
    
    def get_agent_position(self, habitat_data):
        """获取智能体位置"""
        try:
            agent = habitat_data._sim.get_agent(0)
            state = agent.get_state()
            position = state.position
            return {
                'x': float(position[0]),
                'y': float(position[1]),
                'z': float(position[2])
            }
        except:
            return None
    
    def get_agent_heading(self, habitat_data):
        """获取智能体朝向"""
        try:
            heading = habitat_data.get_heading_angle()
            return float(heading)
        except:
            return None
    
    def save_steps_json(self):
        """保存步骤数据到JSON文件"""
        try:
            json_path = self.task_path / "steps_debug_info.json"
            data = {
                'experiment_id': self.experiment_id,
                'total_steps': len(self.steps_data),
                'steps': self.steps_data
            }
            
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving steps JSON: {e}")
    
    def generate_debug_report(self):
        """生成调试报告"""
        try:
            report_path = self.task_path / "debug_report.html"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Debug Report - {self.experiment_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .step {{ border: 1px solid #ddd; margin: 10px 0; padding: 10px; }}
                    .step-header {{ background: #f0f0f0; padding: 5px; font-weight: bold; }}
                    .images {{ display: flex; gap: 10px; margin: 10px 0; }}
                    .image-container {{ border: 1px solid #ccc; padding: 5px; }}
                    .image-container img {{ max-width: 300px; max-height: 200px; }}
                    .info {{ background: #f9f9f9; padding: 5px; margin: 5px 0; }}
                </style>
            </head>
            <body>
                <h1>Debug Report: {self.experiment_id}</h1>
                <p>Total Steps: {len(self.steps_data)}</p>
            """
            
            for step in self.steps_data:
                html_content += f"""
                <div class="step">
                    <div class="step-header">Step {step['step_number']}: {step['step_type']}</div>
                    <div class="info">
                        <div>Time: {step['timestamp']}</div>
                        <div>Position: {step.get('agent_position', 'N/A')}</div>
                        <div>Heading: {step.get('agent_heading', 'N/A')}°</div>
                    </div>
                """
                
                if step.get('panorama_image'):
                    html_content += f"""
                    <div class="images">
                        <div class="image-container">
                            <h4>Panorama View</h4>
                            <img src="{step['panorama_image']}" alt="Panorama">
                        </div>
                    </div>
                    """
                
                if step.get('current_image'):
                    html_content += f"""
                    <div class="images">
                        <div class="image-container">
                            <h4>Current View</h4>
                            <img src="{step['current_image']}" alt="Current View">
                        </div>
                    </div>
                    """
                
                html_content += "</div>"
            
            html_content += """
            </body>
            </html>
            """
            
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            return str(report_path)
            
        except Exception as e:
            print(f"Error generating debug report: {e}")
            return None