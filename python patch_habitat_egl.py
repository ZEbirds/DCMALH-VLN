import ctypes
import sys
import habitat_sim

# 加载 EGL 库并获取必要函数
egl = ctypes.CDLL('libEGL.so.1')
egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
egl.eglGetProcAddress.restype = ctypes.c_void_p

# 获取扩展函数
query_devices_ext = egl.eglGetProcAddress(b'eglQueryDevicesEXT')
get_platform_display_ext = egl.eglGetProcAddress(b'eglGetPlatformDisplayEXT')
query_device_string_ext = egl.eglGetProcAddress(b'eglQueryDeviceStringEXT')

# 定义函数类型
PFNEGLQUERYDEVICESEXTPROC = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))
PFNEGLGETPLATFORMDISPLAYEXTPROC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))
PFNEGLQUERYDEVICESTRINGEXTPROC = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)

eglQueryDevicesEXT = PFNEGLQUERYDEVICESEXTPROC(query_devices_ext)
eglGetPlatformDisplayEXT = PFNEGLGETPLATFORMDISPLAYEXTPROC(get_platform_display_ext)
eglQueryDeviceStringEXT = PFNEGLQUERYDEVICESTRINGEXTPROC(query_device_string_ext)

def find_cuda_egl_mapping():
    """诊断函数：找出 CUDA 设备与 EGL 设备的对应关系"""
    # 查询 EGL 设备数量
    num_egl_devices = ctypes.c_int(0)
    eglQueryDevicesEXT(0, None, ctypes.byref(num_egl_devices))
    print(f"[诊断] 系统共有 {num_egl_devices.value} 个 EGL 设备")
    
    # 获取 EGL 设备句柄
    egl_devices = (ctypes.c_void_p * num_egl_devices.value)()
    eglQueryDevicesEXT(num_egl_devices.value, egl_devices, ctypes.byref(num_egl_devices))
    
    # 尝试获取每个 EGL 设备的属性字符串（如包含 CUDA 信息）
    for i in range(num_egl_devices.value):
        device_str = ""
        if eglQueryDeviceStringEXT:
            # EGL_EXTENSIONS = 0x3055, EGL_DEVICE_EXT = 0x322C
            # 注意：此调用可能因驱动而异，是实验性的
            try:
                ptr = eglQueryDeviceStringEXT(egl_devices[i], 0x3055)
                if ptr:
                    device_str = ctypes.string_at(ptr).decode('utf-8', errors='ignore')[:100]
            except:
                pass
        
        # 尝试为该设备创建 Display 来测试是否有效
        display = eglGetPlatformDisplayEXT(0x313F, egl_devices[i], None) # 0x313F: EGL_PLATFORM_DEVICE_EXT
        status = "成功" if display and display != ctypes.c_void_p(0).value else "失败"
        print(f"[诊断] EGL 设备 {i}: 句柄={hex(egl_devices[i])}, 创建Display={status}, 扩展信息={device_str}")
        
        if display and hasattr(egl, 'eglTerminate'):
            egl.eglTerminate(display)
    
    # 这里需要你的逻辑：根据诊断信息，决定哪个 EGL 设备索引对应 CUDA device 0
    # 例如，如果你发现 EGL 设备 1 看起来像是主 GPU，就返回 1
    # 这是一个需要你根据打印信息判断的示例
    print(f"[诊断] 请根据以上信息，判断哪个 EGL 设备索引最可能对应 CUDA device 0。")
    # 例如：return 1

# 2. 核心补丁：替换 Habitat-Sim 内部创建上下文的选择逻辑
# 注意：此补丁需要根据 Habitat-Sim 的实际源码进行调整，以下为概念示例
def create_custom_context_patch(original_function, chosen_egl_device_id):
    """创建一个包装函数，强制使用我们选择的 EGL 设备"""
    def patched_function(*args, **kwargs):
        # 这里模拟：在原始函数被调用前，强制设置设备ID
        # 实际补丁需要查看 Habitat-Sim 的 'tryCreateContext' 源码
        print(f"[补丁] 拦截上下文创建，强制使用 EGL 设备 {chosen_egl_device_id}")
        # 一种可能的方法是临时设置一个进程级的环境变量
        import os
        os.environ['FORCE_EGL_DEVICE_ID'] = str(chosen_egl_device_id)
        
        # 调用原始函数
        result = original_function(*args, **kwargs)
        return result
    return patched_function

if __name__ == "__main__":
    print("="*60)
    print("Habitat-Sim EGL 设备映射诊断与补丁工具")
    print("="*60)
    find_cuda_egl_mapping()
    print("\n[下一步] 请根据诊断信息，在上方函数中指定返回的 EGL 设备索引，并取消注释补丁应用代码。")
