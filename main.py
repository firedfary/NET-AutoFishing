import cv2
import numpy as np
import mss
import dxcam
import pydirectinput
import time
import random

# ==========================================
# 配置区域 (TODO: 请根据你的实际截图和测试进行修改)
# ==========================================

# 1. 游戏当前运行的分辨率
GAME_WIDTH = 3840
GAME_HEIGHT = 2160

# 2. 相对坐标 ROI 配置 (左上角X比例, 左上角Y比例, 宽度比例, 高度比例)
# 取值范围 0.0 ~ 1.0
ROI_INIT_BUTTONS = (0.7109,0.8542,0.2435,0.113)  # 图1：右下角4个按钮的大致区域
ROI_FISH_HOOKED = (0.3159,0.2153,0.3935,0.0574)   # 图3：“鱼上钩了”提示框区域
ROI_CATCHING_BAR = (0.3143,0.0546,0.3732,0.0278)  # 图4：顶部绿条和黄线区域
ROI_SETTLEMENT = (0.4435,0.8917,0.1159,0.0319)    # 图5：结算界面的特征区域

# 3. HSV 颜色阈值 (需要你在绘图软件中取色并转换为 OpenCV 的 HSV 范围)
# 注意：OpenCV 中 H 的范围是 0-180，S 和 V 的范围是 0-255
HSV_GREEN_LOWER = np.array([71, 118, 35])     # 绿条的下限
HSV_GREEN_UPPER = np.array([93, 215, 255])   # 绿条的上限
HSV_YELLOW_LOWER = np.array([21, 61, 130])  # 黄线的下限
HSV_YELLOW_UPPER = np.array([45, 205, 255])  # 黄线的上限

# ==========================================
# 辅助函数
# ==========================================

def get_abs_roi(rel_roi):
    """将相对比例坐标转换为绝对像素坐标 (top, left, width, height)"""
    return {
        "left": int(rel_roi[0] * GAME_WIDTH),
        "top": int(rel_roi[1] * GAME_HEIGHT),
        "width": int(rel_roi[2] * GAME_WIDTH),
        "height": int(rel_roi[3] * GAME_HEIGHT)
    }

def random_sleep(min_sec=0.05, max_sec=0.15):
    """加入随机延迟，防封号"""
    time.sleep(random.uniform(min_sec, max_sec))

def tap_key(key):
    """模拟按键点击"""
    pydirectinput.keyDown(key)
    random_sleep(0.02, 0.08)
    pydirectinput.keyUp(key)

# ==========================================
# 核心状态机类
# ==========================================

class FishingBot:
    def __init__(self):
        self.state = "INIT"
        self.sct = mss.MSS()
        self.camera = dxcam.create(output_color="BGR") # 用于高频抓取
        
        # 加载模板图片 (TODO: 请替换为你截好的小图路径)
        # 读取时保留彩色或转灰度都可以，这里以灰度为例
        self.tpl_init = cv2.imread('./templates/init_btn.png', 0)
        self.tpl_hooked = cv2.imread('./templates/hooked_prompt.png', 0)
        self.tpl_settlement = cv2.imread('./templates/settlement_icon.png', 0)
        
        print("辅助程序初始化成功，当前状态: INIT")

    def match_template_in_roi(self, sct_img, template, threshold=0.8):
        """在截图中进行模板匹配"""
        img_gray = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2GRAY)
        
        # 动态缩放模板 (可选：如果你的模板是在 2160p 截的，而当前游戏不是 2160p)
        scale = GAME_HEIGHT / 2160.0
        tpl_resized = cv2.resize(template, (0,0), fx=scale, fy=scale)
        
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        return len(loc[0]) > 0

    def find_hsv_center_x(self, frame, lower, upper, min_area=50):
        """
        寻找指定颜色区域的中心 X 坐标
        增加 min_area 过滤噪点，并解决绿条被截断的问题
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        
        # 可选：如果你觉得黄线切口太大，可以加一句形态学闭运算“粘合”断裂处
        # kernel = np.ones((3,3), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        for c in contours:
            if cv2.contourArea(c) > min_area: # 过滤掉水面的反光噪点
                valid_contours.append(c)
                
        if not valid_contours:
            return None
            
        # 【核心修复】：将所有有效的轮廓点堆叠在一起
        all_points = np.vstack(valid_contours)
        
        # 针对这个包含了左右两半的超级大点集，求一个统一的外接矩形
        x, y, w, h = cv2.boundingRect(all_points)
        
        # 整个绿条的真实中心 X
        center_x = x + w // 2 
        return center_x

    # --- 状态执行逻辑 ---

    def run_init(self):
        roi = get_abs_roi(ROI_INIT_BUTTONS)
        sct_img = self.sct.grab(roi)
        
        if self.match_template_in_roi(sct_img, self.tpl_init):
            print("检测到钓鱼按钮，执行抛竿 (按F)...")
            tap_key('f')
            random_sleep(1.0, 1.5) # 等待动画
            self.state = "WAITING_FISH"
        else:
            print("未检测到准备页面，等待中...")
            time.sleep(1)

    def run_waiting_fish(self):
        roi = get_abs_roi(ROI_FISH_HOOKED)
        timeout_start = time.time()
        
        while time.time() - timeout_start < 60: # 最大等待 60 秒
            sct_img = self.sct.grab(roi)
            if self.match_template_in_roi(sct_img, self.tpl_hooked, threshold=0.75):
                print("鱼上钩了！开始收线...")
                tap_key('f') # 假设收线也需要按F或者其他确认键，根据游戏实际情况调整
                self.state = "CATCHING"
                return
            time.sleep(0.1) # 降低CPU占用
            
        print("等待超时，鱼跑了或者天气不佳，重置状态。")
        self.state = "INIT"

    def run_catching(self):
        # 准备高频截图的 ROI (dxcam 需要 tuple: left, top, right, bottom)
        roi_dict = get_abs_roi(ROI_CATCHING_BAR)
        region = (
            roi_dict['left'], 
            roi_dict['top'], 
            roi_dict['left'] + roi_dict['width'], 
            roi_dict['top'] + roi_dict['height']
        )
        
        self.camera.start(region=region, target_fps=60)
        print("进入溜鱼状态，启动高频检测...")
        
        check_settlement_counter = 0
        
        try:
            while True:
                frame = self.camera.get_latest_frame()
                if frame is None:
                    continue
                
                # 提取绿条和黄线的中心坐标
                green_x = self.find_hsv_center_x(frame, HSV_GREEN_LOWER, HSV_GREEN_UPPER)
                yellow_x = self.find_hsv_center_x(frame, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER)
                
                if green_x is not None and yellow_x is not None:
                    # 容差范围，避免频繁抖动
                    tolerance = 5 
                    if yellow_x < green_x - tolerance:
                        # 黄线偏左，按D向右拉
                        pydirectinput.keyDown('d')
                        pydirectinput.keyUp('a') # 确保释放反方向
                    elif yellow_x > green_x + tolerance:
                        # 黄线偏右，按A向左拉
                        pydirectinput.keyDown('a')
                        pydirectinput.keyUp('d')
                    else:
                        # 保持在中心，释放按键
                        pydirectinput.keyUp('a')
                        pydirectinput.keyUp('d')
                
                # 定期检查是否进入结算界面 (降频检查，每循环20次检查1次)
                check_settlement_counter += 1
                if check_settlement_counter > 20:
                    check_settlement_counter = 0
                    settle_roi = get_abs_roi(ROI_SETTLEMENT)
                    sct_img = self.sct.grab(settle_roi)
                    if self.match_template_in_roi(sct_img, self.tpl_settlement):
                        print("检测到结算界面，溜鱼成功！")
                        pydirectinput.keyUp('a')
                        pydirectinput.keyUp('d')
                        self.camera.stop()
                        self.state = "SETTLEMENT"
                        return
        except KeyboardInterrupt:
            # 异常时确保释放按键
            pydirectinput.keyUp('a')
            pydirectinput.keyUp('d')
            self.camera.stop()
            raise

    def run_settlement(self):
        print("正在结算...")
        random_sleep(1.0, 2.0)
        pydirectinput.click() # 点击鼠标左键关闭结算框
        random_sleep(1.0, 1.5)
        print("结算完成，准备下一杆。")
        self.state = "INIT"

    # --- 主循环 ---

    def run(self):
        print("开始运行，按 Ctrl+C 停止...")
        try:
            while True:
                if self.state == "INIT":
                    self.run_init()
                elif self.state == "WAITING_FISH":
                    self.run_waiting_fish()
                elif self.state == "CATCHING":
                    self.run_catching()
                elif self.state == "SETTLEMENT":
                    self.run_settlement()
        except KeyboardInterrupt:
            print("\n程序已手动停止。")
        finally:
            if self.camera.is_capturing:
                self.camera.stop()

# ==========================================
# 启动入口
# ==========================================
if __name__ == "__main__":
    # 强制给用户3秒钟时间切换到游戏窗口
    print("程序将在 3 秒后启动，请切换到游戏窗口...")
    time.sleep(3)
    
    bot = FishingBot()
    bot.run()