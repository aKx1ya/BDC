"""
配置文件：统一管理时间范围、股票池、路径等参数
"""
import os
from datetime import datetime, timedelta

# === 路径配置 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
FEATURE_DIR = os.path.join(DATA_DIR, "features")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

# === 时间配置 ===
# 获取足够长的历史数据用于计算技术指标（至少需要60个交易日的回看窗口）
END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = "2023-01-01"  # 从2023年开始，保证有足够的训练数据

# === 股票池：沪深300成分股 ===
# 比赛数据源是沪深300，这里用baostock获取最新成分股列表
HS300_CODE = "sh.000300"

# === 特征计算参数 ===
MOMENTUM_WINDOWS = [5, 10, 20]  # 动量计算窗口（交易日）
VOLATILITY_WINDOWS = [5, 10, 20]  # 波动率计算窗口
TURNOVER_MA_WINDOW = 20  # 换手率均值窗口
AMIHUD_WINDOW = 20  # Amihud非流动性计算窗口
