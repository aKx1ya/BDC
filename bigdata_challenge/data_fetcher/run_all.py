"""
run_all.py
====================
一键运行所有数据获取脚本

运行顺序：
1. 01_price_volume.py  → 基础K线数据（其他脚本依赖成分股列表）
2. 02_northbound_flow.py → 北向资金
3. 03_margin_trading.py  → 融资融券
4. 04_money_flow.py      → 资金流向
5. 05_sector_momentum.py → 板块数据
6. 06_fundamental.py     → 基本面
7. 07_feature_engine.py  → 特征计算

使用方法：
    python run_all.py          # 运行全部
    python run_all.py --step 1 # 只运行第1步
    python run_all.py --from 3 # 从第3步开始运行
"""

import sys
import time
import importlib
import os

# 确保当前目录在path中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_step(step_num):
    """运行指定步骤"""
    steps = {
        1: ("01_price_volume", "获取日K线数据（~5分钟）"),
        2: ("02_northbound_flow", "获取北向资金数据（~3分钟）"),
        3: ("03_margin_trading", "获取融资融券数据（~5分钟）"),
        4: ("04_money_flow", "获取资金流向数据（~1分钟）"),
        5: ("05_sector_momentum", "获取板块数据（~3分钟）"),
        6: ("06_fundamental", "获取基本面数据（~10分钟）"),
        7: ("07_feature_engine", "计算特征矩阵（~1分钟）"),
    }

    if step_num not in steps:
        print(f"无效步骤: {step_num}")
        return False

    module_name, desc = steps[step_num]
    print(f"\n{'=' * 60}")
    print(f"步骤 {step_num}/7: {desc}")
    print(f"{'=' * 60}\n")

    try:
        module = importlib.import_module(module_name)

        # 每个模块的主入口
        if step_num == 1:
            module.fetch_all_stocks()
        elif step_num == 2:
            module.get_northbound_daily_summary()
            module.get_northbound_stock_history(days=30)
        elif step_num == 3:
            module.get_market_margin_summary()
            module.get_margin_history(days=30)
        elif step_num == 4:
            module.get_money_flow_rank_periods()
            module.get_sector_money_flow()
        elif step_num == 5:
            module.get_sw_industry_classification()
            module.get_sector_daily_performance()
        elif step_num == 6:
            module.get_fundamental_quarterly()
            module.get_growth_data()
            module.get_earnings_forecast()
        elif step_num == 7:
            module.build_features()

        print(f"\n步骤 {step_num} 完成!")
        return True

    except Exception as e:
        print(f"\n步骤 {step_num} 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    start_step = 1
    end_step = 6  # 默认只执行数据获取，不做特征工程
    single_step = None

    args = sys.argv[1:]
    if '--step' in args:
        idx = args.index('--step')
        single_step = int(args[idx + 1])
    elif '--from' in args:
        idx = args.index('--from')
        start_step = int(args[idx + 1])

    # 只有明确指定时才包含步骤7（特征工程）
    if '--with-feature' in args:
        end_step = 7

    print("=" * 60)
    print("大数据挑战赛 - 数据获取流水线")
    print("=" * 60)
    print()

    # 检查依赖
    try:
        import baostock
        import akshare
        print("依赖检查通过")
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请运行: pip3 install baostock akshare")
        return 1

    start_time = time.time()
    failed_steps = []

    if single_step:
        if not run_step(single_step):
            failed_steps.append(single_step)
    else:
        for step in range(start_step, end_step + 1):
            success = run_step(step)
            if not success:
                failed_steps.append(step)
            if not success and step < end_step:
                print(f"\n警告：步骤{step}失败，继续下一步...")
            time.sleep(1)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"完成! 耗时: {elapsed/60:.1f} 分钟")
    print(f"{'=' * 60}")

    if failed_steps:
        print(f"失败步骤: {failed_steps}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
