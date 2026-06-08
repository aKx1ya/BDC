"""增量更新与重试工具函数"""

import pandas as pd
import os
import time
import functools


def get_last_date(csv_path, date_col='date'):
    """读取CSV中最大日期，返回字符串 '%Y-%m-%d'，文件不存在返回 None"""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, usecols=[date_col])
        if df.empty:
            return None
        last = pd.to_datetime(df[date_col]).max()
        return last.strftime('%Y-%m-%d')
    except Exception:
        return None


def append_csv(csv_path, new_df):
    """将新数据追加到已有CSV（保持表头一致），如文件不存在则新建"""
    if new_df.empty:
        return
    if os.path.exists(csv_path):
        new_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        new_df.to_csv(csv_path, index=False)


def next_day(date_str):
    """返回 date_str 的下一天，格式 '%Y-%m-%d'"""
    d = pd.Timestamp(date_str) + pd.Timedelta(days=1)
    return d.strftime('%Y-%m-%d')


def retry_request(func=None, max_retries=3, base_wait=5):
    """
    装饰器：对 akshare 请求做指数退避重试。
    捕获: RemoteDisconnected, JSONDecodeError, ConnectionError, ReadTimeout, OSError
    """
    if func is None:
        return functools.partial(retry_request, max_retries=max_retries, base_wait=base_wait)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from http.client import RemoteDisconnected
        from json import JSONDecodeError
        from requests.exceptions import ConnectionError as ReqConnErr, ReadTimeout

        retryable = (RemoteDisconnected, JSONDecodeError, ReqConnErr, ReadTimeout, OSError)
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except retryable as e:
                wait = base_wait * (2 ** attempt)
                if attempt < max_retries - 1:
                    print(f"    重试 {attempt+1}/{max_retries}，等待{wait}秒... ({type(e).__name__})")
                    time.sleep(wait)
                else:
                    print(f"    重试{max_retries}次仍失败: {e}")
                    return None
    return wrapper
