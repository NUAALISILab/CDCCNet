import sys
import time


class Logger(object):   # 用于日志记录
    def __init__(self, outfile):
        self.terminal = sys.stdout  # 保存终端的标准输出
        self.log_path = outfile     # 日志文件路径
        now = time.strftime("%c")   # 获取当前时间
        self.write('================ (%s) ================\n' % now)    # 写入日志文件

    def write(self, message):
        self.terminal.write(message)    # 在终端输出信息
        with open(self.log_path, mode='a') as f:
            f.write(message)    # 将信息追加到日志文件中

    def write_dict(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %.7f ' % (k, v) # 格式化字典内容为字符串,v必须是浮点数（或可以转换为浮点数）,保留 7 位小数
        self.write(message) # 调用 write 方法写入日志

    def write_dict_str(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %s ' % (k, v)   # v可以是任意类型（字符串、整数等）,直接转换为字符串
        self.write(message)

    def flush(self):
        self.terminal.flush()   # 刷新终端输出


class Timer:    # 计时功能
    def __init__(self, starting_msg=None):
        self.start = time.time()  # 记录开始时间
        self.stage_start = self.start  # 记录当前阶段的开始时间

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))  # 打印启动消息和时间

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def update_progress(self, progress):
        self.elapsed = time.time() - self.start  # 计算已用时间
        self.est_total = self.elapsed / progress  # 估算总时间
        self.est_remaining = self.est_total - self.elapsed  # 估算剩余时间
        self.est_finish = int(self.start + self.est_total)  # 估算完成时间

    def str_estimated_complete(self):
        return str(time.ctime(self.est_finish))  # 返回完成时间的字符串

    def str_estimated_remaining(self):
        return str(self.est_remaining/3600) + 'h'   # 返回剩余时间的小时数

    def estimated_remaining(self):
        return self.est_remaining/3600  # 返回剩余时间的小时数

    def get_stage_elapsed(self):
        return time.time() - self.stage_start   # 返回当前阶段的已用时间

    def reset_stage(self):
        self.stage_start = time.time()   # 重置当前阶段的开始时间

    def lapse(self):
        out = time.time() - self.stage_start    # 计算当前阶段的已用时间
        self.stage_start = time.time()  # 重置当前阶段的开始时间
        return out  # 返回已用时间

