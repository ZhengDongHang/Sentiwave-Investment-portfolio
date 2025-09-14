import csv
import os

def loss_record(loss_log_path, epoch, loop, loss):
    # 如果文件不存在则写入表头
    if not os.path.exists(loss_log_path):
        with open(loss_log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "batch", "loss"])

    # 写入 CSV
    with open(loss_log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, loop.n, loss.item()])

