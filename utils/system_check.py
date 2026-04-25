"""
현재 os의 cpu 사용률, 메모리 사용률을 10초 단위로 체크

System monitor started...
[MON] CPU=0.0% MEM=69.7%
[MON] CPU=36.2% MEM=70.6%
[MON] CPU=51.8% MEM=71.4%
[MON] CPU=41.8% MEM=71.7%

[Ctrl + C] 로 종료하게 되면 간단한 그래프를 생성한다

===== SUMMARY =====
CPU AVG: 17.39%
CPU MAX: 46.50%
CPU P50: 14.60%   # 중앙값
CPU P90: 31.80%   # 상위 10% 구간
CPU P95: 38.60%   # 상위 5% 구간
MEM AVG: 75.71%
MEM MAX: 76.90%
MEM P90: 76.20%   # 상위 10% 구간


"""

import psutil
import time
import json
import statistics
import matplotlib.pyplot as plt
from datetime import datetime
import os

today = datetime.now().strftime("%Y%m%d")
year = datetime.now().strftime("%Y")
month = datetime.now().strftime("%m")
day = datetime.now().strftime("%d")
print(f"{today} {year} {month} {day}")

INTERVAL = 10  # 초
CPU_WARN = 80  # 경고 기준 %
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "system_usage.jsonl")

cpu_list = []
mem_list = []
time_list = []

CPU_COUNT = psutil.cpu_count()


def normalize_cpu(cpu_percent):
    return cpu_percent / CPU_COUNT


def get_usage():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    return cpu, mem


def log_usage(cpu, mem):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "time": now,
        "cpu": cpu,
        "mem": mem
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")


def percentile(data, p):
    if not data:
        return 0
    k = int(len(data) * p / 100)
    return sorted(data)[k]


def print_summary():
    if not cpu_list:
        return


    print("\n===== SUMMARY =====")
    print(f"CPU AVG: {statistics.mean(cpu_list):.2f}%")
    print(f"CPU MAX: {max(cpu_list):.2f}%")
    print(f"CPU P50: {percentile(cpu_list, 50):.2f}%")
    print(f"CPU P90: {percentile(cpu_list, 90):.2f}%")
    print(f"CPU P95: {percentile(cpu_list, 95):.2f}%")


    print(f"MEM AVG: {statistics.mean(mem_list):.2f}%")
    print(f"MEM MAX: {max(mem_list):.2f}%")
    print(f"MEM P90: {percentile(mem_list, 90):.2f}%")


def save_graph():
    if not cpu_list:
        return

    plt.figure(figsize=(20, 12))

    plt.plot(time_list, cpu_list, label="CPU %")
    plt.plot(time_list, mem_list, label="MEM %")

    # x축 시간 라벨을 최대 20개만 표시
    max_ticks = 20
    total = len(time_list)

    if total <= max_ticks:
        tick_positions = range(total)
    else:
        step = total / (max_ticks - 1)
        tick_positions = [round(i * step) for i in range(max_ticks)]
        tick_positions[-1] = total - 1  # 마지막 시간 보장

    tick_labels = [time_list[i] for i in tick_positions]

    plt.xticks(tick_positions, tick_labels, rotation=45)

    plt.xlabel("Time")
    plt.ylabel("Usage (%)")
    plt.title("System Usage")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"system_usage_{today}.png"))
    print(f"Graph saved: system_usage_{today}.png")


def print_top_processes():
    # CPU 측정 초기화
    procs = []
    for p in psutil.process_iter(['pid', 'name']):
        try:
            p.cpu_percent(None)
            procs.append(p)
        except:
            pass

    # 시간 간격 (중요)
    time.sleep(1)

    processes = []
    for p in procs:
        try:
            processes.append({
                'pid': p.pid,
                'name': p.name(),
                'cpu_percent': p.cpu_percent(None),
                'memory_percent': p.memory_percent()
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    top_cpu = sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:10]
    top_mem = sorted(processes, key=lambda x: x['memory_percent'], reverse=True)[:10]

    print("\n===== TOP 10 CPU =====")
    for p in top_cpu:
        cpu = normalize_cpu(p['cpu_percent'])
        print(f"{p['name']:<25} PID={p['pid']:<6} CPU={cpu:.1f}% MEM={p['memory_percent']:.1f}%")

    print("\n===== TOP 10 MEM =====")
    for p in top_mem:
        print(f"{p['name']:<25} PID={p['pid']:<6} CPU={p['cpu_percent']:.1f}% MEM={p['memory_percent']:.1f}%")


def main():
    print("System monitor started...")
    psutil.cpu_percent(None)  # 초기화


    try:
        while True:
            cpu, mem = get_usage()
            now = datetime.now().strftime("%H:%M:%S")


            cpu_list.append(cpu)
            mem_list.append(mem)
            time_list.append(now)


            log_usage(cpu, mem)


            if cpu >= CPU_WARN:
                print(f"[WARN] CPU HIGH! {cpu:.1f}%")


            print(f"[MON] CPU={cpu:.1f}% MEM={mem:.1f}%")


            time.sleep(INTERVAL)


    except KeyboardInterrupt:
        print_summary()
        print_top_processes()
        save_graph()


if __name__ == "__main__":
    main()