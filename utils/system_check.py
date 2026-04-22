"""
현재 os의 cpu 사용률, 메모리 사용률을 10초 단위로 체크

System monitor started...
[MON] CPU=0.0% MEM=69.7%
[MON] CPU=36.2% MEM=70.6%
[MON] CPU=51.8% MEM=71.4%
[MON] CPU=41.8% MEM=71.7%

[Ctrl + C] 로 종료하게 되면 간단한 그래프를 생성한다
"""

import psutil
import time
import json
import statistics
import matplotlib.pyplot as plt
from datetime import datetime

today = datetime.now().strftime("%Y%m%d")
year = datetime.now().strftime("%Y")
month = datetime.now().strftime("%m")
day = datetime.now().strftime("%d")
print(f"{today} {year} {month} {day}")

INTERVAL = 10  # 초
LOG_FILE = "../system_usage.jsonl"
CPU_WARN = 80  # 경고 기준 %


cpu_list = []
mem_list = []
time_list = []


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

    plt.figure()

    plt.plot(time_list, cpu_list, label="CPU %")
    plt.plot(time_list, mem_list, label="MEM %")

    # x축 시간 라벨을 최대 10개만 표시
    max_ticks = 10
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
    plt.savefig("system_usage.png")
    print("Graph saved: system_usage.png")

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


            # 🔥 경고 추가
            if cpu >= CPU_WARN:
                print(f"[WARN] CPU HIGH! {cpu:.1f}%")


            print(f"[MON] CPU={cpu:.1f}% MEM={mem:.1f}%")


            time.sleep(INTERVAL)


    except KeyboardInterrupt:
        print_summary()
        save_graph()


if __name__ == "__main__":
    main()