import threading
import time
import psutil
from rich.live import Live
from rich.table import Table

def generate_table():
    table = Table()
    table.add_column("Metric", justify="center", style="cyan")
    table.add_column("Value", justify="center", style="green")
    
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent

    table.add_row("CPU Usage", f"{cpu_usage}%")
    table.add_row("Memory Usage", f"{memory_usage}%")
    return table




with Live(generate_table(), refresh_per_second=1) as live:
    try:
        while True:
            time.sleep(1)  # Update 1 time per second
            live.update(generate_table())
    except KeyboardInterrupt:
        pass