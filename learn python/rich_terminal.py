from rich.console import Console
from rich.live import Live
from rich.table import Table
import time
import random

# Initialize the Console
console = Console()

def generate_table():
    table = Table(title="System Metrics")
    table.add_column("Metric", justify="center", style="cyan")
    table.add_column("Value", justify="center", style="green")
    
    table.add_row("CPU Usage", f"{random.randint(0, 100)}%")
    table.add_row("Memory Usage", f"{random.randint(0, 100)}%")
    return table

with Live(generate_table(), refresh_per_second=4) as live:
    try:
        while True:
            time.sleep(1)  # Update 1 times per second
            live.update(generate_table())
    except KeyboardInterrupt:
        pass