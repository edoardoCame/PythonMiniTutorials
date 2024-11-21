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





#with Live(generate_table(), refresh_per_second=4) as live:
#------------------------------------------------------------
#This controls how often the display can be visually updated
#It's the maximum refresh rate for the terminal display
#Think of it as the "screen refresh rate"
#In this case, the screen can update up to 4 times per second


#while True:
#    time.sleep(1)  # Update 1 times per second
#    live.update(generate_table())
# #------------------------------------------------------------
# This controls how often you generate new data
# It determines when you actually want to update the content
# Think of it as your "data update frequency"
# Here, new random values are generated once per second


# Think of it like a digital billboard:

# refresh_per_second is how fast the screen can physically update (hardware capability)
# time.sleep() is how often you send new content to display (content updates)
