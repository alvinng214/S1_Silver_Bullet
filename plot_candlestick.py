import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

# Read the CSV file
df = pd.read_csv('PEPPERSTONE_XAUUSD, 5.csv')

# Parse the datetime column
df['time'] = pd.to_datetime(df['time'])

# Sort by time to ensure proper order
df = df.sort_values('time')

# Create figure and axis
fig, ax = plt.subplots(figsize=(16, 8))

# Calculate candle width (in days) based on the time difference
time_diff = (df['time'].iloc[1] - df['time'].iloc[0]).total_seconds() / (24 * 3600)
width = time_diff * 0.6  # 60% of the time interval for candle width

# Plot each candlestick
for idx, row in df.iterrows():
    time = row['time']
    open_price = row['open']
    high_price = row['high']
    low_price = row['low']
    close_price = row['close']

    # Determine if bullish or bearish
    if close_price >= open_price:
        # Bullish candle - Blue
        color = 'blue'
        body_bottom = open_price
        body_height = close_price - open_price
    else:
        # Bearish candle - Red
        color = 'red'
        body_bottom = close_price
        body_height = open_price - close_price

    # Draw the wicks (high-low line) in black
    ax.plot([time, time], [low_price, high_price], color='black', linewidth=1, zorder=1)

    # Draw the candle body
    rect = Rectangle(
        (mdates.date2num(time) - width/2, body_bottom),
        width,
        body_height,
        facecolor=color,
        edgecolor='black',
        linewidth=0.5,
        zorder=2
    )
    ax.add_patch(rect)

# Format the x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45, ha='right')

# Set labels and title
ax.set_xlabel('Datetime', fontsize=12, fontweight='bold')
ax.set_ylabel('Price', fontsize=12, fontweight='bold')
ax.set_title('XAUUSD Candlestick Chart (5-minute)', fontsize=14, fontweight='bold')

# Add grid for better readability
ax.grid(True, alpha=0.3, linestyle='--')

# Add legend
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',
           markersize=10, label='Bullish Candle'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
           markersize=10, label='Bearish Candle'),
    Line2D([0], [0], color='black', linewidth=1, label='Wick')
]
ax.legend(handles=legend_elements, loc='upper left')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the chart
plt.savefig('candlestick_chart.png', dpi=300, bbox_inches='tight')
print("Candlestick chart saved as 'candlestick_chart.png'")

# Display the chart
plt.show()
