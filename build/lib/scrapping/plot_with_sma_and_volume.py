import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Function to plot candlestick (K) chart with SMAs and volume using matplotlib
def plot_with_sma_and_volume(df, ticker, sma_periods):
    # Ensure the index is in datetime format
    df.index = pd.to_datetime(df.index)
    
    plt.figure(figsize=(12, 8))

    # Create subplots
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=4, colspan=1)
    ax2 = plt.subplot2grid((6, 1), (4, 0), rowspan=2, colspan=1, sharex=ax1)

    # Convert index to numerical dates for plotting
    df['date_num'] = mdates.date2num(df.index)

    # Prepare the ohlc DataFrame for candlestick plotting
    ohlc = df[['date_num', 'open', 'high', 'low', 'close']]

    # Plot candlestick chart
    for idx, row in ohlc.iterrows():
        color = 'g' if row['close'] >= row['open'] else 'r'
        # Plot the high-low line
        ax1.plot([row['date_num'], row['date_num']], [row['low'], row['high']], color='black', linewidth=1)
        # Plot the open-close rectangle (candlestick body)
        rect = Rectangle(
            (row['date_num'] - 0.3, min(row['open'], row['close'])), 0.6,
            abs(row['close'] - row['open']), color=color, alpha=0.8
        )
        ax1.add_patch(rect)

    # Plot each SMA
    for period in sma_periods:
        sma_column = f'SMA_{period}'
        if sma_column in df.columns:
            ax1.plot(df.index, df[sma_column], label=f'SMA{period}', linewidth=2)

    # Set title and labels for the candlestick chart
    ax1.set_title(f'{ticker} Price with SMA')
    ax1.set_ylabel('Price')
    ax1.legend()

    # Plot volume
    ax2.bar(df.index, df['volume'], color='blue', alpha=0.3, width=0.6)
    ax2.set_ylabel('Volume')

    # Format x-axis
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.tight_layout()
    plt.show()
