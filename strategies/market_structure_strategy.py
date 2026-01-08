"""
Market Structure Trading Strategy for Backtrader

This strategy trades based on market structure shifts (CHoCH and BoS) identified
by the Market Structure MTF Trend indicator.

Trading Logic:
1. Enter LONG when bullish CHoCH is detected (trend reversal to bullish)
2. Enter SHORT when bearish CHoCH is detected (trend reversal to bearish)
3. Exit when opposite structure break occurs
4. Optional: Use BoS for confirmation or pyramid into existing positions
"""

import backtrader as bt
import sys
import os

# Add indicators directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'indicators'))
from market_structure_mtf import MarketStructureIndicator


class MarketStructureStrategy(bt.Strategy):
    """
    Strategy that trades market structure shifts (CHoCH signals).
    """

    params = (
        ('pivot_strength', 15),       # Pivot strength for structure detection
        ('use_bos', False),            # Whether to trade BoS signals as well
        ('risk_percent', 1.0),         # Risk per trade as % of capital
        ('stop_loss_atr_mult', 2.0),  # Stop loss ATR multiplier
        ('take_profit_rr', 3.0),      # Take profit risk:reward ratio
        ('print_signals', True),       # Print trade signals
    )

    def __init__(self):
        # Add Market Structure indicator
        self.ms_indicator = MarketStructureIndicator(
            self.data,
            pivot_strength=self.params.pivot_strength
        )

        # Add ATR for stop loss calculation
        self.atr = bt.indicators.ATR(self.data, period=14)

        # Track order and position
        self.order = None
        self.buy_price = None
        self.sell_price = None
        self.stop_loss = None
        self.take_profit = None

    def log(self, txt, dt=None):
        """Logging function for the strategy"""
        dt = dt or self.datas[0].datetime.datetime(0)
        if self.params.print_signals:
            print(f'{dt.isoformat()}: {txt}')

    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buy_price = order.executed.price
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.sell_price = order.executed.price

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        """Handle trade notifications"""
        if not trade.isclosed:
            return

        self.log(f'TRADE PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')

    def next(self):
        """Main strategy logic executed on each bar"""
        # Skip if we have a pending order
        if self.order:
            return

        # Get current indicator values
        trend = self.ms_indicator.lines.trend[0]
        choch = self.ms_indicator.lines.choch[0]
        bos = self.ms_indicator.lines.bos[0]
        pivot_high = self.ms_indicator.lines.pivot_high[0]
        pivot_low = self.ms_indicator.lines.pivot_low[0]

        # Current position
        position_size = self.position.size

        # Log market structure changes
        if choch != 0:
            direction = "BULLISH" if choch > 0 else "BEARISH"
            self.log(f'CHoCH DETECTED: {direction} | Price: {self.data.close[0]:.2f}')

        if bos != 0 and self.params.use_bos:
            direction = "BULLISH" if bos > 0 else "BEARISH"
            self.log(f'BoS DETECTED: {direction} | Price: {self.data.close[0]:.2f}')

        # ENTRY LOGIC
        # ----------------------------------------------------------------------

        # BULLISH CHoCH: Enter LONG
        if choch > 0 and position_size <= 0:
            if position_size < 0:
                # Close short position first
                self.log(f'CLOSE SHORT on Bullish CHoCH at {self.data.close[0]:.2f}')
                self.close()

            # Calculate position size and stops
            stop_loss_distance = self.atr[0] * self.params.stop_loss_atr_mult
            stop_loss_price = self.data.close[0] - stop_loss_distance
            take_profit_distance = stop_loss_distance * self.params.take_profit_rr
            take_profit_price = self.data.close[0] + take_profit_distance

            # Calculate position size based on risk
            risk_amount = self.broker.getvalue() * (self.params.risk_percent / 100.0)
            position_size = risk_amount / stop_loss_distance

            self.log(f'BUY SIGNAL (Bullish CHoCH) | Entry: {self.data.close[0]:.2f} | '
                    f'SL: {stop_loss_price:.2f} | TP: {take_profit_price:.2f}')

            # Place buy order
            self.order = self.buy(size=position_size)
            self.stop_loss = stop_loss_price
            self.take_profit = take_profit_price

        # BEARISH CHoCH: Enter SHORT
        elif choch < 0 and position_size >= 0:
            if position_size > 0:
                # Close long position first
                self.log(f'CLOSE LONG on Bearish CHoCH at {self.data.close[0]:.2f}')
                self.close()

            # Calculate position size and stops
            stop_loss_distance = self.atr[0] * self.params.stop_loss_atr_mult
            stop_loss_price = self.data.close[0] + stop_loss_distance
            take_profit_distance = stop_loss_distance * self.params.take_profit_rr
            take_profit_price = self.data.close[0] - take_profit_distance

            # Calculate position size based on risk
            risk_amount = self.broker.getvalue() * (self.params.risk_percent / 100.0)
            position_size = risk_amount / stop_loss_distance

            self.log(f'SELL SIGNAL (Bearish CHoCH) | Entry: {self.data.close[0]:.2f} | '
                    f'SL: {stop_loss_price:.2f} | TP: {take_profit_price:.2f}')

            # Place sell order
            self.order = self.sell(size=position_size)
            self.stop_loss = stop_loss_price
            self.take_profit = take_profit_price

        # EXIT LOGIC
        # ----------------------------------------------------------------------

        # Check for stop loss and take profit
        if position_size > 0:  # Long position
            if self.data.close[0] <= self.stop_loss:
                self.log(f'STOP LOSS HIT (Long) at {self.data.close[0]:.2f}')
                self.close()
            elif self.data.close[0] >= self.take_profit:
                self.log(f'TAKE PROFIT HIT (Long) at {self.data.close[0]:.2f}')
                self.close()

        elif position_size < 0:  # Short position
            if self.data.close[0] >= self.stop_loss:
                self.log(f'STOP LOSS HIT (Short) at {self.data.close[0]:.2f}')
                self.close()
            elif self.data.close[0] <= self.take_profit:
                self.log(f'TAKE PROFIT HIT (Short) at {self.data.close[0]:.2f}')
                self.close()

    def stop(self):
        """Called when backtest is finished"""
        self.log(f'Final Portfolio Value: {self.broker.getvalue():.2f}', dt=None)


class SimpleMarketStructureStrategy(bt.Strategy):
    """
    Simplified strategy that only trades on CHoCH signals with basic rules.
    """

    params = (
        ('pivot_strength', 15),
        ('print_signals', True),
    )

    def __init__(self):
        self.ms_indicator = MarketStructureIndicator(
            self.data,
            pivot_strength=self.params.pivot_strength
        )
        self.order = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        if self.params.print_signals:
            print(f'{dt.isoformat()}: {txt}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED at {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED at {order.executed.price:.2f}')
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'TRADE PNL: {trade.pnlcomm:.2f}')

    def next(self):
        if self.order:
            return

        choch = self.ms_indicator.lines.choch[0]
        position_size = self.position.size

        # Bullish CHoCH: go long
        if choch > 0 and position_size <= 0:
            if position_size < 0:
                self.close()
            self.log(f'BUY SIGNAL (Bullish CHoCH) at {self.data.close[0]:.2f}')
            self.order = self.buy()

        # Bearish CHoCH: go short
        elif choch < 0 and position_size >= 0:
            if position_size > 0:
                self.close()
            self.log(f'SELL SIGNAL (Bearish CHoCH) at {self.data.close[0]:.2f}')
            self.order = self.sell()

    def stop(self):
        self.log(f'Final Value: {self.broker.getvalue():.2f}')
