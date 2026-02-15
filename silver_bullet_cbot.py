"""
Silver Bullet S1 cBot – cTrader Python cBot.

This cBot implements the ICT Silver Bullet strategy using the cTrader cAlgo
Python API.  It reads signals from 12 custom cTrader indicators (installed
separately) and executes real market orders with broker-managed SL/TP.

CUSTOM INDICATORS REQUIRED (install each as a cTrader custom indicator):
  01. SilverBulletWithSignals  – outputs: EntryTriggerBull, EntryTriggerBear,
                                          EntrySbBull, EntrySbBear
  02. IctSetup01               – outputs: EntrySetup01Bull, EntrySetup01Bear
  03. FibonacciOte             – outputs: EntryOteBull, EntryOteBear
  04. IfvgRealtime             – outputs: EntryIfvgBull, EntryIfvgBear
  05. OrderBlockDetector       – outputs: EntryObdetBull, EntryObdetBear
  06. SmartMoneyZones          – outputs: SmzTrend15m, SmzTrend1h
  07. MarketStructureMtf       – outputs: MsTrend15m, MsTrend1h
  08. OrderBlocksImbalanceMtf  – outputs: ObTouch1h, ObTouch4h
  09. MtfFvgX2                 – outputs: FvgTouch1h, FvgTouch4h
  10. IctSessionFilter         – outputs: SessionActive
  11. SmartMoneyConcept        – outputs: (liquidity context, optional)
  12. LiquidityInducements     – outputs: LiqBuysideTarget, LiqSellsideTarget

cBOT PARAMETERS (set in cTrader UI):
  - RiskPerTrade   (double, default 0.02)  – fraction of equity risked per trade
  - MaxConcurrent  (int,    default 3)     – max simultaneous positions
  - Quantity        (double, default 0.01)  – fallback lot size
  - DebugSignals   (bool,   default False) – verbose signal logging

ARCHITECTURE
  on_start  → initialise all 12 custom indicators, register position events
  on_bar    → read indicator outputs, apply filters, execute trades
  on_stop   → print session summary
"""

import clr

clr.AddReference("cAlgo.API")

from cAlgo.API import *

from robot_wrapper import execute_market_order, modify_position


# ---------------------------------------------------------------------------
# cBot class
# ---------------------------------------------------------------------------

class SilverBulletCBot():
    """ICT Silver Bullet S1 – real cTrader execution."""

    Label = "SilverBullet_S1"

    # ── lifecycle ─────────────────────────────────────────────────────────

    def on_start(self):
        """Initialise indicators and state."""

        # ── custom indicator references ──────────────────────────────────
        # Each line below calls the cTrader API to load a custom indicator.
        # The indicator types (e.g. SilverBulletWithSignals) must be
        # installed as custom cTrader Python/C# indicators beforehand.
        #
        # Syntax:  api.Indicators.GetIndicator[IndicatorType](params…)
        # The output DataSeries are accessed as  indicator.OutputName.Last(n)

        self.ind_sb        = api.Indicators.GetIndicator[SilverBulletWithSignals](api.Bars.ClosePrices)
        self.ind_setup01   = api.Indicators.GetIndicator[IctSetup01](api.Bars.ClosePrices)
        self.ind_ote       = api.Indicators.GetIndicator[FibonacciOte](api.Bars.ClosePrices)
        self.ind_ifvg      = api.Indicators.GetIndicator[IfvgRealtime](api.Bars.ClosePrices)
        self.ind_obdet     = api.Indicators.GetIndicator[OrderBlockDetector](api.Bars.ClosePrices)
        self.ind_smz       = api.Indicators.GetIndicator[SmartMoneyZones](api.Bars.ClosePrices)
        self.ind_ms_mtf    = api.Indicators.GetIndicator[MarketStructureMtf](api.Bars.ClosePrices)
        self.ind_ob_mtf    = api.Indicators.GetIndicator[OrderBlocksImbalanceMtf](api.Bars.ClosePrices)
        self.ind_fvg_mtf   = api.Indicators.GetIndicator[MtfFvgX2](api.Bars.ClosePrices)
        self.ind_session   = api.Indicators.GetIndicator[IctSessionFilter](api.Bars.ClosePrices)
        self.ind_liq       = api.Indicators.GetIndicator[LiquidityInducements](api.Bars.ClosePrices)

        # ── position event tracking ──────────────────────────────────────
        api.Positions.Opened += self.on_position_opened
        api.Positions.Closed += self.on_position_closed

        # ── session counters ─────────────────────────────────────────────
        self.trade_counter = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0

        api.Print("Silver Bullet S1 cBot started.")

    def on_stop(self):
        """Print session summary on shutdown."""
        total = self.wins + self.losses
        wr = (self.wins / total * 100) if total > 0 else 0.0
        api.Print(
            f"Session ended | Trades: {total} | "
            f"Wins: {self.wins} | Losses: {self.losses} | "
            f"Win Rate: {wr:.1f}% | P&L: ${self.total_pnl:.2f}"
        )

    # ── position events ──────────────────────────────────────────────────

    def on_position_opened(self, args):
        pos = args.Position
        if pos.Label != self.Label:
            return
        api.Print(
            f"OPENED #{pos.Id} {pos.TradeType} {pos.SymbolName} "
            f"@ {pos.EntryPrice:.2f}  Vol={pos.VolumeInUnits}"
        )

    def on_position_closed(self, args):
        pos = args.Position
        if pos.Label != self.Label:
            return
        pnl = pos.NetProfit
        self.total_pnl += pnl
        if pnl >= 0:
            self.wins += 1
            result = "WIN"
        else:
            self.losses += 1
            result = "LOSS"
        api.Print(
            f"CLOSED #{pos.Id} {result} {pos.TradeType} {pos.SymbolName} "
            f"@ {pos.EntryPrice:.2f} | P&L: ${pnl:.2f} | Reason: {args.Reason}"
        )

    # ── main bar handler ─────────────────────────────────────────────────

    def on_bar(self):
        """Called on every new bar close.  Read indicator outputs, apply
        filters, and execute trades via the cTrader API."""

        # ── 1. read entry triggers from the Silver Bullet indicator ──────
        entry_trigger_bull = int(self.ind_sb.EntryTriggerBull.Last(1))
        entry_trigger_bear = int(self.ind_sb.EntryTriggerBear.Last(1))

        if entry_trigger_bull == 0 and entry_trigger_bear == 0:
            return

        # ── 2. read filter signals ───────────────────────────────────────
        filter_session_active = int(self.ind_session.SessionActive.Last(1))
        filter_htf_bias_bull  = self._resolve_htf_bias(is_long=True)
        filter_htf_bias_bear  = self._resolve_htf_bias(is_long=False)
        filter_htf_poi_bull   = self._resolve_htf_poi(is_long=True)
        filter_htf_poi_bear   = self._resolve_htf_poi(is_long=False)

        # ── 3. read liquidity targets for stop-loss anchors ──────────────
        liq_sellside = self.ind_liq.LiqSellsideTarget.Last(1)
        liq_buyside  = self.ind_liq.LiqBuysideTarget.Last(1)

        # ── 4. count existing positions under our label ──────────────────
        my_positions = [
            p for p in api.Positions
            if p.Label == self.Label and p.SymbolName == api.SymbolName
        ]
        open_count = len(my_positions)

        # ── 5. attempt LONG entry ────────────────────────────────────────
        if entry_trigger_bull == 1:
            self._try_entry(
                is_long=True,
                filter_session_active=filter_session_active,
                filter_htf_bias=filter_htf_bias_bull,
                filter_htf_poi=filter_htf_poi_bull,
                liq_stop=liq_sellside,
                open_count=open_count,
            )

        # ── 6. attempt SHORT entry ───────────────────────────────────────
        if entry_trigger_bear == 1:
            self._try_entry(
                is_long=False,
                filter_session_active=filter_session_active,
                filter_htf_bias=filter_htf_bias_bear,
                filter_htf_poi=filter_htf_poi_bear,
                liq_stop=liq_buyside,
                open_count=open_count,
            )

    # ── HTF bias resolution ──────────────────────────────────────────────

    def _resolve_htf_bias(self, *, is_long):
        """Combine Smart Money Zones + Market Structure MTF into a single
        HTF bias filter (1 = aligned, 0 = not aligned)."""
        if is_long:
            smz_15m = int(self.ind_smz.SmzTrend15m.Last(1))
            smz_1h  = int(self.ind_smz.SmzTrend1h.Last(1))
            ms_15m  = int(self.ind_ms_mtf.MsTrend15m.Last(1))
            ms_1h   = int(self.ind_ms_mtf.MsTrend1h.Last(1))
            smz_bull = 1 if (smz_15m == 1 or smz_1h == 1) else 0
            ms_bull  = 1 if (ms_15m == 1 or ms_1h == 1) else 0
            return 1 if (smz_bull == 1 or ms_bull == 1) else 0
        else:
            smz_15m = int(self.ind_smz.SmzTrend15m.Last(1))
            smz_1h  = int(self.ind_smz.SmzTrend1h.Last(1))
            ms_15m  = int(self.ind_ms_mtf.MsTrend15m.Last(1))
            ms_1h   = int(self.ind_ms_mtf.MsTrend1h.Last(1))
            smz_bear = 1 if (smz_15m == -1 or smz_1h == -1) else 0
            ms_bear  = 1 if (ms_15m == -1 or ms_1h == -1) else 0
            return 1 if (smz_bear == 1 or ms_bear == 1) else 0

    def _resolve_htf_poi(self, *, is_long):
        """Combine OB touch + FVG touch on 1H/4H into a single HTF POI
        filter (1 = recent touch found, 0 = no touch)."""
        if is_long:
            ob_1h  = int(self.ind_ob_mtf.ObTouch1h.Last(1))
            ob_4h  = int(self.ind_ob_mtf.ObTouch4h.Last(1))
            fvg_1h = int(self.ind_fvg_mtf.FvgTouch1h.Last(1))
            fvg_4h = int(self.ind_fvg_mtf.FvgTouch4h.Last(1))
            return 1 if (ob_1h == 1 or ob_4h == 1 or fvg_1h == 1 or fvg_4h == 1) else 0
        else:
            ob_1h  = int(self.ind_ob_mtf.ObTouch1h.Last(1))
            ob_4h  = int(self.ind_ob_mtf.ObTouch4h.Last(1))
            fvg_1h = int(self.ind_fvg_mtf.FvgTouch1h.Last(1))
            fvg_4h = int(self.ind_fvg_mtf.FvgTouch4h.Last(1))
            return 1 if (ob_1h == -1 or ob_4h == -1 or fvg_1h == -1 or fvg_4h == -1) else 0

    # ── entry logic ──────────────────────────────────────────────────────

    def _try_entry(
        self,
        *,
        is_long,
        filter_session_active,
        filter_htf_bias,
        filter_htf_poi,
        liq_stop,
        open_count,
    ):
        """Apply all filters and, if passed, execute a market order."""
        direction = "LONG" if is_long else "SHORT"

        # ── filter: HTF POI ──────────────────────────────────────────────
        if filter_htf_poi != 1:
            if api.DebugSignals:
                api.Print(f"{direction} rejected: HTF POI filter (no 1H/4H OB/FVG touch)")
            return

        # ── filter: HTF trend bias ───────────────────────────────────────
        if filter_htf_bias != 1:
            if api.DebugSignals:
                api.Print(f"{direction} rejected: Trend filter (15M/1H bias not aligned)")
            return

        # ── filter: ICT session window ───────────────────────────────────
        if filter_session_active != 1:
            if api.DebugSignals:
                api.Print(f"{direction} rejected: ICT session not active")
            return

        # ── filter: max concurrent trades ────────────────────────────────
        if open_count >= api.MaxConcurrent:
            if api.DebugSignals:
                api.Print(f"{direction} rejected: max concurrent trades ({api.MaxConcurrent})")
            return

        # ── resolve SL price from liquidity target ───────────────────────
        stop_price = self._valid_price(liq_stop)
        if stop_price is None:
            if api.DebugSignals:
                api.Print(f"{direction} rejected: no valid liquidity SL anchor")
            return

        entry_price = api.Symbol.Ask if is_long else api.Symbol.Bid

        # validate SL is on the correct side of entry
        if is_long and stop_price >= entry_price:
            if api.DebugSignals:
                api.Print(f"LONG rejected: SL {stop_price:.2f} >= entry {entry_price:.2f}")
            return
        if not is_long and stop_price <= entry_price:
            if api.DebugSignals:
                api.Print(f"SHORT rejected: SL {stop_price:.2f} <= entry {entry_price:.2f}")
            return

        # ── calculate risk and volume ────────────────────────────────────
        risk_distance = abs(entry_price - stop_price)
        risk_cash = api.Account.Equity * api.RiskPerTrade

        # Volume = risk$ / risk-distance  (1 unit = $1 per $1 move on XAUUSD)
        raw_volume = risk_cash / risk_distance
        volume = api.Symbol.NormalizeVolumeInUnits(raw_volume, RoundingMode.Down)

        if volume < api.Symbol.VolumeInUnitsMin:
            if api.DebugSignals:
                api.Print(f"{direction} rejected: calculated volume {volume} below minimum")
            return

        # ── calculate TP price (2:1 RR) ──────────────────────────────────
        if is_long:
            target_price = entry_price + risk_distance * 2.0
        else:
            target_price = entry_price - risk_distance * 2.0

        # ── build signal attribution string ──────────────────────────────
        sig = self._get_signal_type(is_long)

        # ── EXECUTE THE TRADE ────────────────────────────────────────────
        trade_type = TradeType.Buy if is_long else TradeType.Sell
        result = execute_market_order(trade_type, api.SymbolName, volume, self.Label)

        if result.IsSuccessful:
            # Set SL/TP at exact price levels (not pips)
            modify_position(result.Position, stop_loss=stop_price, take_profit=target_price)

            self.trade_counter += 1
            api.Print(
                f"#{self.trade_counter} {direction} ENTRY "
                f"@ {result.Position.EntryPrice:.2f} | "
                f"SL={stop_price:.2f} TP={target_price:.2f} | "
                f"Vol={volume} | Signal={sig}"
            )
        else:
            api.Print(f"{direction} order FAILED: {result.Error}")

    # ── helpers ──────────────────────────────────────────────────────────

    def _get_signal_type(self, is_long):
        """Build a human-readable string of which entry sub-signals fired."""
        parts = []
        if is_long:
            if int(self.ind_sb.EntrySbBull.Last(1)) == 1:
                parts.append("SB_FVG_Retrace")
            if int(self.ind_setup01.EntrySetup01Bull.Last(1)) == 1:
                parts.append("ICT_Setup01")
            if int(self.ind_ote.EntryOteBull.Last(1)) == 1:
                parts.append("Fib_OTE")
            if int(self.ind_ifvg.EntryIfvgBull.Last(1)) == 1:
                parts.append("IFVG_Realtime")
            if int(self.ind_obdet.EntryObdetBull.Last(1)) == 1:
                parts.append("OB_Detector")
        else:
            if int(self.ind_sb.EntrySbBear.Last(1)) == 1:
                parts.append("SB_FVG_Retrace")
            if int(self.ind_setup01.EntrySetup01Bear.Last(1)) == 1:
                parts.append("ICT_Setup01")
            if int(self.ind_ote.EntryOteBear.Last(1)) == 1:
                parts.append("Fib_OTE")
            if int(self.ind_ifvg.EntryIfvgBear.Last(1)) == 1:
                parts.append("IFVG_Realtime")
            if int(self.ind_obdet.EntryObdetBear.Last(1)) == 1:
                parts.append("OB_Detector")
        return " + ".join(parts) if parts else "Unknown"

    @staticmethod
    def _valid_price(value):
        """Return the price if it is a valid positive number, else None."""
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if v != v or v <= 0:      # NaN check + positive check
            return None
        return v
