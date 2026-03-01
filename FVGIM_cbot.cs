// =============================================================================
// FVGIM cBot  –  self-contained, no external indicator references
// =============================================================================
// Signal logic  : mirrors FVG Instantaneous Mitigation Signals [LuxAlgo]
//                 Bullish IMFVG → long; Bearish IMFVG → short
// SL anchor     : mirrors BSL and SSL (BSL and SSL.cs)
//
// Long  entry : market order at next bar open after bullish IMFVG bar closes
//               SL = nearest SSL (pivot low) below entry
// Short entry : market order at next bar open after bearish IMFVG bar closes
//               SL = nearest BSL (pivot high) above entry
// TP          : 2 × SL distance  (1 : 2 risk-to-reward)
// Risk        : 1 % of current account equity per trade
// =============================================================================

using System;
using System.Collections.Generic;
using cAlgo.API;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class FVGIM_cBot : Robot
    {
        // ── IMFVG signal parameters ──────────────────────────────────────────

        [Parameter("FVG Width Filter", DefaultValue = 0.0, MinValue = 0.0, Step = 0.1, Group = "IMFVG Signal")]
        public double FvgWidthFilter { get; set; }

        // ── BSL & SSL parameters ─────────────────────────────────────────────

        [Parameter("Pivot Left", DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotLeft { get; set; }

        [Parameter("Pivot Right", DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotRight { get; set; }

        // ── Risk management ──────────────────────────────────────────────────

        [Parameter("Risk Per Trade (%)", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 10.0, Group = "Risk Management")]
        public double RiskPercent { get; set; }

        [Parameter("Min SL Distance (pips)", DefaultValue = 3.0, MinValue = 0.1, Group = "Risk Management")]
        public double MinSlPips { get; set; }

        [Parameter("Max SL Distance (pips)", DefaultValue = 500.0, MinValue = 10.0, Group = "Risk Management")]
        public double MaxSlPips { get; set; }

        // ── Constants ────────────────────────────────────────────────────────

        private const string BotLabel  = "FVGIM_cBot";
        private const double RrRatio   = 2.0;
        private const int    AtrLength = 200;   // matches LuxAlgo ATR(200)

        // =====================================================================
        // IMFVG signal embedded state
        // =====================================================================

        // ATR (Wilder RMA, length 200) — used by the FVG width filter
        private double _atrSumAccum;
        private double _atrRma = double.NaN;

        // Per-bar signal flags (set by UpdateFvgImSignal, read by OnBar)
        private bool _isLongSignal;
        private bool _isShortSignal;

        // =====================================================================
        // BSL / SSL embedded state
        // =====================================================================

        private sealed class LiquidityLevel
        {
            public double Price;
            public int    PivotIndex;
            public bool   Mitigated;   // marked when price sweeps through; kept for SL reference
        }

        private readonly LinkedList<LiquidityLevel> _bslPool = new LinkedList<LiquidityLevel>();
        private readonly LinkedList<LiquidityLevel> _sslPool = new LinkedList<LiquidityLevel>();
        private const int MaxLiquidityLevels = 10;

        // Returns the nearest SSL price strictly below 'price', including swept levels.
        private double FindNearestSslBelow(double price)
        {
            double nearest = double.NaN;
            foreach (var ssl in _sslPool)
            {
                if (ssl.Price >= price) continue;
                if (double.IsNaN(nearest) || ssl.Price > nearest)
                    nearest = ssl.Price;
            }
            return nearest;
        }

        // Returns the nearest BSL price strictly above 'price', including swept levels.
        private double FindNearestBslAbove(double price)
        {
            double nearest = double.NaN;
            foreach (var bsl in _bslPool)
            {
                if (bsl.Price <= price) continue;
                if (double.IsNaN(nearest) || bsl.Price < nearest)
                    nearest = bsl.Price;
            }
            return nearest;
        }

        // ── Duplicate-entry guards ────────────────────────────────────────────

        private int _lastLongSignalBar  = -1;
        private int _lastShortSignalBar = -1;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        protected override void OnStart()
        {
            // Warm up both embedded indicators over all complete historical bars.
            // Bars.Count - 1 is the currently forming bar; stop at Bars.Count - 2.
            int warmupEnd = Bars.Count - 2;
            for (int i = 0; i <= warmupEnd; i++)
                ProcessBar(i);

            Print("FVGIM cBot started. Risk={0}%, RR=1:{1}, FvgWidthFilter={2}",
                  RiskPercent, RrRatio, FvgWidthFilter);
        }

        protected override void OnStop()
        {
            Print("FVGIM cBot stopped.");
        }

        // =====================================================================
        // Bar event
        // =====================================================================

        protected override void OnBar()
        {
            // OnBar fires when a NEW bar opens.
            // Bars.Count - 2 is the bar that just CLOSED (the signal bar).
            int signalBar = Bars.Count - 2;
            if (signalBar < 3)   // IMFVG needs at least 4 bars (index - 3)
                return;

            // Advance both embedded indicator states for the closed bar.
            ProcessBar(signalBar);

            bool hasLong  = _isLongSignal;
            bool hasShort = _isShortSignal;

            if (!hasLong && !hasShort)
                return;

            if (hasLong && _lastLongSignalBar != signalBar)
            {
                _lastLongSignalBar = signalBar;
                TryEnterLong(signalBar);
            }

            if (hasShort && _lastShortSignalBar != signalBar)
            {
                _lastShortSignalBar = signalBar;
                TryEnterShort(signalBar);
            }
        }

        // =====================================================================
        // Per-bar state update
        // =====================================================================

        private void ProcessBar(int index)
        {
            UpdateAtr(index);
            UpdateFvgImSignal(index);
            UpdateBslSsl(index);
        }

        // ─────────────────────────────────────────────────────────────────────
        // ATR  (Wilder's smoothed moving average, length 200)
        // ─────────────────────────────────────────────────────────────────────

        private void UpdateAtr(int index)
        {
            double high = Bars.HighPrices[index];
            double low  = Bars.LowPrices[index];
            double tr;

            if (index == 0)
            {
                tr           = high - low;
                _atrSumAccum = tr;
                return;
            }

            double prevClose = Bars.ClosePrices[index - 1];
            tr = Math.Max(high - low,
                 Math.Max(Math.Abs(high - prevClose),
                          Math.Abs(low  - prevClose)));

            if (index < AtrLength - 1)
            {
                _atrSumAccum += tr;
            }
            else if (index == AtrLength - 1)
            {
                _atrSumAccum += tr;
                _atrRma = _atrSumAccum / AtrLength;   // SMA seed
            }
            else
            {
                _atrRma = ((_atrRma * (AtrLength - 1)) + tr) / AtrLength;  // Wilder smoothing
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // IMFVG signal logic  (mirrors FVG Instantaneous Mitigation [LuxAlgo])
        //
        // Bullish IMFVG on bar i:
        //   Low[i-3] > High[i-1]          — bullish FVG gap (i-3 is left, i-1 is right)
        //   Close[i-2] < Low[i-3]         — middle bar dipped below FVG bottom
        //   Close[i]   > Low[i-3]         — signal bar recovered above FVG bottom
        //   FVG width  > ATR × filter     — size filter (0.0 default = always passes)
        //
        // Bearish IMFVG on bar i:
        //   Low[i-1]   > High[i-3]        — bearish FVG gap
        //   Close[i-2] > High[i-3]        — middle bar spiked above FVG top
        //   Close[i]   < High[i-3]        — signal bar dropped below FVG top
        //   FVG width  > ATR × filter
        // ─────────────────────────────────────────────────────────────────────

        private void UpdateFvgImSignal(int index)
        {
            _isLongSignal  = false;
            _isShortSignal = false;

            if (index < 3)
                return;

            double atr = double.IsNaN(_atrRma) ? 0.0 : _atrRma;

            bool bull = Bars.LowPrices[index - 3]  > Bars.HighPrices[index - 1]
                     && Bars.ClosePrices[index - 2] < Bars.LowPrices[index - 3]
                     && Bars.ClosePrices[index]     > Bars.LowPrices[index - 3]
                     && (Bars.LowPrices[index - 3] - Bars.HighPrices[index - 1]) > atr * FvgWidthFilter;

            bool bear = Bars.LowPrices[index - 1]  > Bars.HighPrices[index - 3]
                     && Bars.ClosePrices[index - 2] > Bars.HighPrices[index - 3]
                     && Bars.ClosePrices[index]     < Bars.HighPrices[index - 3]
                     && (Bars.LowPrices[index - 1] - Bars.HighPrices[index - 3]) > atr * FvgWidthFilter;

            _isLongSignal  = bull;
            _isShortSignal = bear;
        }

        // ─────────────────────────────────────────────────────────────────────
        // BSL / SSL pivot logic  (mirrors BSL_SSL.Calculate)
        // ─────────────────────────────────────────────────────────────────────

        private void UpdateBslSsl(int index)
        {
            int pivotIndex = index - PivotRight;
            if (pivotIndex <= 0)
                return;

            int leftStart = pivotIndex - PivotLeft;
            int rightEnd  = pivotIndex + PivotRight;

            if (leftStart < 0 || rightEnd >= Bars.Count)
                return;

            double candidateHigh = Bars.HighPrices[pivotIndex];
            double candidateLow  = Bars.LowPrices[pivotIndex];

            if (IsPivotHigh(candidateHigh, leftStart, rightEnd))
                AddToPool(_bslPool, candidateHigh, pivotIndex);

            if (IsPivotLow(candidateLow, leftStart, rightEnd))
                AddToPool(_sslPool, candidateLow, pivotIndex);

            MitigateLevels(index);
        }

        private bool IsPivotHigh(double candidate, int start, int end)
        {
            double max = double.MinValue;
            for (int i = start; i <= end; i++)
                if (Bars.HighPrices[i] > max) max = Bars.HighPrices[i];
            return candidate == max;
        }

        private bool IsPivotLow(double candidate, int start, int end)
        {
            double min = double.MaxValue;
            for (int i = start; i <= end; i++)
                if (Bars.LowPrices[i] < min) min = Bars.LowPrices[i];
            return candidate == min;
        }

        private void AddToPool(LinkedList<LiquidityLevel> pool, double price, int pivotIndex)
        {
            if (pool.First != null &&
                pool.First.Value.PivotIndex == pivotIndex &&
                Math.Abs(pool.First.Value.Price - price) < Symbol.PipSize * 0.1)
                return;

            pool.AddFirst(new LiquidityLevel { Price = price, PivotIndex = pivotIndex });

            while (pool.Count > MaxLiquidityLevels)
                pool.RemoveLast();
        }

        private void MitigateLevels(int index)
        {
            double barHigh = Bars.HighPrices[index];
            double barLow  = Bars.LowPrices[index];

            foreach (var ssl in _sslPool)
                if (!ssl.Mitigated && barLow <= ssl.Price)
                    ssl.Mitigated = true;

            foreach (var bsl in _bslPool)
                if (!bsl.Mitigated && barHigh >= bsl.Price)
                    bsl.Mitigated = true;
        }

        // =====================================================================
        // Trade execution
        // =====================================================================

        private void TryEnterLong(int signalBar)
        {
            double entry    = Symbol.Ask;
            double sslLevel = FindNearestSslBelow(entry);

            // Fallback: signal bar's own low (the IMFVG recovery low)
            if (double.IsNaN(sslLevel) || sslLevel <= 0)
            {
                double barLow = Bars.LowPrices[signalBar];
                if (barLow > 0 && barLow < entry)
                {
                    sslLevel = barLow;
                    Print("Bar {0}: LONG using signal-bar low {1:F5} as SL anchor (no SSL pool level below entry).",
                          signalBar, sslLevel);
                }
            }

            if (double.IsNaN(sslLevel) || sslLevel <= 0)
            {
                Print("Bar {0}: LONG skipped – no SL anchor found below entry {1:F5}.", signalBar, entry);
                return;
            }

            if (sslLevel >= entry)
            {
                Print("Bar {0}: LONG skipped – SL anchor {1:F5} not below entry {2:F5}.",
                      signalBar, sslLevel, entry);
                return;
            }

            double slPips = (entry - sslLevel) / Symbol.PipSize;
            if (!ValidateSlPips(signalBar, "LONG", slPips))
                return;

            double tpPips     = slPips * RrRatio;
            double riskAmount = Account.Equity * (RiskPercent / 100.0);
            double volume     = GetRiskVolume(riskAmount, slPips);

            if (volume <= 0)
            {
                Print("Bar {0}: LONG skipped – volume rounds to 0.", signalBar);
                return;
            }

            Print("Bar {0}: LONG  | Entry={1:F5} | SSL={2:F5} | SL={3:F1}p | TP={4:F1}p | Vol={5}",
                  signalBar, entry, sslLevel, slPips, tpPips, volume);

            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, BotLabel, slPips, tpPips);
        }

        private void TryEnterShort(int signalBar)
        {
            double entry    = Symbol.Bid;
            double bslLevel = FindNearestBslAbove(entry);

            // Fallback: signal bar's own high (the IMFVG rejection high)
            if (double.IsNaN(bslLevel) || bslLevel <= 0)
            {
                double barHigh = Bars.HighPrices[signalBar];
                if (barHigh > 0 && barHigh > entry)
                {
                    bslLevel = barHigh;
                    Print("Bar {0}: SHORT using signal-bar high {1:F5} as SL anchor (no BSL pool level above entry).",
                          signalBar, bslLevel);
                }
            }

            if (double.IsNaN(bslLevel) || bslLevel <= 0)
            {
                Print("Bar {0}: SHORT skipped – no SL anchor found above entry {1:F5}.", signalBar, entry);
                return;
            }

            if (bslLevel <= entry)
            {
                Print("Bar {0}: SHORT skipped – SL anchor {1:F5} not above entry {2:F5}.",
                      signalBar, bslLevel, entry);
                return;
            }

            double slPips = (bslLevel - entry) / Symbol.PipSize;
            if (!ValidateSlPips(signalBar, "SHORT", slPips))
                return;

            double tpPips     = slPips * RrRatio;
            double riskAmount = Account.Equity * (RiskPercent / 100.0);
            double volume     = GetRiskVolume(riskAmount, slPips);

            if (volume <= 0)
            {
                Print("Bar {0}: SHORT skipped – volume rounds to 0.", signalBar);
                return;
            }

            Print("Bar {0}: SHORT | Entry={1:F5} | BSL={2:F5} | SL={3:F1}p | TP={4:F1}p | Vol={5}",
                  signalBar, entry, bslLevel, slPips, tpPips, volume);

            ExecuteMarketOrder(TradeType.Sell, SymbolName, volume, BotLabel, slPips, tpPips);
        }

        // =====================================================================
        // Helpers
        // =====================================================================

        private bool ValidateSlPips(int signalBar, string dir, double slPips)
        {
            if (slPips < MinSlPips)
            {
                Print("Bar {0}: {1} skipped – SL {2:F1}p < min {3:F1}p.",
                      signalBar, dir, slPips, MinSlPips);
                return false;
            }
            if (slPips > MaxSlPips)
            {
                Print("Bar {0}: {1} skipped – SL {2:F1}p > max {3:F1}p.",
                      signalBar, dir, slPips, MaxSlPips);
                return false;
            }
            return true;
        }

        private double GetRiskVolume(double riskAmount, double slPips)
        {
            if (slPips <= 0)
                return 0;

            double raw    = Symbol.VolumeForFixedRisk(riskAmount, slPips);
            double volume = Symbol.NormalizeVolumeInUnits(raw, RoundingMode.Down);

            if (volume < Symbol.VolumeInUnitsMin)
                return 0;

            if (volume > Symbol.VolumeInUnitsMax)
                volume = Symbol.VolumeInUnitsMax;

            return volume;
        }
    }
}
