// =============================================================================
// IFVG_BSL_SSL_RiskBot
// =============================================================================
// Self-contained version of IFVG_BSL_SSL_RiskBot. No external indicator
// references.  Structured identically to ICT_01_cBot_Single.
//
// Signal engine — IFVG_Realtime.cs (verbatim logic)
//   • DownTriangle above bar high  → Short signal  (signalDir = -1)
//   • UpTriangle below  bar low    → Long  signal  (signalDir =  1)
//   Trade opens on the bar immediately after the signal bar.
//
// BSL/SSL engine — BSL and SSL.cs (verbatim, chart objects removed)
//   Stop loss anchored to nearest unmitigated BSL (for shorts)
//   or SSL (for longs).
//
// Architecture mirrors ICT_01_cBot_Single:
//   • Incremental _lastProcessed loop — BSL/SSL and MA state are
//     updated bar-by-bar so every bar's pool state is correct.
//   • MaxOpenPositions (int 1-100) + InstanceName parameter.
//   • _lastLongSignalBar / _lastShortSignalBar dedup.
//   • ValidateSlPips with configurable min/max pip guards.
// =============================================================================

using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Indicators;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class IFVGBslSslRiskBot : Robot
    {
        // ═════════════════════════════════════════════════════════════════════
        //  Parameters — IFVG signal
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("FVG Search Lookback (Bars)", Group = "IFVG", DefaultValue = 15, MinValue = 1)]
        public int FvgGapBars { get; set; }

        [Parameter("Min FVG Size (Pips/Points)", Group = "IFVG", DefaultValue = 0.0, MinValue = 0.0)]
        public double MinFvgPips { get; set; }

        [Parameter("FVG Epsilon (Price Units)", Group = "IFVG", DefaultValue = 0.0, MinValue = 0.0)]
        public double FvgEpsilonPoints { get; set; }

        [Parameter("MA Period", Group = "IFVG", DefaultValue = 21, MinValue = 1)]
        public int MaPeriod { get; set; }

        [Parameter("MA Type (SMA / EMA)", Group = "IFVG", DefaultValue = "EMA")]
        public string MaType { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  Parameters — BSL & SSL
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Pivot Left", Group = "BSL & SSL", DefaultValue = 5, MinValue = 1)]
        public int PivotLeft { get; set; }

        [Parameter("Pivot Right", Group = "BSL & SSL", DefaultValue = 5, MinValue = 1)]
        public int PivotRight { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  Parameters — Risk Management
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Risk % per trade", Group = "Risk Management", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 100.0)]
        public double RiskPercent { get; set; }

        [Parameter("Risk:Reward Ratio", Group = "Risk Management", DefaultValue = 2.0, MinValue = 0.1, Step = 0.1)]
        public double RiskRewardRatio { get; set; }

        [Parameter("Max Open Positions", Group = "Risk Management", DefaultValue = 3, MinValue = 1, MaxValue = 100)]
        public int MaxOpenPositions { get; set; }

        [Parameter("Min SL Distance (pips)", Group = "Risk Management", DefaultValue = 3.0, MinValue = 0.1)]
        public double MinSlPips { get; set; }

        [Parameter("Max SL Distance (pips)", Group = "Risk Management", DefaultValue = 500.0, MinValue = 1.0)]
        public double MaxSlPips { get; set; }

        [Parameter("Instance Name", Group = "Risk Management", DefaultValue = "IFVG_BSL_SSL")]
        public string InstanceName { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  Constants
        // ═════════════════════════════════════════════════════════════════════

        private const int MaxPivotsToKeep = 10;

        // ═════════════════════════════════════════════════════════════════════
        //  Inner types — BSL / SSL pivot pools
        // ═════════════════════════════════════════════════════════════════════

        private sealed class BslPivot
        {
            public double Price;
            public int    BarIndex;
            public int    Type;    // 1 = pivot high (BSL), -1 = pivot low (SSL)
        }

        private sealed class BslPool
        {
            public double Price;
            public int    PivotIndex;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  IFVG engine fields
        //
        //  _maSeries must be populated bar-by-bar in the incremental loop
        //  because TryProcessFvgCandidate reads _maSeries[index - 1].
        // ═════════════════════════════════════════════════════════════════════

        private IndicatorDataSeries      _maSeries;
        private SimpleMovingAverage      _sma;
        private ExponentialMovingAverage _ema;

        // ═════════════════════════════════════════════════════════════════════
        //  BSL / SSL engine fields
        // ═════════════════════════════════════════════════════════════════════

        private readonly LinkedList<BslPivot> _bslPivots        = new LinkedList<BslPivot>();
        private readonly LinkedList<BslPool>  _bslBuysidePools  = new LinkedList<BslPool>();
        private readonly LinkedList<BslPool>  _bslSellsidePools = new LinkedList<BslPool>();

        private double _bslCurrentBsl = double.NaN;   // BSL → SL for shorts
        private double _bslCurrentSsl = double.NaN;   // SSL → SL for longs

        // ═════════════════════════════════════════════════════════════════════
        //  cBot state
        // ═════════════════════════════════════════════════════════════════════

        private int _lastLongSignalBar  = -1;
        private int _lastShortSignalBar = -1;
        private int _lastProcessed      = -1;

        // ═════════════════════════════════════════════════════════════════════
        //  Lifecycle
        // ═════════════════════════════════════════════════════════════════════

        protected override void OnStart()
        {
            _maSeries = CreateDataSeries();
            _sma      = Indicators.SimpleMovingAverage(Bars.ClosePrices, MaPeriod);
            _ema      = Indicators.ExponentialMovingAverage(Bars.ClosePrices, MaPeriod);

            Print("IFVG_BSL_SSL_RiskBot started. MaxPositions={0}, Risk={1}%, RR={2}",
                  MaxOpenPositions, RiskPercent, RiskRewardRatio);
        }

        protected override void OnStop()
        {
            Print("IFVG_BSL_SSL_RiskBot stopped.");
        }

        // ═════════════════════════════════════════════════════════════════════
        //  OnBar
        //
        //  Step 1 — Incremental fill: BSL/SSL pools and MA series are updated
        //           bar-by-bar so every bar's state is correct when we later
        //           read _maSeries[signalBar - 1] inside DetectIfvgSignal.
        //  Step 2 — Detect IFVG signal on the just-closed bar (signalBar).
        //  Step 3 — MaxOpenPositions cap.
        //  Step 4 — Execute trade on the next bar open (Symbol.Ask / Bid).
        // ═════════════════════════════════════════════════════════════════════

        protected override void OnBar()
        {
            int signalBar = Bars.Count - 2;   // last fully-closed bar

            // ── Step 1: build BSL/SSL pools and MA up to signalBar ────────────
            for (int i = _lastProcessed + 1; i <= signalBar; i++)
            {
                RunBslSsl(i);      // update pivot pools
                UpdateMa(i);       // populate _maSeries[i] for MA condition check
            }
            _lastProcessed = signalBar;

            if (signalBar < Math.Max(PivotLeft + PivotRight + 1, 3)) return;

            // ── Step 2: IFVG signal detection ─────────────────────────────────
            // maValue is _maSeries[signalBar], already set by UpdateMa above.
            var maValue   = _maSeries[signalBar];
            var signalDir = DetectIfvgSignal(signalBar, maValue);
            if (signalDir == 0) return;

            // ── Step 3: position cap ──────────────────────────────────────────
            int openCount = Positions.FindAll(InstanceName, SymbolName).Length;
            if (openCount >= MaxOpenPositions)
            {
                Print("Bar {0}: max positions ({1}) reached.", signalBar, MaxOpenPositions);
                return;
            }

            // ── Step 4: execute trade ─────────────────────────────────────────
            if (signalDir == 1 && _lastLongSignalBar != signalBar)
            {
                _lastLongSignalBar = signalBar;
                TryEnterLong(signalBar);
            }

            // Re-check capacity in case long just filled it.
            openCount = Positions.FindAll(InstanceName, SymbolName).Length;
            if (openCount >= MaxOpenPositions) return;

            if (signalDir == -1 && _lastShortSignalBar != signalBar)
            {
                _lastShortSignalBar = signalBar;
                TryEnterShort(signalBar);
            }
        }

        // ═════════════════════════════════════════════════════════════════════
        //  IFVG signal engine — verbatim logic from IFVG_Realtime.cs
        //
        //  Long  signal (UpTriangle   below bar low)  → signalDir =  1
        //  Short signal (DownTriangle above bar high) → signalDir = -1
        // ═════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Populates _maSeries[index] for use in TryProcessFvgCandidate.
        /// Called in the incremental loop so _maSeries[index - 1] is always
        /// valid when DetectIfvgSignal runs on the signal bar.
        /// </summary>
        private void UpdateMa(int index)
        {
            _maSeries[index] = string.Equals(MaType, "SMA", StringComparison.OrdinalIgnoreCase)
                ? _sma.Result[index]
                : _ema.Result[index];
        }

        private int DetectIfvgSignal(int index, double maValue)
        {
            var minSizeValue = MinFvgPips * Symbol.PipSize;

            for (var i = 1; i <= FvgGapBars; i++)
            {
                var fvgType = DetectFvg(index, i, FvgEpsilonPoints);
                if (fvgType == 0) continue;

                int signalDir;
                if (TryProcessFvgCandidate(index, i, fvgType, minSizeValue, maValue, out signalDir))
                    return signalDir;
            }

            return 0;
        }

        private int DetectFvg(int currentIndex, int idx, double epsVal)
        {
            if (idx + 2 > currentIndex) return 0;

            var h2 = Bars.HighPrices[currentIndex - (idx + 2)];
            var l2 = Bars.LowPrices[currentIndex  - (idx + 2)];
            var lt = Bars.LowPrices[currentIndex  - idx];
            var ht = Bars.HighPrices[currentIndex - idx];

            if (lt > h2 - epsVal) return  1;   // bearish gap → fvgType 1
            if (ht < l2 + epsVal) return -1;   // bullish gap → fvgType -1
            return 0;
        }

        private bool TryProcessFvgCandidate(
            int index, int i, int fvgType, double minSizeValue,
            double maValue, out int signalDir)
        {
            signalDir = 0;

            var isBearishGap = fvgType == 1;
            var gapLow  = isBearishGap ? Bars.HighPrices[index - (i + 2)] : Bars.HighPrices[index - i];
            var gapHigh = isBearishGap ? Bars.LowPrices[index  - i]        : Bars.LowPrices[index  - (i + 2)];

            if ((gapHigh - gapLow) < minSizeValue) return false;

            // Reject if the gap was already broken before the signal bar
            if (i > 1)
            {
                for (var k = i - 1; k >= 1; k--)
                {
                    var close = Bars.ClosePrices[index - k];
                    if ((isBearishGap && close < gapLow) || (!isBearishGap && close > gapHigh))
                        return false;
                }
            }

            // Current bar must break out of the gap
            var breakout = isBearishGap
                ? Bars.ClosePrices[index] < gapLow
                : Bars.ClosePrices[index] > gapHigh;
            if (!breakout) return false;

            // MA direction + price below/above MA
            // _maSeries[index - 1] is guaranteed valid because UpdateMa
            // was called for every bar up to index in the incremental loop.
            var maReady     = !double.IsNaN(maValue) && !double.IsNaN(_maSeries[index - 1]);
            var maCondition = isBearishGap
                ? maReady && maValue < _maSeries[index - 1] && Bars.ClosePrices[index] < maValue
                : maReady && maValue > _maSeries[index - 1] && Bars.ClosePrices[index] > maValue;
            if (!maCondition) return false;

            // isBearishGap == true  → close breaks below bearish FVG gap
            //   indicator draws DownTriangle at bar High → Short signal
            // isBearishGap == false → close breaks above bullish FVG gap
            //   indicator draws UpTriangle   at bar Low  → Long  signal
            signalDir = isBearishGap ? -1 : 1;
            return true;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  Trade entry helpers
        // ═════════════════════════════════════════════════════════════════════

        private void TryEnterLong(int signalBar)
        {
            double entry    = Symbol.Ask;
            double sslLevel = _bslCurrentSsl;
            if (double.IsNaN(sslLevel) || sslLevel <= 0)
            { Print("Bar {0}: LONG skipped – SSL unavailable.", signalBar); return; }
            if (sslLevel >= entry)
            { Print("Bar {0}: LONG skipped – SSL {1:F5} not below entry {2:F5}.", signalBar, sslLevel, entry); return; }

            double slPips = (entry - sslLevel) / Symbol.PipSize;
            if (!ValidateSlPips(signalBar, "LONG", slPips)) return;

            double riskAmount = Account.Equity * (RiskPercent / 100.0);
            double volume     = GetRiskVolume(riskAmount, slPips);
            if (volume <= 0) { Print("Bar {0}: LONG skipped – volume is 0.", signalBar); return; }

            double tpPips = slPips * RiskRewardRatio;
            Print("Bar {0}: LONG  | Entry={1:F5} | SSL={2:F5} ({3:F1} pips) | TP={4:F1} pips | Vol={5}",
                  signalBar, entry, sslLevel, slPips, tpPips, volume);
            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, InstanceName, slPips, tpPips);
        }

        private void TryEnterShort(int signalBar)
        {
            double entry    = Symbol.Bid;
            double bslLevel = _bslCurrentBsl;
            if (double.IsNaN(bslLevel) || bslLevel <= 0)
            { Print("Bar {0}: SHORT skipped – BSL unavailable.", signalBar); return; }
            if (bslLevel <= entry)
            { Print("Bar {0}: SHORT skipped – BSL {1:F5} not above entry {2:F5}.", signalBar, bslLevel, entry); return; }

            double slPips = (bslLevel - entry) / Symbol.PipSize;
            if (!ValidateSlPips(signalBar, "SHORT", slPips)) return;

            double riskAmount = Account.Equity * (RiskPercent / 100.0);
            double volume     = GetRiskVolume(riskAmount, slPips);
            if (volume <= 0) { Print("Bar {0}: SHORT skipped – volume is 0.", signalBar); return; }

            double tpPips = slPips * RiskRewardRatio;
            Print("Bar {0}: SHORT | Entry={1:F5} | BSL={2:F5} ({3:F1} pips) | TP={4:F1} pips | Vol={5}",
                  signalBar, entry, bslLevel, slPips, tpPips, volume);
            ExecuteMarketOrder(TradeType.Sell, SymbolName, volume, InstanceName, slPips, tpPips);
        }

        private bool ValidateSlPips(int signalBar, string direction, double slPips)
        {
            if (slPips < MinSlPips)
            { Print("Bar {0}: {1} skipped – SL {2:F1} pips < min {3:F1}.", signalBar, direction, slPips, MinSlPips); return false; }
            if (slPips > MaxSlPips)
            { Print("Bar {0}: {1} skipped – SL {2:F1} pips > max {3:F1}.", signalBar, direction, slPips, MaxSlPips); return false; }
            return true;
        }

        private double GetRiskVolume(double riskAmount, double slPips)
        {
            if (slPips <= 0) return 0;
            double volume = Symbol.NormalizeVolumeInUnits(
                Symbol.VolumeForFixedRisk(riskAmount, slPips), RoundingMode.Down);
            if (volume < Symbol.VolumeInUnitsMin) return 0;
            if (volume > Symbol.VolumeInUnitsMax) volume = Symbol.VolumeInUnitsMax;
            return volume;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  BSL / SSL engine — ported from BSL and SSL.cs
        //  (chart drawing removed; pool logic verbatim)
        // ═════════════════════════════════════════════════════════════════════

        private void RunBslSsl(int index)
        {
            BslDetectAndStoreConfirmedPivots(index);
            BslAddPoolFromNewPivot(index);
            BslClearMitigated(index);

            _bslCurrentBsl = _bslBuysidePools.First  != null ? _bslBuysidePools.First.Value.Price  : double.NaN;
            _bslCurrentSsl = _bslSellsidePools.First != null ? _bslSellsidePools.First.Value.Price : double.NaN;
        }

        private void BslDetectAndStoreConfirmedPivots(int currentIndex)
        {
            int pivotIndex = currentIndex - PivotRight;
            if (pivotIndex <= 0) return;
            int leftStart = pivotIndex - PivotLeft;
            int rightEnd  = pivotIndex + PivotRight;
            if (leftStart < 0 || rightEnd >= Bars.Count) return;

            double candidateHigh = Bars.HighPrices[pivotIndex];
            double candidateLow  = Bars.LowPrices[pivotIndex];

            if (BslIsPivotHigh(candidateHigh, leftStart, rightEnd))
                BslUnshiftPivot(new BslPivot { Price = candidateHigh, BarIndex = pivotIndex, Type =  1 });
            if (BslIsPivotLow(candidateLow, leftStart, rightEnd))
                BslUnshiftPivot(new BslPivot { Price = candidateLow,  BarIndex = pivotIndex, Type = -1 });
        }

        private bool BslIsPivotHigh(double candidate, int start, int end)
        {
            double max = double.MinValue;
            for (int i = start; i <= end; i++) if (Bars.HighPrices[i] > max) max = Bars.HighPrices[i];
            return candidate == max;
        }

        private bool BslIsPivotLow(double candidate, int start, int end)
        {
            double min = double.MaxValue;
            for (int i = start; i <= end; i++) if (Bars.LowPrices[i] < min) min = Bars.LowPrices[i];
            return candidate == min;
        }

        private void BslUnshiftPivot(BslPivot p)
        {
            if (_bslPivots.First != null)
            {
                var f = _bslPivots.First.Value;
                if (f.BarIndex == p.BarIndex && f.Type == p.Type &&
                    Math.Abs(f.Price - p.Price) < Symbol.PipSize * 0.1) return;
            }
            _bslPivots.AddFirst(p);
            while (_bslPivots.Count > MaxPivotsToKeep) _bslPivots.RemoveLast();
        }

        private void BslAddPoolFromNewPivot(int currentIndex)
        {
            int confirmedIdx = currentIndex - PivotRight;
            foreach (var pivot in _bslPivots)
            {
                if (pivot.BarIndex != confirmedIdx) continue;
                var pool = new BslPool { Price = pivot.Price, PivotIndex = pivot.BarIndex };
                if (pivot.Type ==  1) _bslBuysidePools.AddFirst(pool);
                if (pivot.Type == -1) _bslSellsidePools.AddFirst(pool);
            }
        }

        private void BslClearMitigated(int index)
        {
            // SSL (sellside): mitigated when Low ≤ pool price
            var node = _bslSellsidePools.First;
            while (node != null)
            { var next = node.Next; if (Bars.LowPrices[index] <= node.Value.Price) _bslSellsidePools.Remove(node); node = next; }

            // BSL (buyside): mitigated when High ≥ pool price
            node = _bslBuysidePools.First;
            while (node != null)
            { var next = node.Next; if (Bars.HighPrices[index] >= node.Value.Price) _bslBuysidePools.Remove(node); node = next; }
        }
    }
}
