// =============================================================================
// ICT_01_Liq_cBot.cs
// =============================================================================
// Self-contained cBot — all ICT_01 signal logic and BSL/SSL pivot logic are
// embedded directly.  No Indicators.GetIndicator<T>() is used because the
// linked-indicator approach suffers from a cTrader backtesting timing issue
// where the indicator's Calculate() may not have run on the just-closed bar's
// final OHLC before OnBar() fires.  Calling ProcessBar(signalBar) explicitly
// in OnBar() guarantees that the signal is computed from the fully-closed bar.
//
// Signal logic  : mirrors ICT_01 (ICT Setup 01 TFlab_ct.cs)
// SL anchor     : mirrors BSL and SSL (BSL and SSL.cs)
//                 Fallback: sweep low/high from the signal bar itself when the
//                 BSL/SSL pool has been cleared by the signal-bar's sweep.
//
// Long  entry : market order at next bar open after long signal bar closes
//               SL ref = nearest active SSL (pivot low); fallback = bar low
//               Actual SL placed SlBufferPips below the reference price.
// Short entry : market order at next bar open after short signal bar closes
//               SL ref = nearest active BSL (pivot high); fallback = bar high
//               Actual SL placed SlBufferPips above the reference price.
// TP          : SL distance × RewardRiskRatio  (default 1:2)
// Risk        : RiskPercent % of current account equity per trade
// Capacity    : max MaxOpenPositions simultaneous positions
//
// WHY THE FALLBACK IS NEEDED
//   The ICT_01 long signal fires precisely because price swept below the FVG
//   proximal zone.  That same sweep triggers BSL_SSL's ClearMitigated() which
//   removes every pivot-low pool entry whose price >= the bar's low.  When all
//   entries are cleared the output is NaN.  The natural ICT stop-loss in that
//   case is below the sweep low itself — which is exactly what the signal bar's
//   low price provides.  Likewise for short signals.
// =============================================================================

using System;
using System.Collections.Generic;
using cAlgo.API;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class ICT_01_Liq_cBot : Robot
    {
        // =====================================================================
        // Parameters – ICT_01 signal settings
        // =====================================================================

        [Parameter("FVG Detector Multiplier", DefaultValue = 1.0, MinValue = 1.0,
            Group = "ICT_01 Signal Settings")]
        public double FvgDetectorMultiplier { get; set; }

        [Parameter("FVG Validity Period (bars)", DefaultValue = 15, MinValue = 2,
            Group = "ICT_01 Signal Settings")]
        public int FvgValidityPeriod { get; set; }

        [Parameter("Use Discount/Premium Zone", DefaultValue = false,
            Group = "ICT_01 Signal Settings")]
        public bool UseDiscountAndPremium { get; set; }

        [Parameter("Signal Method (Hunt / Sweeps)", DefaultValue = "Hunt",
            Group = "ICT_01 Signal Settings")]
        public string SignalMethod { get; set; }

        [Parameter("Max Signals Per Zone", DefaultValue = 3, MinValue = 1,
            Group = "ICT_01 Signal Settings")]
        public int SignalsAllowedPerZone { get; set; }

        [Parameter("Signal After Hunts/Sweeps", DefaultValue = false,
            Group = "ICT_01 Signal Settings")]
        public bool SignalAfterHunts { get; set; }

        [Parameter("Required Hunts/Sweeps Count", DefaultValue = 2, MinValue = 1,
            Group = "ICT_01 Signal Settings")]
        public int RequiredHunts { get; set; }

        // =====================================================================
        // Parameters – BSL/SSL liquidity settings
        // =====================================================================

        [Parameter("Pivot Left", DefaultValue = 5, MinValue = 1,
            Group = "BSL/SSL Liquidity Settings")]
        public int PivotLeft { get; set; }

        [Parameter("Pivot Right", DefaultValue = 5, MinValue = 1,
            Group = "BSL/SSL Liquidity Settings")]
        public int PivotRight { get; set; }

        // =====================================================================
        // Parameters – Risk management
        // =====================================================================

        [Parameter("Risk Per Trade (%)", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 10.0,
            Group = "Risk Management")]
        public double RiskPercent { get; set; }

        [Parameter("Reward : Risk Ratio", DefaultValue = 2.0, MinValue = 1.0,
            Group = "Risk Management")]
        public double RewardRiskRatio { get; set; }

        [Parameter("Max Open Positions", DefaultValue = 3, MinValue = 1, MaxValue = 10,
            Group = "Risk Management")]
        public int MaxOpenPositions { get; set; }

        [Parameter("SL Buffer (pips)", DefaultValue = 3.0, MinValue = 0.0,
            Group = "Risk Management")]
        public double SlBufferPips { get; set; }

        [Parameter("Min SL Distance (pips)", DefaultValue = 3.0, MinValue = 0.1,
            Group = "Risk Management")]
        public double MinSlPips { get; set; }

        [Parameter("Max SL Distance (pips)", DefaultValue = 10000.0, MinValue = 10.0,
            Group = "Risk Management")]
        public double MaxSlPips { get; set; }

        // =====================================================================
        // Constants
        // =====================================================================

        private const string BotLabel  = "ICT01_LIQ";
        private const int    AtrLength = 55;

        // =====================================================================
        // ICT Setup 01 embedded state
        // =====================================================================

        // ATR (Wilder RMA, length 55)
        private double _atrSumAccum;
        private double _atrRma = double.NaN;

        // Active FVG zone levels carried forward each bar
        private double _bullishDistal;
        private double _bullishProximal;
        private double _bearishDistal;
        private double _bearishProximal;

        // Geometry of the most-recently-detected FVG
        private double _bullishFvgDistal;
        private double _bullishFvgProximal;
        private int    _bullishFvgPoint;
        private double _bullishPremium, _bullishDiscount, _bullishEquilibrium;

        private double _bearishFvgDistal;
        private double _bearishFvgProximal;
        private int    _bearishFvgPoint;
        private double _bearishPremium, _bearishDiscount, _bearishEquilibrium;

        // Zone state
        private bool   _isBullishFvgValid = true;
        private bool   _isBearishFvgValid = true;
        private int    _prevBullishFvgPoint;
        private int    _prevBearishFvgPoint;
        private double _lowTracker;
        private double _highTracker;
        private int    _longSignalCount;
        private int    _shortSignalCount;

        // Per-bar signal flags (set by UpdateIctSignal, read by OnBar)
        private bool   _isLongSignal;
        private bool   _isShortSignal;

        // Signal bar's price at the time the signal fires (used for SL fallback)
        private double _longSignalBarLow;
        private double _shortSignalBarHigh;

        // =====================================================================
        // BSL / SSL embedded state
        // =====================================================================

        private sealed class LiquidityLevel
        {
            public double Price;
            public int    PivotIndex;
        }

        private readonly LinkedList<LiquidityLevel> _bslPool = new LinkedList<LiquidityLevel>();
        private readonly LinkedList<LiquidityLevel> _sslPool = new LinkedList<LiquidityLevel>();
        private const int MaxLiquidityLevels = 10;

        private double CurrentBslPrice => _bslPool.First != null ? _bslPool.First.Value.Price : double.NaN;
        private double CurrentSslPrice => _sslPool.First != null ? _sslPool.First.Value.Price : double.NaN;

        // ── Duplicate-entry guards ───────────────────────────────────────────
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

            Print("{0} started — FVG mult={1}, Pivot L/R={2}/{3}, Risk={4}%, RR={5}, MaxPos={6}",
                BotLabel, FvgDetectorMultiplier, PivotLeft, PivotRight,
                RiskPercent, RewardRiskRatio, MaxOpenPositions);
        }

        protected override void OnStop()
        {
            Print("{0} stopped.", BotLabel);
        }

        // =====================================================================
        // Bar event
        // =====================================================================

        protected override void OnBar()
        {
            // OnBar fires when a NEW bar opens.
            // Bars.Count - 2 is the bar that just CLOSED (the signal bar).
            int signalBar = Bars.Count - 2;
            if (signalBar < 2)
                return;

            // Advance both embedded indicator states for the closed bar.
            ProcessBar(signalBar);

            bool hasLong  = _isLongSignal;
            bool hasShort = _isShortSignal;

            if (!hasLong && !hasShort)
                return;

            int openCount = Positions.FindAll(BotLabel, SymbolName).Length;
            if (openCount >= MaxOpenPositions)
            {
                Print("Bar {0}: max positions ({1}) reached, signal skipped.",
                    signalBar, MaxOpenPositions);
                return;
            }

            if (hasLong && _lastLongSignalBar != signalBar)
            {
                _lastLongSignalBar = signalBar;
                TryEnterLong(signalBar);
            }

            // Re-check capacity before acting on a coincident short signal.
            openCount = Positions.FindAll(BotLabel, SymbolName).Length;
            if (openCount >= MaxOpenPositions)
                return;

            if (hasShort && _lastShortSignalBar != signalBar)
            {
                _lastShortSignalBar = signalBar;
                TryEnterShort(signalBar);
            }
        }

        // =====================================================================
        // Per-bar state update  (mirrors ICT_01.Calculate + BSL_SSL.Calculate)
        // =====================================================================

        private void ProcessBar(int index)
        {
            UpdateAtr(index);
            UpdateIctSignal(index);
            UpdateBslSsl(index);
        }

        // ─────────────────────────────────────────────────────────────────────
        // ATR  (Wilder's smoothed moving average, length 55)
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
                _atrRma = _atrSumAccum / AtrLength;
            }
            else
            {
                _atrRma = ((_atrRma * (AtrLength - 1)) + tr) / AtrLength;
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // ICT Setup 01 signal logic  (mirrors ICT_01.Calculate exactly)
        // ─────────────────────────────────────────────────────────────────────

        private void UpdateIctSignal(int index)
        {
            _isLongSignal  = false;
            _isShortSignal = false;

            if (index == 0)
                return;

            double high  = Bars.HighPrices[index];
            double low   = Bars.LowPrices[index];
            double close = Bars.ClosePrices[index];

            // Save previous bar's zone levels before any update this bar
            double prevBullishDistal   = _bullishDistal;
            double prevBullishProximal = _bullishProximal;
            double prevBearishDistal   = _bearishDistal;
            double prevBearishProximal = _bearishProximal;

            // ── FVG detection ─────────────────────────────────────────────────
            bool isBullishFvg = false;
            bool isBearishFvg = false;

            if (index >= 2)
            {
                double high2 = Bars.HighPrices[index - 2];
                double low2  = Bars.LowPrices[index - 2];
                double high1 = Bars.HighPrices[index - 1];
                double low1  = Bars.LowPrices[index - 1];
                double atr   = double.IsNaN(_atrRma) ? 0.0 : _atrRma;

                if ((high - low2) > FvgDetectorMultiplier * atr)
                {
                    if (low > high2 && low2 < low1 && high1 < high &&
                        (high + low2) / 2.0 >= high2)
                    {
                        _bullishFvgDistal   = high2;
                        _bullishFvgProximal = low;
                        _bullishFvgPoint    = index;
                        _bullishDiscount    = low2;
                        _bullishPremium     = high;
                        _bullishEquilibrium = (high + low2) / 2.0;
                        isBullishFvg        = true;
                    }
                }

                if ((high2 - low) > FvgDetectorMultiplier * atr)
                {
                    if (low2 > high && high2 > high1 && low1 > low &&
                        (low + high2) / 2.0 <= low2)
                    {
                        _bearishFvgDistal   = low2;
                        _bearishFvgProximal = high;
                        _bearishFvgPoint    = index;
                        _bearishDiscount    = low;
                        _bearishPremium     = high2;
                        _bearishEquilibrium = (low + high2) / 2.0;
                        isBearishFvg        = true;
                    }
                }
            }

            // Update active zone levels when a new FVG is found
            if (isBullishFvg)
            {
                _bullishDistal   = _bullishFvgDistal;
                _bullishProximal = UseDiscountAndPremium
                    ? (_bullishEquilibrium >= _bullishFvgProximal
                           ? _bullishFvgProximal
                           : _bullishEquilibrium)
                    : _bullishFvgProximal;
            }

            if (isBearishFvg)
            {
                _bearishDistal   = _bearishFvgDistal;
                _bearishProximal = UseDiscountAndPremium
                    ? (_bearishEquilibrium <= _bearishFvgProximal
                           ? _bearishFvgProximal
                           : _bearishEquilibrium)
                    : _bearishFvgProximal;
            }

            // Zone validity update — uses the PREVIOUS bar's distal/proximal
            double body1 = Bars.ClosePrices[index - 1] - Bars.OpenPrices[index - 1];

            if (_isBullishFvgValid)
                _isBullishFvgValid = CheckZoneValidity(
                    index, body1, true,
                    _bullishFvgPoint,
                    prevBullishDistal, prevBullishProximal,
                    _longSignalCount,
                    ref _bullishProximal);

            if (_isBearishFvgValid)
                _isBearishFvgValid = CheckZoneValidity(
                    index, body1, false,
                    _bearishFvgPoint,
                    prevBearishDistal, prevBearishProximal,
                    _shortSignalCount,
                    ref _bearishProximal);

            // New FVG detected → reset zone state
            if (_prevBullishFvgPoint != _bullishFvgPoint)
            {
                _isBullishFvgValid = true;
                _lowTracker        = 0.0;
                _longSignalCount   = 0;
            }

            if (_prevBearishFvgPoint != _bearishFvgPoint)
            {
                _isBearishFvgValid = true;
                _highTracker       = 0.0;
                _shortSignalCount  = 0;
            }

            _prevBullishFvgPoint = _bullishFvgPoint;
            _prevBearishFvgPoint = _bearishFvgPoint;

            // ── Long signal detection ─────────────────────────────────────────
            if (_isBullishFvgValid)
            {
                if (_lowTracker == 0.0 && low < _bullishProximal)
                    _lowTracker = low;

                if (low < _lowTracker && _lowTracker > 0.0)
                {
                    _lowTracker = low;
                    if (close >= _bullishProximal)
                    {
                        _longSignalCount++;
                        _isLongSignal = SignalAfterHunts
                            ? (_longSignalCount == RequiredHunts)
                            : true;
                        if (_isLongSignal)
                            _longSignalBarLow = low;
                    }
                }
            }
            else
            {
                _lowTracker      = 0.0;
                _longSignalCount = 0;
            }

            // ── Short signal detection ────────────────────────────────────────
            if (_isBearishFvgValid)
            {
                if (_highTracker == 0.0 && high > _bearishProximal)
                    _highTracker = high;

                if (high > _highTracker && _highTracker > 0.0)
                {
                    _highTracker = high;
                    if (close <= _bearishProximal)
                    {
                        _shortSignalCount++;
                        _isShortSignal = SignalAfterHunts
                            ? (_shortSignalCount == RequiredHunts)
                            : true;
                        if (_isShortSignal)
                            _shortSignalBarHigh = high;
                    }
                }
            }
            else
            {
                _highTracker       = 0.0;
                _shortSignalCount  = 0;
            }
        }

        private bool CheckZoneValidity(
            int index, double body1, bool isBull,
            int zonePoint,
            double prevDistal, double prevProximal,
            int signalCount,
            ref double proximal)
        {
            bool useOpen    = isBull ? body1 > 0 : body1 <= 0;
            double selected = useOpen
                ? Bars.OpenPrices[index - 1]
                : Bars.ClosePrices[index - 1];

            double sweepPrice = isBull
                ? (SignalMethod == "Sweeps" ? selected : Bars.LowPrices[index - 1])
                : (SignalMethod == "Sweeps" ? selected : Bars.HighPrices[index - 1]);

            bool swept   = isBull ? sweepPrice < prevDistal : sweepPrice > prevDistal;
            bool expired = index > zonePoint + FvgValidityPeriod;
            bool limited = !SignalAfterHunts && signalCount > SignalsAllowedPerZone - 1;

            if (swept || expired || limited)
                return false;

            bool movedInside = isBull
                ? selected < prevProximal && selected > prevDistal
                : selected > prevProximal && selected < prevDistal;

            if (movedInside)
                proximal = selected;

            return true;
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
            // SSL is mitigated when the bar's low trades at or below the level
            var node = _sslPool.First;
            while (node != null)
            {
                var next = node.Next;
                if (Bars.LowPrices[index] <= node.Value.Price)
                    _sslPool.Remove(node);
                node = next;
            }

            // BSL is mitigated when the bar's high trades at or above the level
            node = _bslPool.First;
            while (node != null)
            {
                var next = node.Next;
                if (Bars.HighPrices[index] >= node.Value.Price)
                    _bslPool.Remove(node);
                node = next;
            }
        }

        // =====================================================================
        // Trade execution
        // =====================================================================

        private void TryEnterLong(int signalBar)
        {
            // ── Determine SL reference ────────────────────────────────────────
            // Primary: most-recent active SSL (pivot low) from the embedded pool.
            // Fallback: signal bar's sweep low — the ICT long signal fires because
            //   price swept below the FVG proximal zone; that same sweep clears all
            //   SSL pool entries (MitigateLevels removes every level at or above
            //   the bar's low), leaving the pool empty.  The natural ICT SL in that
            //   case is below the sweep low itself, which _longSignalBarLow holds.
            double sslLevel = CurrentSslPrice;
            string slSource;

            if (!double.IsNaN(sslLevel) && sslLevel > 0)
            {
                slSource = "SSL active pivot low";
            }
            else
            {
                sslLevel = _longSignalBarLow;
                slSource = "sweep low fallback (SSL pool cleared)";
            }

            double entry   = Symbol.Ask;
            double slPrice = sslLevel - SlBufferPips * Symbol.PipSize;

            if (slPrice >= entry)
            {
                Print("Bar {0}: LONG skip — SL {1:F5} ({2}) >= entry {3:F5}.",
                    signalBar, slPrice, slSource, entry);
                return;
            }

            double slPips = (entry - slPrice) / Symbol.PipSize;

            if (!ValidateSlPips(signalBar, "LONG", slPips))
                return;

            double tpPips     = slPips * RewardRiskRatio;
            double riskAmount = Account.Equity * (RiskPercent / 100.0);
            double volume     = GetRiskVolume(riskAmount, slPips);

            if (volume <= 0)
            {
                Print("Bar {0}: LONG skip — volume rounds to 0.", signalBar);
                return;
            }

            Print("Bar {0}: LONG  | Ask={1:F5} | SLref={2:F5} ({3}) | SL={4:F1}p | TP={5:F1}p | Vol={6}",
                signalBar, entry, sslLevel, slSource, slPips, tpPips, volume);

            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, BotLabel, slPips, tpPips);
        }

        private void TryEnterShort(int signalBar)
        {
            // ── Determine SL reference ────────────────────────────────────────
            // Primary: most-recent active BSL (pivot high) from the embedded pool.
            // Fallback: signal bar's sweep high — the ICT short signal fires because
            //   price swept above the FVG proximal zone; that same sweep clears all
            //   BSL pool entries, leaving the pool empty.  The natural ICT SL is
            //   above the sweep high, which _shortSignalBarHigh holds.
            double bslLevel = CurrentBslPrice;
            string slSource;

            if (!double.IsNaN(bslLevel) && bslLevel > 0)
            {
                slSource = "BSL active pivot high";
            }
            else
            {
                bslLevel = _shortSignalBarHigh;
                slSource = "sweep high fallback (BSL pool cleared)";
            }

            double entry   = Symbol.Bid;
            double slPrice = bslLevel + SlBufferPips * Symbol.PipSize;

            if (slPrice <= entry)
            {
                Print("Bar {0}: SHORT skip — SL {1:F5} ({2}) <= entry {3:F5}.",
                    signalBar, slPrice, slSource, entry);
                return;
            }

            double slPips = (slPrice - entry) / Symbol.PipSize;

            if (!ValidateSlPips(signalBar, "SHORT", slPips))
                return;

            double tpPips     = slPips * RewardRiskRatio;
            double riskAmount = Account.Equity * (RiskPercent / 100.0);
            double volume     = GetRiskVolume(riskAmount, slPips);

            if (volume <= 0)
            {
                Print("Bar {0}: SHORT skip — volume rounds to 0.", signalBar);
                return;
            }

            Print("Bar {0}: SHORT | Bid={1:F5} | SLref={2:F5} ({3}) | SL={4:F1}p | TP={5:F1}p | Vol={6}",
                signalBar, entry, bslLevel, slSource, slPips, tpPips, volume);

            ExecuteMarketOrder(TradeType.Sell, SymbolName, volume, BotLabel, slPips, tpPips);
        }

        // =====================================================================
        // Helpers
        // =====================================================================

        private bool ValidateSlPips(int signalBar, string dir, double slPips)
        {
            if (slPips < MinSlPips)
            {
                Print("Bar {0}: {1} skip — SL {2:F1}p < min {3:F1}p.",
                    signalBar, dir, slPips, MinSlPips);
                return false;
            }
            if (slPips > MaxSlPips)
            {
                Print("Bar {0}: {1} skip — SL {2:F1}p > max {3:F1}p.",
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
