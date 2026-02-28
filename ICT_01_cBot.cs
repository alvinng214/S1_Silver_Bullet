// =============================================================================
// ICT Setup 01 cBot  –  self-contained, no external indicator references
// =============================================================================
// Signal logic  : mirrors ICT_01 (ICT Setup 01 TFlab_ct.cs)
// SL anchor     : mirrors BSL and SSL (BSL and SSL.cs)
//
// Long  entry : market order at next bar open after long signal bar closes
//               SL = nearest unmitigated SSL (pivot low) below entry
// Short entry : market order at next bar open after short signal bar closes
//               SL = nearest unmitigated BSL (pivot high) above entry
// TP          : 2 × SL distance  (1 : 2 risk-to-reward)
// Risk        : 1 % of current account equity per trade
// Capacity    : max 3 simultaneous open positions
// =============================================================================

using System;
using System.Collections.Generic;
using cAlgo.API;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class ICT_01_cBot : Robot
    {
        // ── ICT Setup 01 parameters ──────────────────────────────────────────

        [Parameter("FVG Detector Multiplier", DefaultValue = 1.0, MinValue = 1.0, Group = "ICT Setup 01")]
        public double FvgDetectorMultiplier { get; set; }

        [Parameter("FVG Validity Period (bars)", DefaultValue = 15, MinValue = 2, Group = "ICT Setup 01")]
        public int FvgValidityPeriod { get; set; }

        [Parameter("Use Discount & Premium Zone", DefaultValue = false, Group = "ICT Setup 01")]
        public bool UseDiscountAndPremium { get; set; }

        [Parameter("Signal Method (Hunt / Sweeps)", DefaultValue = "Hunt", Group = "ICT Setup 01")]
        public string SignalMethod { get; set; }

        [Parameter("Signals Allowed Per Zone", DefaultValue = 3, MinValue = 1, Group = "ICT Setup 01")]
        public int SignalsAllowedPerZone { get; set; }

        [Parameter("Signal After Hunts", DefaultValue = false, Group = "ICT Setup 01")]
        public bool SignalAfterHunts { get; set; }

        [Parameter("Required Hunts Count", DefaultValue = 2, MinValue = 1, Group = "ICT Setup 01")]
        public int RequiredHunts { get; set; }

        // ── BSL & SSL parameters ─────────────────────────────────────────────

        [Parameter("Pivot Left", DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotLeft { get; set; }

        [Parameter("Pivot Right", DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotRight { get; set; }

        // ── Risk management ──────────────────────────────────────────────────

        [Parameter("Risk Per Trade (%)", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 10.0, Group = "Risk Management")]
        public double RiskPercent { get; set; }

        [Parameter("Max Simultaneous Positions", DefaultValue = 3, MinValue = 1, MaxValue = 10, Group = "Risk Management")]
        public int MaxOpenPositions { get; set; }

        [Parameter("Min SL Distance (pips)", DefaultValue = 3.0, MinValue = 0.1, Group = "Risk Management")]
        public double MinSlPips { get; set; }

        [Parameter("Max SL Distance (pips)", DefaultValue = 500.0, MinValue = 10.0, Group = "Risk Management")]
        public double MaxSlPips { get; set; }

        // ── Constants ────────────────────────────────────────────────────────

        private const string BotLabel  = "ICT01_cBot";
        private const double RrRatio   = 2.0;
        private const int    AtrLength = 55;

        // =====================================================================
        // ICT Setup 01 embedded state
        // =====================================================================

        // ATR (Wilder RMA, length 55)
        private double _atrSumAccum;           // accumulates TR during SMA seed phase
        private double _atrRma = double.NaN;   // running ATR value

        // Active FVG zone levels (carried forward each bar)
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
        private int    _prevBullishFvgPoint;     // FVG point stored at end of previous bar
        private int    _prevBearishFvgPoint;
        private double _lowTracker;
        private double _highTracker;
        private int    _longSignalCount;
        private int    _shortSignalCount;

        // Per-bar signal flags (set by UpdateIctSignal, read by OnBar)
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

        // Returns the nearest SSL price that is strictly below 'price', searching all
        // levels including swept ones. The swept SSL is the natural SL anchor for a long.
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

        // Returns the nearest BSL price that is strictly above 'price', searching all
        // levels including swept ones. The swept BSL is the natural SL anchor for a short.
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

            Print("ICT Setup 01 cBot started. MaxPositions={0}, Risk={1}%",
                  MaxOpenPositions, RiskPercent);
        }

        protected override void OnStop()
        {
            Print("ICT Setup 01 cBot stopped.");
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
                // _atrRma stays NaN until the seed window is complete
                return;
            }

            double prevClose = Bars.ClosePrices[index - 1];
            tr = Math.Max(high - low,
                 Math.Max(Math.Abs(high - prevClose),
                          Math.Abs(low  - prevClose)));

            if (index < AtrLength - 1)
            {
                _atrSumAccum += tr;
                // _atrRma stays NaN
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

            // ── Save previous bar's zone levels before any update this bar ────
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

            // ── Update active zone levels when a new FVG is found ─────────────
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

            // ── Zone validity update (mirrors UpdateZoneValidity) ─────────────
            // Uses the PREVIOUS bar's distal/proximal, exactly as the indicator does.
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

            // ── New FVG detected → reset zone state (mirrors indicator step 4) ─
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

            // Store FVG points for the next bar's change-detection
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
                    }
                    // else: _isLongSignal stays false
                }
                // else: _isLongSignal stays false
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
                    }
                }
            }
            else
            {
                _highTracker       = 0.0;
                _shortSignalCount  = 0;
            }
        }

        /// <summary>
        /// Mirrors ICT_01.UpdateZoneValidity.
        /// Returns false if the zone has been swept, expired, or hit its signal limit.
        /// May tighten <paramref name="proximal"/> when price moves inside the zone.
        /// </summary>
        private bool CheckZoneValidity(
            int index, double body1, bool isBull,
            int zonePoint,
            double prevDistal, double prevProximal,
            int signalCount,
            ref double proximal)
        {
            bool useOpen     = isBull ? body1 > 0 : body1 <= 0;
            double selected  = useOpen
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
            // Guard against exact duplicates (mirrors BSL_SSL.UnshiftPivot)
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

            // SSL is swept when the bar's low trades at or below the level.
            // Mark it instead of removing so it remains available as a SL reference
            // (the sweep low IS the natural stop anchor in ICT methodology).
            foreach (var ssl in _sslPool)
                if (!ssl.Mitigated && barLow <= ssl.Price)
                    ssl.Mitigated = true;

            // BSL is swept when the bar's high trades at or above the level.
            foreach (var bsl in _bslPool)
                if (!bsl.Mitigated && barHigh >= bsl.Price)
                    bsl.Mitigated = true;
        }

        // =====================================================================
        // Trade execution
        // =====================================================================

        private void TryEnterLong(int signalBar)
        {
            // Entry at current Ask  (≈ open of the new bar in OnBar context)
            double entry    = Symbol.Ask;

            // Search all SSL levels (including swept ones) for the nearest below entry.
            // The ICT signal fires because price swept through an SSL, so the swept level
            // IS the correct stop anchor. FindNearestSslBelow includes mitigated levels.
            double sslLevel = FindNearestSslBelow(entry);

            // Fallback: signal bar's own low is the sweep low – the natural ICT stop
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

            // Fallback: FVG distal level
            if (double.IsNaN(sslLevel) || sslLevel <= 0)
            {
                if (_bullishDistal > 0 && _bullishDistal < entry)
                {
                    sslLevel = _bullishDistal;
                    Print("Bar {0}: LONG using FVG distal {1:F5} as SL anchor.", signalBar, sslLevel);
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
            // Entry at current Bid  (≈ open of the new bar in OnBar context)
            double entry    = Symbol.Bid;

            // Search all BSL levels (including swept ones) for the nearest above entry.
            double bslLevel = FindNearestBslAbove(entry);

            // Fallback: signal bar's own high is the sweep high – the natural ICT stop
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

            // Fallback: FVG distal level
            if (double.IsNaN(bslLevel) || bslLevel <= 0)
            {
                if (_bearishDistal > 0 && _bearishDistal > entry)
                {
                    bslLevel = _bearishDistal;
                    Print("Bar {0}: SHORT using FVG distal {1:F5} as SL anchor.", signalBar, bslLevel);
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
