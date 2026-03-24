// =============================================================================
// ICT_01_cBot_Single
// =============================================================================
// Self-contained version of ICT_01_cBot.  No external indicator references.
// The logic of both ICT_01 (signal generation) and BSL_SSL (SL anchor)
// is embedded directly in this file.
//
// ICT_01 engine  — ported from ICT_01_LongShortSignals.cs
//   • All IndicatorDataSeries replaced with scalar carry-forward fields.
//   • Chart drawing and alert emission removed.
//   • ATR uses the same Wilder seed + smoothing as the original.
//
// BSL_SSL engine — ported from BSL and SSL.cs
//   • ExternalLiquidity.Line/Label/Hidden removed (no chart drawing).
//   • ApplyShowRules removed (display only).
//   • CurrentBSL/SSL stored in scalar fields _bslCurrentBsl/_bslCurrentSsl.
//
// Trade logic    — verbatim from ICT_01_cBot.cs
// =============================================================================

using System;
using System.Collections.Generic;
using cAlgo.API;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class ICT_01_cBot_Single : Robot
    {
        // ═════════════════════════════════════════════════════════════════════
        //  Parameters
        // ═════════════════════════════════════════════════════════════════════

        // ── ICT Setup 01 ─────────────────────────────────────────────────────

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

        // ── BSL & SSL ─────────────────────────────────────────────────────────

        [Parameter("Pivot Left", DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotLeft { get; set; }

        [Parameter("Pivot Right", DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotRight { get; set; }

        // ── Risk Management ───────────────────────────────────────────────────

        [Parameter("Risk Per Trade (%)", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 10.0, Group = "Risk Management")]
        public double RiskPercent { get; set; }

        [Parameter("Max Simultaneous Positions", DefaultValue = 3, MinValue = 1, MaxValue = 10, Group = "Risk Management")]
        public int MaxOpenPositions { get; set; }

        [Parameter("Min SL Distance (pips)", DefaultValue = 3.0, MinValue = 0.1, Group = "Risk Management")]
        public double MinSlPips { get; set; }

        [Parameter("Max SL Distance (pips)", DefaultValue = 500.0, MinValue = 10.0, Group = "Risk Management")]
        public double MaxSlPips { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  Constants
        // ═════════════════════════════════════════════════════════════════════

        private const string BotLabel     = "ICT01_cBot_Single";
        private const double RrRatio      = 2.0;
        private const int    IctAtrLength = 55;    // hardcoded in ICT_01 indicator
        private const int    MaxBslPivots = 10;    // MaxPivotsToKeep in BSL_SSL

        // ═════════════════════════════════════════════════════════════════════
        //  Inner types — BSL/SSL (chart-object fields removed)
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
        //  ICT_01 embedded fields
        //
        //  All IndicatorDataSeries replaced with scalar carry-forward fields.
        //  Design rule: at the START of every RunIct(index) call, each field
        //  holds the value that the original series stored at [index-1].
        //  At the END of RunIct(index) the fields hold the [index] values,
        //  ready for the next call.
        // ═════════════════════════════════════════════════════════════════════

        // ATR — Wilder's smoothing, period 55 (matches ICT_01.CalculateAtr)
        private double _ictTrueRange;
        private double _ictAtr    = double.NaN;
        private double _ictAtrSum = 0.0;          // running sum for seed SMA

        // FVG zone carry-forward levels (replace four IndicatorDataSeries)
        // Saved at start of each RunIct call so they can be passed as
        // prevDistal/prevProximal to IctUpdateZoneValidity even after a
        // potential overwrite by new FVG detection within the same bar.
        private double _ictBullDistal;
        private double _ictBullProximal;
        private double _ictBearDistal;
        private double _ictBearProximal;

        // FVG point saved at end of each bar (replaces _bullishFvgPointSeries[index-1])
        private int _ictBullFvgPointSaved;
        private int _ictBearFvgPointSaved;

        // FVG detection results for the current bar
        private double _ictBullFvgDistal;
        private double _ictBullFvgProximal;
        private int    _ictBullFvgPoint;
        private double _ictBullPremium;
        private double _ictBullDiscount;
        private double _ictBullEquilibrium;

        private double _ictBearFvgDistal;
        private double _ictBearFvgProximal;
        private int    _ictBearFvgPoint;
        private double _ictBearPremium;
        private double _ictBearDiscount;
        private double _ictBearEquilibrium;

        // Zone state
        private bool   _ictBullFvgValid = true;
        private bool   _ictBearFvgValid = true;
        private bool   _ictIsBullFvg;
        private bool   _ictIsBearFvg;
        private double _ictLowTracker;
        private double _ictHighTracker;
        private int    _ictLongSignalCount;
        private int    _ictShortSignalCount;

        // Signal outputs (read by OnBar after RunIct completes)
        private bool _ictIsLongSignal;
        private bool _ictIsShortSignal;

        // ═════════════════════════════════════════════════════════════════════
        //  BSL/SSL embedded fields
        // ═════════════════════════════════════════════════════════════════════

        private readonly LinkedList<BslPivot> _bslPivots        = new LinkedList<BslPivot>();
        private readonly LinkedList<BslPool>  _bslBuysidePools  = new LinkedList<BslPool>();
        private readonly LinkedList<BslPool>  _bslSellsidePools = new LinkedList<BslPool>();

        // Live BSL/SSL prices read by TryEnterLong/TryEnterShort
        private double _bslCurrentBsl = double.NaN;
        private double _bslCurrentSsl = double.NaN;

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
            Print("ICT_01_cBot_Single started. MaxPositions={0}, Risk={1}%",
                  MaxOpenPositions, RiskPercent);
        }

        protected override void OnStop()
        {
            Print("ICT_01_cBot_Single stopped.");
        }

        // ═════════════════════════════════════════════════════════════════════
        //  OnBar
        // ═════════════════════════════════════════════════════════════════════

        protected override void OnBar()
        {
            int signalBar = Bars.Count - 2;   // last closed bar

            // Incrementally process every bar not yet seen.
            // During backtesting OnBar fires sequentially so this loop normally
            // executes exactly once per call.  The loop also handles any gaps.
            for (int i = _lastProcessed + 1; i <= signalBar; i++)
            {
                RunBslSsl(i);    // update BSL/SSL levels first
                RunIct(i);       // then detect ICT signals
            }
            _lastProcessed = signalBar;

            if (signalBar < 2)
                return;

            bool isLong  = _ictIsLongSignal;
            bool isShort = _ictIsShortSignal;

            if (!isLong && !isShort)
                return;

            // ── Global capacity guard ─────────────────────────────────────────
            int openCount = Positions.FindAll(BotLabel, SymbolName).Length;
            if (openCount >= MaxOpenPositions)
            {
                Print("Bar {0}: max positions ({1}) reached.", signalBar, MaxOpenPositions);
                return;
            }

            // ── Long ──────────────────────────────────────────────────────────
            if (isLong && _lastLongSignalBar != signalBar)
            {
                _lastLongSignalBar = signalBar;
                TryEnterLong(signalBar);
            }

            // Re-check capacity in case long just filled it.
            openCount = Positions.FindAll(BotLabel, SymbolName).Length;
            if (openCount >= MaxOpenPositions) return;

            // ── Short ─────────────────────────────────────────────────────────
            if (isShort && _lastShortSignalBar != signalBar)
            {
                _lastShortSignalBar = signalBar;
                TryEnterShort(signalBar);
            }
        }

        // ═════════════════════════════════════════════════════════════════════
        //  Trade entry helpers — verbatim from ICT_01_cBot
        //  BSL/SSL levels now come from _bslCurrentBsl/_bslCurrentSsl instead
        //  of indicator output series.
        // ═════════════════════════════════════════════════════════════════════

        private void TryEnterLong(int signalBar)
        {
            double entry    = Symbol.Ask;
            double sslLevel = _bslCurrentSsl;

            if (double.IsNaN(sslLevel) || sslLevel <= 0)
            {
                Print("Bar {0}: LONG skipped – SSL unavailable (NaN/zero).", signalBar);
                return;
            }
            if (sslLevel >= entry)
            {
                Print("Bar {0}: LONG skipped – SSL {1:F5} not below entry {2:F5}.",
                      signalBar, sslLevel, entry);
                return;
            }

            double slPips = (entry - sslLevel) / Symbol.PipSize;
            if (!ValidateSlPips(signalBar, "LONG", slPips)) return;

            double tpPips    = slPips * RrRatio;
            double riskAmount = Account.Equity * (RiskPercent / 100.0);
            double volume    = GetRiskVolume(riskAmount, slPips);
            if (volume <= 0)
            { Print("Bar {0}: LONG skipped – volume is 0.", signalBar); return; }

            Print("Bar {0}: LONG  | Entry={1:F5} | SSL SL={2:F5} ({3:F1} pips) | TP={4:F1} pips | Vol={5}",
                  signalBar, entry, sslLevel, slPips, tpPips, volume);
            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, BotLabel, slPips, tpPips);
        }

        private void TryEnterShort(int signalBar)
        {
            double entry    = Symbol.Bid;
            double bslLevel = _bslCurrentBsl;

            if (double.IsNaN(bslLevel) || bslLevel <= 0)
            {
                Print("Bar {0}: SHORT skipped – BSL unavailable (NaN/zero).", signalBar);
                return;
            }
            if (bslLevel <= entry)
            {
                Print("Bar {0}: SHORT skipped – BSL {1:F5} not above entry {2:F5}.",
                      signalBar, bslLevel, entry);
                return;
            }

            double slPips = (bslLevel - entry) / Symbol.PipSize;
            if (!ValidateSlPips(signalBar, "SHORT", slPips)) return;

            double tpPips    = slPips * RrRatio;
            double riskAmount = Account.Equity * (RiskPercent / 100.0);
            double volume    = GetRiskVolume(riskAmount, slPips);
            if (volume <= 0)
            { Print("Bar {0}: SHORT skipped – volume is 0.", signalBar); return; }

            Print("Bar {0}: SHORT | Entry={1:F5} | BSL SL={2:F5} ({3:F1} pips) | TP={4:F1} pips | Vol={5}",
                  signalBar, entry, bslLevel, slPips, tpPips, volume);
            ExecuteMarketOrder(TradeType.Sell, SymbolName, volume, BotLabel, slPips, tpPips);
        }

        private bool ValidateSlPips(int signalBar, string direction, double slPips)
        {
            if (slPips < MinSlPips)
            {
                Print("Bar {0}: {1} skipped – SL {2:F1} pips < minimum {3:F1} pips.",
                      signalBar, direction, slPips, MinSlPips);
                return false;
            }
            if (slPips > MaxSlPips)
            {
                Print("Bar {0}: {1} skipped – SL {2:F1} pips > maximum {3:F1} pips.",
                      signalBar, direction, slPips, MaxSlPips);
                return false;
            }
            return true;
        }

        private double GetRiskVolume(double riskAmount, double slPips)
        {
            if (slPips <= 0) return 0;
            double raw    = Symbol.VolumeForFixedRisk(riskAmount, slPips);
            double volume = Symbol.NormalizeVolumeInUnits(raw, RoundingMode.Down);
            if (volume < Symbol.VolumeInUnitsMin) return 0;
            if (volume > Symbol.VolumeInUnitsMax) volume = Symbol.VolumeInUnitsMax;
            return volume;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  ICT_01 embedded engine
        //  Ported from ICT_01_LongShortSignals.cs › Calculate().
        //  Drawing, alerts, and display parameters removed.
        //  IndicatorDataSeries replaced with carry-forward scalar fields.
        // ═════════════════════════════════════════════════════════════════════

        private void RunIct(int index)
        {
            IctCalculateAtr(index);

            // ── Bar 0: initialise all carry-forward fields ─────────────────────
            if (index == 0)
            {
                _ictBullDistal        = 0.0;
                _ictBullProximal      = 0.0;
                _ictBearDistal        = 0.0;
                _ictBearProximal      = 0.0;
                _ictBullFvgPointSaved = 0;
                _ictBearFvgPointSaved = 0;
                _ictIsLongSignal      = false;
                _ictIsShortSignal     = false;
                return;
            }

            // ── Save [index-1] values before any potential overwrite ───────────
            // These correspond to _bullishDistalLevel[index-1] etc. in the
            // original indicator and are passed to IctUpdateZoneValidity below.
            var prevBullDistal   = _ictBullDistal;
            var prevBullProximal = _ictBullProximal;
            var prevBearDistal   = _ictBearDistal;
            var prevBearProximal = _ictBearProximal;

            var high  = Bars.HighPrices[index];
            var low   = Bars.LowPrices[index];
            var close = Bars.ClosePrices[index];

            _ictIsBullFvg = false;
            _ictIsBearFvg = false;

            // ── FVG detection (mirrors ICT_01 inner block at index >= 2) ───────
            if (index >= 2)
            {
                var high2    = Bars.HighPrices[index - 2];
                var low2     = Bars.LowPrices[index - 2];
                var high1    = Bars.HighPrices[index - 1];
                var low1     = Bars.LowPrices[index - 1];
                var atrValue = double.IsNaN(_ictAtr) ? 0.0 : _ictAtr;

                // Bullish FVG
                if ((high - low2) > (FvgDetectorMultiplier * atrValue))
                {
                    if (low > high2 && low2 < low1 && high1 < high && (high + low2) / 2.0 >= high2)
                    {
                        _ictBullFvgDistal   = high2;
                        _ictBullFvgProximal = low;
                        _ictBullFvgPoint    = index;
                        _ictBullDiscount    = low2;
                        _ictBullPremium     = high;
                        _ictBullEquilibrium = (high + low2) / 2.0;
                        _ictIsBullFvg       = true;
                    }
                }

                // Bearish FVG
                if ((high2 - low) > (FvgDetectorMultiplier * atrValue))
                {
                    if (low2 > high && high2 > high1 && low1 > low && (low + high2) / 2.0 <= low2)
                    {
                        _ictBearFvgDistal   = low2;
                        _ictBearFvgProximal = high;
                        _ictBearFvgPoint    = index;
                        _ictBearDiscount    = low;
                        _ictBearPremium     = high2;
                        _ictBearEquilibrium = (low + high2) / 2.0;
                        _ictIsBearFvg       = true;
                    }
                }
            }

            // ── Update zone levels (carry-forward fields for distal/proximal) ──
            if (UseDiscountAndPremium)
            {
                if (_ictIsBullFvg)
                {
                    _ictBullDistal   = _ictBullFvgDistal;
                    _ictBullProximal = _ictBullEquilibrium >= _ictBullFvgProximal
                        ? _ictBullFvgProximal : _ictBullEquilibrium;
                }
                if (_ictIsBearFvg)
                {
                    _ictBearDistal   = _ictBearFvgDistal;
                    _ictBearProximal = _ictBearEquilibrium <= _ictBearFvgProximal
                        ? _ictBearFvgProximal : _ictBearEquilibrium;
                }
            }
            else
            {
                if (_ictIsBullFvg)
                {
                    _ictBullDistal   = _ictBullFvgDistal;
                    _ictBullProximal = _ictBullFvgProximal;
                }
                if (_ictIsBearFvg)
                {
                    _ictBearDistal   = _ictBearFvgDistal;
                    _ictBearProximal = _ictBearFvgProximal;
                }
            }

            // ── Zone validity update ───────────────────────────────────────────
            // prevBull/BearDistal/Proximal = [index-1] values (saved above).
            // _ictBull/BearDistal/Proximal  = [index]   values (just updated).
            // This mirrors _bullishDistalLevel[index-1] vs [index] in ICT_01.
            var body1 = Bars.ClosePrices[index - 1] - Bars.OpenPrices[index - 1];

            if (_ictBullFvgValid)
            {
                var bullProx = _ictBullProximal;
                _ictBullFvgValid = IctUpdateZoneValidity(index, body1, true,
                    _ictBullFvgPoint, prevBullDistal, prevBullProximal,
                    _ictLongSignalCount, ref bullProx);
                _ictBullProximal = bullProx;
            }

            if (_ictBearFvgValid)
            {
                var bearProx = _ictBearProximal;
                _ictBearFvgValid = IctUpdateZoneValidity(index, body1, false,
                    _ictBearFvgPoint, prevBearDistal, prevBearProximal,
                    _ictShortSignalCount, ref bearProx);
                _ictBearProximal = bearProx;
            }

            // ── Reset on new FVG zone ──────────────────────────────────────────
            // _ict*FvgPointSaved = _bullishFvgPointSeries[index-1] equivalent.
            // Comparison fires when a new FVG zone was detected this bar.
            if (_ictBullFvgPointSaved != _ictBullFvgPoint)
            {
                _ictBullFvgValid    = true;
                _ictLowTracker      = 0.0;
                _ictLongSignalCount = 0;
                _ictIsLongSignal    = false;
            }

            if (_ictBearFvgPointSaved != _ictBearFvgPoint)
            {
                _ictBearFvgValid     = true;
                _ictHighTracker      = 0.0;
                _ictShortSignalCount = 0;
                _ictIsShortSignal    = false;
            }

            // ── Long signal detection ──────────────────────────────────────────
            if (_ictBullFvgValid)
            {
                if (_ictLowTracker == 0.0 && low < _ictBullProximal)
                    _ictLowTracker = low;

                if (low < _ictLowTracker && _ictLowTracker > 0.0)
                {
                    _ictLowTracker = low;
                    if (close >= _ictBullProximal)
                    {
                        _ictLongSignalCount++;
                        _ictIsLongSignal = SignalAfterHunts
                            ? _ictLongSignalCount == RequiredHunts
                            : true;
                    }
                    else
                    {
                        _ictIsLongSignal = false;
                    }
                }
                else
                {
                    _ictIsLongSignal = false;
                }
            }
            else
            {
                _ictLowTracker      = 0.0;
                _ictLongSignalCount = 0;
                _ictIsLongSignal    = false;
            }

            // ── Short signal detection ─────────────────────────────────────────
            if (_ictBearFvgValid)
            {
                if (_ictHighTracker == 0.0 && high > _ictBearProximal)
                    _ictHighTracker = high;

                if (high > _ictHighTracker && _ictHighTracker > 0.0)
                {
                    _ictHighTracker = high;
                    if (close <= _ictBearProximal)
                    {
                        _ictShortSignalCount++;
                        _ictIsShortSignal = SignalAfterHunts
                            ? _ictShortSignalCount == RequiredHunts
                            : true;
                    }
                    else
                    {
                        _ictIsShortSignal = false;
                    }
                }
                else
                {
                    _ictIsShortSignal = false;
                }
            }
            else
            {
                _ictHighTracker      = 0.0;
                _ictShortSignalCount = 0;
                _ictIsShortSignal    = false;
            }

            // ── Save FVG point for next bar's new-zone check ───────────────────
            _ictBullFvgPointSaved = _ictBullFvgPoint;
            _ictBearFvgPointSaved = _ictBearFvgPoint;
        }

        // Wilder's ATR with running-sum seed — identical to ICT_01.CalculateAtr
        // except _ictAtrSum accumulates all TRs so the seed SMA at bar
        // IctAtrLength-1 needs no historical array.
        private void IctCalculateAtr(int index)
        {
            var high = Bars.HighPrices[index];
            var low  = Bars.LowPrices[index];

            if (index == 0)
            {
                _ictTrueRange = high - low;
                _ictAtrSum   += _ictTrueRange;
                _ictAtr       = double.NaN;
                return;
            }

            var prevClose = Bars.ClosePrices[index - 1];
            _ictTrueRange = Math.Max(high - low,
                            Math.Max(Math.Abs(high - prevClose),
                                     Math.Abs(low  - prevClose)));
            _ictAtrSum += _ictTrueRange;

            if (index < IctAtrLength - 1)
            {
                _ictAtr = double.NaN;
            }
            else if (index == IctAtrLength - 1)
            {
                // Seed: SMA of TR[0 .. IctAtrLength-1], all in _ictAtrSum
                _ictAtr = _ictAtrSum / IctAtrLength;
            }
            else
            {
                // Wilder's smoothing — matches ((_prevAtr * (N-1)) + TR) / N
                _ictAtr = ((_ictAtr * (IctAtrLength - 1)) + _ictTrueRange) / IctAtrLength;
            }
        }

        // Verbatim from ICT_01.UpdateZoneValidity — no changes needed
        private bool IctUpdateZoneValidity(
            int index, double body1, bool isBull, int zonePoint,
            double prevDistal, double prevProximal, int signalCount,
            ref double updatedProximal)
        {
            var useOpen       = isBull ? body1 > 0 : body1 <= 0;
            var selectedPrice = useOpen
                ? Bars.OpenPrices[index - 1]
                : Bars.ClosePrices[index - 1];
            var sweepPrice = isBull
                ? (SignalMethod == "Sweeps" ? selectedPrice : Bars.LowPrices[index - 1])
                : (SignalMethod == "Sweeps" ? selectedPrice : Bars.HighPrices[index - 1]);

            var sweepCheck          = isBull ? sweepPrice < prevDistal : sweepPrice > prevDistal;
            var expired             = index > zonePoint + FvgValidityPeriod;
            var signalLimitExceeded = !SignalAfterHunts && signalCount > SignalsAllowedPerZone - 1;

            if (sweepCheck || expired || signalLimitExceeded)
                return false;

            var movedInside = isBull
                ? selectedPrice < prevProximal && selectedPrice > prevDistal
                : selectedPrice > prevProximal && selectedPrice < prevDistal;

            if (movedInside)
                updatedProximal = selectedPrice;

            return true;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  BSL/SSL embedded engine
        //  Ported from BSL and SSL.cs › Calculate().
        //  Chart drawing (DrawTrendLine, DrawText, RemoveObject) removed.
        //  ApplyShowRules removed (display only; does not affect pool contents).
        //  CurrentBSL/SSL stored in _bslCurrentBsl/_bslCurrentSsl.
        // ═════════════════════════════════════════════════════════════════════

        private void RunBslSsl(int index)
        {
            BslDetectAndStoreConfirmedPivots(index);
            BslAddPoolFromNewPivot(index);
            BslClearMitigated(index);

            // Mirrors BSL_SSL.UpdateOutputLevels: most-recent unmitigated level
            _bslCurrentBsl = _bslBuysidePools.First  != null
                ? _bslBuysidePools.First.Value.Price  : double.NaN;
            _bslCurrentSsl = _bslSellsidePools.First != null
                ? _bslSellsidePools.First.Value.Price : double.NaN;
        }

        private void BslDetectAndStoreConfirmedPivots(int currentIndex)
        {
            int pivotIndex = currentIndex - PivotRight;
            if (pivotIndex <= 0) return;

            int leftStart = pivotIndex - PivotLeft;
            int rightEnd  = pivotIndex + PivotRight;   // equals currentIndex

            if (leftStart < 0 || rightEnd >= Bars.Count) return;

            double candidateHigh = Bars.HighPrices[pivotIndex];
            double candidateLow  = Bars.LowPrices[pivotIndex];

            if (BslIsPivotHigh(candidateHigh, leftStart, rightEnd))
                BslUnshiftPivot(new BslPivot { Price = candidateHigh, BarIndex = pivotIndex, Type =  1 });

            if (BslIsPivotLow(candidateLow, leftStart, rightEnd))
                BslUnshiftPivot(new BslPivot { Price = candidateLow,  BarIndex = pivotIndex, Type = -1 });
        }

        // Non-strict: ties allowed — mirrors ta.pivothigh/low behaviour in BSL_SSL
        private bool BslIsPivotHigh(double candidate, int start, int end)
        {
            double max = double.MinValue;
            for (int i = start; i <= end; i++)
                if (Bars.HighPrices[i] > max) max = Bars.HighPrices[i];
            return candidate == max;
        }

        private bool BslIsPivotLow(double candidate, int start, int end)
        {
            double min = double.MaxValue;
            for (int i = start; i <= end; i++)
                if (Bars.LowPrices[i] < min) min = Bars.LowPrices[i];
            return candidate == min;
        }

        // Prepend new pivot; skip exact duplicates; cap list at MaxBslPivots
        private void BslUnshiftPivot(BslPivot p)
        {
            if (_bslPivots.First != null)
            {
                var f = _bslPivots.First.Value;
                if (f.BarIndex == p.BarIndex && f.Type == p.Type &&
                    Math.Abs(f.Price - p.Price) < Symbol.PipSize * 0.1)
                    return;
            }
            _bslPivots.AddFirst(p);
            while (_bslPivots.Count > MaxBslPivots)
                _bslPivots.RemoveLast();
        }

        // Mirrors AddExternalLiquidityFromNewPivot + AddExternalLiquidity
        // (pool.AddFirst; visual hide-of-old-pools omitted)
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

        // Mirrors ClearMitigated in BSL_SSL
        //   Sellside (SSL): mitigated when Low ≤ its price
        //   Buyside  (BSL): mitigated when High ≥ its price
        private void BslClearMitigated(int index)
        {
            var node = _bslSellsidePools.First;
            while (node != null)
            {
                var next = node.Next;
                if (Bars.LowPrices[index] <= node.Value.Price)
                    _bslSellsidePools.Remove(node);
                node = next;
            }

            node = _bslBuysidePools.First;
            while (node != null)
            {
                var next = node.Next;
                if (Bars.HighPrices[index] >= node.Value.Price)
                    _bslBuysidePools.Remove(node);
                node = next;
            }
        }
    }
}
