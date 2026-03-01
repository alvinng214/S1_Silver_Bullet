// =============================================================================
// Order Block Detector – Signal-Reference cBot
// =============================================================================
// Signal logic  : embeds Order-Block Detector.cs (OB + FVG mitigation signals)
// SL            : ATR-based (identical to ICT_01_SignalRef_cBot.cs)
//
// Two signal types from the Order Block Detector are supported:
//   OB  – fires when price mitigates an Order Block zone then closes beyond it
//   FVG – fires when price mitigates a Fair-Value Gap then closes beyond it
// Both OB and FVG signals can be enabled/disabled independently.
//
// Entry  : market order at next bar open after the signal bar closes
// TP     : 2 × SL distance  (1 : 2 risk-to-reward)
// Risk   : 1 % of current account equity per trade (configurable)
// SL     : ATR × multiplier, clamped to [MinSlPips, MaxSlPips]
//
// Runs on chart timeframe only (no MTF). Heikin-Ashi mode mirrors the
// indicator's UseHeikinAshi parameter (affects close used for entry check).
// =============================================================================

using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Indicators;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class OB_Detector_SignalRef_cBot : Robot
    {
        // ── Order Block Detector parameters ──────────────────────────────────

        [Parameter("Enable OB Signals", DefaultValue = true, Group = "Order Block Detector")]
        public bool ShowOb { get; set; }

        [Parameter("Enable FVG Signals", DefaultValue = true, Group = "Order Block Detector")]
        public bool ShowFvg { get; set; }

        [Parameter("Min Dist OB (bars)", DefaultValue = 1, MinValue = 1, Group = "Order Block Detector")]
        public int MinDist { get; set; }

        [Parameter("Min Dist FVG (bars)", DefaultValue = 1, MinValue = 1, Group = "Order Block Detector")]
        public int MinDistFvg { get; set; }

        [Parameter("Use Heikin-Ashi", DefaultValue = false, Group = "Order Block Detector")]
        public bool UseHeikinAshi { get; set; }

        // ── Risk management ───────────────────────────────────────────────────

        [Parameter("Risk Per Trade (%)", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 10.0, Group = "Risk Management")]
        public double RiskPercent { get; set; }

        [Parameter("ATR Period (SL sizing)", DefaultValue = 14, MinValue = 5, Group = "Risk Management")]
        public int AtrPeriod { get; set; }

        [Parameter("ATR Multiplier for SL", DefaultValue = 2.0, MinValue = 0.5, MaxValue = 10.0, Group = "Risk Management")]
        public double AtrMultiplier { get; set; }

        [Parameter("Min SL Distance (pips)", DefaultValue = 5.0, MinValue = 1.0, Group = "Risk Management")]
        public double MinSlPips { get; set; }

        [Parameter("Max SL Distance (pips)", DefaultValue = 500.0, MinValue = 10.0, Group = "Risk Management")]
        public double MaxSlPips { get; set; }

        // ── Constants ─────────────────────────────────────────────────────────

        private const string BotLabel = "OBDet_Ref";
        private const double RrRatio  = 2.0;

        // ── ATR indicator (for SL sizing – identical to ICT_01_SignalRef_cBot) ─

        private AverageTrueRange _atr;

        // =====================================================================
        // Order Block Detector embedded state
        // =====================================================================

        private sealed class ObRecord
        {
            public double Max;
            public double Min;
            public bool   IsBull;
            public int    DetectionIndex;
        }

        private sealed class FvgRecord
        {
            public double Max;
            public double Min;
            public bool   IsBull;
            public int    DetectionIndex;
        }

        // Pending signal waiting for its entry condition to be met.
        // Entry is set true once consumed (prevents re-firing on same signal).
        private sealed class SignalState
        {
            public double Point  = double.NaN;
            public bool   IsBull;
            public bool   Entry;
        }

        private readonly List<ObRecord>  _obRecords  = new List<ObRecord>();
        private readonly List<FvgRecord> _fvgRecords = new List<FvgRecord>();

        private SignalState _obSignal  = new SignalState();
        private SignalState _fvgSignal = new SignalState();

        private int _lastDetectedObIndex  = -1;
        private int _lastDetectedFvgIndex = -1;

        // Heikin-Ashi values (built incrementally during warmup and live)
        private readonly List<double> _haOpen  = new List<double>();
        private readonly List<double> _haClose = new List<double>();

        // Per-bar signal flags written by ProcessBar, read by OnBar
        private bool _isLongSignal;
        private bool _isShortSignal;

        // ── Duplicate-entry guards ────────────────────────────────────────────

        private int _lastLongBar  = -1;
        private int _lastShortBar = -1;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        protected override void OnStart()
        {
            _atr = Indicators.AverageTrueRange(AtrPeriod, MovingAverageType.WilderSmoothing);

            // Warm up OB/FVG detector state over all complete historical bars.
            int warmupEnd = Bars.Count - 2;
            for (int i = 0; i <= warmupEnd; i++)
                ProcessBar(i);

            Print("OB Detector Signal-Reference Bot started. OB={0}, FVG={1}, Risk={2}%, ATR({3})×{4}, SL=[{5},{6}]p",
                  ShowOb, ShowFvg, RiskPercent, AtrPeriod, AtrMultiplier, MinSlPips, MaxSlPips);
        }

        protected override void OnStop()
        {
            Print("OB Detector Signal-Reference Bot stopped.");
        }

        // =====================================================================
        // Bar event – fires when a new bar opens
        // =====================================================================

        protected override void OnBar()
        {
            // OnBar fires when a NEW bar opens.
            // Bars.Count - 2 is the bar that just CLOSED (the signal bar).
            int signalBar = Bars.Count - 2;
            if (signalBar < 2)
                return;

            ProcessBar(signalBar);

            if (_isLongSignal && _lastLongBar != signalBar)
            {
                _lastLongBar = signalBar;
                OpenLong(signalBar);
            }

            if (_isShortSignal && _lastShortBar != signalBar)
            {
                _lastShortBar = signalBar;
                OpenShort(signalBar);
            }
        }

        // =====================================================================
        // Embedded Order Block Detector signal logic
        // Mirrors Order-Block Detector.cs: Calculate() → DetectOrderBlock()
        // → DetectFvg() → HandleMitigationOb() → HandleMitigationFvg()
        // → entry condition checks.
        // =====================================================================

        private void ProcessBar(int index)
        {
            _isLongSignal  = false;
            _isShortSignal = false;

            if (index < 2)
                return;

            // Build Heikin-Ashi values for this bar (needed when UseHeikinAshi = true)
            EnsureHeikinAshi(index);

            // ── Phase 1: detect new OB / FVG zones ───────────────────────────
            if (ShowOb)  DetectOb(index);
            if (ShowFvg) DetectFvg(index);

            // ── Phase 2: check mitigation of existing zones ───────────────────
            double barLow  = Bars.LowPrices[index];
            double barHigh = Bars.HighPrices[index];

            if (ShowOb)  HandleMitigationOb(index, barLow, barHigh);
            if (ShowFvg) HandleMitigationFvg(index, barLow, barHigh);

            // ── Phase 3: check entry conditions ──────────────────────────────
            // signalClose: HA close when UseHeikinAshi, otherwise regular close.
            // candleUp: always uses regular candle direction (mirrors indicator).
            double signalClose = UseHeikinAshi ? _haClose[index] : Bars.ClosePrices[index];
            bool   candleUp    = Bars.ClosePrices[index] > Bars.OpenPrices[index];

            // OB pending signal
            if (!double.IsNaN(_obSignal.Point) && !_obSignal.Entry)
            {
                if (_obSignal.IsBull && candleUp && signalClose > _obSignal.Point)
                {
                    _obSignal.Entry = true;
                    _isLongSignal   = true;
                }
                else if (!_obSignal.IsBull && !candleUp && signalClose < _obSignal.Point)
                {
                    _obSignal.Entry = true;
                    _isShortSignal  = true;
                }
            }

            // FVG pending signal
            if (!double.IsNaN(_fvgSignal.Point) && !_fvgSignal.Entry)
            {
                if (_fvgSignal.IsBull && candleUp && signalClose > _fvgSignal.Point)
                {
                    _fvgSignal.Entry = true;
                    _isLongSignal    = true;
                }
                else if (!_fvgSignal.IsBull && !candleUp && signalClose < _fvgSignal.Point)
                {
                    _fvgSignal.Entry = true;
                    _isShortSignal   = true;
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // OB detection  (mirrors DetectOrderBlock with sourceIndex = index)
        //
        // Bull OB: current bar bullish, prev bar bearish, High[i] > High[i-1]
        //          → zone = [Low[i-1], High[i-1]]
        // Bear OB: current bar bearish, prev bar bullish, Low[i] < Low[i-1]
        //          → zone = [Low[i-1], High[i-1]]
        // ─────────────────────────────────────────────────────────────────────

        private void DetectOb(int index)
        {
            if (index == _lastDetectedObIndex)
                return;

            bool candleUp     = Bars.ClosePrices[index]     > Bars.OpenPrices[index];
            bool prevCandleUp = Bars.ClosePrices[index - 1] > Bars.OpenPrices[index - 1];

            bool   detected = false;
            bool   isBull   = false;
            double max = 0, min = 0;

            if (candleUp && !prevCandleUp && Bars.HighPrices[index] > Bars.HighPrices[index - 1])
            {
                detected = true;
                isBull   = true;
                max      = Bars.HighPrices[index - 1];
                min      = Bars.LowPrices[index - 1];
            }
            else if (!candleUp && prevCandleUp && Bars.LowPrices[index] < Bars.LowPrices[index - 1])
            {
                detected = true;
                isBull   = false;
                max      = Bars.HighPrices[index - 1];
                min      = Bars.LowPrices[index - 1];
            }

            if (!detected)
                return;

            _obRecords.Insert(0, new ObRecord
            {
                Max = max, Min = min, IsBull = isBull, DetectionIndex = index
            });
            _lastDetectedObIndex = index;
        }

        // ─────────────────────────────────────────────────────────────────────
        // FVG detection  (mirrors DetectFvg with sourceIndex = index)
        //
        // Bull FVG: Low[i] > High[i-2]  → gap = [High[i-2], Low[i]]
        //           Max = Low[i]  (top of gap), Min = High[i-2] (bottom of gap)
        // Bear FVG: Low[i-2] > High[i]  → gap = [High[i], Low[i-2]]
        //           Max = Low[i-2] (top of gap), Min = High[i] (bottom of gap)
        // ─────────────────────────────────────────────────────────────────────

        private void DetectFvg(int index)
        {
            if (index == _lastDetectedFvgIndex)
                return;

            bool   detected = false;
            bool   isBull   = false;
            double max = 0, min = 0;

            if (Bars.LowPrices[index] > Bars.HighPrices[index - 2])
            {
                detected = true;
                isBull   = true;
                max      = Bars.LowPrices[index];
                min      = Bars.HighPrices[index - 2];
            }
            else if (Bars.LowPrices[index - 2] > Bars.HighPrices[index])
            {
                detected = true;
                isBull   = false;
                max      = Bars.LowPrices[index - 2];
                min      = Bars.HighPrices[index];
            }

            if (!detected)
                return;

            _fvgRecords.Insert(0, new FvgRecord
            {
                Max = max, Min = min, IsBull = isBull, DetectionIndex = index
            });
            _lastDetectedFvgIndex = index;
        }

        // ─────────────────────────────────────────────────────────────────────
        // OB mitigation  (mirrors HandleMitigationOb)
        //
        // Bull OB mitigated: Low[i] <= zone.Max  → pending bull signal at Max
        // Bear OB mitigated: High[i] >= zone.Min → pending bear signal at Min
        // Signal only set when DetectionIndex + MinDist < index (min age check).
        // ─────────────────────────────────────────────────────────────────────

        private void HandleMitigationOb(int index, double barLow, double barHigh)
        {
            for (int i = _obRecords.Count - 1; i >= 0; i--)
            {
                var r = _obRecords[i];
                if (r.DetectionIndex >= index)   // same bar — not yet eligible
                    continue;

                if (r.IsBull && barLow <= r.Max)
                {
                    if (r.DetectionIndex + MinDist < index)
                        _obSignal = new SignalState { Point = r.Max, IsBull = true, Entry = false };
                    _obRecords.RemoveAt(i);
                }
                else if (!r.IsBull && barHigh >= r.Min)
                {
                    if (r.DetectionIndex + MinDist < index)
                        _obSignal = new SignalState { Point = r.Min, IsBull = false, Entry = false };
                    _obRecords.RemoveAt(i);
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // FVG mitigation  (mirrors HandleMitigationFvg)
        //
        // Bull FVG mitigated: Low[i] <= zone.Max  → pending bull signal at Max
        // Bear FVG mitigated: High[i] >= zone.Min → pending bear signal at Min
        // ─────────────────────────────────────────────────────────────────────

        private void HandleMitigationFvg(int index, double barLow, double barHigh)
        {
            for (int i = _fvgRecords.Count - 1; i >= 0; i--)
            {
                var r = _fvgRecords[i];
                if (r.DetectionIndex >= index)
                    continue;

                if (r.IsBull && barLow <= r.Max)
                {
                    if (r.DetectionIndex + MinDistFvg < index)
                        _fvgSignal = new SignalState { Point = r.Max, IsBull = true, Entry = false };
                    _fvgRecords.RemoveAt(i);
                }
                else if (!r.IsBull && barHigh >= r.Min)
                {
                    if (r.DetectionIndex + MinDistFvg < index)
                        _fvgSignal = new SignalState { Point = r.Min, IsBull = false, Entry = false };
                    _fvgRecords.RemoveAt(i);
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Heikin-Ashi  (mirrors EnsureHeikinAshiSource)
        // Built incrementally; uses regular Bars (chart timeframe only).
        // ─────────────────────────────────────────────────────────────────────

        private void EnsureHeikinAshi(int index)
        {
            while (_haClose.Count <= index)
            {
                int    i     = _haClose.Count;
                double close = (Bars.OpenPrices[i] + Bars.HighPrices[i]
                              + Bars.LowPrices[i]  + Bars.ClosePrices[i]) / 4.0;
                double open  = i == 0
                    ? (Bars.OpenPrices[i] + Bars.ClosePrices[i]) / 2.0
                    : (_haOpen[i - 1] + _haClose[i - 1]) / 2.0;
                _haOpen.Add(open);
                _haClose.Add(close);
            }
        }

        // =====================================================================
        // Trade helpers – identical to ICT_01_SignalRef_cBot.cs
        // =====================================================================

        private void OpenLong(int signalBar)
        {
            double slPips = GetSlPips(signalBar);
            if (slPips <= 0)
            {
                Print("Bar {0}: LONG skipped – SL distance invalid ({1:F1} pips).", signalBar, slPips);
                return;
            }

            double volume = CalculateVolume(slPips);
            if (volume <= 0)
            {
                Print("Bar {0}: LONG skipped – volume rounds to 0.", signalBar);
                return;
            }

            double tpPips = slPips * RrRatio;
            Print("Bar {0}: LONG | Ask={1:F5} | SL={2:F1}p | TP={3:F1}p | Vol={4}",
                  signalBar, Symbol.Ask, slPips, tpPips, volume);

            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, BotLabel, slPips, tpPips);
        }

        private void OpenShort(int signalBar)
        {
            double slPips = GetSlPips(signalBar);
            if (slPips <= 0)
            {
                Print("Bar {0}: SHORT skipped – SL distance invalid ({1:F1} pips).", signalBar, slPips);
                return;
            }

            double volume = CalculateVolume(slPips);
            if (volume <= 0)
            {
                Print("Bar {0}: SHORT skipped – volume rounds to 0.", signalBar);
                return;
            }

            double tpPips = slPips * RrRatio;
            Print("Bar {0}: SHORT | Bid={1:F5} | SL={2:F1}p | TP={3:F1}p | Vol={4}",
                  signalBar, Symbol.Bid, slPips, tpPips, volume);

            ExecuteMarketOrder(TradeType.Sell, SymbolName, volume, BotLabel, slPips, tpPips);
        }

        private double GetSlPips(int signalBar)
        {
            double atrValue = _atr.Result[signalBar];
            if (double.IsNaN(atrValue) || atrValue <= 0)
                atrValue = _atr.Result.LastValue;

            if (double.IsNaN(atrValue) || atrValue <= 0)
                return MinSlPips;

            double slPips = (atrValue * AtrMultiplier) / Symbol.PipSize;
            slPips = Math.Max(slPips, MinSlPips);
            slPips = Math.Min(slPips, MaxSlPips);
            return slPips;
        }

        private double CalculateVolume(double slPips)
        {
            double riskAmount = Account.Equity * (RiskPercent / 100.0);
            double raw        = Symbol.VolumeForFixedRisk(riskAmount, slPips);
            double volume     = Symbol.NormalizeVolumeInUnits(raw, RoundingMode.Down);

            if (volume < Symbol.VolumeInUnitsMin)
                return 0;

            if (volume > Symbol.VolumeInUnitsMax)
                volume = Symbol.VolumeInUnitsMax;

            return volume;
        }
    }
}
