// =============================================================================
// Order Block Detector – Signal-Reference cBot
// =============================================================================
// References the external OrderBlockDetector indicator
// ("Order-Block Detector.cs") for long/short signals.
//
// Entry  : Market order at the open of the bar AFTER the signal fires.
// Exit   : Fixed TP at 2 × SL distance (1:2 risk-to-reward). SL is ATR-based.
// Risk   : 1 % of current account equity per trade (configurable).
// SL     : ATR-based (ATR × multiplier, clamped to min/max pip range).
//
// Difference vs ICT_01_SignalRef_cBot.cs: signal source only.
// ICT_01_SignalRef_cBot uses the ICT_01 FVG-hunt indicator;
// this bot uses the OrderBlockDetector OB/FVG-mitigation indicator.
// SL logic, R:R, and risk management are identical.
// No limit on simultaneous open positions.
// =============================================================================

using System;
using cAlgo.API;
using cAlgo.API.Indicators;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class OB_Detector_SignalRef_cBot : Robot
    {
        // ── Order Block Detector indicator parameters ─────────────────────────
        // Passed through to Indicators.GetIndicator<OrderBlockDetector>() in the
        // same order as the [Parameter] declarations in the indicator source.

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

        // ── Indicator references ──────────────────────────────────────────────

        private OrderBlockDetector _obDetector;
        private AverageTrueRange   _atr;

        // ── Duplicate-entry guards ────────────────────────────────────────────

        private int _lastLongBar  = -1;
        private int _lastShortBar = -1;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        protected override void OnStart()
        {
            // Load the OrderBlockDetector indicator.
            // Parameters passed in the exact order they appear as [Parameter]
            // attributes in Order-Block Detector.cs.
            // Display-only parameters are hardcoded to neutral defaults.
            _obDetector = Indicators.GetIndicator<OrderBlockDetector>(
                true,               // UseChartTimeframe
                TimeFrame.Hour,     // InputTimeFrame (ignored when UseChartTimeframe=true)
                1,                  // LineWidthLiquidated
                80,                 // Transparency
                Color.Green,        // ColorBull
                Color.Red,          // ColorBear
                Color.Blue,         // ColorFvgBull
                Color.Orange,       // ColorFvgBear
                ShowOb,             // ShowOb
                ShowFvg,            // ShowFvg
                false,              // ShowSignalsOb  (chart icons – suppressed in cBot)
                false,              // ShowSignalsFvg (chart icons – suppressed in cBot)
                MinDist,            // MinDist
                MinDistFvg,         // MinDistFvg
                UseHeikinAshi       // UseHeikinAshi
            );

            _atr = Indicators.AverageTrueRange(AtrPeriod, MovingAverageType.WilderSmoothing);

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
            // OnBar fires when a new bar opens; the bar that just *closed* is
            // at index Bars.Count - 2. That is the signal bar we evaluate.
            int signalBar = Bars.Count - 2;
            if (signalBar < 2)
                return;

            bool isLong  = _obDetector.LongSignal[signalBar]  == 1.0;
            bool isShort = _obDetector.ShortSignal[signalBar] == 1.0;

            if (isLong && _lastLongBar != signalBar)
            {
                _lastLongBar = signalBar;
                OpenLong(signalBar);
            }

            if (isShort && _lastShortBar != signalBar)
            {
                _lastShortBar = signalBar;
                OpenShort(signalBar);
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
