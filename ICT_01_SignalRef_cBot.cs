// =============================================================================
// ICT Setup 01 – Signal-Reference cBot
// =============================================================================
// References the external ICT_01 indicator
// ("ICT Setup 01 [TradingFinder] FVG + Liquidity SweepsHunt Alerts,
//  ICT Setup 01 TFlab_ct.cs") for long/short signals.
//
// Entry  : Market order at the open of the bar AFTER the signal fires.
// Exit   : Position is closed (and reversed) when the opposite signal fires.
//          No fixed take-profit is used.
// Risk   : 1 % of current account equity per trade (configurable).
// SL     : ATR-based safety stop (for position sizing and exchange-limit).
//
// Signal priority: long takes precedence if both signals fire on the same bar.
// =============================================================================

using System;
using cAlgo.API;
using cAlgo.API.Indicators;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class ICT_01_SignalRef_cBot : Robot
    {
        // ── ICT Setup 01 indicator parameters ────────────────────────────────
        // These are passed through to Indicators.GetIndicator<ICT_01>() in the
        // same order as the [Parameter] declarations in the indicator source.

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

        private const string BotLabel = "ICT01_Ref";

        // ── Indicator references ──────────────────────────────────────────────

        private ICT_01 _ictIndicator;
        private AverageTrueRange _atr;

        // ── Duplicate-entry guards ────────────────────────────────────────────

        private int _lastLongBar  = -1;
        private int _lastShortBar = -1;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        protected override void OnStart()
        {
            // Load the ICT_01 custom indicator.
            // Parameters are supplied in the exact order they appear as
            // [Parameter] attributes in the indicator source file.
            _ictIndicator = Indicators.GetIndicator<ICT_01>(
                FvgDetectorMultiplier,      // FVG Detector Multiplier Factor
                FvgValidityPeriod,          // FVG Validity Period
                UseDiscountAndPremium,      // Level in Low Risk Zone
                SignalMethod,               // Issuing Signals Method
                SignalsAllowedPerZone,      // The number of signals allowed from a Zone
                SignalAfterHunts,           // Signal after Hunts/Sweeps
                RequiredHunts,              // How Many Hunts/Sweeps?
                false,                      // Show All Long Setup  (ui-only, disabled)
                false,                      // Show All Short Setup (ui-only, disabled)
                "Off",                      // Alert (suppressed inside cBot)
                "ICT Setup 01 Alerts [TradingFinder]",                            // Alert Name
                "Once Per Bar",             // Message Frequency
                "UTC",                      // Show Alert time by Time Zone
                "Long Signal Position Based on ICT Setup 01 [FVG Hunts]",        // Long Position Message
                "Short Signal Position Based on ICT Setup 01 [FVG Hunts]"        // Short Position Message
            );

            // ATR (Wilder) used to size the safety stop-loss distance.
            _atr = Indicators.AverageTrueRange(AtrPeriod, MovingAverageType.WilderSmoothing);

            Print("ICT Setup 01 Signal-Reference Bot started. Risk={0}%, ATR({1})×{2}, SL range=[{3},{4}] pips",
                  RiskPercent, AtrPeriod, AtrMultiplier, MinSlPips, MaxSlPips);
        }

        protected override void OnStop()
        {
            Print("ICT Setup 01 Signal-Reference Bot stopped.");
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

            bool isLong  = _ictIndicator.LongSignal[signalBar]  == 1.0;
            bool isShort = _ictIndicator.ShortSignal[signalBar] == 1.0;

            // ── Long signal ───────────────────────────────────────────────────
            // Close any open short positions, then enter long at current Ask
            // (≈ the open price of the new bar that just started).
            if (isLong && _lastLongBar != signalBar)
            {
                _lastLongBar = signalBar;
                ClosePositionsByType(TradeType.Sell);
                OpenLong(signalBar);
            }
            // ── Short signal ──────────────────────────────────────────────────
            // Close any open long positions, then enter short at current Bid.
            else if (isShort && _lastShortBar != signalBar)
            {
                _lastShortBar = signalBar;
                ClosePositionsByType(TradeType.Buy);
                OpenShort(signalBar);
            }
        }

        // =====================================================================
        // Trade helpers
        // =====================================================================

        private void ClosePositionsByType(TradeType tradeType)
        {
            foreach (var pos in Positions.FindAll(BotLabel, SymbolName, tradeType))
            {
                var result = ClosePosition(pos);
                if (!result.IsSuccessful)
                    Print("Failed to close {0} position #{1}: {2}", tradeType, pos.Id, result.Error);
            }
        }

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

            Print("Bar {0}: LONG | Ask={1:F5} | SL={2:F1} pips | Vol={3}",
                  signalBar, Symbol.Ask, slPips, volume);

            // No take-profit – the position is reversed when the opposite signal fires.
            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, BotLabel, slPips, null);
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

            Print("Bar {0}: SHORT | Bid={1:F5} | SL={2:F1} pips | Vol={3}",
                  signalBar, Symbol.Bid, slPips, volume);

            // No take-profit – the position is reversed when the opposite signal fires.
            ExecuteMarketOrder(TradeType.Sell, SymbolName, volume, BotLabel, slPips, null);
        }

        // =====================================================================
        // SL distance and volume calculation
        // =====================================================================

        /// <summary>
        /// Derives a stop-loss distance in pips from ATR × multiplier,
        /// clamped to [MinSlPips, MaxSlPips].
        /// Falls back to MinSlPips when ATR is not yet available.
        /// </summary>
        private double GetSlPips(int signalBar)
        {
            double atrValue = _atr.Result[signalBar];
            if (double.IsNaN(atrValue) || atrValue <= 0)
                atrValue = _atr.Result.LastValue;       // use most-recent valid ATR

            if (double.IsNaN(atrValue) || atrValue <= 0)
                return MinSlPips;                        // absolute fallback

            double slPips = (atrValue * AtrMultiplier) / Symbol.PipSize;
            slPips = Math.Max(slPips, MinSlPips);
            slPips = Math.Min(slPips, MaxSlPips);
            return slPips;
        }

        /// <summary>
        /// Calculates trade volume so that the risk equals RiskPercent of
        /// current account equity for the given SL distance.
        /// </summary>
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
