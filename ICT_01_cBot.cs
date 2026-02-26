// =============================================================================
// ICT Setup 01 cBot
// =============================================================================
// Signals   : IctSetup01FvgLiquidityHuntIndicator (ICT Setup 01 TFlab_ct.cs)
//               LongSignal[bar]  == 1.0  → enter long
//               ShortSignal[bar] == 1.0  → enter short
//
// Execution : Market order at the OPEN of the bar that follows the signal bar.
//             (OnBar fires when a new bar opens; signal bar = Bars.Count - 2.)
//
// Stop Loss : Long  → CurrentSSL from BSL_SSL indicator (must be below entry)
//             Short → CurrentBSL from BSL_SSL indicator (must be above entry)
//
// Take Profit: 2 × SL distance  (Risk : Reward = 1 : 2)
//
// Risk       : 1 % of current Account Equity per trade
//
// Capacity   : Maximum 3 simultaneous open positions (total, this bot's label)
// =============================================================================

using System;
using cAlgo.API;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class ICT_01_cBot : Robot
    {
        // ── ICT Setup 01 indicator parameters ───────────────────────────────

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

        // ── BSL & SSL indicator parameters ──────────────────────────────────

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

        // ── Internal constants ───────────────────────────────────────────────

        private const string BotLabel   = "ICT01_cBot";
        private const double RrRatio    = 2.0;          // 1 : 2 risk-to-reward

        // ── Indicator references ─────────────────────────────────────────────

        private IctSetup01FvgLiquidityHuntIndicator _ictIndicator;
        private BSL_SSL                             _bslSslIndicator;

        // ── State ────────────────────────────────────────────────────────────

        private int _lastLongSignalBar  = -1;
        private int _lastShortSignalBar = -1;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        protected override void OnStart()
        {
            // ------------------------------------------------------------------
            // Instantiate ICT Setup 01 indicator.
            // Parameters are passed in the exact declaration order of the
            // indicator's [Parameter] attributes.
            //
            //  1  FvgDetectorMultiplier (double)
            //  2  FvgValidityPeriod     (int)
            //  3  UseDiscountAndPremium (bool)
            //  4  SignalMethod          (string)
            //  5  SignalsAllowedPerZone (int)
            //  6  SignalAfterHunts      (bool)
            //  7  RequiredHunts         (int)
            //  8  ShowAllLongSetups     (bool)  ← always false in cBot context
            //  9  ShowAllShortSetups    (bool)  ← always false in cBot context
            // 10  AlertSetting          (string) ← "Off" (alerts not needed)
            // 11  AlertName             (string)
            // 12  Frequency             (string)
            // 13  AlertTimeZone         (string)
            // 14  LongPositionMessage   (string)
            // 15  ShortPositionMessage  (string)
            // ------------------------------------------------------------------
            _ictIndicator = Indicators.GetIndicator<IctSetup01FvgLiquidityHuntIndicator>(
                FvgDetectorMultiplier,
                FvgValidityPeriod,
                UseDiscountAndPremium,
                SignalMethod,
                SignalsAllowedPerZone,
                SignalAfterHunts,
                RequiredHunts,
                false,
                false,
                "Off",
                "ICT01_cBot",
                "Once Per Bar",
                "UTC",
                "Long Signal",
                "Short Signal"
            );

            // ------------------------------------------------------------------
            // Instantiate BSL & SSL indicator.
            // Parameters in declaration order:
            //  1  PivotLeft          (int)
            //  2  PivotRight         (int)
            //  3  ShowPools          (int)
            //  4  LineStyleParam     (BSL_SSL.LiquidityLineStyle enum)
            //  5  BuysideColorName   (string)
            //  6  SellsideColorName  (string)
            //  7  LabelOffsetPips    (double)
            // ------------------------------------------------------------------
            _bslSslIndicator = Indicators.GetIndicator<BSL_SSL>(
                PivotLeft,
                PivotRight,
                1,
                BSL_SSL.LiquidityLineStyle.Dots,
                "Teal",
                "Red",
                2.0
            );

            Print("ICT Setup 01 cBot started. MaxPositions={0}, Risk={1}%", MaxOpenPositions, RiskPercent);
        }

        protected override void OnStop()
        {
            Print("ICT Setup 01 cBot stopped.");
        }

        // =====================================================================
        // Bar event – fires when a NEW bar opens
        // =====================================================================

        protected override void OnBar()
        {
            // The bar that just CLOSED is the signal bar.
            // This bar is at Bars.Count - 2; the bar that just OPENED (entry bar)
            // is at Bars.Count - 1.  We execute a market order now, which fills
            // at approximately the opening price of the new bar.

            int signalBar = Bars.Count - 2;

            // Need at least index 2 for FVG detection inside the indicator.
            if (signalBar < 2)
                return;

            bool isLongSignal  = _ictIndicator.LongSignal[signalBar]  > 0.5;
            bool isShortSignal = _ictIndicator.ShortSignal[signalBar] > 0.5;

            if (!isLongSignal && !isShortSignal)
                return;

            // ── Global position-count guard ──────────────────────────────────
            int openCount = Positions.FindAll(BotLabel, SymbolName).Length;
            if (openCount >= MaxOpenPositions)
            {
                Print("Bar {0}: Max open positions ({1}) reached. Skipping signal.", signalBar, MaxOpenPositions);
                return;
            }

            // ── Process each signal (re-check capacity before each entry) ────

            if (isLongSignal && _lastLongSignalBar != signalBar)
            {
                _lastLongSignalBar = signalBar;   // mark first to prevent re-entry
                TryEnterLong(signalBar);
            }

            // Re-read position count; a successful long entry may have filled capacity.
            openCount = Positions.FindAll(BotLabel, SymbolName).Length;
            if (openCount >= MaxOpenPositions)
                return;

            if (isShortSignal && _lastShortSignalBar != signalBar)
            {
                _lastShortSignalBar = signalBar;
                TryEnterShort(signalBar);
            }
        }

        // =====================================================================
        // Trade entry helpers
        // =====================================================================

        private void TryEnterLong(int signalBar)
        {
            // Entry price: current Ask (≈ open of the new bar in OnBar context)
            double entry = Symbol.Ask;

            // Stop loss anchor: nearest confirmed Sellside Liquidity (SSL) from
            // the BSL & SSL indicator at the signal bar.  This is the most
            // recent unmitigated pivot low at that point in time.
            double sslLevel = _bslSslIndicator.CurrentSSL[signalBar];

            if (double.IsNaN(sslLevel) || sslLevel <= 0)
            {
                Print("Bar {0}: LONG skipped – SSL is unavailable (NaN/zero).", signalBar);
                return;
            }

            if (sslLevel >= entry)
            {
                Print("Bar {0}: LONG skipped – SSL {1:F5} is not below entry {2:F5}.",
                      signalBar, sslLevel, entry);
                return;
            }

            double slPips = (entry - sslLevel) / Symbol.PipSize;

            if (!ValidateSlPips(signalBar, "LONG", slPips))
                return;

            double tpPips    = slPips * RrRatio;
            double riskAmount = Account.Equity * (RiskPercent / 100.0);
            double volume    = GetRiskVolume(riskAmount, slPips);

            if (volume <= 0)
            {
                Print("Bar {0}: LONG skipped – calculated volume is 0.", signalBar);
                return;
            }

            Print("Bar {0}: LONG  | Entry={1:F5} | SSL SL={2:F5} ({3:F1} pips) | TP={4:F1} pips | Vol={5}",
                  signalBar, entry, sslLevel, slPips, tpPips, volume);

            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, BotLabel, slPips, tpPips);
        }

        private void TryEnterShort(int signalBar)
        {
            // Entry price: current Bid (≈ open of the new bar in OnBar context)
            double entry = Symbol.Bid;

            // Stop loss anchor: nearest confirmed Buyside Liquidity (BSL) from
            // the BSL & SSL indicator at the signal bar.  This is the most
            // recent unmitigated pivot high at that point in time.
            double bslLevel = _bslSslIndicator.CurrentBSL[signalBar];

            if (double.IsNaN(bslLevel) || bslLevel <= 0)
            {
                Print("Bar {0}: SHORT skipped – BSL is unavailable (NaN/zero).", signalBar);
                return;
            }

            if (bslLevel <= entry)
            {
                Print("Bar {0}: SHORT skipped – BSL {1:F5} is not above entry {2:F5}.",
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
                Print("Bar {0}: SHORT skipped – calculated volume is 0.", signalBar);
                return;
            }

            Print("Bar {0}: SHORT | Entry={1:F5} | BSL SL={2:F5} ({3:F1} pips) | TP={4:F1} pips | Vol={5}",
                  signalBar, entry, bslLevel, slPips, tpPips, volume);

            ExecuteMarketOrder(TradeType.Sell, SymbolName, volume, BotLabel, slPips, tpPips);
        }

        // =====================================================================
        // Utility helpers
        // =====================================================================

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

        /// <summary>
        /// Returns the normalised volume in units that risks exactly <paramref name="riskAmount"/>
        /// over <paramref name="slPips"/> pips, rounded DOWN to the broker's nearest valid lot step.
        /// Returns 0 when volume cannot be computed or falls below the symbol's minimum.
        /// </summary>
        private double GetRiskVolume(double riskAmount, double slPips)
        {
            if (slPips <= 0)
                return 0;

            // Symbol.VolumeForFixedRisk handles the pip-value conversion for any
            // instrument (forex, gold, indices, etc.) correctly.
            double rawVolume = Symbol.VolumeForFixedRisk(riskAmount, slPips);
            double volume    = Symbol.NormalizeVolumeInUnits(rawVolume, RoundingMode.Down);

            if (volume < Symbol.VolumeInUnitsMin)
                return 0;

            if (volume > Symbol.VolumeInUnitsMax)
                volume = Symbol.VolumeInUnitsMax;

            return volume;
        }
    }
}
