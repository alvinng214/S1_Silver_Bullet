// =============================================================================
// ICT_01_Liq_cBot.cs
// =============================================================================
// cBot that links to two custom indicators via Indicators.GetIndicator<T>():
//
//   1. ICT_01  (ICT Setup 01 [TradingFinder] FVG + Liquidity SweepsHunt Alerts,
//               ICT Setup 01 TFlab_ct.cs)
//      → LongSignal[bar]  = 1.0 when a long  setup fires on that bar
//      → ShortSignal[bar] = 1.0 when a short setup fires on that bar
//
//   2. BSL_SSL (BSL and SSL.cs)
//      → CurrentSSL[bar] = most-recent confirmed Sellside pivot low  (long SL)
//      → CurrentBSL[bar] = most-recent confirmed Buyside  pivot high (short SL)
//
// Execution rules
//   • Signal bar closes  → on the very next bar open execute a market order
//   • Long  SL = CurrentSSL from signal bar; TP = entry + 2 × (entry − SL)
//   • Short SL = CurrentBSL from signal bar; TP = entry − 2 × (SL − entry)
//   • Volume sized to risk exactly RiskPercent% of remaining equity
//   • Maximum MaxOpenPositions simultaneous positions (all directions combined)
//
// IMPORTANT – project setup
//   All three files must be compiled in the same cTrader Algo project:
//     • "ICT Setup 01 [TradingFinder] FVG + Liquidity SweepsHunt Alerts,
//        ICT Setup 01 TFlab_ct.cs"
//     • "BSL and SSL.cs"
//     • This file
// =============================================================================

using System;
using cAlgo.API;
using cAlgo.API.Internals;

// ICT_01 and BSL_SSL live in namespace cAlgo
using cAlgo;

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

        [Parameter("Min SL Distance (pips)", DefaultValue = 3.0, MinValue = 0.5,
            Group = "Risk Management")]
        public double MinSlPips { get; set; }

        [Parameter("Max SL Distance (pips)", DefaultValue = 500.0, MinValue = 10.0,
            Group = "Risk Management")]
        public double MaxSlPips { get; set; }

        // =====================================================================
        // Private state
        // =====================================================================

        private const string BotLabel = "ICT01_LIQ";

        private ICT_01  _ict01;
        private BSL_SSL _bslSsl;

        // Track last bar on which we acted to avoid double-entry
        private int _lastLongEntryBar  = -1;
        private int _lastShortEntryBar = -1;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        protected override void OnStart()
        {
            // -----------------------------------------------------------------
            // Initialise ICT_01
            // Parameters must be passed in the exact order they are declared
            // inside class ICT_01 (matching [Parameter] attribute order).
            //
            //  1  FvgDetectorMultiplier  double
            //  2  FvgValidityPeriod      int
            //  3  UseDiscountAndPremium  bool
            //  4  SignalMethod           string
            //  5  SignalsAllowedPerZone  int
            //  6  SignalAfterHunts       bool
            //  7  RequiredHunts          int
            //  8  ShowAllLongSetups      bool   (visual only → false)
            //  9  ShowAllShortSetups     bool   (visual only → false)
            // 10  AlertSetting           string ("Off" – cBot reads programmatically)
            // 11  AlertName              string (unused)
            // 12  Frequency              string (unused)
            // 13  AlertTimeZone          string (unused)
            // 14  LongPositionMessage    string (unused)
            // 15  ShortPositionMessage   string (unused)
            // -----------------------------------------------------------------
            _ict01 = Indicators.GetIndicator<ICT_01>(
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
                BotLabel,
                "Once Per Bar",
                "UTC",
                string.Empty,
                string.Empty
            );

            // -----------------------------------------------------------------
            // Initialise BSL_SSL
            // Parameters in declaration order:
            //  1  PivotLeft        int
            //  2  PivotRight       int
            //  3  ShowPools        int
            //  4  LineStyleParam   BSL_SSL.LiquidityLineStyle (enum)
            //  5  BuysideColorName string
            //  6  SellsideColorName string
            //  7  LabelOffsetPips  double
            // -----------------------------------------------------------------
            _bslSsl = Indicators.GetIndicator<BSL_SSL>(
                PivotLeft,
                PivotRight,
                1,
                BSL_SSL.LiquidityLineStyle.Dots,
                "Teal",
                "Red",
                2.0
            );

            Print("{0} started — FVG mult={1}, Pivot L/R={2}/{3}, Risk={4}%, MaxPos={5}",
                BotLabel, FvgDetectorMultiplier, PivotLeft, PivotRight,
                RiskPercent, MaxOpenPositions);
        }

        protected override void OnBar()
        {
            // OnBar fires at the open of a new bar.
            // The bar that just CLOSED is Bars.Count - 2  → this is the signal bar.
            // We are now at the open of Bars.Count - 1   → execution bar.
            int signalBar = Bars.Count - 2;
            if (signalBar < 1)
                return;

            // ── Long signal ──────────────────────────────────────────────────
            if (_lastLongEntryBar != signalBar &&
                _ict01.LongSignal[signalBar] >= 1.0)
            {
                _lastLongEntryBar = signalBar;
                TryEnterLong(signalBar);
            }

            // ── Short signal ─────────────────────────────────────────────────
            if (_lastShortEntryBar != signalBar &&
                _ict01.ShortSignal[signalBar] >= 1.0)
            {
                _lastShortEntryBar = signalBar;
                TryEnterShort(signalBar);
            }
        }

        protected override void OnStop()
        {
            Print("{0} stopped.", BotLabel);
        }

        // =====================================================================
        // Trade entry helpers
        // =====================================================================

        private void TryEnterLong(int signalBar)
        {
            if (Positions.FindAll(BotLabel, SymbolName).Length >= MaxOpenPositions)
            {
                Print("Bar {0}: LONG — max positions ({1}) reached, skip.",
                    signalBar, MaxOpenPositions);
                return;
            }

            // SL at most-recent confirmed Sellside liquidity from the signal bar
            double ssl = _bslSsl.CurrentSSL[signalBar];
            if (double.IsNaN(ssl))
            {
                Print("Bar {0}: LONG — no active SSL level found, skip.", signalBar);
                return;
            }

            // Market order fills at current Ask (open of execution bar)
            double entry = Symbol.Ask;

            if (ssl >= entry)
            {
                Print("Bar {0}: LONG — SSL {1:F5} is not below Ask {2:F5}, skip.",
                    signalBar, ssl, entry);
                return;
            }

            double slPips = (entry - ssl) / Symbol.PipSize;

            if (!SlPipsInRange(signalBar, "LONG", slPips))
                return;

            double tpPips   = slPips * RewardRiskRatio;
            double volume   = CalculateVolume(slPips);

            if (volume <= 0)
            {
                Print("Bar {0}: LONG — volume is 0, skip.", signalBar);
                return;
            }

            Print("Bar {0}: LONG entry | Ask={1:F5} | SSL={2:F5} | SL={3:F1} pips | TP={4:F1} pips | Vol={5}",
                signalBar, entry, ssl, slPips, tpPips, volume);

            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, BotLabel, slPips, tpPips);
        }

        private void TryEnterShort(int signalBar)
        {
            if (Positions.FindAll(BotLabel, SymbolName).Length >= MaxOpenPositions)
            {
                Print("Bar {0}: SHORT — max positions ({1}) reached, skip.",
                    signalBar, MaxOpenPositions);
                return;
            }

            // SL at most-recent confirmed Buyside liquidity from the signal bar
            double bsl = _bslSsl.CurrentBSL[signalBar];
            if (double.IsNaN(bsl))
            {
                Print("Bar {0}: SHORT — no active BSL level found, skip.", signalBar);
                return;
            }

            // Market order fills at current Bid (open of execution bar)
            double entry = Symbol.Bid;

            if (bsl <= entry)
            {
                Print("Bar {0}: SHORT — BSL {1:F5} is not above Bid {2:F5}, skip.",
                    signalBar, bsl, entry);
                return;
            }

            double slPips = (bsl - entry) / Symbol.PipSize;

            if (!SlPipsInRange(signalBar, "SHORT", slPips))
                return;

            double tpPips   = slPips * RewardRiskRatio;
            double volume   = CalculateVolume(slPips);

            if (volume <= 0)
            {
                Print("Bar {0}: SHORT — volume is 0, skip.", signalBar);
                return;
            }

            Print("Bar {0}: SHORT entry | Bid={1:F5} | BSL={2:F5} | SL={3:F1} pips | TP={4:F1} pips | Vol={5}",
                signalBar, entry, bsl, slPips, tpPips, volume);

            ExecuteMarketOrder(TradeType.Sell, SymbolName, volume, BotLabel, slPips, tpPips);
        }

        // =====================================================================
        // Helpers
        // =====================================================================

        private bool SlPipsInRange(int signalBar, string direction, double slPips)
        {
            if (slPips < MinSlPips)
            {
                Print("Bar {0}: {1} — SL {2:F1} pips < min {3:F1}, skip.",
                    signalBar, direction, slPips, MinSlPips);
                return false;
            }

            if (slPips > MaxSlPips)
            {
                Print("Bar {0}: {1} — SL {2:F1} pips > max {3:F1}, skip.",
                    signalBar, direction, slPips, MaxSlPips);
                return false;
            }

            return true;
        }

        /// <summary>
        /// Calculates the trade volume for a given SL distance (in pips) to
        /// risk exactly RiskPercent% of current equity.
        /// </summary>
        private double CalculateVolume(double slPips)
        {
            if (slPips <= 0 || Symbol.PipValue <= 0)
                return 0;

            double riskAmount = Account.Equity * (RiskPercent / 100.0);
            // PipValue = monetary value of 1 pip for 1 unit of volume
            double rawVolume  = riskAmount / (slPips * Symbol.PipValue);
            double volume     = Symbol.NormalizeVolumeInUnits(rawVolume, RoundingMode.Down);

            return volume < Symbol.VolumeInUnitsMin ? Symbol.VolumeInUnitsMin : volume;
        }
    }
}
