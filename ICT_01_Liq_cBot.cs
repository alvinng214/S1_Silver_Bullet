// =============================================================================
// ICT_01_Liq_cBot.cs
// =============================================================================
// cBot that links to two custom indicators via Indicators.GetIndicator<T>():
//
//   1. ICT_01  (ICT Setup 01 TFlab_ct.cs)
//      → LongSignal[bar]  = Bars.LowPrices[bar]  when a long  setup fires
//      → ShortSignal[bar] = Bars.HighPrices[bar]  when a short setup fires
//      → NaN on all other bars
//
//   2. BSL_SSL (BSL and SSL.cs)
//      → CurrentSSL[bar] = most-recent active Sellside pivot low  (long SL ref)
//      → CurrentBSL[bar] = most-recent active Buyside  pivot high (short SL ref)
//      → NaN when the pool has been fully cleared by price sweeping all levels
//
// Execution rules
//   • Signal bar closes  → on the very next bar open execute a market order
//   • Long  SL reference = CurrentSSL from signal bar
//                          (fallback: signal bar's sweep low when pool is empty)
//   • Short SL reference = CurrentBSL from signal bar
//                          (fallback: signal bar's sweep high when pool is empty)
//   • Actual SL placed SlBufferPips below/above the reference
//   • TP = SL distance × RewardRiskRatio  (default 1:2)
//   • Volume sized to risk exactly RiskPercent% of remaining equity
//   • Maximum MaxOpenPositions simultaneous positions (all directions combined)
//
// WHY THE FALLBACK IS NEEDED
//   The ICT_01 long signal fires precisely because price swept below the FVG
//   proximal zone.  That same sweep triggers BSL_SSL's ClearMitigated(), which
//   removes every pivot-low pool entry whose price ≥ the bar's low.  When all
//   entries are cleared the output is NaN.  The natural ICT stop-loss in that
//   case is below the sweep low itself — which is exactly what LongSignal
//   already outputs (= Bars.LowPrices[signalBar]).  Likewise for short signals.
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
using cAlgo;                       // ICT_01 and BSL_SSL live in namespace cAlgo

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

        // Bug fix: was 500 — too small for XAUUSD (pip=0.01) where a $10 stop
        // = 1,000 pips.  Default raised to 10,000 to accommodate all instruments.
        [Parameter("Max SL Distance (pips)", DefaultValue = 10000.0, MinValue = 10.0,
            Group = "Risk Management")]
        public double MaxSlPips { get; set; }

        // =====================================================================
        // Private state
        // =====================================================================

        private const string BotLabel = "ICT01_LIQ";

        private ICT_01  _ict01;
        private BSL_SSL _bslSsl;

        // Prevent acting on the same signal bar twice
        private int _lastLongEntryBar  = -1;
        private int _lastShortEntryBar = -1;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        protected override void OnStart()
        {
            // -----------------------------------------------------------------
            // Initialise ICT_01 – parameters in declaration order (15 total).
            // Alert set to "Off" — cBot reads signals programmatically.
            // -----------------------------------------------------------------
            _ict01 = Indicators.GetIndicator<ICT_01>(
                FvgDetectorMultiplier,   //  1 FvgDetectorMultiplier  double
                FvgValidityPeriod,       //  2 FvgValidityPeriod      int
                UseDiscountAndPremium,   //  3 UseDiscountAndPremium  bool
                SignalMethod,            //  4 SignalMethod           string
                SignalsAllowedPerZone,   //  5 SignalsAllowedPerZone  int
                SignalAfterHunts,        //  6 SignalAfterHunts       bool
                RequiredHunts,           //  7 RequiredHunts          int
                false,                   //  8 ShowAllLongSetups      bool  (visual)
                false,                   //  9 ShowAllShortSetups     bool  (visual)
                "Off",                   // 10 AlertSetting           string
                BotLabel,               // 11 AlertName              string (unused)
                "Once Per Bar",          // 12 Frequency              string (unused)
                "UTC",                   // 13 AlertTimeZone          string (unused)
                string.Empty,            // 14 LongPositionMessage    string (unused)
                string.Empty             // 15 ShortPositionMessage   string (unused)
            );

            // -----------------------------------------------------------------
            // Initialise BSL_SSL – parameters in declaration order (7 total).
            // -----------------------------------------------------------------
            _bslSsl = Indicators.GetIndicator<BSL_SSL>(
                PivotLeft,                          // 1 PivotLeft        int
                PivotRight,                         // 2 PivotRight       int
                1,                                  // 3 ShowPools        int
                BSL_SSL.LiquidityLineStyle.Dots,    // 4 LineStyleParam   enum
                "Teal",                             // 5 BuysideColorName string
                "Red",                              // 6 SellsideColorName string
                2.0                                 // 7 LabelOffsetPips  double
            );

            Print("{0} started — FVG mult={1}, Pivot L/R={2}/{3}, Risk={4}%, MaxPos={5}",
                BotLabel, FvgDetectorMultiplier, PivotLeft, PivotRight,
                RiskPercent, MaxOpenPositions);
        }

        protected override void OnBar()
        {
            // OnBar fires at the open of bar N+1.
            // signalBar = bar N = the bar that just closed and may carry a signal.
            int signalBar = Bars.Count - 2;
            if (signalBar < 1)
                return;

            // ── Long signal ──────────────────────────────────────────────────
            // ICT_01.LongSignal[bar] = Bars.LowPrices[bar] on signal bars, NaN otherwise.
            if (_lastLongEntryBar != signalBar &&
                !double.IsNaN(_ict01.LongSignal[signalBar]))
            {
                _lastLongEntryBar = signalBar;
                TryEnterLong(signalBar);
            }

            // ── Short signal ─────────────────────────────────────────────────
            // ICT_01.ShortSignal[bar] = Bars.HighPrices[bar] on signal bars, NaN otherwise.
            if (_lastShortEntryBar != signalBar &&
                !double.IsNaN(_ict01.ShortSignal[signalBar]))
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
        // Trade entry
        // =====================================================================

        private void TryEnterLong(int signalBar)
        {
            if (Positions.FindAll(BotLabel, SymbolName).Length >= MaxOpenPositions)
            {
                Print("Bar {0}: LONG skip — max positions ({1}) reached.", signalBar, MaxOpenPositions);
                return;
            }

            // ── Determine SL reference ────────────────────────────────────────
            // Primary: most-recent active SSL (pivot low) from BSL_SSL.
            // Fallback: signal bar's sweep low — ICT_01.LongSignal outputs
            //           Bars.LowPrices[signalBar] which is that exact price.
            //
            // Why the fallback is needed:
            //   The ICT long signal fires because price swept below the FVG zone.
            //   That same sweep triggers BSL_SSL.ClearMitigated() which removes
            //   all pivot-low entries whose price >= the bar's low.  When all
            //   entries are cleared, CurrentSSL[signalBar] == NaN.  Using the
            //   sweep low as the SL reference is standard ICT methodology.
            double slRef   = _bslSsl.CurrentSSL[signalBar];
            string slSource;

            if (!double.IsNaN(slRef))
            {
                slSource = "BSL_SSL active pivot low";
            }
            else
            {
                // LongSignal[signalBar] == Bars.LowPrices[signalBar]
                slRef    = _ict01.LongSignal[signalBar];
                slSource = "sweep low (SSL pool cleared)";
            }

            // Place actual SL SlBufferPips below the reference
            double entry   = Symbol.Ask;
            double slPrice = slRef - SlBufferPips * Symbol.PipSize;

            if (slPrice >= entry)
            {
                Print("Bar {0}: LONG skip — SL price {1:F5} ({2}) >= entry {3:F5}.",
                    signalBar, slPrice, slSource, entry);
                return;
            }

            double slPips = (entry - slPrice) / Symbol.PipSize;

            if (!SlPipsInRange(signalBar, "LONG", slPips))
                return;

            double tpPips  = slPips * RewardRiskRatio;
            double volume  = CalculateVolume(slPips);

            if (volume <= 0)
            {
                Print("Bar {0}: LONG skip — volume is 0.", signalBar);
                return;
            }

            Print("Bar {0}: LONG | Ask={1:F5} | SLref={2:F5} ({3}) | SL={4:F1} pips | TP={5:F1} pips | Vol={6}",
                signalBar, entry, slRef, slSource, slPips, tpPips, volume);

            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, BotLabel, slPips, tpPips);
        }

        private void TryEnterShort(int signalBar)
        {
            if (Positions.FindAll(BotLabel, SymbolName).Length >= MaxOpenPositions)
            {
                Print("Bar {0}: SHORT skip — max positions ({1}) reached.", signalBar, MaxOpenPositions);
                return;
            }

            // ── Determine SL reference ────────────────────────────────────────
            // Primary: most-recent active BSL (pivot high) from BSL_SSL.
            // Fallback: signal bar's sweep high (= ICT_01.ShortSignal[signalBar]
            //           = Bars.HighPrices[signalBar]).
            double slRef   = _bslSsl.CurrentBSL[signalBar];
            string slSource;

            if (!double.IsNaN(slRef))
            {
                slSource = "BSL_SSL active pivot high";
            }
            else
            {
                // ShortSignal[signalBar] == Bars.HighPrices[signalBar]
                slRef    = _ict01.ShortSignal[signalBar];
                slSource = "sweep high (BSL pool cleared)";
            }

            // Place actual SL SlBufferPips above the reference
            double entry   = Symbol.Bid;
            double slPrice = slRef + SlBufferPips * Symbol.PipSize;

            if (slPrice <= entry)
            {
                Print("Bar {0}: SHORT skip — SL price {1:F5} ({2}) <= entry {3:F5}.",
                    signalBar, slPrice, slSource, entry);
                return;
            }

            double slPips = (slPrice - entry) / Symbol.PipSize;

            if (!SlPipsInRange(signalBar, "SHORT", slPips))
                return;

            double tpPips  = slPips * RewardRiskRatio;
            double volume  = CalculateVolume(slPips);

            if (volume <= 0)
            {
                Print("Bar {0}: SHORT skip — volume is 0.", signalBar);
                return;
            }

            Print("Bar {0}: SHORT | Bid={1:F5} | SLref={2:F5} ({3}) | SL={4:F1} pips | TP={5:F1} pips | Vol={6}",
                signalBar, entry, slRef, slSource, slPips, tpPips, volume);

            ExecuteMarketOrder(TradeType.Sell, SymbolName, volume, BotLabel, slPips, tpPips);
        }

        // =====================================================================
        // Helpers
        // =====================================================================

        private bool SlPipsInRange(int signalBar, string direction, double slPips)
        {
            if (slPips < MinSlPips)
            {
                Print("Bar {0}: {1} skip — SL {2:F1} pips < min {3:F1}.",
                    signalBar, direction, slPips, MinSlPips);
                return false;
            }

            if (slPips > MaxSlPips)
            {
                Print("Bar {0}: {1} skip — SL {2:F1} pips > max {3:F1}.",
                    signalBar, direction, slPips, MaxSlPips);
                return false;
            }

            return true;
        }

        private double CalculateVolume(double slPips)
        {
            if (slPips <= 0 || Symbol.PipValue <= 0)
                return 0;

            double riskAmount = Account.Equity * (RiskPercent / 100.0);
            double rawVolume  = riskAmount / (slPips * Symbol.PipValue);
            double volume     = Symbol.NormalizeVolumeInUnits(rawVolume, RoundingMode.Down);

            return volume < Symbol.VolumeInUnitsMin ? Symbol.VolumeInUnitsMin : volume;
        }
    }
}
