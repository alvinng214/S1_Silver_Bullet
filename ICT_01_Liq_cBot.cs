// =============================================================================
// ICT_01_Liq_cBot.cs
// =============================================================================
// Links to two custom indicators via Indicators.GetIndicator<T>():
//   • ICT_01  (ICT Setup 01 TFlab_ct.cs)  — LongSignal / ShortSignal outputs
//   • BSL_SSL (BSL and SSL.cs)             — CurrentBSL / CurrentSSL outputs
//
// Entry rule
//   Long  : !double.IsNaN(ICT_01.LongSignal[signalBar])  → buy  at next bar open
//   Short : !double.IsNaN(ICT_01.ShortSignal[signalBar]) → sell at next bar open
//   signalBar = Bars.Count - 2  (the bar that just closed when OnBar fires)
//
// Stop-loss
//   Long  : SSL level (BSL_SSL.CurrentSSL[signalBar])
//           Fallback when pool NaN: ICT_01.LongSignal[signalBar] = Bars.LowPrices[signalBar]
//           (The ICT long signal fires because price swept below the FVG proximal zone;
//            that same sweep clears every SSL pool entry → pool empty → natural SL = sweep low)
//   Short : BSL level (BSL_SSL.CurrentBSL[signalBar])
//           Fallback when pool NaN: ICT_01.ShortSignal[signalBar] = Bars.HighPrices[signalBar]
//   Actual SL price = SL reference ∓ SlBufferPips
//
// Take-profit
//   TP distance = SL distance × RewardRiskRatio  (default 2.0 → 1:2 R:R)
//
// Risk
//   RiskPercent % of current account equity per trade
//   Volume via Symbol.VolumeForFixedRisk()
//
// Capacity
//   MaxOpenPositions simultaneous trades
// =============================================================================

using System;
using cAlgo.API;
using cAlgo;          // BSL_SSL.LiquidityLineStyle, ICT_01

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class ICT_01_Liq_cBot : Robot
    {
        // =====================================================================
        // Parameters — ICT_01 signal settings
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
        // Parameters — BSL/SSL liquidity settings
        // =====================================================================

        [Parameter("Pivot Left", DefaultValue = 5, MinValue = 1,
            Group = "BSL/SSL Liquidity Settings")]
        public int PivotLeft { get; set; }

        [Parameter("Pivot Right", DefaultValue = 5, MinValue = 1,
            Group = "BSL/SSL Liquidity Settings")]
        public int PivotRight { get; set; }

        // =====================================================================
        // Parameters — risk management
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
        // Private fields
        // =====================================================================

        private const string BotLabel = "ICT01_LIQ";

        private ICT_01  _ict01;
        private BSL_SSL _bslSsl;

        private int _lastLongSignalBar  = -1;
        private int _lastShortSignalBar = -1;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        protected override void OnStart()
        {
            // Link ICT_01 — all 15 parameters in declaration order.
            // Visual/alert parameters are hardcoded; signal parameters are forwarded.
            _ict01 = Indicators.GetIndicator<ICT_01>(
                FvgDetectorMultiplier,      // 1  FvgDetectorMultiplier   double
                FvgValidityPeriod,          // 2  FvgValidityPeriod        int
                UseDiscountAndPremium,      // 3  UseDiscountAndPremium    bool
                SignalMethod,               // 4  SignalMethod             string
                SignalsAllowedPerZone,      // 5  SignalsAllowedPerZone    int
                SignalAfterHunts,           // 6  SignalAfterHunts         bool
                RequiredHunts,             // 7  RequiredHunts            int
                false,                     // 8  ShowAllLongSetups        bool   (visual)
                false,                     // 9  ShowAllShortSetups       bool   (visual)
                "Off",                     // 10 AlertSetting             string (no alerts)
                "ICT Setup 01",            // 11 AlertName                string
                "Once Per Bar",            // 12 Frequency                string
                "UTC",                     // 13 AlertTimeZone            string
                "Long Signal",             // 14 LongPositionMessage      string
                "Short Signal"             // 15 ShortPositionMessage     string
            );

            // Link BSL_SSL — all 7 parameters in declaration order.
            // Visual parameters are hardcoded; pivot parameters are forwarded.
            _bslSsl = Indicators.GetIndicator<BSL_SSL>(
                PivotLeft,                              // 1 PivotLeft         int
                PivotRight,                             // 2 PivotRight        int
                1,                                      // 3 ShowPools         int    (visual)
                BSL_SSL.LiquidityLineStyle.Dots,        // 4 LineStyleParam    enum   (visual)
                "Teal",                                 // 5 BuysideColorName  string (visual)
                "Red",                                  // 6 SellsideColorName string (visual)
                2.0                                     // 7 LabelOffsetPips   double (visual)
            );

            Print("{0} started — FVG mult={1}, Pivot L/R={2}/{3}, Risk={4}%, RR={5}, MaxPos={6}",
                BotLabel, FvgDetectorMultiplier, PivotLeft, PivotRight,
                RiskPercent, RewardRiskRatio, MaxOpenPositions);
        }

        protected override void OnStop()
        {
            _ict01?.Dispose();
            _bslSsl?.Dispose();
            Print("{0} stopped.", BotLabel);
        }

        // =====================================================================
        // Bar event
        // =====================================================================

        protected override void OnBar()
        {
            // signalBar is the bar that just closed (bar N).
            // Both indicators have been calculated by the framework for bar N
            // before OnBar fires at bar N+1 open.
            int signalBar = Bars.Count - 2;
            if (signalBar < 1)
                return;

            bool hasLong  = !double.IsNaN(_ict01.LongSignal[signalBar]);
            bool hasShort = !double.IsNaN(_ict01.ShortSignal[signalBar]);

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
        // Trade execution
        // =====================================================================

        private void TryEnterLong(int signalBar)
        {
            // ── Determine SL reference ────────────────────────────────────────
            // Primary  : nearest active SSL pivot low (BSL_SSL.CurrentSSL)
            // Fallback : signal bar's sweep low (= ICT_01.LongSignal value)
            //   Why fallback is needed:
            //     The ICT long signal fires precisely because price swept below the
            //     FVG proximal zone.  BSL_SSL.MitigateLevels() removes every SSL
            //     entry at or above the bar's low, so the pool may be empty on the
            //     same bar the signal fires.  The natural ICT SL is below the sweep
            //     low, which ICT_01.LongSignal[signalBar] = Bars.LowPrices[signalBar].
            double sslRef = _bslSsl.CurrentSSL[signalBar];
            string slSrc;

            if (!double.IsNaN(sslRef) && sslRef > 0)
            {
                slSrc = "SSL pivot low";
            }
            else
            {
                sslRef = _ict01.LongSignal[signalBar]; // = Bars.LowPrices[signalBar]
                slSrc  = "sweep low fallback";
            }

            double entry   = Symbol.Ask;
            double slPrice = sslRef - SlBufferPips * Symbol.PipSize;

            if (slPrice >= entry)
            {
                Print("Bar {0}: LONG skip — SL {1:F5} ({2}) >= entry {3:F5}.",
                    signalBar, slPrice, slSrc, entry);
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
                signalBar, entry, sslRef, slSrc, slPips, tpPips, volume);

            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, BotLabel, slPips, tpPips);
        }

        private void TryEnterShort(int signalBar)
        {
            // ── Determine SL reference ────────────────────────────────────────
            // Primary  : nearest active BSL pivot high (BSL_SSL.CurrentBSL)
            // Fallback : signal bar's sweep high (= ICT_01.ShortSignal value)
            //   Why fallback is needed:
            //     The ICT short signal fires because price swept above the FVG
            //     proximal zone.  BSL_SSL.MitigateLevels() removes every BSL
            //     entry at or below the bar's high, so the pool may be empty on
            //     the same bar the signal fires.  Natural ICT SL = sweep high,
            //     which ICT_01.ShortSignal[signalBar] = Bars.HighPrices[signalBar].
            double bslRef = _bslSsl.CurrentBSL[signalBar];
            string slSrc;

            if (!double.IsNaN(bslRef) && bslRef > 0)
            {
                slSrc = "BSL pivot high";
            }
            else
            {
                bslRef = _ict01.ShortSignal[signalBar]; // = Bars.HighPrices[signalBar]
                slSrc  = "sweep high fallback";
            }

            double entry   = Symbol.Bid;
            double slPrice = bslRef + SlBufferPips * Symbol.PipSize;

            if (slPrice <= entry)
            {
                Print("Bar {0}: SHORT skip — SL {1:F5} ({2}) <= entry {3:F5}.",
                    signalBar, slPrice, slSrc, entry);
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
                signalBar, entry, bslRef, slSrc, slPips, tpPips, volume);

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
