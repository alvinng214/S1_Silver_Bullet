// =============================================================================
// Liquidity_Engulfing_and_Displacement_OBImbalance_cBot
// =============================================================================
// Base    : Liquidity_Engulfing_and_Displacement_cBot
// Added   : OB Imbalance filter — logic ported from
//           LEC_Autotl_ifvg_smc_mtfpt_smz_mtffvg_OBwick_Imb_filter_cbot
//
// OB Imbalance Filter:
//   Seed bar (closedIdx-2) forms an OB zone (Top=High, Bottom=Low of seed bar).
//   Bull OB: Low[closedIdx] - High[closedIdx-2] > ATR * threshold  (FVG gap up)
//   Bear OB: Low[closedIdx-2] - High[closedIdx] > ATR * threshold  (FVG gap down)
//   where closedIdx = MtfFOB(tf.Bars, now) - 1 = last CLOSED HTF bar (NOT forming).
//
//   Zone States:
//     Active     = formed, not yet touched
//     Mitigated  = had its first touch; still eligible
//     Invalidated= close crossed far side; excluded from all trades
//
//   Unmitigated toggle: signal within N bars of FIRST touch.
//   Mitigated   toggle: signal within N bars of LAST  touch.
//   Invalidated OBs excluded regardless of toggle.
// =============================================================================

using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Indicators;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class Liquidity_Engulfing_and_Displacement_OBImbalance_cBot : Robot
    {
        // ═════════════════════════════════════════════════════════════════════
        //  Enums
        // ═════════════════════════════════════════════════════════════════════

        public enum DisplacementType
        {
            OpenToClose,
            HighToLow
        }

        public enum LongStopLossSource
        {
            Ssl,
            CandleMinus1Low
        }

        public enum ShortStopLossSource
        {
            Bsl,
            CandleMinus1High
        }

        public enum ImbMitigationMethod
        {
            Close,
            Wick
        }

        public enum ImbZoneState
        {
            Active,
            Mitigated,
            Invalidated
        }

        // ═════════════════════════════════════════════════════════════════════
        //  Inner types
        // ═════════════════════════════════════════════════════════════════════

        private sealed class BslPivot
        {
            public double Price;
            public int BarIndex;
            public int Type;    // 1 = pivot high (BSL), -1 = pivot low (SSL)
        }

        private sealed class BslPool
        {
            public double Price;
            public int PivotIndex;
        }

        private sealed class ImbObZone
        {
            public bool IsBullish;
            public ImbZoneState State = ImbZoneState.Active;
            public double Top, Bottom;
            public int FirstTouchBar = -1, LastTouchBar = -1;
            // Re-arm on every new touch: eligible again whenever price returns
            // to the zone AFTER the previous trade fired (LastTouchBar > LastFiredBar).
            // Replaces the old permanent TradeFired flag which blocked zones
            // from re-firing after a legitimate re-touch.
            public int LastFiredBar = -1;
        }

        private sealed class ImbTfState
        {
            public string Label;
            public Bars Bars;
            public AverageTrueRange Atr;
            public int MaxCount;
            public DateTime LastCreatedSeedTime = DateTime.MinValue;
            public List<ImbObZone> Zones = new List<ImbObZone>();
        }

        // ═════════════════════════════════════════════════════════════════════
        //  Parameters — Signal Engine Enables
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Enable LEC Long Signal", DefaultValue = false, Group = "Signal Engine Enables")]
        public bool EnableLecLong { get; set; }

        [Parameter("Enable LEC Short Signal", DefaultValue = false, Group = "Signal Engine Enables")]
        public bool EnableLecShort { get; set; }

        [Parameter("Enable Displacement Long Signal", DefaultValue = false, Group = "Signal Engine Enables")]
        public bool EnableDispLong { get; set; }

        [Parameter("Enable Displacement Short Signal", DefaultValue = false, Group = "Signal Engine Enables")]
        public bool EnableDispShort { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  Parameters — LEC Settings
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Enable H1 LEC", DefaultValue = true, Group = "LEC Settings")]
        public bool LecEnableH1 { get; set; }

        [Parameter("Enable H4 LEC", DefaultValue = false, Group = "LEC Settings")]
        public bool LecEnableH4 { get; set; }

        [Parameter("Enable Current TF LEC", DefaultValue = false, Group = "LEC Settings")]
        public bool LecEnableCurrent { get; set; }

        [Parameter("Apply Stop Hunt Wick Filter", DefaultValue = true, Group = "LEC Settings")]
        public bool LecFilterLiquidity { get; set; }

        [Parameter("Apply Close Filter", DefaultValue = true, Group = "LEC Settings")]
        public bool LecFilterClose { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  Parameters — Displacement Settings
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Require FVG", DefaultValue = true, Group = "Displacement Settings")]
        public bool DispRequireFvg { get; set; }

        [Parameter("Displacement Type", DefaultValue = DisplacementType.OpenToClose, Group = "Displacement Settings")]
        public DisplacementType DispType { get; set; }

        [Parameter("Displacement Length", DefaultValue = 100, MinValue = 1, Group = "Displacement Settings")]
        public int DispStdLen { get; set; }

        [Parameter("Displacement Strength", DefaultValue = 2, MinValue = 0, Group = "Displacement Settings")]
        public int DispStdX { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  Parameters — BSL & SSL
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Pivot Left", DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotLeft { get; set; }

        [Parameter("Pivot Right", DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotRight { get; set; }

        [Parameter("Long Stop Loss Source", DefaultValue = LongStopLossSource.Ssl, Group = "BSL & SSL")]
        public LongStopLossSource LongSlSource { get; set; }

        [Parameter("Short Stop Loss Source", DefaultValue = ShortStopLossSource.Bsl, Group = "BSL & SSL")]
        public ShortStopLossSource ShortSlSource { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  Parameters — Risk Management
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Risk Per Trade (%)", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 100.0, Group = "Risk Management")]
        public double RiskPercent { get; set; }

        [Parameter("Risk:Reward Ratio", DefaultValue = 2.0, MinValue = 0.1, Step = 0.1, Group = "Risk Management")]
        public double RiskRewardRatio { get; set; }

        [Parameter("Max Simultaneous Positions", DefaultValue = 3, MinValue = 1, MaxValue = 100, Group = "Risk Management")]
        public int MaxOpenPositions { get; set; }

        [Parameter("Min SL Distance (pips)", DefaultValue = 3.0, MinValue = 0.1, Group = "Risk Management")]
        public double MinSlPips { get; set; }

        [Parameter("Max SL Distance (pips)", DefaultValue = 500.0, MinValue = 1.0, Group = "Risk Management")]
        public double MaxSlPips { get; set; }

        [Parameter("Instance Name", DefaultValue = "LED_OBImb_cBot", Group = "Risk Management")]
        public string InstanceName { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  Parameters — OB Imbalance Filter
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Enable OB Imbalance Filter", DefaultValue = false, Group = "OB Imbalance Filter - General")]
        public bool EnableImbFilter { get; set; }

        [Parameter("Touch-to-Signal Window (bars)", DefaultValue = 10, MinValue = 1, Group = "OB Imbalance Filter - General")]
        public int ImbLookbackBars { get; set; }

        [Parameter("Use Unmitigated OB Logic", DefaultValue = true, Group = "OB Imbalance Filter - General")]
        public bool ImbUseUnmitigated { get; set; }

        [Parameter("Use Mitigated OB Logic", DefaultValue = false, Group = "OB Imbalance Filter - General")]
        public bool ImbUseMitigated { get; set; }

        [Parameter("Mitigation Method", DefaultValue = ImbMitigationMethod.Wick, Group = "OB Imbalance Filter - Detection")]
        public ImbMitigationMethod ImbMitigationMethodInput { get; set; }

        [Parameter("Min Imbalance Size (ATR Mult)", DefaultValue = 0.5, Step = 0.1, Group = "OB Imbalance Filter - Detection")]
        public double ImbFvgThreshold { get; set; }

        [Parameter("Enable TF1", DefaultValue = true, Group = "OB Imbalance Filter - Timeframes")]
        public bool ImbEnableTf1 { get; set; }

        [Parameter("TF1", DefaultValue = "Minute15", Group = "OB Imbalance Filter - Timeframes")]
        public TimeFrame ImbTf1 { get; set; }

        [Parameter("Enable TF2", DefaultValue = true, Group = "OB Imbalance Filter - Timeframes")]
        public bool ImbEnableTf2 { get; set; }

        [Parameter("TF2", DefaultValue = "Minute30", Group = "OB Imbalance Filter - Timeframes")]
        public TimeFrame ImbTf2 { get; set; }

        [Parameter("Enable TF3", DefaultValue = true, Group = "OB Imbalance Filter - Timeframes")]
        public bool ImbEnableTf3 { get; set; }

        [Parameter("TF3", DefaultValue = "Hour", Group = "OB Imbalance Filter - Timeframes")]
        public TimeFrame ImbTf3 { get; set; }

        [Parameter("Enable TF4", DefaultValue = true, Group = "OB Imbalance Filter - Timeframes")]
        public bool ImbEnableTf4 { get; set; }

        [Parameter("TF4", DefaultValue = "Hour4", Group = "OB Imbalance Filter - Timeframes")]
        public TimeFrame ImbTf4 { get; set; }

        [Parameter("Max Zones Per TF", DefaultValue = 50, MinValue = 1, MaxValue = 500, Group = "OB Imbalance Filter - Limits")]
        public int ImbMaxZonesPerTf { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  Constants
        // ═════════════════════════════════════════════════════════════════════

        private const int MaxBslPivots = 10;

        // ═════════════════════════════════════════════════════════════════════
        //  Fields — BSL/SSL
        // ═════════════════════════════════════════════════════════════════════

        private readonly LinkedList<BslPivot> _bslPivots = new LinkedList<BslPivot>();
        private readonly LinkedList<BslPool> _bslBuysidePools = new LinkedList<BslPool>();
        private readonly LinkedList<BslPool> _bslSellsidePools = new LinkedList<BslPool>();

        private double _bslCurrentBsl = double.NaN;
        private double _bslCurrentSsl = double.NaN;

        // ═════════════════════════════════════════════════════════════════════
        //  Fields — LEC / Displacement
        // ═════════════════════════════════════════════════════════════════════

        private Bars _h1Bars;
        private Bars _h4Bars;

        // ═════════════════════════════════════════════════════════════════════
        //  Fields — OB Imbalance
        // ═════════════════════════════════════════════════════════════════════

        private readonly List<ImbTfState> _imbStates = new List<ImbTfState>();

        // ═════════════════════════════════════════════════════════════════════
        //  Fields — cBot state
        // ═════════════════════════════════════════════════════════════════════

        private int _lastProcessed = -1;
        private int _lastLongSignalBar = -1;
        private int _lastShortSignalBar = -1;

        // ═════════════════════════════════════════════════════════════════════
        //  Lifecycle
        // ═════════════════════════════════════════════════════════════════════

        protected override void OnStart()
        {
            if (LecEnableH1 && (EnableLecLong || EnableLecShort))
                _h1Bars = MarketData.GetBars(TimeFrame.Hour);

            if (LecEnableH4 && (EnableLecLong || EnableLecShort))
                _h4Bars = MarketData.GetBars(TimeFrame.Hour4);

            if (EnableImbFilter)
            {
                ImbReg(ImbEnableTf1, ImbTf1, "TF1");
                ImbReg(ImbEnableTf2, ImbTf2, "TF2");
                ImbReg(ImbEnableTf3, ImbTf3, "TF3");
                ImbReg(ImbEnableTf4, ImbTf4, "TF4");
                Print("IMB ON Window={0} Unmitigated={1} Mitigated={2} Threshold={3} Method={4}",
                    ImbLookbackBars, ImbUseUnmitigated, ImbUseMitigated, ImbFvgThreshold, ImbMitigationMethodInput);
            }

            Print("Liquidity_Engulfing_and_Displacement_OBImbalance_cBot started. " +
                  "LEC Long={0} Short={1} (H1={2} H4={3} Cur={4}) | " +
                  "Disp Long={5} Short={6} (FVG={7} Len={8} Str={9}) | " +
                  "LongSL={10} ShortSL={11} | MaxPos={12} Risk={13}% RR={14} | " +
                  "IMB={15}",
                  EnableLecLong, EnableLecShort, LecEnableH1, LecEnableH4, LecEnableCurrent,
                  EnableDispLong, EnableDispShort, DispRequireFvg, DispStdLen, DispStdX,
                  LongSlSource, ShortSlSource, MaxOpenPositions, RiskPercent, RiskRewardRatio,
                  EnableImbFilter);
        }

        protected override void OnStop()
        {
            Print("Liquidity_Engulfing_and_Displacement_OBImbalance_cBot stopped.");
        }

        // ═════════════════════════════════════════════════════════════════════
        //  OnBar
        // ═════════════════════════════════════════════════════════════════════

        protected override void OnBar()
        {
            int signalBar = Bars.Count - 2;

            // Catch-up loop — process all unprocessed bars up to signalBar
            for (int i = _lastProcessed + 1; i <= signalBar; i++)
            {
                RunBslSsl(i);
                if (EnableImbFilter) RunImb(i);
            }
            _lastProcessed = signalBar;

            if (signalBar < Math.Max(PivotLeft + PivotRight + 1, 3))
                return;

            bool lecLong = false;
            bool lecShort = false;
            if (EnableLecLong || EnableLecShort)
                DetectLecSignal(signalBar, out lecLong, out lecShort);

            bool dispLong = false;
            bool dispShort = false;
            if (EnableDispLong || EnableDispShort)
                DetectDisplacementSignal(signalBar, out dispLong, out dispShort);

            bool isLong  = (EnableLecLong  && lecLong)  || (EnableDispLong  && dispLong);
            bool isShort = (EnableLecShort && lecShort) || (EnableDispShort && dispShort);

            if (!isLong && !isShort)
                return;

            int openCount = Positions.FindAll(InstanceName, SymbolName).Length;
            if (openCount >= MaxOpenPositions)
            {
                Print("Bar {0}: max positions ({1}) reached.", signalBar, MaxOpenPositions);
                return;
            }

            if (isLong && _lastLongSignalBar != signalBar)
            {
                _lastLongSignalBar = signalBar;
                if (ChkImb(true, signalBar))
                    TryEnterLong(signalBar);
            }

            openCount = Positions.FindAll(InstanceName, SymbolName).Length;
            if (openCount >= MaxOpenPositions)
                return;

            if (isShort && _lastShortSignalBar != signalBar)
            {
                _lastShortSignalBar = signalBar;
                if (ChkImb(false, signalBar))
                    TryEnterShort(signalBar);
            }
        }

        // ═════════════════════════════════════════════════════════════════════
        //  LEC signal detection
        // ═════════════════════════════════════════════════════════════════════

        private void DetectLecSignal(int signalBar, out bool bullOut, out bool bearOut)
        {
            bullOut = false;
            bearOut = false;

            if (signalBar < 1)
                return;

            if (LecEnableH1 && _h1Bars != null)
                CheckLecOnTf(_h1Bars, signalBar, ref bullOut, ref bearOut);

            if (LecEnableH4 && _h4Bars != null)
                CheckLecOnTf(_h4Bars, signalBar, ref bullOut, ref bearOut);

            if (LecEnableCurrent)
                CheckLecOnTf(Bars, signalBar, ref bullOut, ref bearOut);
        }

        private void CheckLecOnTf(Bars sourceBars, int signalBar, ref bool bull, ref bool bear)
        {
            if (sourceBars == null || sourceBars.Count < 2)
                return;

            int idxNow  = FindBarIndexAtOrBefore(sourceBars, Bars.OpenTimes[signalBar]);
            int idxPrev = FindBarIndexAtOrBefore(sourceBars, Bars.OpenTimes[signalBar - 1]);

            if (idxNow < 1 || idxPrev < 1)
                return;

            bool bullNow, bearNow, bullPrev, bearPrev;
            EvaluateLec(sourceBars, idxNow,  out bullNow,  out bearNow);
            EvaluateLec(sourceBars, idxPrev, out bullPrev, out bearPrev);

            if (bullNow && !bullPrev) bull = true;
            if (bearNow && !bearPrev) bear = true;
        }

        private void EvaluateLec(Bars b, int index, out bool bullEngulf, out bool bearEngulf)
        {
            bullEngulf = false;
            bearEngulf = false;

            if (b == null || index < 1 || index >= b.Count)
                return;

            double priorOpen  = b.OpenPrices[index - 1];
            double priorClose = b.ClosePrices[index - 1];
            double curOpen    = b.OpenPrices[index];
            double curClose   = b.ClosePrices[index];

            bullEngulf = (curOpen <= priorClose) && (curOpen < priorOpen) && (curClose > priorOpen);
            bearEngulf = (curOpen >= priorClose) && (curOpen > priorOpen) && (curClose < priorOpen);

            if (LecFilterLiquidity)
            {
                bullEngulf = bullEngulf && b.LowPrices[index]  <= b.LowPrices[index - 1];
                bearEngulf = bearEngulf && b.HighPrices[index] >= b.HighPrices[index - 1];
            }

            if (LecFilterClose)
            {
                bullEngulf = bullEngulf && b.ClosePrices[index] >= b.HighPrices[index - 1];
                bearEngulf = bearEngulf && b.ClosePrices[index] <= b.LowPrices[index - 1];
            }
        }

        // ═════════════════════════════════════════════════════════════════════
        //  Displacement signal detection
        // ═════════════════════════════════════════════════════════════════════

        private void DetectDisplacementSignal(int signalBar, out bool bullOut, out bool bearOut)
        {
            bullOut = false;
            bearOut = false;

            bool isDisplacement;
            int dispBar;

            if (DispRequireFvg)
            {
                if (signalBar < 2)
                    return;
                isDisplacement = IsDisplacementWithFvg(signalBar);
                dispBar = signalBar - 1;
            }
            else
            {
                isDisplacement = IsDisplacementNoFvg(signalBar);
                dispBar = signalBar;
            }

            if (!isDisplacement)
                return;

            bool isBullish = Bars.ClosePrices[dispBar] > Bars.OpenPrices[dispBar];
            bullOut = isBullish;
            bearOut = !isBullish;
        }

        private bool IsDisplacementWithFvg(int index)
        {
            if (index < 2) return false;

            double prevRange = GetCandleRange(index - 1);
            double prevStd   = GetStdDev(index - 1);

            if (double.IsNaN(prevStd) || prevStd <= 0) return false;

            bool fvg = Bars.ClosePrices[index - 1] > Bars.OpenPrices[index - 1]
                ? Bars.HighPrices[index - 2] < Bars.LowPrices[index]
                : Bars.LowPrices[index - 2]  > Bars.HighPrices[index];

            return prevRange > prevStd && fvg;
        }

        private bool IsDisplacementNoFvg(int index)
        {
            double std = GetStdDev(index);
            return !double.IsNaN(std) && std > 0 && GetCandleRange(index) > std;
        }

        private double GetCandleRange(int index)
        {
            if (index < 0 || index >= Bars.Count) return double.NaN;
            return DispType == DisplacementType.OpenToClose
                ? Math.Abs(Bars.OpenPrices[index] - Bars.ClosePrices[index])
                : Bars.HighPrices[index] - Bars.LowPrices[index];
        }

        private double GetStdDev(int index)
        {
            if (index < 0 || DispStdLen <= 0) return double.NaN;
            int start = index - DispStdLen + 1;
            if (start < 0) return double.NaN;

            double sum = 0.0;
            for (int i = start; i <= index; i++) sum += GetCandleRange(i);
            double mean = sum / DispStdLen;

            double varSum = 0.0;
            for (int i = start; i <= index; i++)
            {
                double diff = GetCandleRange(i) - mean;
                varSum += diff * diff;
            }
            return Math.Sqrt(varSum / DispStdLen) * DispStdX;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  Trade entry helpers
        // ═════════════════════════════════════════════════════════════════════

        private void TryEnterLong(int signalBar)
        {
            double entry = Symbol.Ask;
            double slAnchor; string slAnchorName;

            if (!TryGetLongStopAnchor(signalBar, out slAnchor, out slAnchorName))
            {
                Print("Bar {0}: LONG skipped – stop loss anchor unavailable for mode {1}.", signalBar, LongSlSource);
                return;
            }

            if (slAnchor >= entry)
            {
                Print("Bar {0}: LONG skipped – {1} {2:F5} not below entry {3:F5}.", signalBar, slAnchorName, slAnchor, entry);
                return;
            }

            double slPips = (entry - slAnchor) / Symbol.PipSize;
            if (!ValidateSlPips(signalBar, "LONG", slPips)) return;

            double volume = GetRiskVolume(Account.Equity * (RiskPercent / 100.0), slPips);
            if (volume <= 0) { Print("Bar {0}: LONG skipped – volume is 0.", signalBar); return; }

            double slPrice = Math.Round(entry - slPips * Symbol.PipSize, Symbol.Digits);
            double tpPrice = Math.Round(entry + slPips * RiskRewardRatio * Symbol.PipSize, Symbol.Digits);

            Print("Bar {0}: LONG  | Entry={1:F5} | {2}={3:F5} | SL={4:F5}({5:F1}p) | TP={6:F5}({7:F1}p) | Vol={8}",
                  signalBar, entry, slAnchorName, slAnchor, slPrice, slPips, tpPrice, slPips * RiskRewardRatio, volume);

            var result = ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, InstanceName, null, null);
            if (result.IsSuccessful)
                ModifyPosition(result.Position, slPrice, tpPrice);
            else
                Print("Bar {0}: LONG order failed – {1}", signalBar, result.Error);
        }

        private void TryEnterShort(int signalBar)
        {
            double entry = Symbol.Bid;
            double slAnchor; string slAnchorName;

            if (!TryGetShortStopAnchor(signalBar, out slAnchor, out slAnchorName))
            {
                Print("Bar {0}: SHORT skipped – stop loss anchor unavailable for mode {1}.", signalBar, ShortSlSource);
                return;
            }

            if (slAnchor <= entry)
            {
                Print("Bar {0}: SHORT skipped – {1} {2:F5} not above entry {3:F5}.", signalBar, slAnchorName, slAnchor, entry);
                return;
            }

            double slPips = (slAnchor - entry) / Symbol.PipSize;
            if (!ValidateSlPips(signalBar, "SHORT", slPips)) return;

            double volume = GetRiskVolume(Account.Equity * (RiskPercent / 100.0), slPips);
            if (volume <= 0) { Print("Bar {0}: SHORT skipped – volume is 0.", signalBar); return; }

            double slPrice = Math.Round(entry + slPips * Symbol.PipSize, Symbol.Digits);
            double tpPrice = Math.Round(entry - slPips * RiskRewardRatio * Symbol.PipSize, Symbol.Digits);

            Print("Bar {0}: SHORT | Entry={1:F5} | {2}={3:F5} | SL={4:F5}({5:F1}p) | TP={6:F5}({7:F1}p) | Vol={8}",
                  signalBar, entry, slAnchorName, slAnchor, slPrice, slPips, tpPrice, slPips * RiskRewardRatio, volume);

            var result = ExecuteMarketOrder(TradeType.Sell, SymbolName, volume, InstanceName, null, null);
            if (result.IsSuccessful)
                ModifyPosition(result.Position, slPrice, tpPrice);
            else
                Print("Bar {0}: SHORT order failed – {1}", signalBar, result.Error);
        }

        private bool TryGetLongStopAnchor(int signalBar, out double anchor, out string anchorName)
        {
            anchor = double.NaN; anchorName = string.Empty;
            switch (LongSlSource)
            {
                case LongStopLossSource.Ssl:
                    anchor = _bslCurrentSsl; anchorName = "SSL";
                    return !double.IsNaN(anchor) && anchor > 0;
                case LongStopLossSource.CandleMinus1Low:
                    if (signalBar < 0 || signalBar >= Bars.Count) return false;
                    anchor = Bars.LowPrices[signalBar]; anchorName = "Candle-1 Low";
                    return !double.IsNaN(anchor) && anchor > 0;
                default: return false;
            }
        }

        private bool TryGetShortStopAnchor(int signalBar, out double anchor, out string anchorName)
        {
            anchor = double.NaN; anchorName = string.Empty;
            switch (ShortSlSource)
            {
                case ShortStopLossSource.Bsl:
                    anchor = _bslCurrentBsl; anchorName = "BSL";
                    return !double.IsNaN(anchor) && anchor > 0;
                case ShortStopLossSource.CandleMinus1High:
                    if (signalBar < 0 || signalBar >= Bars.Count) return false;
                    anchor = Bars.HighPrices[signalBar]; anchorName = "Candle-1 High";
                    return !double.IsNaN(anchor) && anchor > 0;
                default: return false;
            }
        }

        private bool ValidateSlPips(int signalBar, string direction, double slPips)
        {
            if (slPips < MinSlPips) { Print("Bar {0}: {1} skipped – SL {2:F1}p < min {3:F1}p.", signalBar, direction, slPips, MinSlPips); return false; }
            if (slPips > MaxSlPips) { Print("Bar {0}: {1} skipped – SL {2:F1}p > max {3:F1}p.", signalBar, direction, slPips, MaxSlPips); return false; }
            return true;
        }

        private double GetRiskVolume(double riskAmount, double slPips)
        {
            if (slPips <= 0) return 0;
            double volume = Symbol.NormalizeVolumeInUnits(Symbol.VolumeForFixedRisk(riskAmount, slPips), RoundingMode.Down);
            if (volume < Symbol.VolumeInUnitsMin) return 0;
            if (volume > Symbol.VolumeInUnitsMax) volume = Symbol.VolumeInUnitsMax;
            return volume;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  BSL/SSL engine
        // ═════════════════════════════════════════════════════════════════════

        private void RunBslSsl(int index)
        {
            BslDetectAndStoreConfirmedPivots(index);
            BslAddPoolFromNewPivot(index);
            BslClearMitigated(index);
            _bslCurrentBsl = _bslBuysidePools.First  != null ? _bslBuysidePools.First.Value.Price  : double.NaN;
            _bslCurrentSsl = _bslSellsidePools.First != null ? _bslSellsidePools.First.Value.Price : double.NaN;
        }

        private void BslDetectAndStoreConfirmedPivots(int currentIndex)
        {
            int pivotIndex = currentIndex - PivotRight;
            if (pivotIndex <= 0) return;

            int leftStart = pivotIndex - PivotLeft;
            int rightEnd  = pivotIndex + PivotRight;
            if (leftStart < 0 || rightEnd >= Bars.Count) return;

            double candidateHigh = Bars.HighPrices[pivotIndex];
            double candidateLow  = Bars.LowPrices[pivotIndex];

            if (BslIsPivotHigh(candidateHigh, leftStart, rightEnd))
                BslUnshiftPivot(new BslPivot { Price = candidateHigh, BarIndex = pivotIndex, Type = 1 });

            if (BslIsPivotLow(candidateLow, leftStart, rightEnd))
                BslUnshiftPivot(new BslPivot { Price = candidateLow, BarIndex = pivotIndex, Type = -1 });
        }

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

        private void BslUnshiftPivot(BslPivot p)
        {
            if (_bslPivots.First != null)
            {
                var f = _bslPivots.First.Value;
                if (f.BarIndex == p.BarIndex && f.Type == p.Type && Math.Abs(f.Price - p.Price) < Symbol.PipSize * 0.1)
                    return;
            }
            _bslPivots.AddFirst(p);
            while (_bslPivots.Count > MaxBslPivots) _bslPivots.RemoveLast();
        }

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

        private void BslClearMitigated(int index)
        {
            var node = _bslSellsidePools.First;
            while (node != null) { var next = node.Next; if (Bars.LowPrices[index]  <= node.Value.Price) _bslSellsidePools.Remove(node); node = next; }

            node = _bslBuysidePools.First;
            while (node != null) { var next = node.Next; if (Bars.HighPrices[index] >= node.Value.Price) _bslBuysidePools.Remove(node); node = next; }
        }

        // ═════════════════════════════════════════════════════════════════════
        //  OB Imbalance Filter engine
        // ═════════════════════════════════════════════════════════════════════

        private void ImbReg(bool en, TimeFrame tf, string label)
        {
            if (!en) return;
            var bars = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf);
            var atr  = Indicators.AverageTrueRange(bars, 14, MovingAverageType.Simple);
            _imbStates.Add(new ImbTfState { Label = label, Bars = bars, Atr = atr, MaxCount = ImbMaxZonesPerTf });
        }

        private void RunImb(int ci)
        {
            var now = Bars.OpenTimes[ci];
            foreach (var tf in _imbStates)
            {
                if (tf.Bars == null || tf.Bars.Count < 4) { ImbUpdate(tf, ci); continue; }

                int htfIdx   = MtfFOB(tf.Bars, now);
                int closedIdx = htfIdx - 1;
                if (closedIdx < 2) { ImbUpdate(tf, ci); continue; }

                double h2 = tf.Bars.HighPrices[closedIdx - 2], l2 = tf.Bars.LowPrices[closedIdx - 2];
                double l0 = tf.Bars.LowPrices[closedIdx],      h0 = tf.Bars.HighPrices[closedIdx];
                DateTime seedTime = tf.Bars.OpenTimes[closedIdx - 2];

                double atrRef = tf.Atr.Result[Math.Max(0, closedIdx - 1)];
                if (!double.IsNaN(atrRef) && atrRef > 0)
                {
                    bool isBull = (l0 - h2) > atrRef * ImbFvgThreshold;
                    bool isBear = (l2 - h0) > atrRef * ImbFvgThreshold;

                    if (seedTime != tf.LastCreatedSeedTime)
                    {
                        if (isBull)      { ImbAdd(tf, true,  h2, l2); tf.LastCreatedSeedTime = seedTime; }
                        else if (isBear) { ImbAdd(tf, false, h2, l2); tf.LastCreatedSeedTime = seedTime; }
                    }
                }
                ImbUpdate(tf, ci);
            }
        }

        private void ImbAdd(ImbTfState tf, bool isBull, double high, double low)
        {
            var z = new ImbObZone
            {
                IsBullish = isBull,
                State     = ImbZoneState.Active,
                Top       = Math.Max(high, low),
                Bottom    = Math.Min(high, low)
            };
            tf.Zones.Add(z);
            if (tf.Zones.Count > tf.MaxCount) tf.Zones.RemoveAt(0);
        }

        private void ImbUpdate(ImbTfState tf, int ci)
        {
            double cl = Bars.ClosePrices[ci], lo = Bars.LowPrices[ci], hi = Bars.HighPrices[ci];
            bool useWick = ImbMitigationMethodInput == ImbMitigationMethod.Wick;

            for (int i = 0; i < tf.Zones.Count; i++)
            {
                var z = tf.Zones[i];
                if (z.State == ImbZoneState.Invalidated) continue;

                // Invalidation: close through the far side
                if ( z.IsBullish && cl < z.Bottom) { z.State = ImbZoneState.Invalidated; continue; }
                if (!z.IsBullish && cl > z.Top)    { z.State = ImbZoneState.Invalidated; continue; }

                // Touch detection
                bool touched = z.IsBullish
                    ? (useWick ? lo <= z.Top    : cl <= z.Top)
                    : (useWick ? hi >= z.Bottom : cl >= z.Bottom);

                if (touched)
                {
                    z.LastTouchBar = ci;
                    if (z.FirstTouchBar < 0) z.FirstTouchBar = ci;
                    if (z.State == ImbZoneState.Active) z.State = ImbZoneState.Mitigated;
                }
            }
        }

        private bool ChkImb(bool isLong, int bar)
        {
            if (!EnableImbFilter)                        return true;
            if (_imbStates.Count == 0)                   return true;
            if (!ImbUseUnmitigated && !ImbUseMitigated)  return true;

            foreach (var tf in _imbStates)
            {
                for (int i = tf.Zones.Count - 1; i >= 0; i--)
                {
                    var z = tf.Zones[i];
                    if (z.IsBullish != isLong)               continue;
                    if (z.State == ImbZoneState.Invalidated) continue;
                    // A zone re-arms whenever price returns to it AFTER the last
                    // fired trade (LastTouchBar > LastFiredBar).  This allows the
                    // same zone to trigger multiple trades across distinct touches
                    // while still preventing duplicate fires on the same touch.

                    if (ImbUseUnmitigated
                        && z.State == ImbZoneState.Mitigated
                        && z.FirstTouchBar >= 0
                        && z.FirstTouchBar > z.LastFiredBar)
                    {
                        int d = bar - z.FirstTouchBar;
                        if (d >= 0 && d <= ImbLookbackBars)
                        {
                            z.LastFiredBar = bar;
                            Print("[IMB PASS Unmitigated] {0} bar={1} TF={2} [B={3:F5} T={4:F5}] ft={5} d={6}",
                                isLong ? "Long" : "Short", bar, tf.Label, z.Bottom, z.Top, z.FirstTouchBar, d);
                            return true;
                        }
                    }

                    if (ImbUseMitigated
                        && z.LastTouchBar >= 0
                        && z.LastTouchBar > z.LastFiredBar)
                    {
                        int d = bar - z.LastTouchBar;
                        if (d >= 0 && d <= ImbLookbackBars)
                        {
                            z.LastFiredBar = bar;
                            Print("[IMB PASS Mitigated] {0} bar={1} TF={2} [B={3:F5} T={4:F5}] lt={5} d={6}",
                                isLong ? "Long" : "Short", bar, tf.Label, z.Bottom, z.Top, z.LastTouchBar, d);
                            return true;
                        }
                    }
                }
            }

            Print("[IMB BLOCKED] {0} bar={1} window={2} unmit={3} mit={4}",
                isLong ? "Long" : "Short", bar, ImbLookbackBars, ImbUseUnmitigated, ImbUseMitigated);
            return false;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  MTF utility — binary search for bar at or before a given time
        // ═════════════════════════════════════════════════════════════════════

        private static int MtfFOB(Bars bars, DateTime time)
        {
            int lo = 0, hi = bars.Count - 1;
            while (lo <= hi)
            {
                int mid = (lo + hi) / 2;
                if      (bars.OpenTimes[mid] == time) return mid;
                else if (bars.OpenTimes[mid] <  time) lo = mid + 1;
                else                                   hi = mid - 1;
            }
            return hi;   // last bar whose open <= time
        }

        // ═════════════════════════════════════════════════════════════════════
        //  Utility — bar search
        // ═════════════════════════════════════════════════════════════════════

        private int FindBarIndexAtOrBefore(Bars bars, DateTime time)
        {
            int idx = bars.OpenTimes.GetIndexByTime(time);
            if (idx >= 0) return idx;
            for (int i = bars.Count - 1; i >= 0; i--)
                if (bars.OpenTimes[i] <= time) return i;
            return -1;
        }
    }
}
