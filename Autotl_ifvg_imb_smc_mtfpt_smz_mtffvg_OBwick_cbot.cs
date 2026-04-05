// =============================================================================
// Autotl_ifvg_smc_mtfpt_smz_mtffvg_OBwick_Imb_filter_cbot
// =============================================================================
// Base    : AutoTL_IFVG_SMC_Orderblock_MTFpt_SMZ_MTFFvg_OBwick_filter_cBot
// Added   : OB Imbalance filter — logic ported from "Order Blocks & Imbalance MTF.cs"
//
// OB Imbalance Filter:
//   Seed bar (htfIdx-2) forms an OB zone (Top=High, Bottom=Low of seed bar).
//   Bull OB: Low[htfIdx] - High[htfIdx-2] > ATR * threshold  (FVG gap up)
//   Bear OB: Low[htfIdx-2] - High[htfIdx] > ATR * threshold  (FVG gap down)
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
using cAlgo.API.Internals;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class Autotl__smc_mtfpt_smz_mtffvg_OBwick_Imb_filter_cbot : Robot
    {
        // ENUMS
        public enum ObFilter          { Atr, CumulativeMeanRange }
        public enum MitigationMode    { Close, HighLow }
        public enum MtfLogicMode      { OR, AND }
        public enum SmzLogicMode      { OR, AND }
        public enum FvgMitigationMode { Normal = 1, Dynamic = 2, None = 3, Half = 4 }
        public enum FvgMitigationType { Wicks = 1, Body = 2 }
        public enum ObWickDetectionMethod { Body, Wicks }
        public enum ObWickMitigationMode  { Normal, Dynamic, None, Half }
        public enum ObWickMitigationType  { Wicks, Body }
        public enum ImbMitigationMethod   { Close, Wick }
        public enum ImbZoneState          { Active, Mitigated, Invalidated }
        public enum LongStopLossSource    { Ssl, CandleMinus1Low  }
        public enum ShortStopLossSource   { Bsl, CandleMinus1High }

        // INNER TYPES
        private sealed class AtlTlState
        {
            public bool   PermitSet = false, PermitSetPrev = false;
            public int    LastAnchorX0 = 0, ActiveX0 = 0, ActiveX1 = 0;
            public double ActiveY0 = 0.0, ActiveY1 = 0.0;
        }

        private sealed class BslPivot { public double Price; public int BarIndex, Type; }
        private sealed class BslPool  { public double Price; public int PivotIndex; }

        private sealed class SmcObRecord
        {
            public int Index, StructureBreakIndex;
            public double Top, Bottom;
            public bool Bullish, Internal;
            public DateTime Time;
        }

        private sealed class MtfTfState
        {
            public Bars TfBars; public int PivotLen, TfMinutes, LastProcessedTfBar = -1;
            public bool IsLowerTf, CurrentTrend, HasTrend, LastEventWasBos;
            public bool AllowBullChoch = true, AllowBullBos = true, AllowBearChoch = true, AllowBearBos = true;
            public double LastPivotHigh = double.NaN, LastPivotLow = double.NaN;
            public double LastBrokenHigh = double.NaN, LastBrokenLow = double.NaN;
            public DateTime PivotHighTime = DateTime.MinValue, PivotLowTime = DateTime.MinValue;
        }

        private sealed class FvgFilterZone
        {
            public bool IsBull, IsMitigated;
            public double Top, Bottom;
            public ChartRectangle Rect;
            public string RectId;
        }

        private sealed class ObWickZone { public bool IsBull, IsMitigated; public double Top, Bottom; }

        private sealed class ObWickTfState
        {
            public Bars TfBars; public string Label; public int MaxCount, LastDetectedSrcIdx = -1;
            public List<ObWickZone> Bulls = new List<ObWickZone>();
            public List<ObWickZone> Bears = new List<ObWickZone>();
        }

        // ImbObZone:
        //   FirstTouchBar = chart bar of FIRST touch (-1 = untouched)
        //   LastTouchBar  = chart bar of LAST  touch (-1 = untouched)
        //   Active → Mitigated on first touch; Invalidated if close crosses far side
        private sealed class ImbObZone
        {
            public bool IsBullish;
            public ImbZoneState State = ImbZoneState.Active;
            public double Top, Bottom;
            public int FirstTouchBar = -1, LastTouchBar = -1;
        }

        // LastCreatedSeedTime: DateTime guard matching indicator's seedTime check
        private sealed class ImbTfState
        {
            public string Label; public Bars Bars; public AverageTrueRange Atr; public int MaxCount;
            public DateTime LastCreatedSeedTime = DateTime.MinValue;
            public List<ImbObZone> Zones = new List<ImbObZone>();
        }

        // ═══ PARAMETERS — Signal Engine Enables ═══════════════════════════════

        [Parameter("Enable ATL Long Signal",   DefaultValue = false, Group = "Signal Engine Enables")]
        public bool EnableAtlLong { get; set; }
        [Parameter("Enable ATL Short Signal",  DefaultValue = false, Group = "Signal Engine Enables")]
        public bool EnableAtlShort { get; set; }
        [Parameter("Enable IFVG Long Signal",  DefaultValue = true, Group = "Signal Engine Enables")]
        public bool EnableIfvgLong { get; set; }
        [Parameter("Enable IFVG Short Signal", DefaultValue = true, Group = "Signal Engine Enables")]
        public bool EnableIfvgShort { get; set; }

        // ═══ PARAMETERS — IFVG Signal Engine ══════════════════════════════════

        [Parameter("FVG Search Lookback (bars)", DefaultValue = 15,  MinValue = 1,   Group = "IFVG Signal Engine")]
        public int IfvgGapBars { get; set; }
        [Parameter("Min FVG Size (pips)",        DefaultValue = 0.0, MinValue = 0.0, Group = "IFVG Signal Engine")]
        public double IfvgMinFvgPips { get; set; }
        [Parameter("FVG Epsilon (price units)",  DefaultValue = 0.0, MinValue = 0.0, Group = "IFVG Signal Engine")]
        public double IfvgEpsilonPoints { get; set; }
        [Parameter("MA Period",                  DefaultValue = 21,  MinValue = 1,   Group = "IFVG Signal Engine")]
        public int IfvgMaPeriod { get; set; }
        [Parameter("MA Type (SMA / EMA)",        DefaultValue = "EMA",              Group = "IFVG Signal Engine")]
        public string IfvgMaType { get; set; }

        // ═══ PARAMETERS — ATL Zig Zag ═════════════════════════════════════════

        [Parameter("Pivot Period", DefaultValue = 5, MinValue = 1, Group = "ATL Zig Zag Logic")]
        public int PP { get; set; }

        // ═══ PARAMETERS — Long Signal Enables ═════════════════════════════════

        [Parameter("React Major External Up TL   -> Long", DefaultValue = false, Group = "Long Signal Enables")]
        public bool LongReact_MjExUp { get; set; }
        [Parameter("React Major Internal Up TL   -> Long", DefaultValue = false, Group = "Long Signal Enables")]
        public bool LongReact_MjInUp { get; set; }
        [Parameter("React Minor External Up TL   -> Long", DefaultValue = false, Group = "Long Signal Enables")]
        public bool LongReact_MnExUp { get; set; }
        [Parameter("React Minor Internal Up TL   -> Long", DefaultValue = false, Group = "Long Signal Enables")]
        public bool LongReact_MnInUp { get; set; }
        [Parameter("Break Major External Down TL -> Long", DefaultValue = false, Group = "Long Signal Enables")]
        public bool LongBreak_MjExDown { get; set; }
        [Parameter("Break Major Internal Down TL -> Long", DefaultValue = false, Group = "Long Signal Enables")]
        public bool LongBreak_MjInDown { get; set; }
        [Parameter("Break Minor External Down TL -> Long", DefaultValue = false, Group = "Long Signal Enables")]
        public bool LongBreak_MnExDown { get; set; }
        [Parameter("Break Minor Internal Down TL -> Long", DefaultValue = false, Group = "Long Signal Enables")]
        public bool LongBreak_MnInDown { get; set; }

        // ═══ PARAMETERS — Short Signal Enables ════════════════════════════════

        [Parameter("Break Major External Up TL   -> Short", DefaultValue = false, Group = "Short Signal Enables")]
        public bool ShortBreak_MjExUp { get; set; }
        [Parameter("Break Major Internal Up TL   -> Short", DefaultValue = false, Group = "Short Signal Enables")]
        public bool ShortBreak_MjInUp { get; set; }
        [Parameter("Break Minor External Up TL   -> Short", DefaultValue = false, Group = "Short Signal Enables")]
        public bool ShortBreak_MnExUp { get; set; }
        [Parameter("Break Minor Internal Up TL   -> Short", DefaultValue = false, Group = "Short Signal Enables")]
        public bool ShortBreak_MnInUp { get; set; }
        [Parameter("React Major External Down TL -> Short", DefaultValue = false, Group = "Short Signal Enables")]
        public bool ShortReact_MjExDown { get; set; }
        [Parameter("React Major Internal Down TL -> Short", DefaultValue = false, Group = "Short Signal Enables")]
        public bool ShortReact_MjInDown { get; set; }
        [Parameter("React Minor External Down TL -> Short", DefaultValue = false, Group = "Short Signal Enables")]
        public bool ShortReact_MnExDown { get; set; }
        [Parameter("React Minor Internal Down TL -> Short", DefaultValue = false, Group = "Short Signal Enables")]
        public bool ShortReact_MnInDown { get; set; }

        // ═══ PARAMETERS — BSL & SSL ═══════════════════════════════════════════

        [Parameter("Pivot Left",             DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotLeft { get; set; }
        [Parameter("Pivot Right",            DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotRight { get; set; }
        [Parameter("Long Stop Loss Source",  DefaultValue = LongStopLossSource.Ssl,  Group = "BSL & SSL")]
        public LongStopLossSource LongSlSource { get; set; }
        [Parameter("Short Stop Loss Source", DefaultValue = ShortStopLossSource.Bsl, Group = "BSL & SSL")]
        public ShortStopLossSource ShortSlSource { get; set; }

        // ═══ PARAMETERS — Risk Management ═════════════════════════════════════

        [Parameter("Risk % per trade",       DefaultValue = 1.0,   MinValue = 0.1, MaxValue = 100.0, Group = "Risk Management")]
        public double RiskPercent { get; set; }
        [Parameter("Risk:Reward Ratio",      DefaultValue = 2.0,   MinValue = 0.1, Step = 0.1, Group = "Risk Management")]
        public double RiskRewardRatio { get; set; }
        [Parameter("Max Open Positions",     DefaultValue = 3,     MinValue = 1, MaxValue = 100, Group = "Risk Management")]
        public int MaxOpenPositions { get; set; }
        [Parameter("Min SL Distance (pips)", DefaultValue = 3.0,   MinValue = 0.1, Group = "Risk Management")]
        public double MinSlPips { get; set; }
        [Parameter("Max SL Distance (pips)", DefaultValue = 500.0, MinValue = 1.0, Group = "Risk Management")]
        public double MaxSlPips { get; set; }
        [Parameter("SL Buffer (pips)",       DefaultValue = 0.0,   MinValue = 0.0, Step = 0.1, Group = "Risk Management")]
        public double SlBufferPips { get; set; }
        [Parameter("Instance Name", DefaultValue = "AutoTL_IFVG_SMC_MTFpt_SMZ_FVG_OBwick_Imb_cBot", Group = "Risk Management")]
        public string InstanceName { get; set; }

        // ═══ PARAMETERS — SMC Filter ══════════════════════════════════════════

        [Parameter("Swings Length",          DefaultValue = 50,  MinValue = 10, Group = "SMC Filter - Swing Structure")]
        public int SmcSwingsLengthInput { get; set; }
        [Parameter("Order Block Filter",     DefaultValue = ObFilter.Atr, Group = "SMC Filter - Order Blocks")]
        public ObFilter SmcOrderBlockFilterInput { get; set; }
        [Parameter("OB Filter ATR Period",   DefaultValue = 200, MinValue = 1, MaxValue = 500, Group = "SMC Filter - Order Blocks")]
        public int SmcObFilterAtrPeriod { get; set; }
        [Parameter("Order Block Mitigation", DefaultValue = MitigationMode.HighLow, Group = "SMC Filter - Order Blocks")]
        public MitigationMode SmcOrderBlockMitigationInput { get; set; }
        [Parameter("Enable Filter 1 (Internal OB)",     DefaultValue = false, Group = "Filter 1 - Internal OB")]
        public bool EnableFilter1 { get; set; }
        [Parameter("OB Touch Window - Internal (bars)", DefaultValue = 10, MinValue = 0, Group = "Filter 1 - Internal OB")]
        public int Filter1Lookback { get; set; }
        [Parameter("Enable Filter 2 (Swing OB)",        DefaultValue = false, Group = "Filter 2 - Swing OB")]
        public bool EnableFilter2 { get; set; }
        [Parameter("OB Touch Window - Swing (bars)",    DefaultValue = 10, MinValue = 0, Group = "Filter 2 - Swing OB")]
        public int Filter2Lookback { get; set; }
        [Parameter("Enable Min Bars From OB Origin",    DefaultValue = false, Group = "OB Quality Filters")]
        public bool EnableMinBarsFromOrigin { get; set; }
        [Parameter("Min Bars - Internal OB",            DefaultValue = 5, MinValue = 1, Group = "OB Quality Filters")]
        public int MinBarsFromOriginInternal { get; set; }
        [Parameter("Min Bars - Swing OB",               DefaultValue = 5, MinValue = 1, Group = "OB Quality Filters")]
        public int MinBarsFromOriginSwing { get; set; }
        [Parameter("Enable ATR Distance Filter",        DefaultValue = false, Group = "OB Quality Filters")]
        public bool EnableAtrDistanceFilter { get; set; }
        [Parameter("ATR Distance Multiplier",           DefaultValue = 1.0, MinValue = 0.1, Step = 0.1, Group = "OB Quality Filters")]
        public double AtrDistanceMultiplier { get; set; }

        // ═══ PARAMETERS — MTF Trend Filter ═══════════════════════════════════

        [Parameter("Enable MTF Trend Filter",   DefaultValue = false,           Group = "MTF Trend Filter - General")]
        public bool EnableMtfFilter { get; set; }
        [Parameter("Multi-TF Logic (OR / AND)", DefaultValue = MtfLogicMode.OR, Group = "MTF Trend Filter - General")]
        public MtfLogicMode MtfFilterLogic { get; set; }

        [Parameter("Enable TF1 Filter",     DefaultValue = false, Group = "MTF Trend Filter - TF1")] public bool EnableMtfTf1 { get; set; }
        [Parameter("TF1 Timeframe",         DefaultValue = "15",  Group = "MTF Trend Filter - TF1")] public string MtfTimeframe1 { get; set; }
        [Parameter("TF1 Pivot Strength",    DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter - TF1")] public int MtfPivotStrength1 { get; set; }
        [Parameter("TF1 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter - TF1")] public bool MtfIsLowerTf1 { get; set; }
        [Parameter("TF1 Allow Bull CHoCH",  DefaultValue = false, Group = "MTF Trend Filter - TF1")] public bool MtfAllowBullChoch1 { get; set; }
        [Parameter("TF1 Allow Bull BOS",    DefaultValue = false, Group = "MTF Trend Filter - TF1")] public bool MtfAllowBullBos1 { get; set; }
        [Parameter("TF1 Allow Bear CHoCH",  DefaultValue = false, Group = "MTF Trend Filter - TF1")] public bool MtfAllowBearChoch1 { get; set; }
        [Parameter("TF1 Allow Bear BOS",    DefaultValue = false, Group = "MTF Trend Filter - TF1")] public bool MtfAllowBearBos1 { get; set; }

        [Parameter("Enable TF2 Filter",     DefaultValue = false, Group = "MTF Trend Filter - TF2")] public bool EnableMtfTf2 { get; set; }
        [Parameter("TF2 Timeframe",         DefaultValue = "30",  Group = "MTF Trend Filter - TF2")] public string MtfTimeframe2 { get; set; }
        [Parameter("TF2 Pivot Strength",    DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter - TF2")] public int MtfPivotStrength2 { get; set; }
        [Parameter("TF2 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter - TF2")] public bool MtfIsLowerTf2 { get; set; }
        [Parameter("TF2 Allow Bull CHoCH",  DefaultValue = false, Group = "MTF Trend Filter - TF2")] public bool MtfAllowBullChoch2 { get; set; }
        [Parameter("TF2 Allow Bull BOS",    DefaultValue = false, Group = "MTF Trend Filter - TF2")] public bool MtfAllowBullBos2 { get; set; }
        [Parameter("TF2 Allow Bear CHoCH",  DefaultValue = false, Group = "MTF Trend Filter - TF2")] public bool MtfAllowBearChoch2 { get; set; }
        [Parameter("TF2 Allow Bear BOS",    DefaultValue = false, Group = "MTF Trend Filter - TF2")] public bool MtfAllowBearBos2 { get; set; }

        [Parameter("Enable TF3 Filter",     DefaultValue = false, Group = "MTF Trend Filter - TF3")] public bool EnableMtfTf3 { get; set; }
        [Parameter("TF3 Timeframe",         DefaultValue = "60",  Group = "MTF Trend Filter - TF3")] public string MtfTimeframe3 { get; set; }
        [Parameter("TF3 Pivot Strength",    DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter - TF3")] public int MtfPivotStrength3 { get; set; }
        [Parameter("TF3 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter - TF3")] public bool MtfIsLowerTf3 { get; set; }
        [Parameter("TF3 Allow Bull CHoCH",  DefaultValue = false, Group = "MTF Trend Filter - TF3")] public bool MtfAllowBullChoch3 { get; set; }
        [Parameter("TF3 Allow Bull BOS",    DefaultValue = false, Group = "MTF Trend Filter - TF3")] public bool MtfAllowBullBos3 { get; set; }
        [Parameter("TF3 Allow Bear CHoCH",  DefaultValue = false, Group = "MTF Trend Filter - TF3")] public bool MtfAllowBearChoch3 { get; set; }
        [Parameter("TF3 Allow Bear BOS",    DefaultValue = false, Group = "MTF Trend Filter - TF3")] public bool MtfAllowBearBos3 { get; set; }

        [Parameter("Enable TF4 Filter",     DefaultValue = false, Group = "MTF Trend Filter - TF4")] public bool EnableMtfTf4 { get; set; }
        [Parameter("TF4 Timeframe",         DefaultValue = "240", Group = "MTF Trend Filter - TF4")] public string MtfTimeframe4 { get; set; }
        [Parameter("TF4 Pivot Strength",    DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter - TF4")] public int MtfPivotStrength4 { get; set; }
        [Parameter("TF4 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter - TF4")] public bool MtfIsLowerTf4 { get; set; }
        [Parameter("TF4 Allow Bull CHoCH",  DefaultValue = false, Group = "MTF Trend Filter - TF4")] public bool MtfAllowBullChoch4 { get; set; }
        [Parameter("TF4 Allow Bull BOS",    DefaultValue = false, Group = "MTF Trend Filter - TF4")] public bool MtfAllowBullBos4 { get; set; }
        [Parameter("TF4 Allow Bear CHoCH",  DefaultValue = false, Group = "MTF Trend Filter - TF4")] public bool MtfAllowBearChoch4 { get; set; }
        [Parameter("TF4 Allow Bear BOS",    DefaultValue = false, Group = "MTF Trend Filter - TF4")] public bool MtfAllowBearBos4 { get; set; }

        // ═══ PARAMETERS — SMZ Trend Filter ═══════════════════════════════════

        [Parameter("Enable SMZ Trend Filter", DefaultValue = false,           Group = "SMZ Trend Filter - General")]
        public bool EnableSmzFilter { get; set; }
        [Parameter("SMZ MA Period",           DefaultValue = 50, MinValue = 1, Group = "SMZ Trend Filter - General")]
        public int SmzMaPeriod { get; set; }
        [Parameter("SMZ Logic (OR / AND)",    DefaultValue = SmzLogicMode.OR, Group = "SMZ Trend Filter - General")]
        public SmzLogicMode SmzFilterLogic { get; set; }
        [Parameter("Enable 1m  TF", DefaultValue = false, Group = "SMZ Trend Filter - Timeframes")] public bool SmzEnable1m  { get; set; }
        [Parameter("Enable 5m  TF", DefaultValue = false, Group = "SMZ Trend Filter - Timeframes")] public bool SmzEnable5m  { get; set; }
        [Parameter("Enable 15m TF", DefaultValue = false, Group = "SMZ Trend Filter - Timeframes")] public bool SmzEnable15m { get; set; }
        [Parameter("Enable 30m TF", DefaultValue = false, Group = "SMZ Trend Filter - Timeframes")] public bool SmzEnable30m { get; set; }
        [Parameter("Enable 1H  TF", DefaultValue = false, Group = "SMZ Trend Filter - Timeframes")] public bool SmzEnable1h  { get; set; }
        [Parameter("Enable 4H  TF", DefaultValue = false, Group = "SMZ Trend Filter - Timeframes")] public bool SmzEnable4h  { get; set; }
        [Parameter("Enable 1D  TF", DefaultValue = false, Group = "SMZ Trend Filter - Timeframes")] public bool SmzEnable1d  { get; set; }

        // ═══ PARAMETERS — Combined TF Filter ══════════════════════════════════

        [Parameter("Enable Combined TF Filter",  DefaultValue = false, Group = "Combined TF Filter - General")] public bool EnableCombinedTfFilter { get; set; }
        [Parameter("Enable 1H Condition",        DefaultValue = false, Group = "Combined TF Filter - 1H")]      public bool CmbEnable1h { get; set; }
        [Parameter("Use SMZ SMA (1H)",           DefaultValue = false, Group = "Combined TF Filter - 1H")]      public bool UseCmbSmz1h { get; set; }
        [Parameter("Use MTF Pivot Trend (1H)",   DefaultValue = false, Group = "Combined TF Filter - 1H")]      public bool UseCmbMtf1h { get; set; }
        [Parameter("1H MTF Pivot Strength",      DefaultValue = 15, MinValue = 1, Group = "Combined TF Filter - 1H")] public int CmbMtfPivotStrength1h { get; set; }
        [Parameter("1H MTF Lower than chart?",   DefaultValue = false, Group = "Combined TF Filter - 1H")]      public bool CmbMtfIsLowerTf1h { get; set; }
        [Parameter("Enable 15m Condition",       DefaultValue = false, Group = "Combined TF Filter - 15m")]     public bool CmbEnable15m { get; set; }
        [Parameter("Use SMZ SMA (15m)",          DefaultValue = false, Group = "Combined TF Filter - 15m")]     public bool UseCmbSmz15m { get; set; }
        [Parameter("Use MTF Pivot Trend (15m)",  DefaultValue = false, Group = "Combined TF Filter - 15m")]     public bool UseCmbMtf15m { get; set; }
        [Parameter("15m MTF Pivot Strength",     DefaultValue = 15, MinValue = 1, Group = "Combined TF Filter - 15m")] public int CmbMtfPivotStrength15m { get; set; }
        [Parameter("15m MTF Lower than chart?",  DefaultValue = false, Group = "Combined TF Filter - 15m")]     public bool CmbMtfIsLowerTf15m { get; set; }

        // ═══ PARAMETERS — OB Wick Filter ══════════════════════════════════════

        [Parameter("Enable OB Wick Filter",   DefaultValue = false,                       Group = "OB Wick Filter - General")] public bool EnableObWickFilter { get; set; }
        [Parameter("OB Wick Lookback Bars",   DefaultValue = 10, MinValue = 1,            Group = "OB Wick Filter - General")] public int ObWickLookbackBars { get; set; }
        [Parameter("OB Detection Method",     DefaultValue = ObWickDetectionMethod.Body,  Group = "OB Wick Filter - General")] public ObWickDetectionMethod ObWickDetectionMethodInput { get; set; }
        [Parameter("OB Mitigation Mode",      DefaultValue = ObWickMitigationMode.Normal, Group = "OB Wick Filter - General")] public ObWickMitigationMode ObWickMitigationModeInput { get; set; }
        [Parameter("OB Mitigation Type",      DefaultValue = ObWickMitigationType.Wicks,  Group = "OB Wick Filter - General")] public ObWickMitigationType ObWickMitigationTypeInput { get; set; }
        [Parameter("OB Only Market Hours",    DefaultValue = false,                       Group = "OB Wick Filter - General")] public bool ObWickOnlyMktHrs { get; set; }
        [Parameter("OB Enable 5m",    DefaultValue = false, Group = "OB Wick Filter - Timeframes")] public bool ObWickEnable5m    { get; set; }
        [Parameter("OB Enable 15m",   DefaultValue = false, Group = "OB Wick Filter - Timeframes")] public bool ObWickEnable15m   { get; set; }
        [Parameter("OB Enable 30m",   DefaultValue = false, Group = "OB Wick Filter - Timeframes")] public bool ObWickEnable30m   { get; set; }
        [Parameter("OB Enable 1h",    DefaultValue = false, Group = "OB Wick Filter - Timeframes")] public bool ObWickEnable1h    { get; set; }
        [Parameter("OB Enable 4h",    DefaultValue = false, Group = "OB Wick Filter - Timeframes")] public bool ObWickEnable4h    { get; set; }
        [Parameter("OB Enable Daily", DefaultValue = false, Group = "OB Wick Filter - Timeframes")] public bool ObWickEnableDaily { get; set; }
        [Parameter("OB Max 5m",    DefaultValue = 8, MinValue = 1, Group = "OB Wick Filter - Max Count")] public int ObWickMax5m    { get; set; }
        [Parameter("OB Max 15m",   DefaultValue = 8, MinValue = 1, Group = "OB Wick Filter - Max Count")] public int ObWickMax15m   { get; set; }
        [Parameter("OB Max 30m",   DefaultValue = 8, MinValue = 1, Group = "OB Wick Filter - Max Count")] public int ObWickMax30m   { get; set; }
        [Parameter("OB Max 1h",    DefaultValue = 8, MinValue = 1, Group = "OB Wick Filter - Max Count")] public int ObWickMax1h    { get; set; }
        [Parameter("OB Max 4h",    DefaultValue = 8, MinValue = 1, Group = "OB Wick Filter - Max Count")] public int ObWickMax4h    { get; set; }
        [Parameter("OB Max Daily", DefaultValue = 8, MinValue = 1, Group = "OB Wick Filter - Max Count")] public int ObWickMaxDaily { get; set; }

        // ═══ PARAMETERS — OB Imbalance Filter ════════════════════════════════
        //   Touch-to-Signal Window: max chart bars between OB touch and signal
        //   Unmitigated toggle: uses FIRST touch bar
        //   Mitigated   toggle: uses LAST  touch bar
        //   Invalidated OBs never pass regardless of toggle

        [Parameter("Enable OB Imbalance Filter",    DefaultValue = true, Group = "OB Imbalance Filter - General")] public bool EnableImbFilter { get; set; }
        [Parameter("Touch-to-Signal Window (bars)", DefaultValue = 10, MinValue = 1, Group = "OB Imbalance Filter - General")] public int ImbLookbackBars { get; set; }
        [Parameter("Use Unmitigated OB Logic",      DefaultValue = true,  Group = "OB Imbalance Filter - General")] public bool ImbUseUnmitigated { get; set; }
        [Parameter("Use Mitigated OB Logic",        DefaultValue = false, Group = "OB Imbalance Filter - General")] public bool ImbUseMitigated   { get; set; }
        [Parameter("Mitigation Method",             DefaultValue = ImbMitigationMethod.Wick, Group = "OB Imbalance Filter - Detection")] public ImbMitigationMethod ImbMitigationMethodInput { get; set; }
        [Parameter("Min Imbalance Size (ATR Mult)", DefaultValue = 0.5, Step = 0.1, Group = "OB Imbalance Filter - Detection")] public double ImbFvgThreshold { get; set; }
        [Parameter("Enable TF1", DefaultValue = true,       Group = "OB Imbalance Filter - Timeframes")] public bool ImbEnableTf1 { get; set; }
        [Parameter("TF1",        DefaultValue = "Minute15", Group = "OB Imbalance Filter - Timeframes")] public TimeFrame ImbTf1 { get; set; }
        [Parameter("Enable TF2", DefaultValue = false,       Group = "OB Imbalance Filter - Timeframes")] public bool ImbEnableTf2 { get; set; }
        [Parameter("TF2",        DefaultValue = "Minute30", Group = "OB Imbalance Filter - Timeframes")] public TimeFrame ImbTf2 { get; set; }
        [Parameter("Enable TF3", DefaultValue = false,       Group = "OB Imbalance Filter - Timeframes")] public bool ImbEnableTf3 { get; set; }
        [Parameter("TF3",        DefaultValue = "Hour",     Group = "OB Imbalance Filter - Timeframes")] public TimeFrame ImbTf3 { get; set; }
        [Parameter("Enable TF4", DefaultValue = false,       Group = "OB Imbalance Filter - Timeframes")] public bool ImbEnableTf4 { get; set; }
        [Parameter("TF4",        DefaultValue = "Hour4",    Group = "OB Imbalance Filter - Timeframes")] public TimeFrame ImbTf4 { get; set; }
        [Parameter("Max Zones Per TF", DefaultValue = 50, MinValue = 1, MaxValue = 500, Group = "OB Imbalance Filter - Limits")] public int ImbMaxZonesPerTf { get; set; }

        // ═══ PARAMETERS — MTF FVG Filter ══════════════════════════════════════

        [Parameter("Enable MTF FVG Filter",  DefaultValue = false,                    Group = "MTF FVG Filter - General")] public bool EnableFvgFilter { get; set; }
        [Parameter("FVG Lookback Bars",      DefaultValue = 10, MinValue = 1,         Group = "MTF FVG Filter - General")] public int FvgLookbackBars { get; set; }
        [Parameter("FVG Logic (OR / AND)",   DefaultValue = MtfLogicMode.OR,          Group = "MTF FVG Filter - General")] public MtfLogicMode FvgFilterLogic { get; set; }
        [Parameter("FVG Only Market Hours",  DefaultValue = false,                    Group = "MTF FVG Filter - Detection")] public bool FvgOnlyMktHrs { get; set; }
        [Parameter("FVG Mitigation Action",  DefaultValue = FvgMitigationMode.Normal, Group = "MTF FVG Filter - Detection")] public FvgMitigationMode FvgMitigationAction { get; set; }
        [Parameter("FVG Mitigation Type",    DefaultValue = FvgMitigationType.Wicks,  Group = "MTF FVG Filter - Detection")] public FvgMitigationType FvgMitigationTypeInput { get; set; }
        [Parameter("FVG Enable Chart TF", DefaultValue = false, Group = "MTF FVG Filter - Timeframes")] public bool FvgEnableChartTf { get; set; }
        [Parameter("FVG Enable 5m",       DefaultValue = false, Group = "MTF FVG Filter - Timeframes")] public bool FvgEnable5m      { get; set; }
        [Parameter("FVG Enable 10m",      DefaultValue = false, Group = "MTF FVG Filter - Timeframes")] public bool FvgEnable10m     { get; set; }
        [Parameter("FVG Enable 15m",      DefaultValue = false, Group = "MTF FVG Filter - Timeframes")] public bool FvgEnable15m     { get; set; }
        [Parameter("FVG Enable 30m",      DefaultValue = false, Group = "MTF FVG Filter - Timeframes")] public bool FvgEnable30m     { get; set; }
        [Parameter("FVG Enable 1h",       DefaultValue = false, Group = "MTF FVG Filter - Timeframes")] public bool FvgEnable1h      { get; set; }
        [Parameter("FVG Enable 4h",       DefaultValue = false, Group = "MTF FVG Filter - Timeframes")] public bool FvgEnable4h      { get; set; }
        [Parameter("FVG Enable 8h",       DefaultValue = false, Group = "MTF FVG Filter - Timeframes")] public bool FvgEnable8h      { get; set; }
        [Parameter("FVG Enable 12h",      DefaultValue = false, Group = "MTF FVG Filter - Timeframes")] public bool FvgEnable12h     { get; set; }
        [Parameter("FVG Enable Daily",    DefaultValue = false, Group = "MTF FVG Filter - Timeframes")] public bool FvgEnableDaily   { get; set; }
        [Parameter("FVG Enable Weekly",   DefaultValue = false, Group = "MTF FVG Filter - Timeframes")] public bool FvgEnableWeekly  { get; set; }
        [Parameter("FVG Enable Monthly",  DefaultValue = false, Group = "MTF FVG Filter - Timeframes")] public bool FvgEnableMonthly { get; set; }
        [Parameter("FVG Max Chart",   DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter - Max Count")] public int FvgMaxChart   { get; set; }
        [Parameter("FVG Max 5m",      DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter - Max Count")] public int FvgMax5m      { get; set; }
        [Parameter("FVG Max 10m",     DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter - Max Count")] public int FvgMax10m     { get; set; }
        [Parameter("FVG Max 15m",     DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter - Max Count")] public int FvgMax15m     { get; set; }
        [Parameter("FVG Max 30m",     DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter - Max Count")] public int FvgMax30m     { get; set; }
        [Parameter("FVG Max 1h",      DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter - Max Count")] public int FvgMax1h      { get; set; }
        [Parameter("FVG Max 4h",      DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter - Max Count")] public int FvgMax4h      { get; set; }
        [Parameter("FVG Max 8h",      DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter - Max Count")] public int FvgMax8h      { get; set; }
        [Parameter("FVG Max 12h",     DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter - Max Count")] public int FvgMax12h     { get; set; }
        [Parameter("FVG Max Daily",   DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter - Max Count")] public int FvgMaxDaily   { get; set; }
        [Parameter("FVG Max Weekly",  DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter - Max Count")] public int FvgMaxWeekly  { get; set; }
        [Parameter("FVG Max Monthly", DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter - Max Count")] public int FvgMaxMonthly { get; set; }
        [Parameter("Show Active FVG Zones",    DefaultValue = false,       Group = "MTF FVG Filter - Display")] public bool FvgShowActive    { get; set; }
        [Parameter("Show Mitigated FVG Zones", DefaultValue = false,       Group = "MTF FVG Filter - Display")] public bool FvgShowMitigated { get; set; }
        [Parameter("Active Bull Color",        DefaultValue = "#4D00C800", Group = "MTF FVG Filter - Display")] public Color FvgActiveBullColor { get; set; }
        [Parameter("Active Bear Color",        DefaultValue = "#4DC80000", Group = "MTF FVG Filter - Display")] public Color FvgActiveBearColor { get; set; }
        [Parameter("Mitigated FVG Color",      DefaultValue = "#26FFFF00", Group = "MTF FVG Filter - Display")] public Color FvgMitigatedColor  { get; set; }
        [Parameter("Display Bars Right",       DefaultValue = 10, MinValue = 1, Group = "MTF FVG Filter - Display")] public int FvgDisplayBarsRight { get; set; }

        // ═══ CONSTANTS & FIELDS ════════════════════════════════════════════════

        private const int MaxBslPivots = 10, MaxSmcObs = 500, SmcAtrPeriod = 200;

        private readonly List<string> _atlZzType  = new List<string>();
        private readonly List<double> _atlZzValue = new List<double>();
        private readonly List<int>    _atlZzIndex = new List<int>();
        private readonly List<string> _atlAdvType  = new List<string>();
        private readonly List<double> _atlAdvValue = new List<double>();
        private readonly List<int>    _atlAdvIndex = new List<int>();

        private double _atlMajorHighLevel = double.NaN, _atlMajorLowLevel = double.NaN;
        private bool   _atlMajorLevelsInitialized, _atlLock0 = true, _atlLock1 = true;
        private double _atlLastHighPivotValue = double.NaN, _atlLastLowPivotValue = double.NaN;
        private int    _atlLastHighPivotIndex = -1, _atlLastLowPivotIndex = -1;
        private int    _atlX0; private double _atlY0;
        private string _atlT0 = string.Empty, _atlT0Prev = string.Empty;
        private double _atlPrevZzLastValue = double.NaN;
        private char   _atlPrevZzLastTypeSuffix = '\0';
        private static readonly string[] AtlTlTypeNames = { "MLL","MHH","MHL","MLH","mLL","mHH","mHL","mLH" };
        private readonly int[]        _atlPtrX0    = new int[8];
        private readonly double[]     _atlPtrY0    = new double[8];
        private readonly int[]        _atlPtrX1    = new int[8];
        private readonly double[]     _atlPtrY1    = new double[8];
        private readonly AtlTlState[] _atlTlStates = new AtlTlState[8];
        private bool _atlIsLongSignal, _atlIsShortSignal;

        private IndicatorDataSeries      _ifvgMaSeries;
        private SimpleMovingAverage      _ifvgSma;
        private ExponentialMovingAverage _ifvgEma;

        private readonly LinkedList<BslPivot> _bslPivots        = new LinkedList<BslPivot>();
        private readonly LinkedList<BslPool>  _bslBuysidePools  = new LinkedList<BslPool>();
        private readonly LinkedList<BslPool>  _bslSellsidePools = new LinkedList<BslPool>();
        private double _bslCurrentBsl = double.NaN, _bslCurrentSsl = double.NaN;

        private readonly List<SmcObRecord> _smcInternalBullObs = new List<SmcObRecord>();
        private readonly List<SmcObRecord> _smcInternalBearObs = new List<SmcObRecord>();
        private readonly List<SmcObRecord> _smcSwingBullObs    = new List<SmcObRecord>();
        private readonly List<SmcObRecord> _smcSwingBearObs    = new List<SmcObRecord>();
        private readonly List<double>      _parsedHighs        = new List<double>();
        private readonly List<double>      _parsedLows         = new List<double>();
        private readonly List<DateTime>    _times              = new List<DateTime>();

        private double _smcAtrWilder = double.NaN, _smcAtrWilderSum, _smcCumTr;
        private int    _swingLeg, _swingTrend, _lastSwingHighIndex = -1, _lastSwingLowIndex = -1;
        private double _lastSwingHigh = double.NaN, _lastSwingLow = double.NaN;
        private bool   _swingHighCrossed, _swingLowCrossed;
        private int    _internalLeg, _internalTrend, _internalHighIndex = -1, _internalLowIndex = -1;
        private double _internalHighLevel = double.NaN, _internalLowLevel = double.NaN;
        private bool   _internalHighCrossed, _internalLowCrossed;
        private int    _smcObIdCounter, _lastParsedIndex = -1, _smcWarmup;

        private MtfTfState _mtfState1, _mtfState2, _mtfState3, _mtfState4;

        private Bars _smz1mBars, _smz5mBars, _smz15mBars, _smz30mBars;
        private Bars _smz1hBars, _smz4hBars, _smz1dBars;
        private SimpleMovingAverage _smz1mSma, _smz5mSma, _smz15mSma, _smz30mSma;
        private SimpleMovingAverage _smz1hSma, _smz4hSma, _smz1dSma;

        private Bars                _cmbSmz1hBars, _cmbSmz15mBars;
        private SimpleMovingAverage _cmbSmz1hSma,  _cmbSmz15mSma;
        private MtfTfState          _cmbMtf1h,     _cmbMtf15m;

        private readonly Dictionary<string,Bars>                _fvgBarsByTf      = new Dictionary<string,Bars>();
        private readonly Dictionary<string,List<FvgFilterZone>> _fvgBullByTf      = new Dictionary<string,List<FvgFilterZone>>();
        private readonly Dictionary<string,List<FvgFilterZone>> _fvgBearByTf      = new Dictionary<string,List<FvgFilterZone>>();
        private readonly Dictionary<string,int>                 _fvgMaxByTf       = new Dictionary<string,int>();
        private readonly Dictionary<string,int>                 _fvgLastBullTfIdx = new Dictionary<string,int>();
        private readonly Dictionary<string,int>                 _fvgLastBearTfIdx = new Dictionary<string,int>();
        private readonly Dictionary<string,double>              _fvgPrevBullHigh2 = new Dictionary<string,double>();
        private readonly Dictionary<string,double>              _fvgPrevBullLow   = new Dictionary<string,double>();
        private readonly Dictionary<string,double>              _fvgPrevBearLow2  = new Dictionary<string,double>();
        private readonly Dictionary<string,double>              _fvgPrevBearHigh  = new Dictionary<string,double>();
        private bool FvgUseBodyMitigation => FvgMitigationTypeInput == FvgMitigationType.Body;
        private int  _fvgChartId;

        private readonly List<ObWickTfState> _obWickStates = new List<ObWickTfState>();
        private readonly List<ImbTfState>    _imbStates    = new List<ImbTfState>();

        private int _lastProcessed = -1, _lastLongSignalBar = -1, _lastShortSignalBar = -1;

        // ═══ LIFECYCLE ════════════════════════════════════════════════════════

        protected override void OnStart()
        {
            for (int i = 0; i < 8; i++) _atlTlStates[i] = new AtlTlState();
            _smcWarmup = Math.Max(SmcSwingsLengthInput, 5) + 5;

            if (EnableMtfFilter)
            {
                _mtfState1 = EnableMtfTf1 ? MtfCreateState(MtfTimeframe1,MtfPivotStrength1,MtfIsLowerTf1,MtfAllowBullChoch1,MtfAllowBullBos1,MtfAllowBearChoch1,MtfAllowBearBos1) : null;
                _mtfState2 = EnableMtfTf2 ? MtfCreateState(MtfTimeframe2,MtfPivotStrength2,MtfIsLowerTf2,MtfAllowBullChoch2,MtfAllowBullBos2,MtfAllowBearChoch2,MtfAllowBearBos2) : null;
                _mtfState3 = EnableMtfTf3 ? MtfCreateState(MtfTimeframe3,MtfPivotStrength3,MtfIsLowerTf3,MtfAllowBullChoch3,MtfAllowBullBos3,MtfAllowBearChoch3,MtfAllowBearBos3) : null;
                _mtfState4 = EnableMtfTf4 ? MtfCreateState(MtfTimeframe4,MtfPivotStrength4,MtfIsLowerTf4,MtfAllowBullChoch4,MtfAllowBullBos4,MtfAllowBearChoch4,MtfAllowBearBos4) : null;
                Print("MTF ON Logic={0} TF1={1}({2}) TF2={3}({4}) TF3={5}({6}) TF4={7}({8})", MtfFilterLogic, EnableMtfTf1,MtfTimeframe1, EnableMtfTf2,MtfTimeframe2, EnableMtfTf3,MtfTimeframe3, EnableMtfTf4,MtfTimeframe4);
            }
            if (EnableSmzFilter)
            {
                SmzInit(SmzEnable1m,TimeFrame.Minute,ref _smz1mBars,ref _smz1mSma); SmzInit(SmzEnable5m,TimeFrame.Minute5,ref _smz5mBars,ref _smz5mSma);
                SmzInit(SmzEnable15m,TimeFrame.Minute15,ref _smz15mBars,ref _smz15mSma); SmzInit(SmzEnable30m,TimeFrame.Minute30,ref _smz30mBars,ref _smz30mSma);
                SmzInit(SmzEnable1h,TimeFrame.Hour,ref _smz1hBars,ref _smz1hSma); SmzInit(SmzEnable4h,TimeFrame.Hour4,ref _smz4hBars,ref _smz4hSma);
                SmzInit(SmzEnable1d,TimeFrame.Daily,ref _smz1dBars,ref _smz1dSma);
                Print("SMZ ON Logic={0} MA={1} 1m={2} 5m={3} 15m={4} 30m={5} 1h={6} 4h={7} 1d={8}",SmzFilterLogic,SmzMaPeriod,SmzEnable1m,SmzEnable5m,SmzEnable15m,SmzEnable30m,SmzEnable1h,SmzEnable4h,SmzEnable1d);
            }
            if (EnableCombinedTfFilter)
            {
                if (CmbEnable1h  && UseCmbSmz1h)  { _cmbSmz1hBars  = MarketData.GetBars(TimeFrame.Hour);     _cmbSmz1hSma  = Indicators.SimpleMovingAverage(_cmbSmz1hBars.ClosePrices,  SmzMaPeriod); }
                if (CmbEnable1h  && UseCmbMtf1h)  _cmbMtf1h  = MtfCreateStateFixed(TimeFrame.Hour,     CmbMtfPivotStrength1h,  CmbMtfIsLowerTf1h);
                if (CmbEnable15m && UseCmbSmz15m) { _cmbSmz15mBars = MarketData.GetBars(TimeFrame.Minute15); _cmbSmz15mSma = Indicators.SimpleMovingAverage(_cmbSmz15mBars.ClosePrices, SmzMaPeriod); }
                if (CmbEnable15m && UseCmbMtf15m) _cmbMtf15m = MtfCreateStateFixed(TimeFrame.Minute15, CmbMtfPivotStrength15m, CmbMtfIsLowerTf15m);
                Print("CMB ON 1H={0}(SMZ={1},MTF={2}) 15m={3}(SMZ={4},MTF={5})",CmbEnable1h,UseCmbSmz1h,UseCmbMtf1h,CmbEnable15m,UseCmbSmz15m,UseCmbMtf15m);
            }
            if (EnableFvgFilter)
            {
                FvgRegTf("Chart",Bars.TimeFrame,FvgEnableChartTf,FvgMaxChart); FvgRegTf("5m",TimeFrame.Minute5,FvgEnable5m,FvgMax5m);
                FvgRegTf("10m",TimeFrame.Minute10,FvgEnable10m,FvgMax10m); FvgRegTf("15m",TimeFrame.Minute15,FvgEnable15m,FvgMax15m);
                FvgRegTf("30m",TimeFrame.Minute30,FvgEnable30m,FvgMax30m); FvgRegTf("1hr",TimeFrame.Hour,FvgEnable1h,FvgMax1h);
                FvgRegTf("4hr",TimeFrame.Hour4,FvgEnable4h,FvgMax4h); FvgRegTf("8hr",TimeFrame.Hour8,FvgEnable8h,FvgMax8h);
                FvgRegTf("12hr",TimeFrame.Hour12,FvgEnable12h,FvgMax12h); FvgRegTf("Daily",TimeFrame.Daily,FvgEnableDaily,FvgMaxDaily);
                FvgRegTf("Weekly",TimeFrame.Weekly,FvgEnableWeekly,FvgMaxWeekly); FvgRegTf("Monthly",TimeFrame.Monthly,FvgEnableMonthly,FvgMaxMonthly);
                Print("FVG ON Logic={0} Lookback={1} Mitigation={2}/{3}",FvgFilterLogic,FvgLookbackBars,FvgMitigationAction,FvgMitigationTypeInput);
            }
            _ifvgMaSeries = CreateDataSeries();
            _ifvgSma = Indicators.SimpleMovingAverage(Bars.ClosePrices, IfvgMaPeriod);
            _ifvgEma = Indicators.ExponentialMovingAverage(Bars.ClosePrices, IfvgMaPeriod);
            if (EnableObWickFilter)
            {
                ObWickReg(TimeFrame.Minute5,ObWickEnable5m,ObWickMax5m,"5m"); ObWickReg(TimeFrame.Minute15,ObWickEnable15m,ObWickMax15m,"15m");
                ObWickReg(TimeFrame.Minute30,ObWickEnable30m,ObWickMax30m,"30m"); ObWickReg(TimeFrame.Hour,ObWickEnable1h,ObWickMax1h,"1h");
                ObWickReg(TimeFrame.Hour4,ObWickEnable4h,ObWickMax4h,"4h"); ObWickReg(TimeFrame.Daily,ObWickEnableDaily,ObWickMaxDaily,"Daily");
                Print("OBWick ON Lookback={0} Det={1} Mit={2}/{3} 5m={4} 15m={5} 30m={6} 1h={7} 4h={8} D={9}",ObWickLookbackBars,ObWickDetectionMethodInput,ObWickMitigationModeInput,ObWickMitigationTypeInput,ObWickEnable5m,ObWickEnable15m,ObWickEnable30m,ObWickEnable1h,ObWickEnable4h,ObWickEnableDaily);
            }
            if (EnableImbFilter)
            {
                ImbReg(ImbEnableTf1,ImbTf1,"TF1"); ImbReg(ImbEnableTf2,ImbTf2,"TF2");
                ImbReg(ImbEnableTf3,ImbTf3,"TF3"); ImbReg(ImbEnableTf4,ImbTf4,"TF4");
                Print("IMB ON Window={0} Unmitigated={1} Mitigated={2} Threshold={3} Method={4}",ImbLookbackBars,ImbUseUnmitigated,ImbUseMitigated,ImbFvgThreshold,ImbMitigationMethodInput);
            }
            if (SlBufferPips > 0) Print("SL Buffer={0} pips", SlBufferPips);
            Print("cBot started PP={0} PL={1} PR={2} MaxPos={3} Risk={4}% RR={5} ATL L={6} S={7} IFVG L={8} S={9} MA={10}/{11}",
                PP,PivotLeft,PivotRight,MaxOpenPositions,RiskPercent,RiskRewardRatio,EnableAtlLong,EnableAtlShort,EnableIfvgLong,EnableIfvgShort,IfvgMaType,IfvgMaPeriod);
        }

        protected override void OnStop() => Print("Autotl_ifvg_smc_mtfpt_smz_mtffvg_OBwick_Imb_filter_cbot stopped.");

        // ═══ ONBAR ════════════════════════════════════════════════════════════

        protected override void OnBar()
        {
            int bar = Bars.Count - 2;
            for (int i = _lastProcessed + 1; i <= bar; i++)
            {
                RunBslSsl(i); RunSmcFilter(i); RunAtl(i);
                if (EnableFvgFilter)    RunFvg(i);
                if (EnableObWickFilter) RunObWick(i);
                if (EnableImbFilter)    RunImb(i);
            }
            _lastProcessed = bar;

            if (EnableMtfFilter)
            {
                var t = Bars.OpenTimes[bar];
                MtfAdv(_mtfState1,t); MtfAdv(_mtfState2,t); MtfAdv(_mtfState3,t); MtfAdv(_mtfState4,t);
            }
            if (EnableCombinedTfFilter)
            { var t=Bars.OpenTimes[bar]; MtfAdv(_cmbMtf1h,t); MtfAdv(_cmbMtf15m,t); }

            if (bar < Math.Max(2*PP, PivotLeft+PivotRight+1)) return;

            bool atlLong=EnableAtlLong&&_atlIsLongSignal, atlShort=EnableAtlShort&&_atlIsShortSignal;
            bool ifvgLong=false, ifvgShort=false;
            if (EnableIfvgLong||EnableIfvgShort)
            {
                double ma=IfvgCalcMa(bar); int dir=IfvgDetect(bar,ma);
                ifvgLong=EnableIfvgLong&&dir>0; ifvgShort=EnableIfvgShort&&dir<0;
            }

            bool isLong=atlLong||ifvgLong, isShort=atlShort||ifvgShort;
            if (!isLong && !isShort) return;

            int openCount = Positions.FindAll(InstanceName, SymbolName).Length;
            if (openCount >= MaxOpenPositions) { Print("Bar {0}: max positions reached.", bar); return; }

            if (isLong && _lastLongSignalBar != bar)
            {
                _lastLongSignalBar = bar;
                if (ChkFilters(bar,1) && ChkMtf(true,bar) && ChkSmz(true,bar) &&
                    ChkCmb(true,bar) && ChkFvg(true,bar) && ChkObWick(true,bar) && ChkImb(true,bar))
                    TryLong(bar);
            }
            openCount = Positions.FindAll(InstanceName, SymbolName).Length;
            if (openCount >= MaxOpenPositions) return;
            if (isShort && _lastShortSignalBar != bar)
            {
                _lastShortSignalBar = bar;
                if (ChkFilters(bar,-1) && ChkMtf(false,bar) && ChkSmz(false,bar) &&
                    ChkCmb(false,bar) && ChkFvg(false,bar) && ChkObWick(false,bar) && ChkImb(false,bar))
                    TryShort(bar);
            }
        }

        // ═══ ATL ENGINE ═══════════════════════════════════════════════════════

        private void RunAtl(int index)
        {
            AtlZigZag(index); AtlSyncAdv(); AtlMajMin(index);
            if (_atlAdvType.Count > 2) { int l=_atlAdvType.Count-1; _atlX0=_atlAdvIndex[l]; _atlY0=_atlAdvValue[l]; _atlT0=_atlAdvType[l]; }
            AtlPtrs(); AtlTrendLines(index);
            _atlT0Prev = _atlT0;
            if (_atlZzType.Count > 0) { int n=_atlZzType.Count-1; _atlPrevZzLastValue=_atlZzValue[n]; _atlPrevZzLastTypeSuffix=_atlZzType[n][_atlZzType[n].Length-1]; }
        }

        private bool AtlPivHigh(int index, out double pv)
        {
            pv=double.NaN; if(index<2*PP)return false;
            int pb=index-PP,ws=index-2*PP; double c=Bars.HighPrices[pb],mx=double.MinValue;
            for(int i=ws;i<=index;i++)if(Bars.HighPrices[i]>mx)mx=Bars.HighPrices[i];
            if(c!=mx)return false; int lb=ws;
            for(int i=ws;i<=index;i++)if(Bars.HighPrices[i]==mx)lb=i;
            if(lb!=pb)return false; pv=c;return true;
        }

        private bool AtlPivLow(int index, out double pv)
        {
            pv=double.NaN; if(index<2*PP)return false;
            int pb=index-PP,ws=index-2*PP; double c=Bars.LowPrices[pb],mn=double.MaxValue;
            for(int i=ws;i<=index;i++)if(Bars.LowPrices[i]<mn)mn=Bars.LowPrices[i];
            if(c!=mn)return false; int lb=ws;
            for(int i=ws;i<=index;i++)if(Bars.LowPrices[i]==mn)lb=i;
            if(lb!=pb)return false; pv=c;return true;
        }

        private void AtlZigZag(int index)
        {
            bool hh=AtlPivHigh(index,out double hv),hl=AtlPivLow(index,out double lv);
            if(!hh&&!hl)return;
            int pb=index-PP; double bc=Bars.ClosePrices[index];
            if(hh){_atlLastHighPivotValue=hv;_atlLastHighPivotIndex=pb;}
            if(hl){_atlLastLowPivotValue=lv;_atlLastLowPivotIndex=pb;}

            string LH(double v){int n=_atlZzType.Count;return n>2?(_atlZzValue[n-2]<v?"HH":"LH"):"H";}
            string LL(double v){int n=_atlZzType.Count;return n>2?(_atlZzValue[n-2]<v?"HL":"LL"):"L";}
            void RL(){int n=_atlZzType.Count-1;_atlZzType.RemoveAt(n);_atlZzValue.RemoveAt(n);_atlZzIndex.RemoveAt(n);}
            void PH(double v,int b){_atlZzType.Add(LH(v));_atlZzValue.Add(v);_atlZzIndex.Add(b);}
            void PL(double v,int b){_atlZzType.Add(LL(v));_atlZzValue.Add(v);_atlZzIndex.Add(b);}

            int cnt=_atlZzType.Count;
            if(hh&&hl)
            {
                if(cnt==0){_atlZzType.Add("H");_atlZzValue.Add(hv);_atlZzIndex.Add(pb);}
                else
                {
                    string lt=_atlZzType[cnt-1];double lv2=_atlZzValue[cnt-1];
                    if(lt=="L"||lt=="LL"){if(lv<lv2){RL();PL(lv,pb);}else PH(hv,pb);}
                    else if(lt=="H"||lt=="HH"){if(hv>lv2){RL();PH(hv,pb);}else PL(lv,pb);}
                    else if(lt=="LH"){if(hv<lv2)PL(lv,pb);else if(hv>lv2){if(bc<lv2){RL();PH(hv,pb);}else if(bc>lv2)PL(lv,pb);}}
                    else if(lt=="HL"){if(lv>lv2)PH(hv,pb);else if(lv<lv2){if(bc>lv2){RL();PL(lv,pb);}else if(bc<lv2)PH(hv,pb);}}
                }
            }
            else if(hh)
            {
                cnt=_atlZzType.Count;
                if(cnt==0){_atlZzType.Insert(0,"H");_atlZzValue.Insert(0,hv);_atlZzIndex.Insert(0,pb);}
                else
                {
                    string lt=_atlZzType[cnt-1];double lv2=_atlZzValue[cnt-1];
                    if(lt=="L"||lt=="HL"||lt=="LL"){if(hv>lv2)PH(hv,pb);else if(hv<lv2){RL();if(!double.IsNaN(_atlLastLowPivotValue)&&_atlLastLowPivotIndex>=0)PL(_atlLastLowPivotValue,_atlLastLowPivotIndex);}}
                    else if(lt=="H"||lt=="HH"||lt=="LH"){if(lv2<hv){RL();PH(hv,pb);}}
                }
            }
            else
            {
                cnt=_atlZzType.Count;
                if(cnt==0){_atlZzType.Insert(0,"L");_atlZzValue.Insert(0,lv);_atlZzIndex.Insert(0,pb);}
                else
                {
                    string lt=_atlZzType[cnt-1];double lv2=_atlZzValue[cnt-1];
                    if(lt=="H"||lt=="HH"||lt=="LH"){if(lv<lv2)PL(lv,pb);else if(lv>lv2){RL();if(!double.IsNaN(_atlLastHighPivotValue)&&_atlLastHighPivotIndex>=0)PH(_atlLastHighPivotValue,_atlLastHighPivotIndex);}}
                    else if(lt=="L"||lt=="HL"||lt=="LL"){if(lv2>lv){RL();PL(lv,pb);}}
                }
            }
            if(!_atlMajorLevelsInitialized&&_atlZzType.Count==2)
            {
                if(_atlZzType[0]=="H"){_atlMajorHighLevel=_atlZzValue[0];_atlMajorLowLevel=_atlZzValue[1];}
                else{_atlMajorHighLevel=_atlZzValue[1];_atlMajorLowLevel=_atlZzValue[0];}
                _atlMajorLevelsInitialized=true;
            }
            if(_atlLock0&&_atlZzType.Count>=1){_atlAdvType.Insert(0,"M"+_atlZzType[0]);_atlAdvValue.Insert(0,_atlZzValue[0]);_atlAdvIndex.Insert(0,_atlZzIndex[0]);_atlLock0=false;}
            if(_atlLock1&&_atlZzType.Count>=2){_atlAdvType.Insert(1,"M"+_atlZzType[1]);_atlAdvValue.Insert(1,_atlZzValue[1]);_atlAdvIndex.Insert(1,_atlZzIndex[1]);_atlLock1=false;}
        }

        private void AtlSyncAdv()
        {
            if(_atlZzType.Count<=1||_atlAdvType.Count==0)return;
            int zl=_atlZzType.Count-1;double czv=_atlZzValue[zl];string czt=_atlZzType[zl];char cs=czt[czt.Length-1];
            if(double.IsNaN(_atlPrevZzLastValue)||czv==_atlPrevZzLastValue)return;
            if(cs!=_atlPrevZzLastTypeSuffix){_atlAdvType.Add("m"+czt);_atlAdvValue.Add(czv);_atlAdvIndex.Add(_atlZzIndex[zl]);}
            else{int al=_atlAdvType.Count-1;_atlAdvValue[al]=czv;_atlAdvIndex[al]=_atlZzIndex[zl];}
        }

        private void AtlMajMin(int index)
        {
            if(!_atlMajorLevelsInitialized||_atlAdvType.Count<=1)return;
            double cls=Bars.ClosePrices[index];
            string ZT(int o=0){int n=_atlZzType.Count-1-o;return n>=0?_atlZzType[n]:string.Empty;}
            if(cls>_atlMajorHighLevel)
            {
                int l=_atlAdvType.Count-1;string t=_atlAdvType[l];
                if(t=="mL"){_atlAdvType[l]="ML";_atlMajorLowLevel=_atlAdvValue[l];}
                else if(t=="mHL"||t=="mLL"){string p="M"+ZT();if(p.Length>1)_atlAdvType[l]=p;_atlMajorLowLevel=_atlAdvValue[l];}
                else if(t=="mLH"||t=="mHH"||t=="MLH"||t=="MHH"){if(l>=1){string t2=_atlAdvType[l-1];if(t2=="mHL"||t2=="mLL"){string p="M"+ZT(1);if(p.Length>1)_atlAdvType[l-1]=p;_atlMajorLowLevel=_atlAdvValue[l-1];}}}
            }
            {int l=_atlAdvType.Count-1;string t=_atlAdvType[l];
            if(_atlAdvValue[l]>_atlMajorHighLevel){if(t=="mH"){_atlAdvType[l]="MH";_atlMajorHighLevel=_atlAdvValue[l];}else if(t=="mLH"){string p="M"+ZT();if(p.Length>1)_atlAdvType[l]=p;_atlMajorHighLevel=_atlAdvValue[l];}else if(t=="mHH"||t=="MHH"){string p="M"+ZT();if(p.Length>1)_atlAdvType[l]=p;_atlMajorHighLevel=_atlAdvValue[l];}}}
            if(cls<_atlMajorLowLevel)
            {
                int l=_atlAdvType.Count-1;string t=_atlAdvType[l];
                if(t=="mH"){_atlAdvType[l]="MH";_atlMajorHighLevel=_atlAdvValue[l];}
                else if(t=="mLH"||t=="mHH"){string p="M"+ZT();if(p.Length>1)_atlAdvType[l]=p;_atlMajorHighLevel=_atlAdvValue[l];}
                else if(t=="mHL"||t=="mLL"||t=="MHL"||t=="MLL"){if(l>=1){string t2=_atlAdvType[l-1];if(t2=="mLH"||t2=="mHH"){string p="M"+ZT(1);if(p.Length>1)_atlAdvType[l-1]=p;_atlMajorHighLevel=_atlAdvValue[l-1];}}}
            }
            {int l=_atlAdvType.Count-1;string t=_atlAdvType[l];
            if(_atlAdvValue[l]<_atlMajorLowLevel){if(t=="mL"){_atlAdvType[l]="ML";_atlMajorLowLevel=_atlAdvValue[l];}else if(t=="mHL"||t=="mLL"||t=="MLL"){string p="M"+ZT();if(p.Length>1)_atlAdvType[l]=p;_atlMajorLowLevel=_atlAdvValue[l];}}}
        }

        private void AtlPtrs()
        {
            if(_atlT0==_atlT0Prev)return;
            for(int i=0;i<8;i++){if(_atlT0!=AtlTlTypeNames[i])continue; if(_atlPtrX0[i]==0){_atlPtrX0[i]=_atlX0;_atlPtrY0[i]=_atlY0;}else if(_atlPtrX1[i]==0){_atlPtrX1[i]=_atlX0;_atlPtrY1[i]=_atlY0;}else{_atlPtrX0[i]=_atlPtrX1[i];_atlPtrY0[i]=_atlPtrY1[i];_atlPtrX1[i]=_atlX0;_atlPtrY1[i]=_atlY0;}}
        }

        private void AtlTrendLines(int index)
        {
            _atlIsLongSignal=false;_atlIsShortSignal=false;
            AtlTL(index,0,true, ShortBreak_MjExUp,  LongReact_MjExUp);
            AtlTL(index,1,false,ShortReact_MjExDown,LongBreak_MjExDown);
            AtlTL(index,2,true, ShortBreak_MjInUp,  LongReact_MjInUp);
            AtlTL(index,3,false,ShortReact_MjInDown,LongBreak_MjInDown);
            AtlTL(index,4,true, ShortBreak_MnExUp,  LongReact_MnExUp);
            AtlTL(index,5,false,ShortReact_MnExDown,LongBreak_MnExDown);
            AtlTL(index,6,true, ShortBreak_MnInUp,  LongReact_MnInUp);
            AtlTL(index,7,false,ShortReact_MnInDown,LongBreak_MnInDown);
        }

        private void AtlTL(int index,int ti,bool isUp,bool enBrkShort,bool enReactLong)
        {
            var s=_atlTlStates[ti];
            int x0=_atlPtrX0[ti];double y0=_atlPtrY0[ti];int x1=_atlPtrX1[ti];double y1=_atlPtrY1[ti];
            s.PermitSetPrev=s.PermitSet;
            if(x0!=0&&x1!=0&&x0!=s.LastAnchorX0)
            {
                s.LastAnchorX0=x0; bool cs=isUp?(y1>y0):(y1<y0),permit=false;
                if(cs){permit=true;for(int b=x0+1;b<=index;b++){double lp=AtlLP(x0,y0,x1,y1,b);if(isUp?Bars.ClosePrices[b]<=lp:Bars.ClosePrices[b]>=lp){permit=false;break;}}}
                if(permit){s.ActiveX0=x0;s.ActiveY0=y0;s.ActiveX1=x1;s.ActiveY1=y1;s.PermitSet=true;}
            }
            if(s.PermitSet){if(s.ActiveX0==0)s.PermitSet=false;else{double lp=AtlLP(s.ActiveX0,s.ActiveY0,s.ActiveX1,s.ActiveY1,index);if(isUp?Bars.ClosePrices[index]<=lp:Bars.ClosePrices[index]>=lp)s.PermitSet=false;}}
            bool ab=s.PermitSetPrev&&!s.PermitSet,ar=false;
            if(s.PermitSet&&s.ActiveX0!=0){double lp=AtlLP(s.ActiveX0,s.ActiveY0,s.ActiveX1,s.ActiveY1,index);ar=isUp?(Bars.ClosePrices[index]>lp&&Bars.LowPrices[index]<lp):(Bars.ClosePrices[index]<lp&&Bars.HighPrices[index]>lp);}
            if(isUp){if(ab&&enBrkShort)_atlIsShortSignal=true;if(ar&&enReactLong)_atlIsLongSignal=true;}
            else    {if(ab&&enReactLong)_atlIsLongSignal=true; if(ar&&enBrkShort)_atlIsShortSignal=true;}
        }

        private static double AtlLP(int x0,double y0,int x1,double y1,int at)=>x1==x0?y0:y0+(y1-y0)*(double)(at-x0)/(x1-x0);

        // ═══ IFVG ENGINE ══════════════════════════════════════════════════════

        private double IfvgCalcMa(int i)
        {
            _ifvgMaSeries[i]=string.Equals(IfvgMaType,"SMA",StringComparison.OrdinalIgnoreCase)?_ifvgSma.Result[i]:_ifvgEma.Result[i];
            return _ifvgMaSeries[i];
        }

        private int IfvgDetect(int index,double ma)
        {
            var msv=IfvgMinFvgPips*Symbol.PipSize;
            for(var i=1;i<=IfvgGapBars;i++){var ft=IfvgFvg(index,i,IfvgEpsilonPoints);if(ft==0)continue;int sd;if(IfvgProc(index,i,ft,msv,ma,out sd))return sd;}
            return 0;
        }

        private int IfvgFvg(int ci,int idx,double eps)
        {
            if(idx+2>ci)return 0;
            double h2=Bars.HighPrices[ci-(idx+2)],l2=Bars.LowPrices[ci-(idx+2)],lt=Bars.LowPrices[ci-idx],ht=Bars.HighPrices[ci-idx];
            if(lt>h2-eps)return 1;if(ht<l2+eps)return -1;return 0;
        }

        private bool IfvgProc(int index,int i,int ft,double msv,double ma,out int sd)
        {
            sd=0;bool bg=ft==1;
            var gl=bg?Bars.HighPrices[index-(i+2)]:Bars.HighPrices[index-i];
            var gh=bg?Bars.LowPrices[index-i]:Bars.LowPrices[index-(i+2)];
            if((gh-gl)<msv)return false;
            if(i>1){for(var k=i-1;k>=1;k--){var c=Bars.ClosePrices[index-k];if(bg&&c<gl)return false;if(!bg&&c>gh)return false;}}
            var bo=bg?Bars.ClosePrices[index]<gl:Bars.ClosePrices[index]>gh;if(!bo)return false;
            var mr=!double.IsNaN(ma)&&!double.IsNaN(_ifvgMaSeries[index-1]);
            var mc=bg?(mr&&ma<_ifvgMaSeries[index-1]&&Bars.ClosePrices[index]<ma):(mr&&ma>_ifvgMaSeries[index-1]&&Bars.ClosePrices[index]>ma);
            if(!mc)return false;sd=bg?-1:1;return true;
        }

        // ═══ BSL/SSL ENGINE ═══════════════════════════════════════════════════

        private void RunBslSsl(int index)
        {
            BslDetPivots(index);BslAddPool(index);BslClear(index);
            _bslCurrentBsl=_bslBuysidePools.First!=null?_bslBuysidePools.First.Value.Price:double.NaN;
            _bslCurrentSsl=_bslSellsidePools.First!=null?_bslSellsidePools.First.Value.Price:double.NaN;
        }

        private void BslDetPivots(int ci)
        {
            int pi=ci-PivotRight;if(pi<=0)return;
            int ls=pi-PivotLeft,re=pi+PivotRight;if(ls<0||re>=Bars.Count)return;
            double ch=Bars.HighPrices[pi],cl=Bars.LowPrices[pi];
            if(BslIsH(ch,ls,re))BslUnshift(new BslPivot{Price=ch,BarIndex=pi,Type=1});
            if(BslIsL(cl,ls,re)) BslUnshift(new BslPivot{Price=cl,BarIndex=pi,Type=-1});
        }

        private bool BslIsH(double c,int s,int e){double m=double.MinValue;for(int i=s;i<=e;i++)if(Bars.HighPrices[i]>m)m=Bars.HighPrices[i];return c==m;}
        private bool BslIsL(double c,int s,int e){double m=double.MaxValue;for(int i=s;i<=e;i++)if(Bars.LowPrices[i]<m)m=Bars.LowPrices[i];return c==m;}

        private void BslUnshift(BslPivot p)
        {
            if(_bslPivots.First!=null){var f=_bslPivots.First.Value;if(f.BarIndex==p.BarIndex&&f.Type==p.Type&&Math.Abs(f.Price-p.Price)<Symbol.PipSize*0.1)return;}
            _bslPivots.AddFirst(p);while(_bslPivots.Count>MaxBslPivots)_bslPivots.RemoveLast();
        }

        private void BslAddPool(int ci)
        {
            int ci2=ci-PivotRight;
            foreach(var p in _bslPivots){if(p.BarIndex!=ci2)continue;var pool=new BslPool{Price=p.Price,PivotIndex=p.BarIndex};if(p.Type==1)_bslBuysidePools.AddFirst(pool);if(p.Type==-1)_bslSellsidePools.AddFirst(pool);}
        }

        private void BslClear(int i)
        {
            var n=_bslSellsidePools.First;while(n!=null){var nx=n.Next;if(Bars.LowPrices[i]<=n.Value.Price)_bslSellsidePools.Remove(n);n=nx;}
            n=_bslBuysidePools.First;while(n!=null){var nx=n.Next;if(Bars.HighPrices[i]>=n.Value.Price)_bslBuysidePools.Remove(n);n=nx;}
        }

        // ═══ SMC OB FILTER ════════════════════════════════════════════════════

        private bool ChkFilters(int index,int cond)
        {
            bool bull=cond>0;bool? f1=null,f2=null;
            if(EnableFilter1){var p=bull?_smcInternalBullObs:_smcInternalBearObs;f1=SmcTouch(p,index,Filter1Lookback,bull);}
            if(EnableFilter2){var p=bull?_smcSwingBullObs:_smcSwingBearObs;f2=SmcTouch(p,index,Filter2Lookback,bull);}
            if(f1.HasValue&&f2.HasValue){if(!f1.Value&&!f2.Value){Print("[Filter BLOCKED] {0} bar={1}",bull?"Long":"Short",index);return false;}return true;}
            if(f1.HasValue&&!f1.Value){Print("[Filter1 BLOCKED] {0} bar={1}",bull?"Long":"Short",index);return false;}
            if(f2.HasValue&&!f2.Value){Print("[Filter2 BLOCKED] {0} bar={1}",bull?"Long":"Short",index);return false;}
            return true;
        }

        private bool SmcTouch(List<SmcObRecord> pool,int bar,int lookback,bool bull)
        {
            if(pool.Count==0)return false;
            var atr=double.IsNaN(_smcAtrWilder)?0.0:_smcAtrWilder;var cn=Bars.ClosePrices[bar];
            foreach(var ob in pool)
            {
                if(EnableMinBarsFromOrigin){var mb=ob.Internal?MinBarsFromOriginInternal:MinBarsFromOriginSwing;if(bar-ob.Index<mb)continue;}
                if(EnableAtrDistanceFilter&&AtrDistanceMultiplier>0&&atr>0){var adv=bull?cn-ob.Top:ob.Bottom-cn;if(adv<AtrDistanceMultiplier*atr)continue;}
                int ltb=-1;
                for(var b=ob.StructureBreakIndex+1;b<=bar;b++){if(bull&&Bars.LowPrices[b]<=ob.Top)ltb=b;if(!bull&&Bars.HighPrices[b]>=ob.Bottom)ltb=b;}
                if(ltb<0)continue;if(bar-ltb<=lookback)return true;
            }
            return false;
        }

        private void RunSmcFilter(int index)
        {
            for(var i=_lastParsedIndex+1;i<=index;i++)SmcArr(i);_lastParsedIndex=index;
            if(index<_smcWarmup)return;
            const int iLen=5;var sLen=Math.Max(5,SmcSwingsLengthInput);
            var iln=SmcLeg(index,iLen,_internalLeg);var idc=iln-_internalLeg;
            if(idc!=0){if(idc==1){_internalLowLevel=Bars.LowPrices[index-iLen];_internalLowIndex=index-iLen;_internalLowCrossed=false;}else{_internalHighLevel=Bars.HighPrices[index-iLen];_internalHighIndex=index-iLen;_internalHighCrossed=false;}}
            _internalLeg=iln;
            var sln=SmcLeg(index,sLen,_swingLeg);var sdc=sln-_swingLeg;
            if(sdc!=0){if(sdc==1){_lastSwingLow=Bars.LowPrices[index-sLen];_lastSwingLowIndex=index-sLen;_swingLowCrossed=false;}else{_lastSwingHigh=Bars.HighPrices[index-sLen];_lastSwingHighIndex=index-sLen;_swingHighCrossed=false;}}
            _swingLeg=sln;
            var cl=Bars.ClosePrices[index];
            if(!double.IsNaN(_internalHighLevel)&&!_internalHighCrossed&&cl>_internalHighLevel){_internalHighCrossed=true;_internalTrend=1; SmcStore(_internalHighIndex,true,1,index);}
            if(!double.IsNaN(_internalLowLevel) &&!_internalLowCrossed &&cl<_internalLowLevel) {_internalLowCrossed=true;_internalTrend=-1;SmcStore(_internalLowIndex,true,-1,index);}
            if(!double.IsNaN(_lastSwingHigh)    &&!_swingHighCrossed   &&cl>_lastSwingHigh)    {_swingHighCrossed=true;_swingTrend=1;    SmcStore(_lastSwingHighIndex,false,1,index);}
            if(!double.IsNaN(_lastSwingLow)     &&!_swingLowCrossed    &&cl<_lastSwingLow)     {_swingLowCrossed=true;_swingTrend=-1;   SmcStore(_lastSwingLowIndex,false,-1,index);}
            SmcMng(_smcInternalBullObs,index,true);SmcMng(_smcInternalBearObs,index,false);
            SmcMng(_smcSwingBullObs,index,true);SmcMng(_smcSwingBearObs,index,false);
        }

        private void SmcArr(int index)
        {
            double tr;
            if(index==0){_smcCumTr=0;_smcAtrWilderSum=0;_smcAtrWilder=double.NaN;tr=Bars.HighPrices[0]-Bars.LowPrices[0];}
            else{var pc=Bars.ClosePrices[index-1];tr=Math.Max(Bars.HighPrices[index]-Bars.LowPrices[index],Math.Max(Math.Abs(Bars.HighPrices[index]-pc),Math.Abs(Bars.LowPrices[index]-pc)));_smcCumTr+=tr;
            if(index<SmcAtrPeriod){_smcAtrWilderSum+=tr;_smcAtrWilder=double.NaN;}
            else if(index==SmcAtrPeriod){_smcAtrWilderSum+=tr;_smcAtrWilder=_smcAtrWilderSum/SmcAtrPeriod;}
            else _smcAtrWilder=(_smcAtrWilder*(SmcAtrPeriod-1)+tr)/SmcAtrPeriod;}
            var vm=SmcOrderBlockFilterInput==ObFilter.Atr?(double.IsNaN(_smcAtrWilder)?double.MaxValue:_smcAtrWilder):(_smcCumTr/Math.Max(1,index));
            var hv=(Bars.HighPrices[index]-Bars.LowPrices[index])>=2.0*vm;
            _parsedHighs.Add(hv?Bars.LowPrices[index]:Bars.HighPrices[index]);
            _parsedLows.Add(hv?Bars.HighPrices[index]:Bars.LowPrices[index]);
            _times.Add(Bars.OpenTimes[index]);
        }

        private void SmcStore(int pi,bool isInt,int bias,int bi)
        {
            if(pi<0||pi>=bi||bi>=_parsedHighs.Count)return;
            int pix=pi;
            if(bias==-1){double mv=double.MinValue;for(var i=pi;i<bi;i++)if(_parsedHighs[i]>mv){mv=_parsedHighs[i];pix=i;}}
            else{double mv=double.MaxValue;for(var i=pi;i<bi;i++)if(_parsedLows[i]<mv){mv=_parsedLows[i];pix=i;}}
            bool bull=bias==1;
            var ob=new SmcObRecord{Index=pix,Top=_parsedHighs[pix],Bottom=_parsedLows[pix],Bullish=bull,Internal=isInt,Time=_times[pix],StructureBreakIndex=bi};
            _smcObIdCounter++;
            var list=isInt?(bull?_smcInternalBullObs:_smcInternalBearObs):(bull?_smcSwingBullObs:_smcSwingBearObs);
            if(list.Count>=MaxSmcObs)list.RemoveAt(list.Count-1);list.Insert(0,ob);
        }

        private void SmcMng(List<SmcObRecord> list,int index,bool bull)
        {
            var bs=SmcOrderBlockMitigationInput==MitigationMode.Close?Bars.ClosePrices[index]:Bars.HighPrices[index];
            var bls=SmcOrderBlockMitigationInput==MitigationMode.Close?Bars.ClosePrices[index]:Bars.LowPrices[index];
            for(var i=list.Count-1;i>=0;i--){var ob=list[i];if((bull&&bls<ob.Bottom)||(!bull&&bs>ob.Top))list.RemoveAt(i);}
        }

        private int SmcLeg(int index,int size,int prev)
        {
            if(index-size<1)return prev;
            double rh=Bars.HighPrices[index-size],rl=Bars.LowPrices[index-size];
            double hi=double.MinValue,lo=double.MaxValue;
            for(var i=Math.Max(0,index-size+1);i<=index;i++){if(Bars.HighPrices[i]>hi)hi=Bars.HighPrices[i];if(Bars.LowPrices[i]<lo)lo=Bars.LowPrices[i];}
            if(rh>hi)return 0;if(rl<lo)return 1;return prev;
        }

        // ═══ TRADE ENTRY ══════════════════════════════════════════════════════

        private void TryLong(int bar)
        {
            double entry=Symbol.Ask,anc;string an;
            if(!LongAnc(bar,out anc,out an)){Print("Bar {0}: LONG skip no anchor",bar);return;}
            if(anc>=entry){Print("Bar {0}: LONG skip anchor not below entry",bar);return;}
            double slp=(entry-anc)/Symbol.PipSize+SlBufferPips;
            if(!VldSl(bar,"LONG",slp))return;
            double vol=Vol(Account.Equity*(RiskPercent/100.0),slp);
            if(vol<=0){Print("Bar {0}: LONG skip vol=0",bar);return;}
            double sl=Math.Round(entry-slp*Symbol.PipSize,Symbol.Digits),tp=Math.Round(entry+slp*RiskRewardRatio*Symbol.PipSize,Symbol.Digits);
            Print("Bar {0}: LONG Entry={1:F5} {2}={3:F5} SL={4:F5}({5:F1}p buf={6:F1}) TP={7:F5} Vol={8}",bar,entry,an,anc,sl,slp,SlBufferPips,tp,vol);
            var r=ExecuteMarketOrder(TradeType.Buy,SymbolName,vol,InstanceName,null,null);
            if(r.IsSuccessful)ModifyPosition(r.Position,sl,tp);else Print("Bar {0}: LONG fail {1}",bar,r.Error);
        }

        private void TryShort(int bar)
        {
            double entry=Symbol.Bid,anc;string an;
            if(!ShortAnc(bar,out anc,out an)){Print("Bar {0}: SHORT skip no anchor",bar);return;}
            if(anc<=entry){Print("Bar {0}: SHORT skip anchor not above entry",bar);return;}
            double slp=(anc-entry)/Symbol.PipSize+SlBufferPips;
            if(!VldSl(bar,"SHORT",slp))return;
            double vol=Vol(Account.Equity*(RiskPercent/100.0),slp);
            if(vol<=0){Print("Bar {0}: SHORT skip vol=0",bar);return;}
            double sl=Math.Round(entry+slp*Symbol.PipSize,Symbol.Digits),tp=Math.Round(entry-slp*RiskRewardRatio*Symbol.PipSize,Symbol.Digits);
            Print("Bar {0}: SHORT Entry={1:F5} {2}={3:F5} SL={4:F5}({5:F1}p buf={6:F1}) TP={7:F5} Vol={8}",bar,entry,an,anc,sl,slp,SlBufferPips,tp,vol);
            var r=ExecuteMarketOrder(TradeType.Sell,SymbolName,vol,InstanceName,null,null);
            if(r.IsSuccessful)ModifyPosition(r.Position,sl,tp);else Print("Bar {0}: SHORT fail {1}",bar,r.Error);
        }

        private bool LongAnc(int bar,out double a,out string n)
        {
            a=double.NaN;n=string.Empty;
            if(LongSlSource==LongStopLossSource.Ssl){a=_bslCurrentSsl;n="SSL";return !double.IsNaN(a)&&a>0;}
            if(bar<1||bar>=Bars.Count)return false;a=Bars.LowPrices[bar];n="Candle-1 Low";return !double.IsNaN(a)&&a>0;
        }

        private bool ShortAnc(int bar,out double a,out string n)
        {
            a=double.NaN;n=string.Empty;
            if(ShortSlSource==ShortStopLossSource.Bsl){a=_bslCurrentBsl;n="BSL";return !double.IsNaN(a)&&a>0;}
            if(bar<1||bar>=Bars.Count)return false;a=Bars.HighPrices[bar];n="Candle-1 High";return !double.IsNaN(a)&&a>0;
        }

        private bool VldSl(int bar,string dir,double slp)
        {
            if(slp<MinSlPips){Print("Bar {0}: {1} skip SL {2:F1}<min {3:F1}",bar,dir,slp,MinSlPips);return false;}
            if(slp>MaxSlPips){Print("Bar {0}: {1} skip SL {2:F1}>max {3:F1}",bar,dir,slp,MaxSlPips);return false;}
            return true;
        }

        private double Vol(double risk,double slp)
        {
            if(slp<=0)return 0;
            double v=Symbol.NormalizeVolumeInUnits(Symbol.VolumeForFixedRisk(risk,slp),RoundingMode.Down);
            if(v<Symbol.VolumeInUnitsMin)return 0;if(v>Symbol.VolumeInUnitsMax)v=Symbol.VolumeInUnitsMax;return v;
        }

        // ═══ SMZ FILTER ═══════════════════════════════════════════════════════

        private void SmzInit(bool en,TimeFrame tf,ref Bars b,ref SimpleMovingAverage s)
        {if(!en)return;b=MarketData.GetBars(tf);s=Indicators.SimpleMovingAverage(b.ClosePrices,SmzMaPeriod);}

        private bool ChkSmz(bool isLong,int bar)
        {
            if(!EnableSmzFilter)return true;
            bool isAnd=SmzFilterLogic==SmzLogicMode.AND;int en=0,pa=0;
            SmzChk("1m",_smz1mBars,_smz1mSma,SmzEnable1m,isLong,ref en,ref pa);
            SmzChk("5m",_smz5mBars,_smz5mSma,SmzEnable5m,isLong,ref en,ref pa);
            SmzChk("15m",_smz15mBars,_smz15mSma,SmzEnable15m,isLong,ref en,ref pa);
            SmzChk("30m",_smz30mBars,_smz30mSma,SmzEnable30m,isLong,ref en,ref pa);
            SmzChk("1H",_smz1hBars,_smz1hSma,SmzEnable1h,isLong,ref en,ref pa);
            SmzChk("4H",_smz4hBars,_smz4hSma,SmzEnable4h,isLong,ref en,ref pa);
            SmzChk("1D",_smz1dBars,_smz1dSma,SmzEnable1d,isLong,ref en,ref pa);
            if(en==0)return true;var r=isAnd?(pa==en):(pa>0);
            if(!r)Print("[SMZ BLOCKED] {0} bar={1} {2}/{3} {4}",isLong?"Long":"Short",bar,pa,en,isAnd?"AND":"OR");
            return r;
        }

        private void SmzChk(string lbl,Bars bars,SimpleMovingAverage sma,bool en,bool isLong,ref int ec,ref int pc)
        {
            if(!en||bars==null||sma==null)return;ec++;
            var idx=bars.Count-1;if(idx<SmzMaPeriod-1){pc++;return;}
            var sv=sma.Result[idx];if(double.IsNaN(sv)){pc++;return;}
            if(isLong==(bars.ClosePrices[idx]>sv))pc++;
        }

        // ═══ COMBINED TF FILTER ════════════════════════════════════════════════

        private bool ChkCmb(bool isLong,int bar)
        {
            if(!EnableCombinedTfFilter)return true;
            var p1=!CmbEnable1h ||CmbCond(isLong,UseCmbSmz1h, _cmbSmz1hBars, _cmbSmz1hSma, UseCmbMtf1h, _cmbMtf1h);
            var p2=!CmbEnable15m||CmbCond(isLong,UseCmbSmz15m,_cmbSmz15mBars,_cmbSmz15mSma,UseCmbMtf15m,_cmbMtf15m);
            if(!p1||!p2)Print("[CMB BLOCKED] {0} bar={1} 1H={2} 15m={3}",isLong?"Long":"Short",bar,p1?"PASS":"FAIL",p2?"PASS":"FAIL");
            return p1&&p2;
        }

        private bool CmbCond(bool isLong,bool useSmz,Bars sb,SimpleMovingAverage ss,bool useMtf,MtfTfState ms)
        {
            if(!useSmz&&!useMtf)return true;
            bool sp=false;
            if(useSmz&&sb!=null&&ss!=null){var idx=sb.Count-1;if(idx<SmzMaPeriod-1)sp=true;else{var sv=ss.Result[idx];sp=double.IsNaN(sv)||(isLong==(sb.ClosePrices[idx]>sv));}}
            bool mp=false;if(useMtf&&ms!=null)mp=!ms.HasTrend||(ms.CurrentTrend==isLong);
            return sp||mp;
        }

        // ═══ MTF PIVOT TREND ENGINE ═══════════════════════════════════════════

        private MtfTfState MtfCreateState(string tfIn,int ps,bool lower,bool bc,bool bb,bool bearC,bool bearB)
        {
            var tf=MtfParse(tfIn);var bars=tf==Bars.TimeFrame?Bars:MarketData.GetBars(tf);
            return new MtfTfState{TfBars=bars,PivotLen=Math.Max(1,ps),IsLowerTf=lower,TfMinutes=MtfMins(tf),AllowBullChoch=bc,AllowBullBos=bb,AllowBearChoch=bearC,AllowBearBos=bearB};
        }

        private MtfTfState MtfCreateStateFixed(TimeFrame tf,int ps,bool lower)
        {var bars=tf==Bars.TimeFrame?Bars:MarketData.GetBars(tf);return new MtfTfState{TfBars=bars,PivotLen=Math.Max(1,ps),IsLowerTf=lower,TfMinutes=MtfMins(tf)};}

        private void MtfAdv(MtfTfState s,DateTime ct)
        {
            if(s==null)return;var ti=MtfResolve(s,ct);if(ti<0)return;
            for(var i=s.LastProcessedTfBar+1;i<=ti;i++)MtfBar(s,i);
            if(ti>s.LastProcessedTfBar)s.LastProcessedTfBar=ti;
        }

        private void MtfBar(MtfTfState s,int ti)
        {
            var bars=s.TfBars;double pph=s.LastPivotHigh,ppl=s.LastPivotLow;
            if(ti>=s.PivotLen*2)
            {
                var pi=ti-s.PivotLen;
                if(MtfIsH(bars,pi,s.PivotLen)){var pp=bars.HighPrices[pi];s.LastPivotHigh=s.CurrentTrend?(double.IsNaN(s.LastPivotHigh)?pp:Math.Max(pp,s.LastPivotHigh)):pp;if(s.LastPivotHigh!=pph)s.PivotHighTime=bars.OpenTimes[pi];}
                if(MtfIsL(bars,pi,s.PivotLen)){var pp=bars.LowPrices[pi]; s.LastPivotLow=!s.CurrentTrend?(double.IsNaN(s.LastPivotLow)?pp:Math.Min(pp,s.LastPivotLow)):pp; if(s.LastPivotLow!=ppl)s.PivotLowTime=bars.OpenTimes[pi];}
            }
            double cl=bars.ClosePrices[ti]; double pc=ti>0?bars.ClosePrices[ti-1]:cl;
            if(!double.IsNaN(s.LastPivotHigh)&&!double.IsNaN(pph)&&pc<=pph&&cl>s.LastPivotHigh)
            {bool bos=s.CurrentTrend&&s.LastPivotHigh!=s.LastBrokenHigh;s.CurrentTrend=true;s.HasTrend=true;s.LastBrokenHigh=s.LastPivotHigh;s.LastBrokenLow=double.NaN;s.LastEventWasBos=bos;}
            if(!double.IsNaN(s.LastPivotLow)&&!double.IsNaN(ppl)&&pc>=ppl&&cl<s.LastPivotLow)
            {bool bos=!s.CurrentTrend&&s.LastPivotLow!=s.LastBrokenLow;s.CurrentTrend=false;s.HasTrend=true;s.LastBrokenLow=s.LastPivotLow;s.LastBrokenHigh=double.NaN;s.LastEventWasBos=bos;}
        }

        private int MtfResolve(MtfTfState s,DateTime ct)
        {
            if(!s.IsLowerTf)return MtfFOB(s.TfBars,ct.AddMinutes(-(s.TfMinutes>0?s.TfMinutes:1)));
            var cbi=MtfFOB(Bars,ct);if(cbi<0)return -1;
            var co=Bars.OpenTimes[cbi];DateTime cno;
            if(cbi+1<Bars.Count)cno=Bars.OpenTimes[cbi+1];else{var m=MtfMins(Bars.TimeFrame);cno=co.AddMinutes(m>0?m:1);}
            var first=MtfFOA(s.TfBars,co);if(first<0)return MtfFOB(s.TfBars,ct);
            if(s.TfBars.OpenTimes[first]>=cno)return MtfFOB(s.TfBars,ct);
            return first;
        }

        private bool ChkMtf(bool isLong,int bar)
        {
            if(!EnableMtfFilter)return true;
            bool isAnd=MtfFilterLogic==MtfLogicMode.AND;int en=0,pa=0;
            if(_mtfState1!=null){en++;if(MtfPass(_mtfState1,isLong))pa++;}
            if(_mtfState2!=null){en++;if(MtfPass(_mtfState2,isLong))pa++;}
            if(_mtfState3!=null){en++;if(MtfPass(_mtfState3,isLong))pa++;}
            if(_mtfState4!=null){en++;if(MtfPass(_mtfState4,isLong))pa++;}
            if(en==0)return true;var r=isAnd?(pa==en):(pa>0);
            if(!r)Print("[MTF BLOCKED] {0} bar={1} {2}/{3} {4}",isLong?"Long":"Short",bar,pa,en,isAnd?"AND":"OR");
            return r;
        }

        private static bool MtfPass(MtfTfState s,bool isLong)
        {
            if(!s.HasTrend)return true;if(s.CurrentTrend!=isLong)return false;
            bool ac=isLong?s.AllowBullChoch:s.AllowBearChoch,ab=isLong?s.AllowBullBos:s.AllowBearBos;
            if(!ac&&!ab)return true;return s.LastEventWasBos?ab:ac;
        }

        private static bool MtfIsH(Bars b,int idx,int len){int l=idx-len,r=idx+len;if(l<0||r>=b.Count)return false;var p=b.HighPrices[idx];for(var i=l;i<=r;i++)if(i!=idx&&b.HighPrices[i]>=p)return false;return true;}
        private static bool MtfIsL(Bars b,int idx,int len){int l=idx-len,r=idx+len;if(l<0||r>=b.Count)return false;var p=b.LowPrices[idx]; for(var i=l;i<=r;i++)if(i!=idx&&b.LowPrices[i]<=p)return false; return true;}

        private static int MtfFOB(Bars bars,DateTime time)
        {int lo=0,hi=bars.Count-1;while(lo<=hi){int mid=(lo+hi)/2;if(bars.OpenTimes[mid]==time)return mid;else if(bars.OpenTimes[mid]<time)lo=mid+1;else hi=mid-1;}return hi;}

        private static int MtfFOA(Bars bars,DateTime time)
        {int lo=0,hi=bars.Count-1,ans=-1;while(lo<=hi){int mid=(lo+hi)/2;if(bars.OpenTimes[mid]>=time){ans=mid;hi=mid-1;}else lo=mid+1;}return ans;}

        private static TimeFrame MtfParse(string t)
        {
            switch((t??"").Trim().ToUpperInvariant())
            {
                case"1":return TimeFrame.Minute;case"2":return TimeFrame.Minute2;case"3":return TimeFrame.Minute3;
                case"4":return TimeFrame.Minute4;case"5":return TimeFrame.Minute5;case"10":return TimeFrame.Minute10;
                case"15":return TimeFrame.Minute15;case"30":return TimeFrame.Minute30;case"45":return TimeFrame.Minute45;
                case"60":case"1H":return TimeFrame.Hour;case"120":case"2H":return TimeFrame.Hour2;
                case"240":case"4H":return TimeFrame.Hour4;case"480":case"8H":return TimeFrame.Hour8;
                case"720":case"12H":return TimeFrame.Hour12;case"D":case"1D":return TimeFrame.Daily;
                case"W":case"1W":return TimeFrame.Weekly;case"M":case"1M":return TimeFrame.Monthly;
                default:return TimeFrame.Minute15;
            }
        }

        private static int MtfMins(TimeFrame tf)
        {
            if(tf==TimeFrame.Minute)return 1;if(tf==TimeFrame.Minute2)return 2;if(tf==TimeFrame.Minute3)return 3;
            if(tf==TimeFrame.Minute4)return 4;if(tf==TimeFrame.Minute5)return 5;if(tf==TimeFrame.Minute10)return 10;
            if(tf==TimeFrame.Minute15)return 15;if(tf==TimeFrame.Minute30)return 30;if(tf==TimeFrame.Minute45)return 45;
            if(tf==TimeFrame.Hour)return 60;if(tf==TimeFrame.Hour2)return 120;if(tf==TimeFrame.Hour4)return 240;
            if(tf==TimeFrame.Hour8)return 480;if(tf==TimeFrame.Hour12)return 720;if(tf==TimeFrame.Daily)return 1440;
            if(tf==TimeFrame.Weekly)return 10080;if(tf==TimeFrame.Monthly)return 43200;return 0;
        }

        // ═══ OB WICK FILTER ENGINE ════════════════════════════════════════════

        private void ObWickReg(TimeFrame tf,bool en,int max,string label)
        {if(!en)return;var bars=tf==Bars.TimeFrame?Bars:MarketData.GetBars(tf);_obWickStates.Add(new ObWickTfState{TfBars=bars,Label=label,MaxCount=max});}

        private void RunObWick(int ci)
        {
            DateTime ct=Bars.OpenTimes[ci]; double lo=Bars.LowPrices[ci],hi=Bars.HighPrices[ci],cl=Bars.ClosePrices[ci];
            bool inS=true;if(ObWickOnlyMktHrs){var m=ct.Hour*60+ct.Minute;inS=m>=570&&m<=960;}
            foreach(var st in _obWickStates)
            {
                var tb=st.TfBars;var si=MtfFOB(tb,ct);if(si<3)continue;
                int cc=si-1,pv=cc-1;if(pv<0)continue;
                if(inS&&cc>st.LastDetectedSrcIdx)
                {
                    st.LastDetectedSrcIdx=cc;
                    double o1=tb.OpenPrices[pv],c1=tb.ClosePrices[pv],h1=tb.HighPrices[pv],l1=tb.LowPrices[pv];
                    double op=tb.OpenPrices[cc],cl2=tb.ClosePrices[cc];
                    bool isBull=ObWickDetectionMethodInput==ObWickDetectionMethod.Body?(o1>c1&&op<cl2&&cl2>h1):(op<h1);
                    bool isBear=ObWickDetectionMethodInput==ObWickDetectionMethod.Body?(o1<c1&&op>cl2&&cl2<l1):(op<l1);
                    if(isBull){double zt=Math.Max(h1,o1),zb=Math.Min(h1,o1);if(!ObWickDup(st.Bulls,zt)){if(st.Bulls.Count>=st.MaxCount)st.Bulls.RemoveAt(0);st.Bulls.Add(new ObWickZone{IsBull=true,Top=zt,Bottom=zb});}}
                    if(isBear){double zt=Math.Max(o1,l1),zb=Math.Min(o1,l1);if(!ObWickDup(st.Bears,zt)){if(st.Bears.Count>=st.MaxCount)st.Bears.RemoveAt(0);st.Bears.Add(new ObWickZone{IsBull=false,Top=zt,Bottom=zb});}}
                }
                ObWickMitBull(st,lo,cl);ObWickMitBear(st,hi,cl);
            }
        }

        private void ObWickMitBull(ObWickTfState st,double lo,double cl)
        {
            var mode=ObWickMitigationModeInput;bool ub=ObWickMitigationTypeInput==ObWickMitigationType.Body;
            for(int k=st.Bulls.Count-1;k>=0;k--)
            {
                var z=st.Bulls[k];var mid=(z.Top+z.Bottom)/2.0;
                if(mode==ObWickMitigationMode.Dynamic){if(ub&&cl<z.Top)z.Top=cl;else if(!ub&&lo<z.Top)z.Top=lo;}
                if(mode==ObWickMitigationMode.None&&!z.IsMitigated){bool p=ub?cl<z.Bottom:lo<z.Bottom;if(p)z.IsMitigated=true;continue;}
                bool rm=false;
                if(mode==ObWickMitigationMode.Normal||mode==ObWickMitigationMode.Dynamic)rm=ub?cl<z.Bottom:lo<z.Bottom;
                else if(mode==ObWickMitigationMode.Half)rm=ub?cl<mid:lo<mid;
                if(rm)st.Bulls.RemoveAt(k);
            }
        }

        private void ObWickMitBear(ObWickTfState st,double hi,double cl)
        {
            var mode=ObWickMitigationModeInput;bool ub=ObWickMitigationTypeInput==ObWickMitigationType.Body;
            for(int k=st.Bears.Count-1;k>=0;k--)
            {
                var z=st.Bears[k];var mid=(z.Top+z.Bottom)/2.0;
                if(mode==ObWickMitigationMode.Dynamic){if(ub&&cl>z.Bottom)z.Bottom=cl;else if(!ub&&hi>z.Bottom)z.Bottom=hi;}
                if(mode==ObWickMitigationMode.None&&!z.IsMitigated){bool p=ub?cl>z.Top:hi>z.Top;if(p)z.IsMitigated=true;continue;}
                bool rm=false;
                if(mode==ObWickMitigationMode.Normal||mode==ObWickMitigationMode.Dynamic)rm=ub?cl>z.Top:hi>z.Top;
                else if(mode==ObWickMitigationMode.Half)rm=ub?cl>mid:hi>mid;
                if(rm)st.Bears.RemoveAt(k);
            }
        }

        private static bool ObWickDup(List<ObWickZone> zones,double top){for(int i=zones.Count-1;i>=0;i--)if(zones[i].Top==top)return true;return false;}

        private bool ChkObWick(bool isLong,int bar)
        {
            if(!EnableObWickFilter)return true;if(_obWickStates.Count==0)return true;
            int start=Math.Max(0,bar-ObWickLookbackBars+1);
            foreach(var st in _obWickStates)
            {
                var zones=isLong?st.Bulls:st.Bears;
                foreach(var z in zones){if(z.IsMitigated)continue;for(int b=start;b<=bar;b++){bool t=isLong?Bars.LowPrices[b]<=z.Top:Bars.HighPrices[b]>=z.Bottom;if(t)return true;}}
            }
            Print("[OBWick BLOCKED] {0} bar={1} lookback={2}",isLong?"Long":"Short",bar,ObWickLookbackBars);return false;
        }

        // ═══ OB IMBALANCE FILTER ENGINE ══════════════════════════════════════
        //
        //  Detection mirrors "Order Blocks & Imbalance MTF.cs":
        //    htfIdx    = MtfFOB(tf.Bars, now) — last HTF bar whose open <= chart bar open
        //    seedBar   = htfIdx - 2            — same convention as indicator
        //    FVG up    = Low[htfIdx] - High[htfIdx-2] > ATR * threshold  → Bull OB
        //    FVG down  = Low[htfIdx-2] - High[htfIdx] > ATR * threshold  → Bear OB
        //    Zone      = [High[seed], Low[seed]] (Top/Bottom of seed bar)
        //    Seed guard: LastCreatedSeedTime (DateTime) — matches indicator's guard
        //
        //  State machine per zone per chart bar:
        //    1. Skip Invalidated zones immediately
        //    2. Invalidation check FIRST (matches indicator's UpdateZoneStates order):
        //         Bull: close < Bottom  → Invalidated, skip rest
        //         Bear: close > Top     → Invalidated, skip rest
        //    3. Touch check:
        //         Bull: useWick ? Low<=Top  : Close<=Top
        //         Bear: useWick ? High>=Bot : Close>=Bot
        //    4. On touch: update LastTouchBar; set FirstTouchBar on first touch;
        //                 Active → Mitigated on first touch
        //
        //  CheckImbFilter gate:
        //    Unmitigated: zone.State==Mitigated AND 0<=(bar-FirstTouchBar)<=ImbLookbackBars
        //    Mitigated  : zone.LastTouchBar>=0  AND 0<=(bar-LastTouchBar) <=ImbLookbackBars
        //    Either path passing is sufficient. Both OFF → no-op (returns true).

        private void ImbReg(bool en,TimeFrame tf,string label)
        {
            if(!en)return;
            var bars=tf==Bars.TimeFrame?Bars:MarketData.GetBars(tf);
            var atr=Indicators.AverageTrueRange(bars,14,MovingAverageType.Simple);
            _imbStates.Add(new ImbTfState{Label=label,Bars=bars,Atr=atr,MaxCount=ImbMaxZonesPerTf});
        }

        private void RunImb(int ci)
        {
            var now=Bars.OpenTimes[ci];
            foreach(var tf in _imbStates)
            {
                if(tf.Bars==null||tf.Bars.Count<3){ImbUpdate(tf,ci);continue;}

                // htfIdx: last HTF bar whose open <= current chart bar open
                int htfIdx=MtfFOB(tf.Bars,now);
                if(htfIdx<2){ImbUpdate(tf,ci);continue;}

                double h2=tf.Bars.HighPrices[htfIdx-2],l2=tf.Bars.LowPrices[htfIdx-2];
                double l0=tf.Bars.LowPrices[htfIdx],   h0=tf.Bars.HighPrices[htfIdx];
                DateTime seedTime=tf.Bars.OpenTimes[htfIdx-2];

                double atrRef=tf.Atr.Result[Math.Max(0,htfIdx-1)];
                if(!double.IsNaN(atrRef)&&atrRef>0)
                {
                    // FVG conditions — mirror indicator exactly
                    bool isBull=(l0-h2)>atrRef*ImbFvgThreshold;  // gap up  → Bull OB
                    bool isBear=(l2-h0)>atrRef*ImbFvgThreshold;  // gap dn  → Bear OB

                    // DateTime seed guard — mirrors indicator's seedTime!=LastCreatedSeedTime
                    if(seedTime!=tf.LastCreatedSeedTime)
                    {
                        if(isBull)      {ImbAdd(tf,true, h2,l2);tf.LastCreatedSeedTime=seedTime;}
                        else if(isBear) {ImbAdd(tf,false,h2,l2);tf.LastCreatedSeedTime=seedTime;}
                    }
                }

                ImbUpdate(tf,ci);
            }
        }

        private void ImbAdd(ImbTfState tf,bool isBull,double high,double low)
        {
            var z=new ImbObZone{IsBullish=isBull,State=ImbZoneState.Active,Top=Math.Max(high,low),Bottom=Math.Min(high,low)};
            tf.Zones.Add(z);if(tf.Zones.Count>tf.MaxCount)tf.Zones.RemoveAt(0);
        }

        private void ImbUpdate(ImbTfState tf,int ci)
        {
            double cl=Bars.ClosePrices[ci],lo=Bars.LowPrices[ci],hi=Bars.HighPrices[ci];
            bool useWick=ImbMitigationMethodInput==ImbMitigationMethod.Wick;
            for(int i=0;i<tf.Zones.Count;i++)
            {
                var z=tf.Zones[i];
                if(z.State==ImbZoneState.Invalidated)continue;

                // Invalidation check FIRST (mirrors indicator's UpdateZoneStates)
                if( z.IsBullish&&cl<z.Bottom){z.State=ImbZoneState.Invalidated;continue;}
                if(!z.IsBullish&&cl>z.Top)   {z.State=ImbZoneState.Invalidated;continue;}

                // Touch check
                bool touched=z.IsBullish?(useWick?lo<=z.Top:cl<=z.Top):(useWick?hi>=z.Bottom:cl>=z.Bottom);
                if(touched)
                {
                    z.LastTouchBar=ci;
                    if(z.FirstTouchBar<0)z.FirstTouchBar=ci;
                    if(z.State==ImbZoneState.Active)z.State=ImbZoneState.Mitigated;
                }
            }
        }

        // CheckImbFilter:
        //   Unmitigated path: zone state is Mitigated (had its first touch),
        //                     signal within ImbLookbackBars of FirstTouchBar.
        //   Mitigated path  : zone has been touched at least once (LastTouchBar>=0),
        //                     signal within ImbLookbackBars of LastTouchBar.
        //   Invalidated zones never pass either path.
        private bool ChkImb(bool isLong,int bar)
        {
            if(!EnableImbFilter)return true;
            if(_imbStates.Count==0)return true;
            if(!ImbUseUnmitigated&&!ImbUseMitigated)return true;

            foreach(var tf in _imbStates)
            {
                for(int i=tf.Zones.Count-1;i>=0;i--)
                {
                    var z=tf.Zones[i];
                    if(z.IsBullish!=isLong)continue;
                    if(z.State==ImbZoneState.Invalidated)continue;

                    // Unmitigated path: Active->Mitigated on first touch,
                    // signal must be within ImbLookbackBars of that first touch
                    if(ImbUseUnmitigated&&z.State==ImbZoneState.Mitigated&&z.FirstTouchBar>=0)
                    {
                        int d=bar-z.FirstTouchBar;
                        if(d>=0&&d<=ImbLookbackBars)
                        {
                            Print("[IMB PASS Unmitigated] {0} bar={1} TF={2} [B={3:F5} T={4:F5}] ft={5} d={6}",
                                isLong?"Long":"Short",bar,tf.Label,z.Bottom,z.Top,z.FirstTouchBar,d);
                            return true;
                        }
                    }

                    // Mitigated path: zone touched at least once,
                    // signal within ImbLookbackBars of last touch
                    if(ImbUseMitigated&&z.LastTouchBar>=0)
                    {
                        int d=bar-z.LastTouchBar;
                        if(d>=0&&d<=ImbLookbackBars)
                        {
                            Print("[IMB PASS Mitigated] {0} bar={1} TF={2} [B={3:F5} T={4:F5}] lt={5} d={6}",
                                isLong?"Long":"Short",bar,tf.Label,z.Bottom,z.Top,z.LastTouchBar,d);
                            return true;
                        }
                    }
                }
            }

            Print("[IMB BLOCKED] {0} bar={1} window={2} unmit={3} mit={4}",
                isLong?"Long":"Short",bar,ImbLookbackBars,ImbUseUnmitigated,ImbUseMitigated);
            return false;
        }

        // ═══ MTF FVG FILTER ENGINE ════════════════════════════════════════════

        private void FvgRegTf(string key,TimeFrame tf,bool en,int max)
        {
            if(!en)return;
            _fvgBarsByTf[key]=tf==Bars.TimeFrame?Bars:MarketData.GetBars(tf);
            _fvgBullByTf[key]=new List<FvgFilterZone>();_fvgBearByTf[key]=new List<FvgFilterZone>();
            _fvgMaxByTf[key]=max;_fvgLastBullTfIdx[key]=-1;_fvgLastBearTfIdx[key]=-1;
            _fvgPrevBullHigh2[key]=double.NaN;_fvgPrevBullLow[key]=double.NaN;
            _fvgPrevBearLow2[key]=double.NaN;_fvgPrevBearHigh[key]=double.NaN;
        }

        private void RunFvg(int ci)
        {
            foreach(var kv in _fvgBarsByTf)
            {
                var key=kv.Key;var tb=kv.Value;
                var idx=MtfFOB(tb,Bars.OpenTimes[ci]);if(idx<3)continue;
                if(FvgOnlyMktHrs){var m=Bars.OpenTimes[ci].Hour*60+Bars.OpenTimes[ci].Minute;if(m<570||m>960)continue;}
                double h=tb.HighPrices[idx-1],h2=tb.HighPrices[idx-3],l=tb.LowPrices[idx-1],l2=tb.LowPrices[idx-3];
                bool nb=h2<l,nd=l2>h;
                double ph2=_fvgPrevBullHigh2[key],pl=_fvgPrevBullLow[key],pl2=_fvgPrevBearLow2[key],ph=_fvgPrevBearHigh[key];
                var bd=(double.IsNaN(ph2)||h2!=ph2)&&(double.IsNaN(pl)||l!=pl);
                var dd=(double.IsNaN(pl2)||l2!=pl2)&&(double.IsNaN(ph)||h!=ph);
                if(nb&&_fvgLastBullTfIdx[key]!=idx&&bd)
                {var bulls=_fvgBullByTf[key];if(bulls.Count>=_fvgMaxByTf[key]){FvgRm(bulls[0]);bulls.RemoveAt(0);}var z=new FvgFilterZone{IsBull=true,Top=l,Bottom=h2};if(FvgShowActive)FvgDraw(z,ci);bulls.Add(z);_fvgLastBullTfIdx[key]=idx;}
                if(nd&&_fvgLastBearTfIdx[key]!=idx&&dd)
                {var bears=_fvgBearByTf[key];if(bears.Count>=_fvgMaxByTf[key]){FvgRm(bears[0]);bears.RemoveAt(0);}var z=new FvgFilterZone{IsBull=false,Top=l2,Bottom=h};if(FvgShowActive)FvgDraw(z,ci);bears.Add(z);_fvgLastBearTfIdx[key]=idx;}
                _fvgPrevBullHigh2[key]=h2;_fvgPrevBullLow[key]=l;_fvgPrevBearLow2[key]=l2;_fvgPrevBearHigh[key]=h;
                FvgUpd(_fvgBullByTf[key],true,ci);FvgUpd(_fvgBearByTf[key],false,ci);
            }
        }

        private void FvgUpd(List<FvgFilterZone> zones,bool bull,int ci)
        {
            var mode=FvgMitigationAction;
            for(int k=zones.Count-1;k>=0;k--)
            {
                var z=zones[k];var mid=(z.Top+z.Bottom)/2.0;
                if(mode==FvgMitigationMode.Dynamic)
                {
                    if(bull){if(FvgUseBodyMitigation&&Bars.ClosePrices[ci]<z.Top)z.Top=Bars.ClosePrices[ci];else if(!FvgUseBodyMitigation&&Bars.LowPrices[ci]<z.Top)z.Top=Bars.LowPrices[ci];}
                    else{if(FvgUseBodyMitigation&&Bars.ClosePrices[ci]>z.Bottom)z.Bottom=Bars.ClosePrices[ci];else if(!FvgUseBodyMitigation&&Bars.HighPrices[ci]>z.Bottom)z.Bottom=Bars.HighPrices[ci];}
                    if(z.Rect!=null){z.Rect.Y1=z.Top;z.Rect.Y2=z.Bottom;}
                }
                if(mode==FvgMitigationMode.None&&!z.IsMitigated)
                {
                    var pen=bull?(FvgUseBodyMitigation?Bars.ClosePrices[ci]<z.Bottom:Bars.LowPrices[ci]<z.Bottom):(FvgUseBodyMitigation?Bars.ClosePrices[ci]>z.Top:Bars.HighPrices[ci]>z.Top);
                    if(pen){z.IsMitigated=true;if(z.Rect!=null){if(FvgShowMitigated)z.Rect.Color=FvgMitigatedColor;else FvgRm(z);}}
                }
                if(z.Rect!=null&&!z.IsMitigated)z.Rect.Time2=FvgRight(ci);
                var rf=bull?(FvgUseBodyMitigation?Bars.ClosePrices[ci]<z.Bottom:Bars.LowPrices[ci]<z.Bottom):(FvgUseBodyMitigation?Bars.ClosePrices[ci]>z.Top:Bars.HighPrices[ci]>z.Top);
                var rh=bull?(FvgUseBodyMitigation?Bars.ClosePrices[ci]<mid:Bars.LowPrices[ci]<mid):(FvgUseBodyMitigation?Bars.ClosePrices[ci]>mid:Bars.HighPrices[ci]>mid);
                if(((mode==FvgMitigationMode.Normal||mode==FvgMitigationMode.Dynamic)&&rf)||(mode==FvgMitigationMode.Half&&rh)){FvgRm(z);zones.RemoveAt(k);}
            }
        }

        private void FvgDraw(FvgFilterZone z,int ci)
        {
            var id="fvg_"+(z.IsBull?"b":"s")+"_"+(_fvgChartId++);
            var r=Chart.DrawRectangle(id,Bars.OpenTimes[ci],z.Top,FvgRight(ci),z.Bottom,z.IsBull?FvgActiveBullColor:FvgActiveBearColor);
            r.IsFilled=true;r.IsInteractive=false;z.Rect=r;z.RectId=id;
        }

        private void FvgRm(FvgFilterZone z){if(z.RectId!=null){Chart.RemoveObject(z.RectId);z.Rect=null;z.RectId=null;}}

        private DateTime FvgRight(int ci)
        {
            if(Bars.Count<2)return Bars.OpenTimes[ci].AddMinutes(FvgDisplayBarsRight);
            var sp=Bars.OpenTimes[Bars.Count-1]-Bars.OpenTimes[Bars.Count-2];if(sp<=TimeSpan.Zero)sp=TimeSpan.FromMinutes(1);
            return Bars.OpenTimes[ci]+TimeSpan.FromTicks(sp.Ticks*FvgDisplayBarsRight);
        }

        private bool ChkFvg(bool isLong,int bar)
        {
            if(!EnableFvgFilter)return true;if(_fvgBarsByTf.Count==0)return true;
            bool isAnd=FvgFilterLogic==MtfLogicMode.AND;int en=_fvgBarsByTf.Count,pa=0;
            foreach(var key in _fvgBarsByTf.Keys){var zones=isLong?_fvgBullByTf[key]:_fvgBearByTf[key];if(FvgTouch(zones,isLong,bar))pa++;}
            var r=isAnd?(pa==en):(pa>0);
            if(!r)Print("[FVG BLOCKED] {0} bar={1} {2}/{3} {4}",isLong?"Long":"Short",bar,pa,en,isAnd?"AND":"OR");
            return r;
        }

        private bool FvgTouch(List<FvgFilterZone> zones,bool isBull,int bar)
        {
            if(zones.Count==0)return false;int start=Math.Max(0,bar-FvgLookbackBars+1);
            foreach(var z in zones){if(z.IsMitigated)continue;for(int b=start;b<=bar;b++){bool t=isBull?Bars.LowPrices[b]<=z.Top:Bars.HighPrices[b]>=z.Bottom;if(t)return true;}}
            return false;
        }
    }
}
