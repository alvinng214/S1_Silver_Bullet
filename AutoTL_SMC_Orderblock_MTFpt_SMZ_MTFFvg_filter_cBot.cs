// =============================================================================
// AutoTL_SMC_Orderblock_MTFpt_SMZ_MTFFvg_filter_cBot
// =============================================================================
// Base    : AutoTL_SMC_Orderblock_MTFpt_SMZ_filter_cBot (all logic preserved)
// Added   : MTF FVG Filter — embeds FVG detection from MTF FVG x2 [MK].
//
// FVG Filter logic:
//   Long  allowed when >= 1 active Bull FVG on any enabled TF was touched
//   within the last FvgLookbackBars chart bars.
//   Short allowed when >= 1 active Bear FVG on any enabled TF was touched
//   within the last FvgLookbackBars chart bars.
//
//   Touch definition:
//     Bull FVG: Bars.Low[b]  <= zone.Top    (low enters gap from above)
//     Bear FVG: Bars.High[b] >= zone.Bottom (high enters gap from below)
//
//   Mitigation modes (Normal/Dynamic/Half) remove zones when fully penetrated.
//   None mode: zone is kept alive permanently but flagged IsMitigated after
//   full penetration and therefore no longer satisfies the touch check.
//
//   OR/AND logic across enabled TFs matches the pattern of all other filters.
//
// Display / alert parameters from MTF FVG x2 [MK] that are NOT included:
//   Colors, labels, visibility min/max, overlay subsystem, IncursionAlerts,
//   IncursionPct, EntryChangeColor, MtfImb/MtfPo master toggles, TfInput.
// =============================================================================

using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Indicators;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class AutoTL_SMC_Orderblock_MTFpt_SMZ_MTFFvg_filter_cBot : Robot
    {
        // ═════════════════════════════════════════════════════════════════════
        //  ENUMS
        // ═════════════════════════════════════════════════════════════════════

        public enum ObFilter          { Atr, CumulativeMeanRange }
        public enum MitigationMode    { Close, HighLow }
        public enum MtfLogicMode      { OR, AND }
        public enum SmzLogicMode      { OR, AND }
        public enum FvgMitigationMode { Normal = 1, Dynamic = 2, None = 3, Half = 4 }
        public enum FvgMitigationType { Wicks = 1, Body = 2 }

        // ═════════════════════════════════════════════════════════════════════
        //  INNER TYPES — ATL TL state
        // ═════════════════════════════════════════════════════════════════════

        private sealed class AtlTlState
        {
            public bool   PermitSet     = false;
            public bool   PermitSetPrev = false;
            public int    LastAnchorX0  = 0;
            public int    ActiveX0      = 0;
            public double ActiveY0      = 0.0;
            public int    ActiveX1      = 0;
            public double ActiveY1      = 0.0;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  INNER TYPES — BSL/SSL, SMC, MTF
        // ═════════════════════════════════════════════════════════════════════

        private sealed class BslPivot
        {
            public double Price;
            public int    BarIndex;
            public int    Type;
        }

        private sealed class BslPool
        {
            public double Price;
            public int    PivotIndex;
        }

        private sealed class SmcObRecord
        {
            public int      Index;
            public double   Top;
            public double   Bottom;
            public bool     Bullish;
            public bool     Internal;
            public int      StructureBreakIndex;
            public DateTime Time;
        }

        private sealed class MtfTfState
        {
            public Bars     TfBars;
            public int      PivotLen;
            public bool     IsLowerTf;
            public int      TfMinutes;
            public int      LastProcessedTfBar = -1;
            public bool     CurrentTrend;
            public bool     HasTrend;
            public double   LastPivotHigh    = double.NaN;
            public double   LastPivotLow     = double.NaN;
            public double   LastBrokenHigh   = double.NaN;
            public double   LastBrokenLow    = double.NaN;
            public DateTime PivotHighTime    = DateTime.MinValue;
            public DateTime PivotLowTime     = DateTime.MinValue;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  INNER TYPE — FVG Filter Zone (no chart objects)
        // ═════════════════════════════════════════════════════════════════════

        private sealed class FvgFilterZone
        {
            public bool           IsBull;
            public double         Top;         // higher price boundary
            public double         Bottom;      // lower price boundary
            public bool           IsMitigated; // None mode: crossed but kept alive
            public ChartRectangle Rect;        // null = no chart object drawn
            public string         RectId;      // for Chart.RemoveObject
        }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — ATL Zig Zag Logic
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Pivot Period", DefaultValue = 5, MinValue = 1, Group = "ATL Zig Zag Logic")]
        public int PP { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — Long Signal Enables
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("React Major External Up TL   → Long", DefaultValue = true, Group = "Long Signal Enables")]
        public bool LongReact_MjExUp { get; set; }

        [Parameter("React Major Internal Up TL   → Long", DefaultValue = true, Group = "Long Signal Enables")]
        public bool LongReact_MjInUp { get; set; }

        [Parameter("React Minor External Up TL   → Long", DefaultValue = true, Group = "Long Signal Enables")]
        public bool LongReact_MnExUp { get; set; }

        [Parameter("React Minor Internal Up TL   → Long", DefaultValue = true, Group = "Long Signal Enables")]
        public bool LongReact_MnInUp { get; set; }

        [Parameter("Break Major External Down TL → Long", DefaultValue = true, Group = "Long Signal Enables")]
        public bool LongBreak_MjExDown { get; set; }

        [Parameter("Break Major Internal Down TL → Long", DefaultValue = true, Group = "Long Signal Enables")]
        public bool LongBreak_MjInDown { get; set; }

        [Parameter("Break Minor External Down TL → Long", DefaultValue = true, Group = "Long Signal Enables")]
        public bool LongBreak_MnExDown { get; set; }

        [Parameter("Break Minor Internal Down TL → Long", DefaultValue = true, Group = "Long Signal Enables")]
        public bool LongBreak_MnInDown { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — Short Signal Enables
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Break Major External Up TL   → Short", DefaultValue = true, Group = "Short Signal Enables")]
        public bool ShortBreak_MjExUp { get; set; }

        [Parameter("Break Major Internal Up TL   → Short", DefaultValue = true, Group = "Short Signal Enables")]
        public bool ShortBreak_MjInUp { get; set; }

        [Parameter("Break Minor External Up TL   → Short", DefaultValue = true, Group = "Short Signal Enables")]
        public bool ShortBreak_MnExUp { get; set; }

        [Parameter("Break Minor Internal Up TL   → Short", DefaultValue = true, Group = "Short Signal Enables")]
        public bool ShortBreak_MnInUp { get; set; }

        [Parameter("React Major External Down TL → Short", DefaultValue = true, Group = "Short Signal Enables")]
        public bool ShortReact_MjExDown { get; set; }

        [Parameter("React Major Internal Down TL → Short", DefaultValue = true, Group = "Short Signal Enables")]
        public bool ShortReact_MjInDown { get; set; }

        [Parameter("React Minor External Down TL → Short", DefaultValue = true, Group = "Short Signal Enables")]
        public bool ShortReact_MnExDown { get; set; }

        [Parameter("React Minor Internal Down TL → Short", DefaultValue = true, Group = "Short Signal Enables")]
        public bool ShortReact_MnInDown { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — BSL & SSL
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Pivot Left",  DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotLeft { get; set; }

        [Parameter("Pivot Right", DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotRight { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — Risk Management
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Risk % per trade",         DefaultValue = 1.0,   MinValue = 0.1,  MaxValue = 100.0, Group = "Risk Management")]
        public double RiskPercent { get; set; }

        [Parameter("Risk:Reward Ratio",        DefaultValue = 2.0,   MinValue = 0.1,  Step = 0.1,       Group = "Risk Management")]
        public double RiskRewardRatio { get; set; }

        [Parameter("Max Open Positions",        DefaultValue = 3,     MinValue = 1,    MaxValue = 100,   Group = "Risk Management")]
        public int MaxOpenPositions { get; set; }

        [Parameter("Min SL Distance (pips)",    DefaultValue = 3.0,   MinValue = 0.1,                   Group = "Risk Management")]
        public double MinSlPips { get; set; }

        [Parameter("Max SL Distance (pips)",    DefaultValue = 500.0, MinValue = 1.0,                   Group = "Risk Management")]
        public double MaxSlPips { get; set; }

        [Parameter("SL Buffer (pips)",          DefaultValue = 0.0,   MinValue = 0.0, Step = 0.1,       Group = "Risk Management")]
        public double SlBufferPips { get; set; }

        [Parameter("Instance Name",             DefaultValue = "AutoTL_SMC_MTFpt_SMZ_FVG_cBot",         Group = "Risk Management")]
        public string InstanceName { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — SMC Filter
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Swings Length", DefaultValue = 50, MinValue = 10, Group = "SMC Filter — Swing Structure")]
        public int SmcSwingsLengthInput { get; set; }

        [Parameter("Order Block Filter",   DefaultValue = ObFilter.Atr,        Group = "SMC Filter — Order Blocks")]
        public ObFilter SmcOrderBlockFilterInput { get; set; }

        [Parameter("OB Filter ATR Period", DefaultValue = 200, MinValue = 1, MaxValue = 500, Group = "SMC Filter — Order Blocks")]
        public int SmcObFilterAtrPeriod { get; set; }

        [Parameter("Order Block Mitigation", DefaultValue = MitigationMode.HighLow, Group = "SMC Filter — Order Blocks")]
        public MitigationMode SmcOrderBlockMitigationInput { get; set; }

        [Parameter("Enable Filter 1 (Internal OB)",          DefaultValue = true,  Group = "Filter 1 — Internal OB")]
        public bool EnableFilter1 { get; set; }

        [Parameter("OB Touch Window — Internal (bars)",      DefaultValue = 10, MinValue = 0, Group = "Filter 1 — Internal OB")]
        public int Filter1Lookback { get; set; }

        [Parameter("Enable Filter 2 (Swing OB)",             DefaultValue = false, Group = "Filter 2 — Swing OB")]
        public bool EnableFilter2 { get; set; }

        [Parameter("OB Touch Window — Swing (bars)",         DefaultValue = 10, MinValue = 0, Group = "Filter 2 — Swing OB")]
        public int Filter2Lookback { get; set; }

        [Parameter("Enable Min Bars From OB Origin",         DefaultValue = false, Group = "OB Quality Filters")]
        public bool EnableMinBarsFromOrigin { get; set; }

        [Parameter("Min Bars — Internal OB",                 DefaultValue = 5, MinValue = 1, Group = "OB Quality Filters")]
        public int MinBarsFromOriginInternal { get; set; }

        [Parameter("Min Bars — Swing OB",                    DefaultValue = 5, MinValue = 1, Group = "OB Quality Filters")]
        public int MinBarsFromOriginSwing { get; set; }

        [Parameter("Enable ATR Distance Filter",             DefaultValue = false, Group = "OB Quality Filters")]
        public bool EnableAtrDistanceFilter { get; set; }

        [Parameter("ATR Distance Multiplier",                DefaultValue = 1.0, MinValue = 0.1, Step = 0.1, Group = "OB Quality Filters")]
        public double AtrDistanceMultiplier { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — MTF Trend Filter
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Enable MTF Trend Filter",   DefaultValue = false,             Group = "MTF Trend Filter — General")]
        public bool EnableMtfFilter { get; set; }

        [Parameter("Multi-TF Logic (OR / AND)", DefaultValue = MtfLogicMode.OR,  Group = "MTF Trend Filter — General")]
        public MtfLogicMode MtfFilterLogic { get; set; }

        [Parameter("Enable TF1 Filter",  DefaultValue = true,  Group = "MTF Trend Filter — TF1")]
        public bool EnableMtfTf1 { get; set; }
        [Parameter("TF1 Timeframe",      DefaultValue = "15",  Group = "MTF Trend Filter — TF1")]
        public string MtfTimeframe1 { get; set; }
        [Parameter("TF1 Pivot Strength", DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter — TF1")]
        public int MtfPivotStrength1 { get; set; }
        [Parameter("TF1 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter — TF1")]
        public bool MtfIsLowerTf1 { get; set; }

        [Parameter("Enable TF2 Filter",  DefaultValue = false, Group = "MTF Trend Filter — TF2")]
        public bool EnableMtfTf2 { get; set; }
        [Parameter("TF2 Timeframe",      DefaultValue = "30",  Group = "MTF Trend Filter — TF2")]
        public string MtfTimeframe2 { get; set; }
        [Parameter("TF2 Pivot Strength", DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter — TF2")]
        public int MtfPivotStrength2 { get; set; }
        [Parameter("TF2 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter — TF2")]
        public bool MtfIsLowerTf2 { get; set; }

        [Parameter("Enable TF3 Filter",  DefaultValue = false, Group = "MTF Trend Filter — TF3")]
        public bool EnableMtfTf3 { get; set; }
        [Parameter("TF3 Timeframe",      DefaultValue = "60",  Group = "MTF Trend Filter — TF3")]
        public string MtfTimeframe3 { get; set; }
        [Parameter("TF3 Pivot Strength", DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter — TF3")]
        public int MtfPivotStrength3 { get; set; }
        [Parameter("TF3 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter — TF3")]
        public bool MtfIsLowerTf3 { get; set; }

        [Parameter("Enable TF4 Filter",  DefaultValue = false, Group = "MTF Trend Filter — TF4")]
        public bool EnableMtfTf4 { get; set; }
        [Parameter("TF4 Timeframe",      DefaultValue = "240", Group = "MTF Trend Filter — TF4")]
        public string MtfTimeframe4 { get; set; }
        [Parameter("TF4 Pivot Strength", DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter — TF4")]
        public int MtfPivotStrength4 { get; set; }
        [Parameter("TF4 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter — TF4")]
        public bool MtfIsLowerTf4 { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — SMZ Trend Filter
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Enable SMZ Trend Filter",       DefaultValue = false,           Group = "SMZ Trend Filter — General")]
        public bool EnableSmzFilter { get; set; }

        [Parameter("SMZ MA Period",                 DefaultValue = 50, MinValue = 1, Group = "SMZ Trend Filter — General")]
        public int SmzMaPeriod { get; set; }

        [Parameter("SMZ Logic (OR / AND)",          DefaultValue = SmzLogicMode.OR, Group = "SMZ Trend Filter — General")]
        public SmzLogicMode SmzFilterLogic { get; set; }

        [Parameter("Enable 1m  TF",  DefaultValue = false, Group = "SMZ Trend Filter — Timeframes")]
        public bool SmzEnable1m  { get; set; }

        [Parameter("Enable 5m  TF",  DefaultValue = false, Group = "SMZ Trend Filter — Timeframes")]
        public bool SmzEnable5m  { get; set; }

        [Parameter("Enable 15m TF",  DefaultValue = true,  Group = "SMZ Trend Filter — Timeframes")]
        public bool SmzEnable15m { get; set; }

        [Parameter("Enable 30m TF",  DefaultValue = false, Group = "SMZ Trend Filter — Timeframes")]
        public bool SmzEnable30m { get; set; }

        [Parameter("Enable 1H  TF",  DefaultValue = true,  Group = "SMZ Trend Filter — Timeframes")]
        public bool SmzEnable1h  { get; set; }

        [Parameter("Enable 4H  TF",  DefaultValue = false, Group = "SMZ Trend Filter — Timeframes")]
        public bool SmzEnable4h  { get; set; }

        [Parameter("Enable 1D  TF",  DefaultValue = false, Group = "SMZ Trend Filter — Timeframes")]
        public bool SmzEnable1d  { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — Combined TF Filter
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Enable Combined TF Filter",  DefaultValue = false, Group = "Combined TF Filter — General")]
        public bool EnableCombinedTfFilter { get; set; }

        [Parameter("Enable 1H Condition",        DefaultValue = true,  Group = "Combined TF Filter — 1H")]
        public bool CmbEnable1h { get; set; }

        [Parameter("Use SMZ SMA (1H)",           DefaultValue = true,  Group = "Combined TF Filter — 1H")]
        public bool UseCmbSmz1h { get; set; }

        [Parameter("Use MTF Pivot Trend (1H)",   DefaultValue = true,  Group = "Combined TF Filter — 1H")]
        public bool UseCmbMtf1h { get; set; }

        [Parameter("1H MTF Pivot Strength",      DefaultValue = 15, MinValue = 1, Group = "Combined TF Filter — 1H")]
        public int CmbMtfPivotStrength1h { get; set; }

        [Parameter("1H MTF Lower than chart?",   DefaultValue = false, Group = "Combined TF Filter — 1H")]
        public bool CmbMtfIsLowerTf1h { get; set; }

        [Parameter("Enable 15m Condition",       DefaultValue = true,  Group = "Combined TF Filter — 15m")]
        public bool CmbEnable15m { get; set; }

        [Parameter("Use SMZ SMA (15m)",          DefaultValue = true,  Group = "Combined TF Filter — 15m")]
        public bool UseCmbSmz15m { get; set; }

        [Parameter("Use MTF Pivot Trend (15m)",  DefaultValue = true,  Group = "Combined TF Filter — 15m")]
        public bool UseCmbMtf15m { get; set; }

        [Parameter("15m MTF Pivot Strength",     DefaultValue = 15, MinValue = 1, Group = "Combined TF Filter — 15m")]
        public int CmbMtfPivotStrength15m { get; set; }

        [Parameter("15m MTF Lower than chart?",  DefaultValue = false, Group = "Combined TF Filter — 15m")]
        public bool CmbMtfIsLowerTf15m { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — MTF FVG Filter
        //
        //  Embeds FVG detection from MTF FVG x2 [MK] (display stripped).
        //  Long  allowed when a Bull FVG on any enabled TF was touched within
        //  FvgLookbackBars chart bars (OR logic) or all enabled TFs (AND logic).
        //  Short allowed symmetrically for Bear FVGs.
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Enable MTF FVG Filter",   DefaultValue = false,                    Group = "MTF FVG Filter — General")]
        public bool EnableFvgFilter { get; set; }

        [Parameter("FVG Lookback Bars",       DefaultValue = 10,   MinValue = 1,       Group = "MTF FVG Filter — General")]
        public int FvgLookbackBars { get; set; }

        [Parameter("FVG Logic (OR / AND)",    DefaultValue = MtfLogicMode.OR,          Group = "MTF FVG Filter — General")]
        public MtfLogicMode FvgFilterLogic { get; set; }

        [Parameter("FVG Only Market Hours",   DefaultValue = false,                    Group = "MTF FVG Filter — Detection")]
        public bool FvgOnlyMktHrs { get; set; }

        [Parameter("FVG Mitigation Action",   DefaultValue = FvgMitigationMode.Normal, Group = "MTF FVG Filter — Detection")]
        public FvgMitigationMode FvgMitigationAction { get; set; }

        [Parameter("FVG Mitigation Type",     DefaultValue = FvgMitigationType.Wicks,  Group = "MTF FVG Filter — Detection")]
        public FvgMitigationType FvgMitigationTypeInput { get; set; }

        // Per-TF enable toggles (mirrors "Enabled Timeframes" group in MTF FVG x2 [MK])
        [Parameter("FVG Enable Chart TF",  DefaultValue = false, Group = "MTF FVG Filter — Timeframes")]
        public bool FvgEnableChartTf { get; set; }

        [Parameter("FVG Enable 5m",        DefaultValue = false, Group = "MTF FVG Filter — Timeframes")]
        public bool FvgEnable5m { get; set; }

        [Parameter("FVG Enable 10m",       DefaultValue = false, Group = "MTF FVG Filter — Timeframes")]
        public bool FvgEnable10m { get; set; }

        [Parameter("FVG Enable 15m",       DefaultValue = true,  Group = "MTF FVG Filter — Timeframes")]
        public bool FvgEnable15m { get; set; }

        [Parameter("FVG Enable 30m",       DefaultValue = false, Group = "MTF FVG Filter — Timeframes")]
        public bool FvgEnable30m { get; set; }

        [Parameter("FVG Enable 1h",        DefaultValue = true,  Group = "MTF FVG Filter — Timeframes")]
        public bool FvgEnable1h { get; set; }

        [Parameter("FVG Enable 4h",        DefaultValue = true,  Group = "MTF FVG Filter — Timeframes")]
        public bool FvgEnable4h { get; set; }

        [Parameter("FVG Enable 8h",        DefaultValue = false, Group = "MTF FVG Filter — Timeframes")]
        public bool FvgEnable8h { get; set; }

        [Parameter("FVG Enable 12h",       DefaultValue = false, Group = "MTF FVG Filter — Timeframes")]
        public bool FvgEnable12h { get; set; }

        [Parameter("FVG Enable Daily",     DefaultValue = true,  Group = "MTF FVG Filter — Timeframes")]
        public bool FvgEnableDaily { get; set; }

        [Parameter("FVG Enable Weekly",    DefaultValue = true,  Group = "MTF FVG Filter — Timeframes")]
        public bool FvgEnableWeekly { get; set; }

        [Parameter("FVG Enable Monthly",   DefaultValue = true,  Group = "MTF FVG Filter — Timeframes")]
        public bool FvgEnableMonthly { get; set; }

        // Per-TF max zone counts (mirrors "Max FVG Settings" group in MTF FVG x2 [MK])
        [Parameter("FVG Max Chart",   DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter — Max Count")]
        public int FvgMaxChart { get; set; }

        [Parameter("FVG Max 5m",      DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter — Max Count")]
        public int FvgMax5m { get; set; }

        [Parameter("FVG Max 10m",     DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter — Max Count")]
        public int FvgMax10m { get; set; }

        [Parameter("FVG Max 15m",     DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter — Max Count")]
        public int FvgMax15m { get; set; }

        [Parameter("FVG Max 30m",     DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter — Max Count")]
        public int FvgMax30m { get; set; }

        [Parameter("FVG Max 1h",      DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter — Max Count")]
        public int FvgMax1h { get; set; }

        [Parameter("FVG Max 4h",      DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter — Max Count")]
        public int FvgMax4h { get; set; }

        [Parameter("FVG Max 8h",      DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter — Max Count")]
        public int FvgMax8h { get; set; }

        [Parameter("FVG Max 12h",     DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter — Max Count")]
        public int FvgMax12h { get; set; }

        [Parameter("FVG Max Daily",   DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter — Max Count")]
        public int FvgMaxDaily { get; set; }

        [Parameter("FVG Max Weekly",  DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter — Max Count")]
        public int FvgMaxWeekly { get; set; }

        [Parameter("FVG Max Monthly", DefaultValue = 8, MinValue = 1, Group = "MTF FVG Filter — Max Count")]
        public int FvgMaxMonthly { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — MTF FVG Filter — Display
        //
        //  Draws FVG zones directly on the chart for visual confirmation.
        //  "Active"   = zone is alive and eligible for the touch filter.
        //  "Mitigated"= zone has been fully penetrated (None mitigation mode
        //               only — other modes delete zones on full penetration,
        //               so mitigated zones only accumulate under None mode).
        //
        //  Rectangle right edge is extended FvgDisplayBarsRight bars past the
        //  current bar each tick so zones appear to grow forward in real-time.
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Show Active FVG Zones",    DefaultValue = false, Group = "MTF FVG Filter — Display")]
        public bool FvgShowActive { get; set; }

        [Parameter("Show Mitigated FVG Zones", DefaultValue = false, Group = "MTF FVG Filter — Display")]
        public bool FvgShowMitigated { get; set; }

        [Parameter("Active Bull Color",        DefaultValue = "#4D00C800", Group = "MTF FVG Filter — Display")]
        public Color FvgActiveBullColor { get; set; }

        [Parameter("Active Bear Color",        DefaultValue = "#4DC80000", Group = "MTF FVG Filter — Display")]
        public Color FvgActiveBearColor { get; set; }

        [Parameter("Mitigated FVG Color",      DefaultValue = "#26FFFF00", Group = "MTF FVG Filter — Display")]
        public Color FvgMitigatedColor { get; set; }

        [Parameter("Display Bars Right",       DefaultValue = 10, MinValue = 1, Group = "MTF FVG Filter — Display")]
        public int FvgDisplayBarsRight { get; set; }
        // ═════════════════════════════════════════════════════════════════════

        private const int MaxBslPivots = 10;
        private const int MaxSmcObs    = 500;
        private const int SmcAtrPeriod = 200;

        // ═════════════════════════════════════════════════════════════════════
        //  ATL EMBEDDED STATE FIELDS
        // ═════════════════════════════════════════════════════════════════════

        private readonly List<string> _atlZzType  = new List<string>();
        private readonly List<double> _atlZzValue = new List<double>();
        private readonly List<int>    _atlZzIndex = new List<int>();

        private readonly List<string> _atlAdvType  = new List<string>();
        private readonly List<double> _atlAdvValue = new List<double>();
        private readonly List<int>    _atlAdvIndex = new List<int>();

        private double _atlMajorHighLevel         = double.NaN;
        private double _atlMajorLowLevel          = double.NaN;
        private bool   _atlMajorLevelsInitialized = false;

        private bool _atlLock0 = true;
        private bool _atlLock1 = true;

        private double _atlLastHighPivotValue = double.NaN;
        private int    _atlLastHighPivotIndex = -1;
        private double _atlLastLowPivotValue  = double.NaN;
        private int    _atlLastLowPivotIndex  = -1;

        private int    _atlX0     = 0;
        private double _atlY0     = 0.0;
        private string _atlT0     = string.Empty;
        private string _atlT0Prev = string.Empty;

        private double _atlPrevZzLastValue      = double.NaN;
        private char   _atlPrevZzLastTypeSuffix = '\0';

        private static readonly string[] AtlTlTypeNames =
            { "MLL", "MHH", "MHL", "MLH", "mLL", "mHH", "mHL", "mLH" };

        private readonly int[]    _atlPtrX0 = new int[8];
        private readonly double[] _atlPtrY0 = new double[8];
        private readonly int[]    _atlPtrX1 = new int[8];
        private readonly double[] _atlPtrY1 = new double[8];

        private readonly AtlTlState[] _atlTlStates = new AtlTlState[8];

        private bool _atlIsLongSignal;
        private bool _atlIsShortSignal;

        // ═════════════════════════════════════════════════════════════════════
        //  BSL/SSL FIELDS
        // ═════════════════════════════════════════════════════════════════════

        private readonly LinkedList<BslPivot> _bslPivots        = new LinkedList<BslPivot>();
        private readonly LinkedList<BslPool>  _bslBuysidePools  = new LinkedList<BslPool>();
        private readonly LinkedList<BslPool>  _bslSellsidePools = new LinkedList<BslPool>();
        private double _bslCurrentBsl = double.NaN;
        private double _bslCurrentSsl = double.NaN;

        // ═════════════════════════════════════════════════════════════════════
        //  SMC FILTER FIELDS
        // ═════════════════════════════════════════════════════════════════════

        private readonly List<SmcObRecord> _smcInternalBullObs = new List<SmcObRecord>();
        private readonly List<SmcObRecord> _smcInternalBearObs = new List<SmcObRecord>();
        private readonly List<SmcObRecord> _smcSwingBullObs    = new List<SmcObRecord>();
        private readonly List<SmcObRecord> _smcSwingBearObs    = new List<SmcObRecord>();
        private readonly List<double>      _parsedHighs        = new List<double>();
        private readonly List<double>      _parsedLows         = new List<double>();
        private readonly List<DateTime>    _times              = new List<DateTime>();

        private double _smcAtrWilder    = double.NaN;
        private double _smcAtrWilderSum = 0.0;
        private double _smcCumTr        = 0.0;

        private int    _swingLeg;
        private int    _swingTrend;
        private double _lastSwingHigh      = double.NaN;
        private double _lastSwingLow       = double.NaN;
        private int    _lastSwingHighIndex = -1;
        private int    _lastSwingLowIndex  = -1;
        private bool   _swingHighCrossed;
        private bool   _swingLowCrossed;

        private int    _internalLeg;
        private int    _internalTrend;
        private double _internalHighLevel   = double.NaN;
        private double _internalLowLevel    = double.NaN;
        private int    _internalHighIndex   = -1;
        private int    _internalLowIndex    = -1;
        private bool   _internalHighCrossed;
        private bool   _internalLowCrossed;

        private int _smcObIdCounter;
        private int _lastParsedIndex = -1;
        private int _smcWarmup;

        // ═════════════════════════════════════════════════════════════════════
        //  MTF FILTER FIELDS
        // ═════════════════════════════════════════════════════════════════════

        private MtfTfState _mtfState1;
        private MtfTfState _mtfState2;
        private MtfTfState _mtfState3;
        private MtfTfState _mtfState4;

        // ═════════════════════════════════════════════════════════════════════
        //  SMZ FILTER FIELDS
        // ═════════════════════════════════════════════════════════════════════

        private Bars _smz1mBars,  _smz5mBars,  _smz15mBars, _smz30mBars;
        private Bars _smz1hBars,  _smz4hBars,  _smz1dBars;

        private SimpleMovingAverage _smz1mSma,  _smz5mSma,  _smz15mSma, _smz30mSma;
        private SimpleMovingAverage _smz1hSma,  _smz4hSma,  _smz1dSma;

        // ═════════════════════════════════════════════════════════════════════
        //  COMBINED TF FILTER FIELDS
        // ═════════════════════════════════════════════════════════════════════

        private Bars _cmbSmz1hBars,  _cmbSmz15mBars;
        private SimpleMovingAverage  _cmbSmz1hSma,  _cmbSmz15mSma;
        private MtfTfState           _cmbMtf1h,     _cmbMtf15m;

        // ═════════════════════════════════════════════════════════════════════
        //  FVG FILTER FIELDS
        // ═════════════════════════════════════════════════════════════════════

        private readonly Dictionary<string, Bars>                _fvgBarsByTf      = new Dictionary<string, Bars>();
        private readonly Dictionary<string, List<FvgFilterZone>> _fvgBullByTf      = new Dictionary<string, List<FvgFilterZone>>();
        private readonly Dictionary<string, List<FvgFilterZone>> _fvgBearByTf      = new Dictionary<string, List<FvgFilterZone>>();
        private readonly Dictionary<string, int>                 _fvgMaxByTf       = new Dictionary<string, int>();
        private readonly Dictionary<string, int>                 _fvgLastBullTfIdx = new Dictionary<string, int>();
        private readonly Dictionary<string, int>                 _fvgLastBearTfIdx = new Dictionary<string, int>();
        private readonly Dictionary<string, double>              _fvgPrevBullHigh2 = new Dictionary<string, double>();
        private readonly Dictionary<string, double>              _fvgPrevBullLow   = new Dictionary<string, double>();
        private readonly Dictionary<string, double>              _fvgPrevBearLow2  = new Dictionary<string, double>();
        private readonly Dictionary<string, double>              _fvgPrevBearHigh  = new Dictionary<string, double>();

        private bool FvgUseBodyMitigation => FvgMitigationTypeInput == FvgMitigationType.Body;

        private int _fvgChartId; // monotonic ID counter for chart object names

        // ═════════════════════════════════════════════════════════════════════
        //  CBOT STATE
        // ═════════════════════════════════════════════════════════════════════

        private int _lastProcessed      = -1;
        private int _lastLongSignalBar  = -1;
        private int _lastShortSignalBar = -1;

        // ═════════════════════════════════════════════════════════════════════
        //  LIFECYCLE
        // ═════════════════════════════════════════════════════════════════════

        protected override void OnStart()
        {
            for (int i = 0; i < 8; i++)
                _atlTlStates[i] = new AtlTlState();

            _smcWarmup = Math.Max(SmcSwingsLengthInput, 5) + 5;

            if (EnableMtfFilter)
            {
                _mtfState1 = EnableMtfTf1 ? MtfCreateState(MtfTimeframe1, MtfPivotStrength1, MtfIsLowerTf1) : null;
                _mtfState2 = EnableMtfTf2 ? MtfCreateState(MtfTimeframe2, MtfPivotStrength2, MtfIsLowerTf2) : null;
                _mtfState3 = EnableMtfTf3 ? MtfCreateState(MtfTimeframe3, MtfPivotStrength3, MtfIsLowerTf3) : null;
                _mtfState4 = EnableMtfTf4 ? MtfCreateState(MtfTimeframe4, MtfPivotStrength4, MtfIsLowerTf4) : null;
                Print("MTF filter ON. Logic={0}. TF1={1}({2}) TF2={3}({4}) TF3={5}({6}) TF4={7}({8})",
                    MtfFilterLogic,
                    EnableMtfTf1, MtfTimeframe1, EnableMtfTf2, MtfTimeframe2,
                    EnableMtfTf3, MtfTimeframe3, EnableMtfTf4, MtfTimeframe4);
            }

            if (EnableSmzFilter)
            {
                SmzInitTf(SmzEnable1m,  TimeFrame.Minute,   ref _smz1mBars,  ref _smz1mSma);
                SmzInitTf(SmzEnable5m,  TimeFrame.Minute5,  ref _smz5mBars,  ref _smz5mSma);
                SmzInitTf(SmzEnable15m, TimeFrame.Minute15, ref _smz15mBars, ref _smz15mSma);
                SmzInitTf(SmzEnable30m, TimeFrame.Minute30, ref _smz30mBars, ref _smz30mSma);
                SmzInitTf(SmzEnable1h,  TimeFrame.Hour,     ref _smz1hBars,  ref _smz1hSma);
                SmzInitTf(SmzEnable4h,  TimeFrame.Hour4,    ref _smz4hBars,  ref _smz4hSma);
                SmzInitTf(SmzEnable1d,  TimeFrame.Daily,    ref _smz1dBars,  ref _smz1dSma);
                Print("SMZ filter ON. Logic={0}. MA={1}. TFs: 1m={2} 5m={3} 15m={4} 30m={5} 1h={6} 4h={7} 1d={8}",
                    SmzFilterLogic, SmzMaPeriod,
                    SmzEnable1m, SmzEnable5m, SmzEnable15m, SmzEnable30m,
                    SmzEnable1h, SmzEnable4h, SmzEnable1d);
            }

            if (EnableCombinedTfFilter)
            {
                if (CmbEnable1h && UseCmbSmz1h)
                {
                    _cmbSmz1hBars = MarketData.GetBars(TimeFrame.Hour);
                    _cmbSmz1hSma  = Indicators.SimpleMovingAverage(_cmbSmz1hBars.ClosePrices, SmzMaPeriod);
                }
                if (CmbEnable1h && UseCmbMtf1h)
                    _cmbMtf1h = MtfCreateStateFixed(TimeFrame.Hour, CmbMtfPivotStrength1h, CmbMtfIsLowerTf1h);
                if (CmbEnable15m && UseCmbSmz15m)
                {
                    _cmbSmz15mBars = MarketData.GetBars(TimeFrame.Minute15);
                    _cmbSmz15mSma  = Indicators.SimpleMovingAverage(_cmbSmz15mBars.ClosePrices, SmzMaPeriod);
                }
                if (CmbEnable15m && UseCmbMtf15m)
                    _cmbMtf15m = MtfCreateStateFixed(TimeFrame.Minute15, CmbMtfPivotStrength15m, CmbMtfIsLowerTf15m);
                Print("Combined TF filter ON. 1H={0}(SmzSMA={1},MtfPivot={2}) 15m={3}(SmzSMA={4},MtfPivot={5})",
                    CmbEnable1h, UseCmbSmz1h, UseCmbMtf1h,
                    CmbEnable15m, UseCmbSmz15m, UseCmbMtf15m);
            }

            if (EnableFvgFilter)
            {
                FvgRegisterTf("Chart",   Bars.TimeFrame,     FvgEnableChartTf, FvgMaxChart);
                FvgRegisterTf("5m",      TimeFrame.Minute5,  FvgEnable5m,      FvgMax5m);
                FvgRegisterTf("10m",     TimeFrame.Minute10, FvgEnable10m,     FvgMax10m);
                FvgRegisterTf("15m",     TimeFrame.Minute15, FvgEnable15m,     FvgMax15m);
                FvgRegisterTf("30m",     TimeFrame.Minute30, FvgEnable30m,     FvgMax30m);
                FvgRegisterTf("1hr",     TimeFrame.Hour,     FvgEnable1h,      FvgMax1h);
                FvgRegisterTf("4hr",     TimeFrame.Hour4,    FvgEnable4h,      FvgMax4h);
                FvgRegisterTf("8hr",     TimeFrame.Hour8,    FvgEnable8h,      FvgMax8h);
                FvgRegisterTf("12hr",    TimeFrame.Hour12,   FvgEnable12h,     FvgMax12h);
                FvgRegisterTf("Daily",   TimeFrame.Daily,    FvgEnableDaily,   FvgMaxDaily);
                FvgRegisterTf("Weekly",  TimeFrame.Weekly,   FvgEnableWeekly,  FvgMaxWeekly);
                FvgRegisterTf("Monthly", TimeFrame.Monthly,  FvgEnableMonthly, FvgMaxMonthly);
                Print("FVG filter ON. Logic={0} Lookback={1} Mitigation={2}/{3}. " +
                      "TFs: Chart={4} 5m={5} 10m={6} 15m={7} 30m={8} 1h={9} 4h={10} 8h={11} 12h={12} D={13} W={14} M={15}",
                    FvgFilterLogic, FvgLookbackBars, FvgMitigationAction, FvgMitigationTypeInput,
                    FvgEnableChartTf, FvgEnable5m, FvgEnable10m, FvgEnable15m, FvgEnable30m,
                    FvgEnable1h, FvgEnable4h, FvgEnable8h, FvgEnable12h,
                    FvgEnableDaily, FvgEnableWeekly, FvgEnableMonthly);
            }

            if (SlBufferPips > 0)
                Print("SL Buffer = {0} pips", SlBufferPips);

            Print("AutoTL_SMC_Orderblock_MTFpt_SMZ_MTFFvg_filter_cBot started. " +
                  "PP={0}, PivotL={1}, PivotR={2}, MaxPos={3}, Risk={4}%, RR={5}",
                  PP, PivotLeft, PivotRight, MaxOpenPositions, RiskPercent, RiskRewardRatio);
        }

        protected override void OnStop()
        {
            Print("AutoTL_SMC_Orderblock_MTFpt_SMZ_MTFFvg_filter_cBot stopped.");
        }

        // ═════════════════════════════════════════════════════════════════════
        //  ONBAR
        // ═════════════════════════════════════════════════════════════════════

        protected override void OnBar()
        {
            int signalBar = Bars.Count - 2;

            for (int i = _lastProcessed + 1; i <= signalBar; i++)
            {
                RunBslSsl(i);
                RunSmcFilter(i);
                RunAtl(i);
                if (EnableFvgFilter) RunFvgFilter(i);
            }
            _lastProcessed = signalBar;

            if (EnableMtfFilter)
            {
                var chartTime = Bars.OpenTimes[signalBar];
                MtfAdvanceState(_mtfState1, chartTime);
                MtfAdvanceState(_mtfState2, chartTime);
                MtfAdvanceState(_mtfState3, chartTime);
                MtfAdvanceState(_mtfState4, chartTime);
            }

            if (EnableCombinedTfFilter)
            {
                var chartTime = Bars.OpenTimes[signalBar];
                MtfAdvanceState(_cmbMtf1h,  chartTime);
                MtfAdvanceState(_cmbMtf15m, chartTime);
            }

            if (signalBar < Math.Max(2 * PP, PivotLeft + PivotRight + 1)) return;

            bool isLong  = _atlIsLongSignal;
            bool isShort = _atlIsShortSignal;
            if (!isLong && !isShort) return;

            int openCount = Positions.FindAll(InstanceName, SymbolName).Length;
            if (openCount >= MaxOpenPositions)
            {
                Print("Bar {0}: max positions ({1}) reached.", signalBar, MaxOpenPositions);
                return;
            }

            // ── Long path ─────────────────────────────────────────────────────
            if (isLong && _lastLongSignalBar != signalBar)
            {
                _lastLongSignalBar = signalBar;
                if (CheckFilters(signalBar, 1) &&
                    CheckMtfFilter(true, signalBar) &&
                    CheckSmzFilter(true, signalBar) &&
                    CheckCombinedTfFilter(true, signalBar) &&
                    CheckFvgFilter(true, signalBar))
                {
                    TryEnterLong(signalBar);
                }
            }

            openCount = Positions.FindAll(InstanceName, SymbolName).Length;
            if (openCount >= MaxOpenPositions) return;

            // ── Short path ────────────────────────────────────────────────────
            if (isShort && _lastShortSignalBar != signalBar)
            {
                _lastShortSignalBar = signalBar;
                if (CheckFilters(signalBar, -1) &&
                    CheckMtfFilter(false, signalBar) &&
                    CheckSmzFilter(false, signalBar) &&
                    CheckCombinedTfFilter(false, signalBar) &&
                    CheckFvgFilter(false, signalBar))
                {
                    TryEnterShort(signalBar);
                }
            }
        }

        // ═════════════════════════════════════════════════════════════════════
        //  ATL EMBEDDED ENGINE (verbatim from Autotrendline_BSLSSL_cBot.cs)
        // ═════════════════════════════════════════════════════════════════════

        private void RunAtl(int index)
        {
            AtlUpdateZigZag(index);
            AtlSyncAdvArray();
            AtlUpdateMajorMinor(index);

            if (_atlAdvType.Count > 2)
            {
                int last  = _atlAdvType.Count - 1;
                _atlX0    = _atlAdvIndex[last];
                _atlY0    = _atlAdvValue[last];
                _atlT0    = _atlAdvType[last];
            }

            AtlUpdatePointers();
            AtlProcessAllTrendLines(index);

            _atlT0Prev = _atlT0;
            if (_atlZzType.Count > 0)
            {
                int n = _atlZzType.Count - 1;
                _atlPrevZzLastValue      = _atlZzValue[n];
                _atlPrevZzLastTypeSuffix = _atlZzType[n][_atlZzType[n].Length - 1];
            }
        }

        private bool AtlDetectPivotHigh(int index, out double pivotValue)
        {
            pivotValue = double.NaN;
            if (index < 2 * PP) return false;
            int    pivotBar    = index - PP;
            int    windowStart = index - 2 * PP;
            double candidate   = Bars.HighPrices[pivotBar];
            double max = double.MinValue;
            for (int i = windowStart; i <= index; i++)
                if (Bars.HighPrices[i] > max) max = Bars.HighPrices[i];
            if (candidate != max) return false;
            int lastMaxBar = windowStart;
            for (int i = windowStart; i <= index; i++)
                if (Bars.HighPrices[i] == max) lastMaxBar = i;
            if (lastMaxBar != pivotBar) return false;
            pivotValue = candidate;
            return true;
        }

        private bool AtlDetectPivotLow(int index, out double pivotValue)
        {
            pivotValue = double.NaN;
            if (index < 2 * PP) return false;
            int    pivotBar    = index - PP;
            int    windowStart = index - 2 * PP;
            double candidate   = Bars.LowPrices[pivotBar];
            double min = double.MaxValue;
            for (int i = windowStart; i <= index; i++)
                if (Bars.LowPrices[i] < min) min = Bars.LowPrices[i];
            if (candidate != min) return false;
            int lastMinBar = windowStart;
            for (int i = windowStart; i <= index; i++)
                if (Bars.LowPrices[i] == min) lastMinBar = i;
            if (lastMinBar != pivotBar) return false;
            pivotValue = candidate;
            return true;
        }

        private void AtlUpdateZigZag(int index)
        {
            bool hasHigh = AtlDetectPivotHigh(index, out double highValue);
            bool hasLow  = AtlDetectPivotLow(index,  out double lowValue);
            if (!hasHigh && !hasLow) return;

            int    pivotBar = index - PP;
            double barClose = Bars.ClosePrices[index];

            if (hasHigh) { _atlLastHighPivotValue = highValue; _atlLastHighPivotIndex = pivotBar; }
            if (hasLow)  { _atlLastLowPivotValue  = lowValue;  _atlLastLowPivotIndex  = pivotBar; }

            string LabelHigh(double v) { int n = _atlZzType.Count; return n > 2 ? (_atlZzValue[n-2] < v ? "HH" : "LH") : "H"; }
            string LabelLow(double v)  { int n = _atlZzType.Count; return n > 2 ? (_atlZzValue[n-2] < v ? "HL" : "LL") : "L"; }
            void RemoveLast() { int n = _atlZzType.Count-1; _atlZzType.RemoveAt(n); _atlZzValue.RemoveAt(n); _atlZzIndex.RemoveAt(n); }
            void PushHigh(double v, int bar) { _atlZzType.Add(LabelHigh(v)); _atlZzValue.Add(v); _atlZzIndex.Add(bar); }
            void PushLow(double v, int bar)  { _atlZzType.Add(LabelLow(v));  _atlZzValue.Add(v); _atlZzIndex.Add(bar); }

            int cnt = _atlZzType.Count;

            if (hasHigh && hasLow)
            {
                if (cnt == 0) { _atlZzType.Add("H"); _atlZzValue.Add(highValue); _atlZzIndex.Add(pivotBar); }
                else
                {
                    string last = _atlZzType[cnt-1]; double lastVal = _atlZzValue[cnt-1];
                    if (last == "L" || last == "LL")
                    { if (lowValue < lastVal) { RemoveLast(); PushLow(lowValue, pivotBar); } else PushHigh(highValue, pivotBar); }
                    else if (last == "H" || last == "HH")
                    { if (highValue > lastVal) { RemoveLast(); PushHigh(highValue, pivotBar); } else PushLow(lowValue, pivotBar); }
                    else if (last == "LH")
                    {
                        if (highValue < lastVal) PushLow(lowValue, pivotBar);
                        else if (highValue > lastVal)
                        { if (barClose < lastVal) { RemoveLast(); PushHigh(highValue, pivotBar); } else if (barClose > lastVal) PushLow(lowValue, pivotBar); }
                    }
                    else if (last == "HL")
                    {
                        if (lowValue > lastVal) PushHigh(highValue, pivotBar);
                        else if (lowValue < lastVal)
                        { if (barClose > lastVal) { RemoveLast(); PushLow(lowValue, pivotBar); } else if (barClose < lastVal) PushHigh(highValue, pivotBar); }
                    }
                }
            }
            else if (hasHigh)
            {
                cnt = _atlZzType.Count;
                if (cnt == 0) { _atlZzType.Insert(0,"H"); _atlZzValue.Insert(0,highValue); _atlZzIndex.Insert(0,pivotBar); }
                else
                {
                    string last = _atlZzType[cnt-1]; double lastVal = _atlZzValue[cnt-1];
                    if (last == "L" || last == "HL" || last == "LL")
                    {
                        if (highValue > lastVal) PushHigh(highValue, pivotBar);
                        else if (highValue < lastVal) { RemoveLast(); if (!double.IsNaN(_atlLastLowPivotValue) && _atlLastLowPivotIndex >= 0) PushLow(_atlLastLowPivotValue, _atlLastLowPivotIndex); }
                    }
                    else if (last == "H" || last == "HH" || last == "LH")
                    { if (lastVal < highValue) { RemoveLast(); PushHigh(highValue, pivotBar); } }
                }
            }
            else
            {
                cnt = _atlZzType.Count;
                if (cnt == 0) { _atlZzType.Insert(0,"L"); _atlZzValue.Insert(0,lowValue); _atlZzIndex.Insert(0,pivotBar); }
                else
                {
                    string last = _atlZzType[cnt-1]; double lastVal = _atlZzValue[cnt-1];
                    if (last == "H" || last == "HH" || last == "LH")
                    {
                        if (lowValue < lastVal) PushLow(lowValue, pivotBar);
                        else if (lowValue > lastVal) { RemoveLast(); if (!double.IsNaN(_atlLastHighPivotValue) && _atlLastHighPivotIndex >= 0) PushHigh(_atlLastHighPivotValue, _atlLastHighPivotIndex); }
                    }
                    else if (last == "L" || last == "HL" || last == "LL")
                    { if (lastVal > lowValue) { RemoveLast(); PushLow(lowValue, pivotBar); } }
                }
            }

            if (!_atlMajorLevelsInitialized && _atlZzType.Count == 2)
            {
                if (_atlZzType[0] == "H") { _atlMajorHighLevel = _atlZzValue[0]; _atlMajorLowLevel = _atlZzValue[1]; }
                else                       { _atlMajorHighLevel = _atlZzValue[1]; _atlMajorLowLevel = _atlZzValue[0]; }
                _atlMajorLevelsInitialized = true;
            }

            if (_atlLock0 && _atlZzType.Count >= 1)
            {
                _atlAdvType.Insert(0, "M" + _atlZzType[0]);
                _atlAdvValue.Insert(0, _atlZzValue[0]);
                _atlAdvIndex.Insert(0, _atlZzIndex[0]);
                _atlLock0 = false;
            }

            if (_atlLock1 && _atlZzType.Count >= 2)
            {
                _atlAdvType.Insert(1, "M" + _atlZzType[1]);
                _atlAdvValue.Insert(1, _atlZzValue[1]);
                _atlAdvIndex.Insert(1, _atlZzIndex[1]);
                _atlLock1 = false;
            }
        }

        private void AtlSyncAdvArray()
        {
            if (_atlZzType.Count <= 1 || _atlAdvType.Count == 0) return;
            int    zzLast        = _atlZzType.Count - 1;
            double currentZzVal  = _atlZzValue[zzLast];
            string currentZzType = _atlZzType[zzLast];
            char   currentSuffix = currentZzType[currentZzType.Length - 1];
            if (double.IsNaN(_atlPrevZzLastValue) || currentZzVal == _atlPrevZzLastValue) return;
            if (currentSuffix != _atlPrevZzLastTypeSuffix)
            { _atlAdvType.Add("m" + currentZzType); _atlAdvValue.Add(currentZzVal); _atlAdvIndex.Add(_atlZzIndex[zzLast]); }
            else
            { int advLast = _atlAdvType.Count - 1; _atlAdvValue[advLast] = currentZzVal; _atlAdvIndex[advLast] = _atlZzIndex[zzLast]; }
        }

        private void AtlUpdateMajorMinor(int index)
        {
            if (!_atlMajorLevelsInitialized || _atlAdvType.Count <= 1) return;
            double cls = Bars.ClosePrices[index];
            string ZzType(int offset = 0) { int n = _atlZzType.Count - 1 - offset; return n >= 0 ? _atlZzType[n] : string.Empty; }

            if (cls > _atlMajorHighLevel)
            {
                int last = _atlAdvType.Count - 1; string t = _atlAdvType[last];
                if (t == "mL") { _atlAdvType[last] = "ML"; _atlMajorLowLevel = _atlAdvValue[last]; }
                else if (t == "mHL" || t == "mLL") { string p = "M" + ZzType(); if (p.Length > 1) _atlAdvType[last] = p; _atlMajorLowLevel = _atlAdvValue[last]; }
                else if (t == "mLH" || t == "mHH" || t == "MLH" || t == "MHH")
                { if (last >= 1) { string t2 = _atlAdvType[last-1]; if (t2 == "mHL" || t2 == "mLL") { string p = "M" + ZzType(1); if (p.Length > 1) _atlAdvType[last-1] = p; _atlMajorLowLevel = _atlAdvValue[last-1]; } } }
            }
            {
                int last = _atlAdvType.Count - 1; string t = _atlAdvType[last];
                if (_atlAdvValue[last] > _atlMajorHighLevel)
                {
                    if (t == "mH") { _atlAdvType[last] = "MH"; _atlMajorHighLevel = _atlAdvValue[last]; }
                    else if (t == "mLH") { string p = "M" + ZzType(); if (p.Length > 1) _atlAdvType[last] = p; _atlMajorHighLevel = _atlAdvValue[last]; }
                    else if (t == "mHH" || t == "MHH") { string p = "M" + ZzType(); if (p.Length > 1) _atlAdvType[last] = p; _atlMajorHighLevel = _atlAdvValue[last]; }
                }
            }

            if (cls < _atlMajorLowLevel)
            {
                int last = _atlAdvType.Count - 1; string t = _atlAdvType[last];
                if (t == "mH") { _atlAdvType[last] = "MH"; _atlMajorHighLevel = _atlAdvValue[last]; }
                else if (t == "mLH" || t == "mHH") { string p = "M" + ZzType(); if (p.Length > 1) _atlAdvType[last] = p; _atlMajorHighLevel = _atlAdvValue[last]; }
                else if (t == "mHL" || t == "mLL" || t == "MHL" || t == "MLL")
                { if (last >= 1) { string t2 = _atlAdvType[last-1]; if (t2 == "mLH" || t2 == "mHH") { string p = "M" + ZzType(1); if (p.Length > 1) _atlAdvType[last-1] = p; _atlMajorHighLevel = _atlAdvValue[last-1]; } } }
            }
            {
                int last = _atlAdvType.Count - 1; string t = _atlAdvType[last];
                if (_atlAdvValue[last] < _atlMajorLowLevel)
                {
                    if (t == "mL") { _atlAdvType[last] = "ML"; _atlMajorLowLevel = _atlAdvValue[last]; }
                    else if (t == "mHL" || t == "mLL" || t == "MLL") { string p = "M" + ZzType(); if (p.Length > 1) _atlAdvType[last] = p; _atlMajorLowLevel = _atlAdvValue[last]; }
                }
            }
        }

        private void AtlUpdatePointers()
        {
            if (_atlT0 == _atlT0Prev) return;
            for (int i = 0; i < 8; i++)
            {
                if (_atlT0 != AtlTlTypeNames[i]) continue;
                if (_atlPtrX0[i] == 0) { _atlPtrX0[i] = _atlX0; _atlPtrY0[i] = _atlY0; }
                else if (_atlPtrX1[i] == 0) { _atlPtrX1[i] = _atlX0; _atlPtrY1[i] = _atlY0; }
                else { _atlPtrX0[i] = _atlPtrX1[i]; _atlPtrY0[i] = _atlPtrY1[i]; _atlPtrX1[i] = _atlX0; _atlPtrY1[i] = _atlY0; }
            }
        }

        private void AtlProcessAllTrendLines(int index)
        {
            _atlIsLongSignal  = false;
            _atlIsShortSignal = false;
            AtlProcessTrendLine(index, 0, true,  ShortBreak_MjExUp,   LongReact_MjExUp);
            AtlProcessTrendLine(index, 1, false, ShortReact_MjExDown, LongBreak_MjExDown);
            AtlProcessTrendLine(index, 2, true,  ShortBreak_MjInUp,   LongReact_MjInUp);
            AtlProcessTrendLine(index, 3, false, ShortReact_MjInDown, LongBreak_MjInDown);
            AtlProcessTrendLine(index, 4, true,  ShortBreak_MnExUp,   LongReact_MnExUp);
            AtlProcessTrendLine(index, 5, false, ShortReact_MnExDown, LongBreak_MnExDown);
            AtlProcessTrendLine(index, 6, true,  ShortBreak_MnInUp,   LongReact_MnInUp);
            AtlProcessTrendLine(index, 7, false, ShortReact_MnInDown, LongBreak_MnInDown);
        }

        private void AtlProcessTrendLine(int index, int tlIdx, bool isUp,
            bool enableBreakShort, bool enableReactLong)
        {
            AtlTlState state = _atlTlStates[tlIdx];
            int x0 = _atlPtrX0[tlIdx]; double y0 = _atlPtrY0[tlIdx];
            int x1 = _atlPtrX1[tlIdx]; double y1 = _atlPtrY1[tlIdx];

            state.PermitSetPrev = state.PermitSet;

            if (x0 != 0 && x1 != 0 && x0 != state.LastAnchorX0)
            {
                state.LastAnchorX0 = x0;
                bool correctSlope = isUp ? (y1 > y0) : (y1 < y0);
                bool permit = false;
                if (correctSlope)
                {
                    permit = true;
                    for (int barI = x0 + 1; barI <= index; barI++)
                    {
                        double lp = AtlLinePrice(x0, y0, x1, y1, barI);
                        if (isUp ? Bars.ClosePrices[barI] <= lp : Bars.ClosePrices[barI] >= lp) { permit = false; break; }
                    }
                }
                if (permit) { state.ActiveX0 = x0; state.ActiveY0 = y0; state.ActiveX1 = x1; state.ActiveY1 = y1; state.PermitSet = true; }
            }

            if (state.PermitSet)
            {
                if (state.ActiveX0 == 0) state.PermitSet = false;
                else
                {
                    double lp = AtlLinePrice(state.ActiveX0, state.ActiveY0, state.ActiveX1, state.ActiveY1, index);
                    if (isUp ? Bars.ClosePrices[index] <= lp : Bars.ClosePrices[index] >= lp) state.PermitSet = false;
                }
            }

            bool alertBreak = state.PermitSetPrev && !state.PermitSet;
            bool alertReact = false;
            if (state.PermitSet && state.ActiveX0 != 0)
            {
                double lp = AtlLinePrice(state.ActiveX0, state.ActiveY0, state.ActiveX1, state.ActiveY1, index);
                alertReact = isUp
                    ? (Bars.ClosePrices[index] > lp && Bars.LowPrices[index]  < lp)
                    : (Bars.ClosePrices[index] < lp && Bars.HighPrices[index] > lp);
            }

            if (isUp)
            {
                if (alertBreak && enableBreakShort) _atlIsShortSignal = true;
                if (alertReact && enableReactLong)  _atlIsLongSignal  = true;
            }
            else
            {
                if (alertBreak && enableReactLong)  _atlIsLongSignal  = true;
                if (alertReact && enableBreakShort) _atlIsShortSignal = true;
            }
        }

        private static double AtlLinePrice(int x0, double y0, int x1, double y1, int atBar)
        {
            if (x1 == x0) return y0;
            return y0 + (y1 - y0) * (double)(atBar - x0) / (x1 - x0);
        }

        // ═════════════════════════════════════════════════════════════════════
        //  BSL/SSL ENGINE
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
                BslUnshiftPivot(new BslPivot { Price = candidateHigh, BarIndex = pivotIndex, Type =  1 });
            if (BslIsPivotLow(candidateLow, leftStart, rightEnd))
                BslUnshiftPivot(new BslPivot { Price = candidateLow,  BarIndex = pivotIndex, Type = -1 });
        }

        private bool BslIsPivotHigh(double candidate, int start, int end)
        {
            double max = double.MinValue;
            for (int i = start; i <= end; i++) if (Bars.HighPrices[i] > max) max = Bars.HighPrices[i];
            return candidate == max;
        }

        private bool BslIsPivotLow(double candidate, int start, int end)
        {
            double min = double.MaxValue;
            for (int i = start; i <= end; i++) if (Bars.LowPrices[i] < min) min = Bars.LowPrices[i];
            return candidate == min;
        }

        private void BslUnshiftPivot(BslPivot p)
        {
            if (_bslPivots.First != null)
            {
                var f = _bslPivots.First.Value;
                if (f.BarIndex == p.BarIndex && f.Type == p.Type && Math.Abs(f.Price - p.Price) < Symbol.PipSize * 0.1) return;
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
            while (node != null) { var next = node.Next; if (Bars.LowPrices[index] <= node.Value.Price) _bslSellsidePools.Remove(node); node = next; }
            node = _bslBuysidePools.First;
            while (node != null) { var next = node.Next; if (Bars.HighPrices[index] >= node.Value.Price) _bslBuysidePools.Remove(node); node = next; }
        }

        // ═════════════════════════════════════════════════════════════════════
        //  SMC OB FILTER
        // ═════════════════════════════════════════════════════════════════════

        private bool CheckFilters(int index, int cond)
        {
            var isBull = cond > 0;
            bool? f1 = null, f2 = null;
            if (EnableFilter1) { var pool = isBull ? _smcInternalBullObs : _smcInternalBearObs; f1 = HasActiveTouchInLookback(pool, index, Filter1Lookback, isBull); }
            if (EnableFilter2) { var pool = isBull ? _smcSwingBullObs    : _smcSwingBearObs;    f2 = HasActiveTouchInLookback(pool, index, Filter2Lookback, isBull); }
            if (f1.HasValue && f2.HasValue) { if (!f1.Value && !f2.Value) { Print("[Filter BLOCKED] {0} at bar {1}: no OB touched within {2}/{3} bars.", isBull?"Long":"Short", index, Filter1Lookback, Filter2Lookback); return false; } return true; }
            if (f1.HasValue && !f1.Value) { Print("[Filter1 BLOCKED] {0} at bar {1}: no internal OB touched within {2} bars.", isBull?"Long":"Short", index, Filter1Lookback); return false; }
            if (f2.HasValue && !f2.Value) { Print("[Filter2 BLOCKED] {0} at bar {1}: no swing OB touched within {2} bars.",    isBull?"Long":"Short", index, Filter2Lookback); return false; }
            return true;
        }

        private bool HasActiveTouchInLookback(List<SmcObRecord> pool, int signalBar, int lookback, bool bullish)
        {
            if (pool.Count == 0) return false;
            var currentAtr = double.IsNaN(_smcAtrWilder) ? 0.0 : _smcAtrWilder;
            var closeNow   = Bars.ClosePrices[signalBar];
            foreach (var ob in pool)
            {
                if (EnableMinBarsFromOrigin) { var minBars = ob.Internal ? MinBarsFromOriginInternal : MinBarsFromOriginSwing; if (signalBar - ob.Index < minBars) continue; }
                if (EnableAtrDistanceFilter && AtrDistanceMultiplier > 0 && currentAtr > 0)
                { var advance = bullish ? closeNow - ob.Top : ob.Bottom - closeNow; if (advance < AtrDistanceMultiplier * currentAtr) continue; }
                var lastTouchBar = -1;
                for (var b = ob.StructureBreakIndex + 1; b <= signalBar; b++)
                { if (bullish && Bars.LowPrices[b] <= ob.Top) lastTouchBar = b; if (!bullish && Bars.HighPrices[b] >= ob.Bottom) lastTouchBar = b; }
                if (lastTouchBar < 0) continue;
                if (signalBar - lastTouchBar <= lookback) return true;
            }
            return false;
        }

        private void RunSmcFilter(int index)
        {
            for (var i = _lastParsedIndex + 1; i <= index; i++) UpdateSmcParsedArrays(i);
            _lastParsedIndex = index;
            if (index < _smcWarmup) return;

            const int iLen = 5;
            var sLen = Math.Max(5, SmcSwingsLengthInput);

            var internalLegNow = ComputeLeg(index, iLen, _internalLeg);
            var internalDc     = internalLegNow - _internalLeg;
            if (internalDc != 0)
            { if (internalDc == 1) { _internalLowLevel = Bars.LowPrices[index-iLen]; _internalLowIndex = index-iLen; _internalLowCrossed = false; } else { _internalHighLevel = Bars.HighPrices[index-iLen]; _internalHighIndex = index-iLen; _internalHighCrossed = false; } }
            _internalLeg = internalLegNow;

            var swingLegNow = ComputeLeg(index, sLen, _swingLeg);
            var swingDc     = swingLegNow - _swingLeg;
            if (swingDc != 0)
            { if (swingDc == 1) { _lastSwingLow = Bars.LowPrices[index-sLen]; _lastSwingLowIndex = index-sLen; _swingLowCrossed = false; } else { _lastSwingHigh = Bars.HighPrices[index-sLen]; _lastSwingHighIndex = index-sLen; _swingHighCrossed = false; } }
            _swingLeg = swingLegNow;

            var close = Bars.ClosePrices[index];
            if (!double.IsNaN(_internalHighLevel) && !_internalHighCrossed && close > _internalHighLevel) { _internalHighCrossed = true; _internalTrend = 1;  StoreSmcOrderBlock(_internalHighIndex, true,  1,  index); }
            if (!double.IsNaN(_internalLowLevel)  && !_internalLowCrossed  && close < _internalLowLevel)  { _internalLowCrossed  = true; _internalTrend = -1; StoreSmcOrderBlock(_internalLowIndex,  true,  -1, index); }
            if (!double.IsNaN(_lastSwingHigh)      && !_swingHighCrossed    && close > _lastSwingHigh)     { _swingHighCrossed    = true; _swingTrend    = 1;  StoreSmcOrderBlock(_lastSwingHighIndex, false, 1,  index); }
            if (!double.IsNaN(_lastSwingLow)       && !_swingLowCrossed     && close < _lastSwingLow)      { _swingLowCrossed     = true; _swingTrend    = -1; StoreSmcOrderBlock(_lastSwingLowIndex,  false, -1, index); }

            ManageSmcObList(_smcInternalBullObs, index, true);
            ManageSmcObList(_smcInternalBearObs, index, false);
            ManageSmcObList(_smcSwingBullObs,    index, true);
            ManageSmcObList(_smcSwingBearObs,    index, false);
        }

        private void UpdateSmcParsedArrays(int index)
        {
            double tr;
            if (index == 0) { _smcCumTr = 0; _smcAtrWilderSum = 0; _smcAtrWilder = double.NaN; tr = Bars.HighPrices[0] - Bars.LowPrices[0]; }
            else
            {
                var pc = Bars.ClosePrices[index-1];
                tr = Math.Max(Bars.HighPrices[index]-Bars.LowPrices[index], Math.Max(Math.Abs(Bars.HighPrices[index]-pc), Math.Abs(Bars.LowPrices[index]-pc)));
                _smcCumTr += tr;
                if      (index <  SmcAtrPeriod) { _smcAtrWilderSum += tr; _smcAtrWilder = double.NaN; }
                else if (index == SmcAtrPeriod) { _smcAtrWilderSum += tr; _smcAtrWilder = _smcAtrWilderSum / SmcAtrPeriod; }
                else                            { _smcAtrWilder = (_smcAtrWilder * (SmcAtrPeriod-1) + tr) / SmcAtrPeriod; }
            }
            var vm = SmcOrderBlockFilterInput == ObFilter.Atr ? (double.IsNaN(_smcAtrWilder) ? double.MaxValue : _smcAtrWilder) : (_smcCumTr / Math.Max(1, index));
            var hv = (Bars.HighPrices[index] - Bars.LowPrices[index]) >= 2.0 * vm;
            _parsedHighs.Add(hv ? Bars.LowPrices[index]  : Bars.HighPrices[index]);
            _parsedLows.Add( hv ? Bars.HighPrices[index] : Bars.LowPrices[index]);
            _times.Add(Bars.OpenTimes[index]);
        }

        private void StoreSmcOrderBlock(int pivotIndex, bool isInternal, int bias, int breakIndex)
        {
            if (pivotIndex < 0 || pivotIndex >= breakIndex || breakIndex >= _parsedHighs.Count) return;
            var parsedIndex = pivotIndex;
            if (bias == -1) { var maxV = double.MinValue; for (var i = pivotIndex; i < breakIndex; i++) if (_parsedHighs[i] > maxV) { maxV = _parsedHighs[i]; parsedIndex = i; } }
            else            { var minV = double.MaxValue; for (var i = pivotIndex; i < breakIndex; i++) if (_parsedLows[i]  < minV) { minV = _parsedLows[i];  parsedIndex = i; } }
            var bullish = bias == 1;
            var ob = new SmcObRecord { Index = parsedIndex, Top = _parsedHighs[parsedIndex], Bottom = _parsedLows[parsedIndex], Bullish = bullish, Internal = isInternal, Time = _times[parsedIndex], StructureBreakIndex = breakIndex };
            _smcObIdCounter++;
            var list = isInternal ? (bullish ? _smcInternalBullObs : _smcInternalBearObs) : (bullish ? _smcSwingBullObs : _smcSwingBearObs);
            if (list.Count >= MaxSmcObs) list.RemoveAt(list.Count - 1);
            list.Insert(0, ob);
        }

        private void ManageSmcObList(List<SmcObRecord> list, int index, bool bullish)
        {
            var bearSrc = SmcOrderBlockMitigationInput == MitigationMode.Close ? Bars.ClosePrices[index] : Bars.HighPrices[index];
            var bullSrc = SmcOrderBlockMitigationInput == MitigationMode.Close ? Bars.ClosePrices[index] : Bars.LowPrices[index];
            for (var i = list.Count-1; i >= 0; i--) { var ob = list[i]; if ((bullish && bullSrc < ob.Bottom) || (!bullish && bearSrc > ob.Top)) list.RemoveAt(i); }
        }

        private int ComputeLeg(int index, int size, int previousLeg)
        {
            if (index - size < 1) return previousLeg;
            var refHigh = Bars.HighPrices[index-size]; var refLow = Bars.LowPrices[index-size];
            var highest = double.MinValue; var lowest = double.MaxValue;
            for (var i = Math.Max(0, index-size+1); i <= index; i++) { if (Bars.HighPrices[i] > highest) highest = Bars.HighPrices[i]; if (Bars.LowPrices[i] < lowest) lowest = Bars.LowPrices[i]; }
            if (refHigh > highest) return 0;
            if (refLow  < lowest)  return 1;
            return previousLeg;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  TRADE ENTRY
        // ═════════════════════════════════════════════════════════════════════

        private void TryEnterLong(int signalBar)
        {
            double entry = Symbol.Ask; double sslLevel = _bslCurrentSsl;
            if (double.IsNaN(sslLevel) || sslLevel <= 0) { Print("Bar {0}: LONG skipped – SSL unavailable.", signalBar); return; }
            if (sslLevel >= entry) { Print("Bar {0}: LONG skipped – SSL {1:F5} not below entry {2:F5}.", signalBar, sslLevel, entry); return; }
            double slPips = (entry - sslLevel) / Symbol.PipSize + SlBufferPips;
            if (!ValidateSlPips(signalBar, "LONG", slPips)) return;
            double volume = GetRiskVolume(Account.Equity * (RiskPercent / 100.0), slPips);
            if (volume <= 0) { Print("Bar {0}: LONG skipped – volume is 0.", signalBar); return; }
            Print("Bar {0}: LONG  | Entry={1:F5} | SSL={2:F5} | SL={3:F1}p (buf={4:F1}) | TP={5:F1}p | Vol={6}", signalBar, entry, sslLevel, slPips, SlBufferPips, slPips * RiskRewardRatio, volume);
            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, InstanceName, slPips, slPips * RiskRewardRatio);
        }

        private void TryEnterShort(int signalBar)
        {
            double entry = Symbol.Bid; double bslLevel = _bslCurrentBsl;
            if (double.IsNaN(bslLevel) || bslLevel <= 0) { Print("Bar {0}: SHORT skipped – BSL unavailable.", signalBar); return; }
            if (bslLevel <= entry) { Print("Bar {0}: SHORT skipped – BSL {1:F5} not above entry {2:F5}.", signalBar, bslLevel, entry); return; }
            double slPips = (bslLevel - entry) / Symbol.PipSize + SlBufferPips;
            if (!ValidateSlPips(signalBar, "SHORT", slPips)) return;
            double volume = GetRiskVolume(Account.Equity * (RiskPercent / 100.0), slPips);
            if (volume <= 0) { Print("Bar {0}: SHORT skipped – volume is 0.", signalBar); return; }
            Print("Bar {0}: SHORT | Entry={1:F5} | BSL={2:F5} | SL={3:F1}p (buf={4:F1}) | TP={5:F1}p | Vol={6}", signalBar, entry, bslLevel, slPips, SlBufferPips, slPips * RiskRewardRatio, volume);
            ExecuteMarketOrder(TradeType.Sell, SymbolName, volume, InstanceName, slPips, slPips * RiskRewardRatio);
        }

        private bool ValidateSlPips(int signalBar, string direction, double slPips)
        {
            if (slPips < MinSlPips) { Print("Bar {0}: {1} skipped – SL {2:F1} < min {3:F1} pips.", signalBar, direction, slPips, MinSlPips); return false; }
            if (slPips > MaxSlPips) { Print("Bar {0}: {1} skipped – SL {2:F1} > max {3:F1} pips.", signalBar, direction, slPips, MaxSlPips); return false; }
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
        //  SMZ TREND FILTER
        // ═════════════════════════════════════════════════════════════════════

        private void SmzInitTf(bool enabled, TimeFrame tf, ref Bars barsField, ref SimpleMovingAverage smaField)
        {
            if (!enabled) return;
            barsField = MarketData.GetBars(tf);
            smaField  = Indicators.SimpleMovingAverage(barsField.ClosePrices, SmzMaPeriod);
        }

        private bool CheckSmzFilter(bool isLong, int signalBar)
        {
            if (!EnableSmzFilter) return true;
            var isAnd = SmzFilterLogic == SmzLogicMode.AND;
            var enabled = 0; var passing = 0;
            SmzCheckTf("1m",  _smz1mBars,  _smz1mSma,  SmzEnable1m,  isLong, ref enabled, ref passing);
            SmzCheckTf("5m",  _smz5mBars,  _smz5mSma,  SmzEnable5m,  isLong, ref enabled, ref passing);
            SmzCheckTf("15m", _smz15mBars, _smz15mSma, SmzEnable15m, isLong, ref enabled, ref passing);
            SmzCheckTf("30m", _smz30mBars, _smz30mSma, SmzEnable30m, isLong, ref enabled, ref passing);
            SmzCheckTf("1H",  _smz1hBars,  _smz1hSma,  SmzEnable1h,  isLong, ref enabled, ref passing);
            SmzCheckTf("4H",  _smz4hBars,  _smz4hSma,  SmzEnable4h,  isLong, ref enabled, ref passing);
            SmzCheckTf("1D",  _smz1dBars,  _smz1dSma,  SmzEnable1d,  isLong, ref enabled, ref passing);
            if (enabled == 0) return true;
            var result = isAnd ? (passing == enabled) : (passing > 0);
            if (!result)
            {
                var sb = new System.Text.StringBuilder();
                SmzAppendStatus(sb, "1m", _smz1mBars, _smz1mSma, SmzEnable1m); SmzAppendStatus(sb, "5m", _smz5mBars, _smz5mSma, SmzEnable5m);
                SmzAppendStatus(sb, "15m", _smz15mBars, _smz15mSma, SmzEnable15m); SmzAppendStatus(sb, "30m", _smz30mBars, _smz30mSma, SmzEnable30m);
                SmzAppendStatus(sb, "1H", _smz1hBars, _smz1hSma, SmzEnable1h); SmzAppendStatus(sb, "4H", _smz4hBars, _smz4hSma, SmzEnable4h);
                SmzAppendStatus(sb, "1D", _smz1dBars, _smz1dSma, SmzEnable1d);
                Print("[SMZ BLOCKED] {0} bar={1} passing={2}/{3} logic={4} | {5}", isLong?"Long":"Short", signalBar, passing, enabled, isAnd?"AND":"OR", sb);
            }
            return result;
        }

        private void SmzCheckTf(string label, Bars bars, SimpleMovingAverage sma, bool tfEnabled, bool isLong, ref int enabledCount, ref int passingCount)
        {
            if (!tfEnabled || bars == null || sma == null) return;
            enabledCount++;
            var idx = bars.Count - 1;
            if (idx < SmzMaPeriod - 1) { passingCount++; return; }
            var smaValue = sma.Result[idx];
            if (double.IsNaN(smaValue)) { passingCount++; return; }
            if (isLong == (bars.ClosePrices[idx] > smaValue)) passingCount++;
        }

        private void SmzAppendStatus(System.Text.StringBuilder sb, string label, Bars bars, SimpleMovingAverage sma, bool tfEnabled)
        {
            if (!tfEnabled) return;
            if (bars == null || sma == null) { sb.Append(label).Append(":Off "); return; }
            var idx = bars.Count - 1;
            if (idx < SmzMaPeriod - 1) { sb.Append(label).Append(":Warm "); return; }
            var smaVal = sma.Result[idx];
            if (double.IsNaN(smaVal)) { sb.Append(label).Append(":NaN "); return; }
            sb.Append(label).Append(bars.ClosePrices[idx] > smaVal ? ":Bull " : ":Bear ");
        }

        // ═════════════════════════════════════════════════════════════════════
        //  COMBINED TF FILTER
        // ═════════════════════════════════════════════════════════════════════

        private bool CheckCombinedTfFilter(bool isLong, int signalBar)
        {
            if (!EnableCombinedTfFilter) return true;
            var pass1h  = !CmbEnable1h  || CheckCombinedTfCondition(isLong, "1H",  UseCmbSmz1h,  _cmbSmz1hBars,  _cmbSmz1hSma,  UseCmbMtf1h,  _cmbMtf1h);
            var pass15m = !CmbEnable15m || CheckCombinedTfCondition(isLong, "15m", UseCmbSmz15m, _cmbSmz15mBars, _cmbSmz15mSma, UseCmbMtf15m, _cmbMtf15m);
            var result = pass1h && pass15m;
            if (!result)
            {
                var sb = new System.Text.StringBuilder();
                if (CmbEnable1h)  { sb.Append("1H:");  CmbAppendConditionStatus(sb, isLong, UseCmbSmz1h,  _cmbSmz1hBars,  _cmbSmz1hSma,  UseCmbMtf1h,  _cmbMtf1h);  sb.Append(' '); }
                if (CmbEnable15m) { sb.Append("15m:"); CmbAppendConditionStatus(sb, isLong, UseCmbSmz15m, _cmbSmz15mBars, _cmbSmz15mSma, UseCmbMtf15m, _cmbMtf15m); }
                Print("[CMB BLOCKED] {0} bar={1} 1H={2} 15m={3} | {4}", isLong?"Long":"Short", signalBar, pass1h?"PASS":"FAIL", pass15m?"PASS":"FAIL", sb);
            }
            return result;
        }

        private bool CheckCombinedTfCondition(bool isLong, string label, bool useSmz, Bars smzBars, SimpleMovingAverage smzSma, bool useMtf, MtfTfState mtfState)
        {
            if (!useSmz && !useMtf) return true;
            bool smzPass = false;
            if (useSmz && smzBars != null && smzSma != null)
            {
                var idx = smzBars.Count - 1;
                if (idx < SmzMaPeriod - 1) smzPass = true;
                else { var smaVal = smzSma.Result[idx]; smzPass = double.IsNaN(smaVal) || (isLong == (smzBars.ClosePrices[idx] > smaVal)); }
            }
            bool mtfPass = false;
            if (useMtf && mtfState != null) mtfPass = !mtfState.HasTrend || (mtfState.CurrentTrend == isLong);
            return smzPass || mtfPass;
        }

        private void CmbAppendConditionStatus(System.Text.StringBuilder sb, bool isLong, bool useSmz, Bars smzBars, SimpleMovingAverage smzSma, bool useMtf, MtfTfState mtfState)
        {
            if (useSmz)
            {
                sb.Append("SMZ=");
                if (smzBars == null || smzSma == null) sb.Append("Off");
                else { var idx = smzBars.Count-1; if (idx < SmzMaPeriod-1) sb.Append("Warm"); else { var v = smzSma.Result[idx]; sb.Append(double.IsNaN(v) ? "NaN" : (smzBars.ClosePrices[idx] > v ? "Bull" : "Bear")); } }
                sb.Append(' ');
            }
            if (useMtf) { sb.Append("MTF="); if (mtfState == null) sb.Append("Off"); else if (!mtfState.HasTrend) sb.Append("Warm"); else sb.Append(mtfState.CurrentTrend ? "Bull" : "Bear"); }
        }

        private MtfTfState MtfCreateStateFixed(TimeFrame tf, int pivotStrength, bool isLowerTf)
        {
            var bars = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf);
            return new MtfTfState { TfBars = bars, PivotLen = Math.Max(1, pivotStrength), IsLowerTf = isLowerTf, TfMinutes = MtfTfMinutes(tf) };
        }

        // ═════════════════════════════════════════════════════════════════════
        //  MTF PIVOT-TREND FILTER ENGINE
        // ═════════════════════════════════════════════════════════════════════

        private MtfTfState MtfCreateState(string tfInput, int pivotStrength, bool isLowerTf)
        {
            var tf = MtfParseTimeFrame(tfInput);
            var bars = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf);
            return new MtfTfState { TfBars = bars, PivotLen = Math.Max(1, pivotStrength), IsLowerTf = isLowerTf, TfMinutes = MtfTfMinutes(tf) };
        }

        private void MtfAdvanceState(MtfTfState s, DateTime chartTime)
        {
            if (s == null) return;
            var tfBarIndex = MtfResolveTfBar(s, chartTime);
            if (tfBarIndex < 0) return;
            for (var i = s.LastProcessedTfBar + 1; i <= tfBarIndex; i++) MtfProcessCalcBar(s, i);
            if (tfBarIndex > s.LastProcessedTfBar) s.LastProcessedTfBar = tfBarIndex;
        }

        private void MtfProcessCalcBar(MtfTfState s, int tfBarIndex)
        {
            var bars = s.TfBars;
            var prevLastPivotHigh = s.LastPivotHigh;
            var prevLastPivotLow  = s.LastPivotLow;

            if (tfBarIndex >= s.PivotLen * 2)
            {
                var pivotIdx = tfBarIndex - s.PivotLen;
                if (MtfIsPivotHigh(bars, pivotIdx, s.PivotLen))
                {
                    var pp = bars.HighPrices[pivotIdx];
                    s.LastPivotHigh = s.CurrentTrend ? (double.IsNaN(s.LastPivotHigh) ? pp : Math.Max(pp, s.LastPivotHigh)) : pp;
                    if (s.LastPivotHigh != prevLastPivotHigh) s.PivotHighTime = bars.OpenTimes[pivotIdx];
                }
                if (MtfIsPivotLow(bars, pivotIdx, s.PivotLen))
                {
                    var pp = bars.LowPrices[pivotIdx];
                    s.LastPivotLow = !s.CurrentTrend ? (double.IsNaN(s.LastPivotLow) ? pp : Math.Min(pp, s.LastPivotLow)) : pp;
                    if (s.LastPivotLow != prevLastPivotLow) s.PivotLowTime = bars.OpenTimes[pivotIdx];
                }
            }

            var close     = bars.ClosePrices[tfBarIndex];
            var prevClose = tfBarIndex > 0 ? bars.ClosePrices[tfBarIndex-1] : close;

            if (!double.IsNaN(s.LastPivotHigh) && !double.IsNaN(prevLastPivotHigh))
                if (prevClose <= prevLastPivotHigh && close > s.LastPivotHigh)
                { s.CurrentTrend = true; s.HasTrend = true; s.LastBrokenHigh = s.LastPivotHigh; s.LastBrokenLow = double.NaN; }

            if (!double.IsNaN(s.LastPivotLow) && !double.IsNaN(prevLastPivotLow))
                if (prevClose >= prevLastPivotLow && close < s.LastPivotLow)
                { s.CurrentTrend = false; s.HasTrend = true; s.LastBrokenLow = s.LastPivotLow; s.LastBrokenHigh = double.NaN; }
        }

        private int MtfResolveTfBar(MtfTfState s, DateTime chartTime)
        {
            if (!s.IsLowerTf) return MtfFindAtOrBefore(s.TfBars, chartTime.AddMinutes(-(s.TfMinutes > 0 ? s.TfMinutes : 1)));
            var chartBarIndex = MtfFindAtOrBefore(Bars, chartTime);
            if (chartBarIndex < 0) return -1;
            var chartOpen = Bars.OpenTimes[chartBarIndex];
            DateTime chartNextOpen;
            if (chartBarIndex + 1 < Bars.Count) chartNextOpen = Bars.OpenTimes[chartBarIndex+1];
            else { var m = MtfTfMinutes(Bars.TimeFrame); chartNextOpen = chartOpen.AddMinutes(m > 0 ? m : 1); }
            var first = MtfFindAtOrAfter(s.TfBars, chartOpen);
            if (first < 0) return MtfFindAtOrBefore(s.TfBars, chartTime);
            if (s.TfBars.OpenTimes[first] >= chartNextOpen) return MtfFindAtOrBefore(s.TfBars, chartTime);
            return first;
        }

        private bool CheckMtfFilter(bool isLong, int signalBar)
        {
            if (!EnableMtfFilter) return true;
            var isAnd = MtfFilterLogic == MtfLogicMode.AND;
            var enabled = 0; var passing = 0;
            if (_mtfState1 != null) { enabled++; if (!_mtfState1.HasTrend || _mtfState1.CurrentTrend == isLong) passing++; }
            if (_mtfState2 != null) { enabled++; if (!_mtfState2.HasTrend || _mtfState2.CurrentTrend == isLong) passing++; }
            if (_mtfState3 != null) { enabled++; if (!_mtfState3.HasTrend || _mtfState3.CurrentTrend == isLong) passing++; }
            if (_mtfState4 != null) { enabled++; if (!_mtfState4.HasTrend || _mtfState4.CurrentTrend == isLong) passing++; }
            if (enabled == 0) return true;
            var result = isAnd ? (passing == enabled) : (passing > 0);
            if (!result)
                Print("[MTF BLOCKED] {0} bar={1} passing={2}/{3} logic={4} | TF1:{5}({6}) TF2:{7}({8}) TF3:{9}({10}) TF4:{11}({12})",
                    isLong?"Long":"Short", signalBar, passing, enabled, isAnd?"AND":"OR",
                    _mtfState1!=null?(_mtfState1.HasTrend?(_mtfState1.CurrentTrend?"Bull":"Bear"):"Warm"):"Off", _mtfState1!=null?_mtfState1.LastProcessedTfBar:-1,
                    _mtfState2!=null?(_mtfState2.HasTrend?(_mtfState2.CurrentTrend?"Bull":"Bear"):"Warm"):"Off", _mtfState2!=null?_mtfState2.LastProcessedTfBar:-1,
                    _mtfState3!=null?(_mtfState3.HasTrend?(_mtfState3.CurrentTrend?"Bull":"Bear"):"Warm"):"Off", _mtfState3!=null?_mtfState3.LastProcessedTfBar:-1,
                    _mtfState4!=null?(_mtfState4.HasTrend?(_mtfState4.CurrentTrend?"Bull":"Bear"):"Warm"):"Off", _mtfState4!=null?_mtfState4.LastProcessedTfBar:-1);
            return result;
        }

        private static bool MtfIsPivotHigh(Bars bars, int idx, int len)
        {
            var left = idx-len; var right = idx+len;
            if (left < 0 || right >= bars.Count) return false;
            var pivot = bars.HighPrices[idx];
            for (var i = left; i <= right; i++) if (i != idx && bars.HighPrices[i] >= pivot) return false;
            return true;
        }

        private static bool MtfIsPivotLow(Bars bars, int idx, int len)
        {
            var left = idx-len; var right = idx+len;
            if (left < 0 || right >= bars.Count) return false;
            var pivot = bars.LowPrices[idx];
            for (var i = left; i <= right; i++) if (i != idx && bars.LowPrices[i] <= pivot) return false;
            return true;
        }

        private static int MtfFindAtOrBefore(Bars bars, DateTime time)
        {
            var lo = 0; var hi = bars.Count - 1;
            while (lo <= hi) { var mid = (lo+hi)/2; if (bars.OpenTimes[mid] == time) return mid; else if (bars.OpenTimes[mid] < time) lo = mid+1; else hi = mid-1; }
            return hi;
        }

        private static int MtfFindAtOrAfter(Bars bars, DateTime time)
        {
            var lo = 0; var hi = bars.Count - 1; var ans = -1;
            while (lo <= hi) { var mid = (lo+hi)/2; if (bars.OpenTimes[mid] >= time) { ans = mid; hi = mid-1; } else lo = mid+1; }
            return ans;
        }

        private static TimeFrame MtfParseTimeFrame(string text)
        {
            switch ((text ?? string.Empty).Trim().ToUpperInvariant())
            {
                case "1": return TimeFrame.Minute; case "2": return TimeFrame.Minute2; case "3": return TimeFrame.Minute3;
                case "4": return TimeFrame.Minute4; case "5": return TimeFrame.Minute5; case "10": return TimeFrame.Minute10;
                case "15": return TimeFrame.Minute15; case "30": return TimeFrame.Minute30; case "45": return TimeFrame.Minute45;
                case "60": case "1H": return TimeFrame.Hour; case "120": case "2H": return TimeFrame.Hour2;
                case "240": case "4H": return TimeFrame.Hour4; case "480": case "8H": return TimeFrame.Hour8;
                case "720": case "12H": return TimeFrame.Hour12; case "D": case "1D": return TimeFrame.Daily;
                case "W": case "1W": return TimeFrame.Weekly; case "M": case "1M": return TimeFrame.Monthly;
                default: return TimeFrame.Minute15;
            }
        }

        private static int MtfTfMinutes(TimeFrame tf)
        {
            if (tf == TimeFrame.Minute) return 1; if (tf == TimeFrame.Minute2) return 2; if (tf == TimeFrame.Minute3) return 3;
            if (tf == TimeFrame.Minute4) return 4; if (tf == TimeFrame.Minute5) return 5; if (tf == TimeFrame.Minute10) return 10;
            if (tf == TimeFrame.Minute15) return 15; if (tf == TimeFrame.Minute30) return 30; if (tf == TimeFrame.Minute45) return 45;
            if (tf == TimeFrame.Hour) return 60; if (tf == TimeFrame.Hour2) return 120; if (tf == TimeFrame.Hour4) return 240;
            if (tf == TimeFrame.Hour8) return 480; if (tf == TimeFrame.Hour12) return 720; if (tf == TimeFrame.Daily) return 1440;
            if (tf == TimeFrame.Weekly) return 10080; if (tf == TimeFrame.Monthly) return 43200;
            return 0;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  MTF FVG FILTER ENGINE
        //  Ported from MTF FVG x2 [MK] indicator.
        //  Display / alert code stripped; detection + mitigation state only.
        // ═════════════════════════════════════════════════════════════════════

        private void FvgRegisterTf(string key, TimeFrame tf, bool enabled, int max)
        {
            if (!enabled) return;
            _fvgBarsByTf[key]      = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf);
            _fvgBullByTf[key]      = new List<FvgFilterZone>();
            _fvgBearByTf[key]      = new List<FvgFilterZone>();
            _fvgMaxByTf[key]       = max;
            _fvgLastBullTfIdx[key] = -1;
            _fvgLastBearTfIdx[key] = -1;
            _fvgPrevBullHigh2[key] = double.NaN;
            _fvgPrevBullLow[key]   = double.NaN;
            _fvgPrevBearLow2[key]  = double.NaN;
            _fvgPrevBearHigh[key]  = double.NaN;
        }

        /// <summary>
        /// Detects new FVG zones from each enabled HTF and updates existing
        /// zones for mitigation. Called once per chart bar from the incremental loop.
        /// </summary>
        private void RunFvgFilter(int chartIndex)
        {
            foreach (var kv in _fvgBarsByTf)
            {
                var tfKey  = kv.Key;
                var tfBars = kv.Value;

                // Locate the most recent confirmed (closed) HTF bar
                var i = MtfFindAtOrBefore(tfBars, Bars.OpenTimes[chartIndex]);
                if (i < 3) continue;

                // Optional market-hours gate (09:30–16:00 ET = UTC mins 570-960)
                if (FvgOnlyMktHrs)
                {
                    var totalMins = Bars.OpenTimes[chartIndex].Hour * 60
                                  + Bars.OpenTimes[chartIndex].Minute;
                    if (totalMins < 570 || totalMins > 960) continue;
                }

                // HTF bar data — mirrors indicator variable names exactly
                var h   = tfBars.HighPrices[i - 1];   // High of bar i-1
                var h2  = tfBars.HighPrices[i - 3];   // High of bar i-3
                var l   = tfBars.LowPrices[i - 1];    // Low  of bar i-1
                var l2  = tfBars.LowPrices[i - 3];    // Low  of bar i-3

                // FVG conditions (mirrors IsFvgBull / IsFvgBear in indicator)
                var newBull = h2 < l;   // gap up:   High[i-3] < Low[i-1]
                var newBear = l2 > h;   // gap down: Low[i-3]  > High[i-1]

                // Duplicate-detection guard (mirrors Pine's != checks)
                var prevH2 = _fvgPrevBullHigh2[tfKey];
                var prevL  = _fvgPrevBullLow[tfKey];
                var prevL2 = _fvgPrevBearLow2[tfKey];
                var prevH  = _fvgPrevBearHigh[tfKey];

                var bullDistinct = (double.IsNaN(prevH2) || h2 != prevH2) &&
                                   (double.IsNaN(prevL)  || l  != prevL);
                var bearDistinct = (double.IsNaN(prevL2) || l2 != prevL2) &&
                                   (double.IsNaN(prevH)  || h  != prevH);

                // Add new Bull FVG
                // zone.Top = l (Low of right bar), zone.Bottom = h2 (High of left bar)
                if (newBull && _fvgLastBullTfIdx[tfKey] != i && bullDistinct)
                {
                    var bulls = _fvgBullByTf[tfKey];
                    if (bulls.Count >= _fvgMaxByTf[tfKey])
                    {
                        FvgRemoveZoneChart(bulls[0]);
                        bulls.RemoveAt(0);
                    }
                    var zone = new FvgFilterZone { IsBull = true, Top = l, Bottom = h2 };
                    if (FvgShowActive) FvgDrawZone(zone, chartIndex);
                    bulls.Add(zone);
                    _fvgLastBullTfIdx[tfKey] = i;
                }

                // Add new Bear FVG
                // zone.Top = l2 (Low of left bar), zone.Bottom = h (High of right bar)
                if (newBear && _fvgLastBearTfIdx[tfKey] != i && bearDistinct)
                {
                    var bears = _fvgBearByTf[tfKey];
                    if (bears.Count >= _fvgMaxByTf[tfKey])
                    {
                        FvgRemoveZoneChart(bears[0]);
                        bears.RemoveAt(0);
                    }
                    var zone = new FvgFilterZone { IsBull = false, Top = l2, Bottom = h };
                    if (FvgShowActive) FvgDrawZone(zone, chartIndex);
                    bears.Add(zone);
                    _fvgLastBearTfIdx[tfKey] = i;
                }

                // Save state for next call's duplicate check
                _fvgPrevBullHigh2[tfKey] = h2;
                _fvgPrevBullLow[tfKey]   = l;
                _fvgPrevBearLow2[tfKey]  = l2;
                _fvgPrevBearHigh[tfKey]  = h;

                // Update existing zones (mitigation)
                FvgUpdateZones(_fvgBullByTf[tfKey], true,  chartIndex);
                FvgUpdateZones(_fvgBearByTf[tfKey], false, chartIndex);
            }
        }

        /// <summary>
        /// Applies per-bar mitigation to a zone list and manages chart rectangles.
        /// Mirrors UpdateExistingFvgs from the indicator.
        /// </summary>
        private void FvgUpdateZones(List<FvgFilterZone> zones, bool bull, int chartIndex)
        {
            var mode = FvgMitigationAction;

            for (int k = zones.Count - 1; k >= 0; k--)
            {
                var z   = zones[k];
                var mid = (z.Top + z.Bottom) / 2.0;

                // Dynamic: shrink zone boundary toward current price
                if (mode == FvgMitigationMode.Dynamic)
                {
                    if (bull)
                    {
                        if  (FvgUseBodyMitigation && Bars.ClosePrices[chartIndex] < z.Top) z.Top = Bars.ClosePrices[chartIndex];
                        else if (!FvgUseBodyMitigation && Bars.LowPrices[chartIndex] < z.Top) z.Top = Bars.LowPrices[chartIndex];
                    }
                    else
                    {
                        if  (FvgUseBodyMitigation && Bars.ClosePrices[chartIndex] > z.Bottom) z.Bottom = Bars.ClosePrices[chartIndex];
                        else if (!FvgUseBodyMitigation && Bars.HighPrices[chartIndex] > z.Bottom) z.Bottom = Bars.HighPrices[chartIndex];
                    }
                    // Sync shrunken boundaries to chart object
                    if (z.Rect != null) { z.Rect.Y1 = z.Top; z.Rect.Y2 = z.Bottom; }
                }

                // None: keep zone alive but flag it once fully penetrated
                if (mode == FvgMitigationMode.None && !z.IsMitigated)
                {
                    var penetrated = bull
                        ? (FvgUseBodyMitigation ? Bars.ClosePrices[chartIndex] < z.Bottom : Bars.LowPrices[chartIndex]  < z.Bottom)
                        : (FvgUseBodyMitigation ? Bars.ClosePrices[chartIndex] > z.Top    : Bars.HighPrices[chartIndex] > z.Top);
                    if (penetrated)
                    {
                        z.IsMitigated = true;
                        if (z.Rect != null)
                        {
                            if (FvgShowMitigated)
                                z.Rect.Color = FvgMitigatedColor;  // recolour to mitigated
                            else
                            {
                                FvgRemoveZoneChart(z);             // hide if toggle is off
                            }
                        }
                    }
                }

                // Extend chart rectangle right edge each bar for active (non-mitigated) zones
                if (z.Rect != null && !z.IsMitigated)
                    z.Rect.Time2 = FvgChartRight(chartIndex);

                // Full-removal condition (Normal & Dynamic)
                var removeFull = bull
                    ? (FvgUseBodyMitigation ? Bars.ClosePrices[chartIndex] < z.Bottom : Bars.LowPrices[chartIndex]  < z.Bottom)
                    : (FvgUseBodyMitigation ? Bars.ClosePrices[chartIndex] > z.Top    : Bars.HighPrices[chartIndex] > z.Top);

                // Half-removal condition
                var removeHalf = bull
                    ? (FvgUseBodyMitigation ? Bars.ClosePrices[chartIndex] < mid : Bars.LowPrices[chartIndex]  < mid)
                    : (FvgUseBodyMitigation ? Bars.ClosePrices[chartIndex] > mid : Bars.HighPrices[chartIndex] > mid);

                if (((mode == FvgMitigationMode.Normal || mode == FvgMitigationMode.Dynamic) && removeFull) ||
                    (mode == FvgMitigationMode.Half && removeHalf))
                {
                    FvgRemoveZoneChart(z);
                    zones.RemoveAt(k);
                }
            }
        }

        // ─── FVG display helpers ───────────────────────────────────────────────

        /// <summary>
        /// Draws a chart rectangle for the zone and stores the reference.
        /// Left edge = bar where FVG was confirmed; right edge = current bar + FvgDisplayBarsRight.
        /// </summary>
        private void FvgDrawZone(FvgFilterZone zone, int chartIndex)
        {
            var id    = "fvg_" + (zone.IsBull ? "b" : "s") + "_" + (_fvgChartId++);
            var color = zone.IsBull ? FvgActiveBullColor : FvgActiveBearColor;
            var left  = Bars.OpenTimes[chartIndex];
            var right = FvgChartRight(chartIndex);

            var rect = Chart.DrawRectangle(id, left, zone.Top, right, zone.Bottom, color);
            rect.IsFilled      = true;
            rect.IsInteractive = false;

            zone.Rect  = rect;
            zone.RectId = id;
        }

        /// <summary>Removes the chart object for a zone if one exists.</summary>
        private void FvgRemoveZoneChart(FvgFilterZone zone)
        {
            if (zone.RectId != null)
            {
                Chart.RemoveObject(zone.RectId);
                zone.Rect   = null;
                zone.RectId = null;
            }
        }

        /// <summary>
        /// Returns a DateTime FvgDisplayBarsRight bars ahead of the given chart bar.
        /// Matches the right-extension pattern from MTF FVG x2 [MK] indicator.
        /// </summary>
        private DateTime FvgChartRight(int chartIndex)
        {
            if (Bars.Count < 2)
                return Bars.OpenTimes[chartIndex].AddMinutes(FvgDisplayBarsRight);
            var span = Bars.OpenTimes[Bars.Count - 1] - Bars.OpenTimes[Bars.Count - 2];
            if (span <= TimeSpan.Zero) span = TimeSpan.FromMinutes(1);
            return Bars.OpenTimes[chartIndex] + TimeSpan.FromTicks(span.Ticks * FvgDisplayBarsRight);
        }

        /// <summary>
        /// FVG filter gate.  Returns true when the filter is off or when the
        /// enabled-TF vote (OR/AND) is satisfied.
        /// Satisfied = at least one non-mitigated zone of the correct polarity
        /// was touched by a chart bar within [signalBar - FvgLookbackBars + 1, signalBar].
        ///
        /// Touch definition:
        ///   Bull FVG: Bars.Low[b]  &lt;= zone.Top    (low enters gap from above)
        ///   Bear FVG: Bars.High[b] &gt;= zone.Bottom (high enters gap from below)
        /// </summary>
        private bool CheckFvgFilter(bool isLong, int signalBar)
        {
            if (!EnableFvgFilter) return true;
            if (_fvgBarsByTf.Count == 0) return true;   // no TFs registered → pass

            var isAnd   = FvgFilterLogic == MtfLogicMode.AND;
            var enabled = _fvgBarsByTf.Count;
            var passing = 0;

            foreach (var tfKey in _fvgBarsByTf.Keys)
            {
                var zones = isLong ? _fvgBullByTf[tfKey] : _fvgBearByTf[tfKey];
                if (FvgHasTouchInLookback(zones, isLong, signalBar))
                    passing++;
            }

            var result = isAnd ? (passing == enabled) : (passing > 0);

            if (!result)
                Print("[FVG BLOCKED] {0} bar={1} passing={2}/{3} logic={4} lookback={5}",
                    isLong ? "Long" : "Short", signalBar,
                    passing, enabled, isAnd ? "AND" : "OR", FvgLookbackBars);

            return result;
        }

        /// <summary>
        /// Returns true if any active (non-mitigated) zone in the list
        /// was touched by a chart bar in [signalBar - FvgLookbackBars + 1, signalBar].
        /// </summary>
        private bool FvgHasTouchInLookback(List<FvgFilterZone> zones, bool isBull, int signalBar)
        {
            if (zones.Count == 0) return false;
            int start = Math.Max(0, signalBar - FvgLookbackBars + 1);
            foreach (var z in zones)
            {
                if (z.IsMitigated) continue;
                for (int b = start; b <= signalBar; b++)
                {
                    bool touched = isBull
                        ? Bars.LowPrices[b]  <= z.Top      // low dips into bull gap from above
                        : Bars.HighPrices[b] >= z.Bottom;  // high rallies into bear gap from below
                    if (touched) return true;
                }
            }
            return false;
        }
    }
}
