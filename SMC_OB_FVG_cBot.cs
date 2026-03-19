using System;
using System.ComponentModel;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    // ─────────────────────────────────────────────────────────────────────────────
    //  SMC OB & FVG cBot
    //
    //  Reads entry signals from the SMC_OrderBlock_Detector indicator and places
    //  market orders with risk-based position sizing:
    //
    //  • Internal OB long  : Signal OB long  → Buy  at bar close, SL = OB low  − SlBufferPips
    //  • Internal OB short : Signal OB short → Sell at bar close, SL = OB high + SlBufferPips
    //  • Swing    OB long  : Signal OB long  → Buy  at bar close, SL = OB low  − SlBufferPips
    //  • Swing    OB short : Signal OB short → Sell at bar close, SL = OB high + SlBufferPips
    //  • FVG long          : FVG long signal → Buy  at bar close, SL = trigger bar low  − SlBufferPips
    //  • FVG short         : FVG short signal→ Sell at bar close, SL = trigger bar high + SlBufferPips
    //
    //  Risk per trade = 0.5 % of Account.Balance.
    //
    //  Two indicator instances are used internally:
    //    _obIndicator  – ShowInternalOrderBlocks / ShowSwingOrderBlocks controlled by
    //                    the TradeInternalOb / TradeSwingOb toggles; FVG disabled.
    //    _fvgIndicator – OBs disabled; FVG enabled.
    //  This separates OB SL levels (stored in LongObBottom / ShortObTop) from FVG
    //  signals (whose SL is derived from the trigger bar's low / high).
    // ─────────────────────────────────────────────────────────────────────────────

    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class SMC_OB_FVG_cBot : Robot
    {
        // ════════════════════════════════════════════════════════════════════════
        //  TRADE TRIGGER TOGGLES
        // ════════════════════════════════════════════════════════════════════════

        public enum RiskRewardOption
        {
            [Description("1:1")]   OneToOne      = 0,
            [Description("1:1.5")] OneToOneHalf  = 1,
            [Description("1:2")]   OneToTwo      = 2,
            [Description("1:3")]   OneToThree    = 3,
            [Description("1:4")]   OneToFour     = 4,
            [Description("1:5")]   OneToFive     = 5
        }

        [Parameter("Trade on Internal OB Signals", DefaultValue = true, Group = "Trade Triggers")]
        public bool TradeInternalOb { get; set; }

        [Parameter("Trade on Swing OB Signals", DefaultValue = true, Group = "Trade Triggers")]
        public bool TradeSwingOb { get; set; }

        [Parameter("Trade on FVG Signals", DefaultValue = true, Group = "Trade Triggers")]
        public bool TradeFvg { get; set; }

        [Parameter("SL Buffer (pips)", DefaultValue = 5.0, MinValue = 0.0, Step = 0.1, Group = "Trade Triggers")]
        public double SlBufferPips { get; set; }

        [Parameter("Enable Take Profit", DefaultValue = true, Group = "Trade Triggers")]
        public bool TakeProfitEnabled { get; set; }

        [Parameter("Risk : Reward Ratio", DefaultValue = RiskRewardOption.OneToTwo, Group = "Trade Triggers")]
        public RiskRewardOption RiskReward { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  SMART MONEY CONCEPTS — mirrors every parameter in SMC_OrderBlock_Detector
        // ════════════════════════════════════════════════════════════════════════

        // ── General ──────────────────────────────────────────────────────────────

        [Parameter("Mode", DefaultValue = SMC_OrderBlock_Detector.DisplayMode.Historical, Group = "Smart Money Concepts")]
        public SMC_OrderBlock_Detector.DisplayMode ModeInput { get; set; }

        [Parameter("Style", DefaultValue = SMC_OrderBlock_Detector.ThemeStyle.Colored, Group = "Smart Money Concepts")]
        public SMC_OrderBlock_Detector.ThemeStyle StyleInput { get; set; }

        [Parameter("Color Candles", DefaultValue = false, Group = "Smart Money Concepts")]
        public bool ShowTrendInput { get; set; }

        [Parameter("Plot Signal Series", DefaultValue = false, Group = "Smart Money Concepts")]
        public bool PlotSignalSeriesInput { get; set; }

        // ── Real Time Internal Structure ──────────────────────────────────────────

        [Parameter("Show Internal Structure", DefaultValue = true, Group = "Real Time Internal Structure")]
        public bool ShowInternalsInput { get; set; }

        [Parameter("Bullish Structure", DefaultValue = SMC_OrderBlock_Detector.StructureFilter.All, Group = "Real Time Internal Structure")]
        public SMC_OrderBlock_Detector.StructureFilter ShowInternalBullInput { get; set; }

        [Parameter("Internal Bull Color", DefaultValue = "#089981", Group = "Real Time Internal Structure")]
        public Color InternalBullColorInput { get; set; }

        [Parameter("Bearish Structure", DefaultValue = SMC_OrderBlock_Detector.StructureFilter.All, Group = "Real Time Internal Structure")]
        public SMC_OrderBlock_Detector.StructureFilter ShowInternalBearInput { get; set; }

        [Parameter("Internal Bear Color", DefaultValue = "#F23645", Group = "Real Time Internal Structure")]
        public Color InternalBearColorInput { get; set; }

        [Parameter("Confluence Filter", DefaultValue = false, Group = "Real Time Internal Structure")]
        public bool InternalFilterConfluenceInput { get; set; }

        [Parameter("Internal Label Size", DefaultValue = SMC_OrderBlock_Detector.LabelSizeOpt.Tiny, Group = "Real Time Internal Structure")]
        public SMC_OrderBlock_Detector.LabelSizeOpt InternalStructureSize { get; set; }

        [Parameter("iBOS/iCHoCH Source", DefaultValue = SMC_OrderBlock_Detector.StructureSource.Close, Group = "Real Time Internal Structure")]
        public SMC_OrderBlock_Detector.StructureSource InternalStructureSourceInput { get; set; }

        [Parameter("Internal Pivot Length", DefaultValue = 5, MinValue = 1, MaxValue = 50, Group = "Real Time Internal Structure")]
        public int InternalPivotLength { get; set; }

        // ── Real Time Swing Structure ─────────────────────────────────────────────

        [Parameter("Show Swing Structure", DefaultValue = true, Group = "Real Time Swing Structure")]
        public bool ShowStructureInput { get; set; }

        [Parameter("Bullish Structure", DefaultValue = SMC_OrderBlock_Detector.StructureFilter.All, Group = "Real Time Swing Structure")]
        public SMC_OrderBlock_Detector.StructureFilter ShowSwingBullInput { get; set; }

        [Parameter("Swing Bull Color", DefaultValue = "#089981", Group = "Real Time Swing Structure")]
        public Color SwingBullishColorInput { get; set; }

        [Parameter("Bearish Structure", DefaultValue = SMC_OrderBlock_Detector.StructureFilter.All, Group = "Real Time Swing Structure")]
        public SMC_OrderBlock_Detector.StructureFilter ShowSwingBearInput { get; set; }

        [Parameter("Swing Bear Color", DefaultValue = "#F23645", Group = "Real Time Swing Structure")]
        public Color SwingBearishColorInput { get; set; }

        [Parameter("Swing Label Size", DefaultValue = SMC_OrderBlock_Detector.LabelSizeOpt.Small, Group = "Real Time Swing Structure")]
        public SMC_OrderBlock_Detector.LabelSizeOpt SwingStructureSize { get; set; }

        [Parameter("Show Swings Points", DefaultValue = false, Group = "Real Time Swing Structure")]
        public bool ShowSwingsInput { get; set; }

        [Parameter("Swings Length", DefaultValue = 50, MinValue = 10, Group = "Real Time Swing Structure")]
        public int SwingsLengthInput { get; set; }

        [Parameter("Show Strong/Weak High/Low", DefaultValue = true, Group = "Real Time Swing Structure")]
        public bool ShowHighLowSwingsInput { get; set; }

        [Parameter("BOS/CHoCH Source", DefaultValue = SMC_OrderBlock_Detector.StructureSource.Close, Group = "Real Time Swing Structure")]
        public SMC_OrderBlock_Detector.StructureSource SwingStructureSourceInput { get; set; }

        // ── Order Blocks ──────────────────────────────────────────────────────────
        //  ShowInternalOrderBlocksInput / ShowSwingOrderBlocksInput control display.
        //  When trading is enabled (TradeInternalOb / TradeSwingOb = true) OBs are
        //  always detected even if the display toggle is off, so signals can still fire.

        [Parameter("Internal Order Blocks", DefaultValue = true, Group = "Order Blocks")]
        public bool ShowInternalOrderBlocksInput { get; set; }

        [Parameter("Swing Order Blocks", DefaultValue = true, Group = "Order Blocks")]
        public bool ShowSwingOrderBlocksInput { get; set; }

        [Parameter("Internal OB Size", DefaultValue = 5, MinValue = 1, MaxValue = 20, Group = "Order Blocks")]
        public int InternalOrderBlocksSizeInput { get; set; }

        [Parameter("Swing OB Size", DefaultValue = 5, MinValue = 1, MaxValue = 20, Group = "Order Blocks")]
        public int SwingOrderBlocksSizeInput { get; set; }

        [Parameter("Order Block Filter", DefaultValue = SMC_OrderBlock_Detector.ObFilter.Atr, Group = "Order Blocks")]
        public SMC_OrderBlock_Detector.ObFilter OrderBlockFilterInput { get; set; }

        [Parameter("OB Filter ATR Period", DefaultValue = 200, MinValue = 1, MaxValue = 500, Group = "Order Blocks")]
        public int ObFilterAtrPeriod { get; set; }

        [Parameter("OB Filter CMR Period", DefaultValue = 0, MinValue = 0, MaxValue = 500, Group = "Order Blocks")]
        public int ObFilterCmrPeriod { get; set; }

        [Parameter("Order Block Mitigation", DefaultValue = SMC_OrderBlock_Detector.MitigationMode.HighLow, Group = "Order Blocks")]
        public SMC_OrderBlock_Detector.MitigationMode OrderBlockMitigationInput { get; set; }

        [Parameter("Internal Bullish OB Color", DefaultValue = "#CC3179F5", Group = "Order Blocks")]
        public Color InternalBullishOrderBlockColor { get; set; }

        [Parameter("Internal Bearish OB Color", DefaultValue = "#CCF77C80", Group = "Order Blocks")]
        public Color InternalBearishOrderBlockColor { get; set; }

        [Parameter("Swing Bullish OB Color", DefaultValue = "#CC1848CC", Group = "Order Blocks")]
        public Color SwingBullishOrderBlockColor { get; set; }

        [Parameter("Swing Bearish OB Color", DefaultValue = "#CCB22833", Group = "Order Blocks")]
        public Color SwingBearishOrderBlockColor { get; set; }

        [Parameter("Show All Historical OBs", DefaultValue = true, Group = "Order Blocks")]
        public bool ShowAllHistoricalObs { get; set; }

        [Parameter("Show Mitigated OBs", DefaultValue = true, Group = "Order Blocks")]
        public bool ShowMitigatedObs { get; set; }

        [Parameter("Mitigated OB Opacity (%)", DefaultValue = 30, MinValue = 1, MaxValue = 99, Group = "Order Blocks")]
        public int MitigatedObOpacity { get; set; }

        // ── EQH / EQL ─────────────────────────────────────────────────────────────

        [Parameter("Equal High/Low", DefaultValue = true, Group = "EQH/EQL")]
        public bool ShowEqualHighsLowsInput { get; set; }

        [Parameter("Bars Confirmation", DefaultValue = 3, MinValue = 1, Group = "EQH/EQL")]
        public int EqualHighsLowsLengthInput { get; set; }

        [Parameter("Threshold", DefaultValue = 0.1, MinValue = 0, MaxValue = 0.5, Group = "EQH/EQL")]
        public double EqualHighsLowsThresholdInput { get; set; }

        // ── Fair Value Gaps ───────────────────────────────────────────────────────
        //  ShowFairValueGapsInput controls display; TradeFvg controls trading.
        //  When TradeFvg = true, FVGs are always detected even if the display is off.

        [Parameter("Fair Value Gaps", DefaultValue = true, Group = "Fair Value Gaps")]
        public bool ShowFairValueGapsInput { get; set; }

        [Parameter("Auto Threshold", DefaultValue = true, Group = "Fair Value Gaps")]
        public bool FairValueGapsThresholdInput { get; set; }

        [Parameter("Bullish FVG Color", DefaultValue = "#7000FF68", Group = "Fair Value Gaps")]
        public Color FairValueGapsBullColorInput { get; set; }

        [Parameter("Bearish FVG Color", DefaultValue = "#70FF0008", Group = "Fair Value Gaps")]
        public Color FairValueGapsBearColorInput { get; set; }

        [Parameter("Extend FVG", DefaultValue = 1, MinValue = 0, Group = "Fair Value Gaps")]
        public int FairValueGapsExtendInput { get; set; }

        // ── Highs & Lows MTF ──────────────────────────────────────────────────────

        [Parameter("Daily", DefaultValue = false, Group = "Highs & Lows MTF")]
        public bool ShowDailyLevelsInput { get; set; }

        [Parameter("Daily Style", DefaultValue = SMC_OrderBlock_Detector.LineStyleOpt.Solid, Group = "Highs & Lows MTF")]
        public SMC_OrderBlock_Detector.LineStyleOpt DailyLevelsStyleInput { get; set; }

        [Parameter("Daily Color", DefaultValue = "#2157F3", Group = "Highs & Lows MTF")]
        public Color DailyLevelsColorInput { get; set; }

        [Parameter("Weekly", DefaultValue = false, Group = "Highs & Lows MTF")]
        public bool ShowWeeklyLevelsInput { get; set; }

        [Parameter("Weekly Style", DefaultValue = SMC_OrderBlock_Detector.LineStyleOpt.Solid, Group = "Highs & Lows MTF")]
        public SMC_OrderBlock_Detector.LineStyleOpt WeeklyLevelsStyleInput { get; set; }

        [Parameter("Weekly Color", DefaultValue = "#2157F3", Group = "Highs & Lows MTF")]
        public Color WeeklyLevelsColorInput { get; set; }

        [Parameter("Monthly", DefaultValue = false, Group = "Highs & Lows MTF")]
        public bool ShowMonthlyLevelsInput { get; set; }

        [Parameter("Monthly Style", DefaultValue = SMC_OrderBlock_Detector.LineStyleOpt.Solid, Group = "Highs & Lows MTF")]
        public SMC_OrderBlock_Detector.LineStyleOpt MonthlyLevelsStyleInput { get; set; }

        [Parameter("Monthly Color", DefaultValue = "#2157F3", Group = "Highs & Lows MTF")]
        public Color MonthlyLevelsColorInput { get; set; }

        // ── Premium & Discount Zones ──────────────────────────────────────────────

        [Parameter("Premium/Discount Zones", DefaultValue = false, Group = "Premium & Discount Zones")]
        public bool ShowPremiumDiscountZonesInput { get; set; }

        [Parameter("Premium Zone Color", DefaultValue = "#F23645", Group = "Premium & Discount Zones")]
        public Color PremiumZoneColorInput { get; set; }

        [Parameter("Equilibrium Zone Color", DefaultValue = "#878B94", Group = "Premium & Discount Zones")]
        public Color EquilibriumZoneColorInput { get; set; }

        [Parameter("Discount Zone Color", DefaultValue = "#089981", Group = "Premium & Discount Zones")]
        public Color DiscountZoneColorInput { get; set; }

        // ── Signal Display ────────────────────────────────────────────────────────

        [Parameter("Line Width Liquidated", DefaultValue = 1, MinValue = 1, MaxValue = 4, Group = "Signal Display")]
        public int LineWidthLiquidated { get; set; }

        [Parameter("Show Signal Dots", DefaultValue = false, Group = "Signal Display")]
        public bool ShowSignalDots { get; set; }

        [Parameter("Show Signals OB", DefaultValue = true, Group = "Signal Display")]
        public bool ShowSignalsOb { get; set; }

        [Parameter("Show Signals FVG", DefaultValue = false, Group = "Signal Display")]
        public bool ShowSignalsFvg { get; set; }

        [Parameter("Signal Offset (pips)", DefaultValue = 2.0, MinValue = 0.0, Step = 0.1, Group = "Signal Display")]
        public double SignalOffsetPips { get; set; }

        // ── Signals ───────────────────────────────────────────────────────────────

        [Parameter("Min Dist OB (bars)", DefaultValue = 1, MinValue = 1, Group = "Signals")]
        public int MinDist { get; set; }

        [Parameter("Min Dist FVG (bars)", DefaultValue = 1, MinValue = 1, Group = "Signals")]
        public int MinDistFvg { get; set; }

        [Parameter("Use Heikin-Ashi", DefaultValue = false, Group = "Signals")]
        public bool UseHeikinAshi { get; set; }

        [Parameter("Min Bars After Structure Break", DefaultValue = 0, MinValue = 0, MaxValue = 200, Group = "Signals")]
        public int MinBarsAfterStructureBreak { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  PRIVATE FIELDS
        // ════════════════════════════════════════════════════════════════════════

        // Indicator instance that fires OB signals.
        // Uses per-type output series so Internal and Swing OBs fire independently:
        //   LongInternalObBottom / ShortInternalObTop  → Internal OB signals (SL = OB level)
        //   LongSwingObBottom    / ShortSwingObTop     → Swing   OB signals (SL = OB level)
        private SMC_OrderBlock_Detector _obIndicator;

        // Indicator instance that fires FVG signals.
        // LongObBottom[bar]  → non-NaN when a bullish FVG long signal fired (value = FVG top level).
        // ShortObTop[bar]    → non-NaN when a bearish FVG short signal fired (value = FVG bottom level).
        // For FVG trades the SL is NOT taken from these values; instead it uses
        // the trigger bar's Low (long) or High (short) directly.
        private SMC_OrderBlock_Detector _fvgIndicator;

        // Warm-up guard: minimum bars needed before the indicator produces valid output.
        private int _minBarsWarmup;

        // ════════════════════════════════════════════════════════════════════════
        //  LIFECYCLE
        // ════════════════════════════════════════════════════════════════════════

        protected override void OnStart()
        {
            _minBarsWarmup = Math.Max(SwingsLengthInput, EqualHighsLowsLengthInput) + 10;

            bool useOb = TradeInternalOb || TradeSwingOb;

            if (useOb)
            {
                // OB indicator instance:
                //   ShowInternalOrderBlocksInput = TradeInternalOb
                //   ShowSwingOrderBlocksInput    = TradeSwingOb
                //   ShowFairValueGapsInput       = false  (FVG handled by _fvgIndicator)
                //   ShowSignalsOb                = true   (draw OB arrows on chart)
                //   ShowSignalsFvg               = false
                _obIndicator = Indicators.GetIndicator<SMC_OrderBlock_Detector>(
                    /* 01 ModeInput                    */ ModeInput,
                    /* 02 StyleInput                   */ StyleInput,
                    /* 03 ShowTrendInput                */ ShowTrendInput,
                    /* 04 PlotSignalSeriesInput         */ PlotSignalSeriesInput,
                    /* 05 ShowInternalsInput            */ ShowInternalsInput,
                    /* 06 ShowInternalBullInput         */ ShowInternalBullInput,
                    /* 07 InternalBullColorInput        */ InternalBullColorInput,
                    /* 08 ShowInternalBearInput         */ ShowInternalBearInput,
                    /* 09 InternalBearColorInput        */ InternalBearColorInput,
                    /* 10 InternalFilterConfluenceInput */ InternalFilterConfluenceInput,
                    /* 11 InternalStructureSize         */ InternalStructureSize,
                    /* 12 InternalStructureSourceInput  */ InternalStructureSourceInput,
                    /* 13 InternalPivotLength           */ InternalPivotLength,
                    /* 14 ShowStructureInput            */ ShowStructureInput,
                    /* 15 ShowSwingBullInput            */ ShowSwingBullInput,
                    /* 16 SwingBullishColorInput        */ SwingBullishColorInput,
                    /* 17 ShowSwingBearInput            */ ShowSwingBearInput,
                    /* 18 SwingBearishColorInput        */ SwingBearishColorInput,
                    /* 19 SwingStructureSize            */ SwingStructureSize,
                    /* 20 ShowSwingsInput               */ ShowSwingsInput,
                    /* 21 SwingsLengthInput             */ SwingsLengthInput,
                    /* 22 ShowHighLowSwingsInput        */ ShowHighLowSwingsInput,
                    /* 23 SwingStructureSourceInput     */ SwingStructureSourceInput,
                    /* 24 ShowInternalOrderBlocksInput  */ TradeInternalOb,
                    /* 25 InternalOrderBlocksSizeInput  */ InternalOrderBlocksSizeInput,
                    /* 26 ShowSwingOrderBlocksInput     */ TradeSwingOb,
                    /* 27 SwingOrderBlocksSizeInput     */ SwingOrderBlocksSizeInput,
                    /* 28 OrderBlockFilterInput         */ OrderBlockFilterInput,
                    /* 29 ObFilterAtrPeriod             */ ObFilterAtrPeriod,
                    /* 30 ObFilterCmrPeriod             */ ObFilterCmrPeriod,
                    /* 31 OrderBlockMitigationInput     */ OrderBlockMitigationInput,
                    /* 32 InternalBullishOrderBlockColor*/ InternalBullishOrderBlockColor,
                    /* 33 InternalBearishOrderBlockColor*/ InternalBearishOrderBlockColor,
                    /* 34 SwingBullishOrderBlockColor   */ SwingBullishOrderBlockColor,
                    /* 35 SwingBearishOrderBlockColor   */ SwingBearishOrderBlockColor,
                    /* 36 ShowAllHistoricalObs          */ ShowAllHistoricalObs,
                    /* 37 ShowMitigatedObs              */ ShowMitigatedObs,
                    /* 38 MitigatedObOpacity            */ MitigatedObOpacity,
                    /* 39 ShowEqualHighsLowsInput       */ ShowEqualHighsLowsInput,
                    /* 40 EqualHighsLowsLengthInput     */ EqualHighsLowsLengthInput,
                    /* 41 EqualHighsLowsThresholdInput  */ EqualHighsLowsThresholdInput,
                    /* 42 ShowFairValueGapsInput        */ false,
                    /* 43 FairValueGapsThresholdInput   */ FairValueGapsThresholdInput,
                    /* 44 FairValueGapsBullColorInput   */ FairValueGapsBullColorInput,
                    /* 45 FairValueGapsBearColorInput   */ FairValueGapsBearColorInput,
                    /* 46 FairValueGapsExtendInput      */ FairValueGapsExtendInput,
                    /* 47 ShowDailyLevelsInput          */ ShowDailyLevelsInput,
                    /* 48 DailyLevelsStyleInput         */ DailyLevelsStyleInput,
                    /* 49 DailyLevelsColorInput         */ DailyLevelsColorInput,
                    /* 50 ShowWeeklyLevelsInput         */ ShowWeeklyLevelsInput,
                    /* 51 WeeklyLevelsStyleInput        */ WeeklyLevelsStyleInput,
                    /* 52 WeeklyLevelsColorInput        */ WeeklyLevelsColorInput,
                    /* 53 ShowMonthlyLevelsInput        */ ShowMonthlyLevelsInput,
                    /* 54 MonthlyLevelsStyleInput       */ MonthlyLevelsStyleInput,
                    /* 55 MonthlyLevelsColorInput       */ MonthlyLevelsColorInput,
                    /* 56 ShowPremiumDiscountZonesInput */ ShowPremiumDiscountZonesInput,
                    /* 57 PremiumZoneColorInput         */ PremiumZoneColorInput,
                    /* 58 EquilibriumZoneColorInput     */ EquilibriumZoneColorInput,
                    /* 59 DiscountZoneColorInput        */ DiscountZoneColorInput,
                    /* 60 LineWidthLiquidated           */ LineWidthLiquidated,
                    /* 61 ShowSignalDots                */ ShowSignalDots,
                    /* 62 ShowSignalsOb                 */ ShowSignalsOb,
                    /* 63 ShowSignalsFvg                */ false,
                    /* 64 SignalOffsetPips              */ SignalOffsetPips,
                    /* 65 MinDist                       */ MinDist,
                    /* 66 MinDistFvg                    */ MinDistFvg,
                    /* 67 UseHeikinAshi                 */ UseHeikinAshi,
                    /* 68 MinBarsAfterStructureBreak    */ MinBarsAfterStructureBreak
                );
            }

            if (TradeFvg)
            {
                // FVG indicator instance:
                //   ShowInternalOrderBlocksInput = false  (OBs handled by _obIndicator)
                //   ShowSwingOrderBlocksInput    = false
                //   ShowFairValueGapsInput       = true
                //   ShowSignalsOb                = false
                //   ShowSignalsFvg               = true   (draw FVG arrows on chart)
                _fvgIndicator = Indicators.GetIndicator<SMC_OrderBlock_Detector>(
                    /* 01 ModeInput                    */ ModeInput,
                    /* 02 StyleInput                   */ StyleInput,
                    /* 03 ShowTrendInput                */ ShowTrendInput,
                    /* 04 PlotSignalSeriesInput         */ PlotSignalSeriesInput,
                    /* 05 ShowInternalsInput            */ ShowInternalsInput,
                    /* 06 ShowInternalBullInput         */ ShowInternalBullInput,
                    /* 07 InternalBullColorInput        */ InternalBullColorInput,
                    /* 08 ShowInternalBearInput         */ ShowInternalBearInput,
                    /* 09 InternalBearColorInput        */ InternalBearColorInput,
                    /* 10 InternalFilterConfluenceInput */ InternalFilterConfluenceInput,
                    /* 11 InternalStructureSize         */ InternalStructureSize,
                    /* 12 InternalStructureSourceInput  */ InternalStructureSourceInput,
                    /* 13 InternalPivotLength           */ InternalPivotLength,
                    /* 14 ShowStructureInput            */ ShowStructureInput,
                    /* 15 ShowSwingBullInput            */ ShowSwingBullInput,
                    /* 16 SwingBullishColorInput        */ SwingBullishColorInput,
                    /* 17 ShowSwingBearInput            */ ShowSwingBearInput,
                    /* 18 SwingBearishColorInput        */ SwingBearishColorInput,
                    /* 19 SwingStructureSize            */ SwingStructureSize,
                    /* 20 ShowSwingsInput               */ ShowSwingsInput,
                    /* 21 SwingsLengthInput             */ SwingsLengthInput,
                    /* 22 ShowHighLowSwingsInput        */ ShowHighLowSwingsInput,
                    /* 23 SwingStructureSourceInput     */ SwingStructureSourceInput,
                    /* 24 ShowInternalOrderBlocksInput  */ false,
                    /* 25 InternalOrderBlocksSizeInput  */ InternalOrderBlocksSizeInput,
                    /* 26 ShowSwingOrderBlocksInput     */ false,
                    /* 27 SwingOrderBlocksSizeInput     */ SwingOrderBlocksSizeInput,
                    /* 28 OrderBlockFilterInput         */ OrderBlockFilterInput,
                    /* 29 ObFilterAtrPeriod             */ ObFilterAtrPeriod,
                    /* 30 ObFilterCmrPeriod             */ ObFilterCmrPeriod,
                    /* 31 OrderBlockMitigationInput     */ OrderBlockMitigationInput,
                    /* 32 InternalBullishOrderBlockColor*/ InternalBullishOrderBlockColor,
                    /* 33 InternalBearishOrderBlockColor*/ InternalBearishOrderBlockColor,
                    /* 34 SwingBullishOrderBlockColor   */ SwingBullishOrderBlockColor,
                    /* 35 SwingBearishOrderBlockColor   */ SwingBearishOrderBlockColor,
                    /* 36 ShowAllHistoricalObs          */ ShowAllHistoricalObs,
                    /* 37 ShowMitigatedObs              */ ShowMitigatedObs,
                    /* 38 MitigatedObOpacity            */ MitigatedObOpacity,
                    /* 39 ShowEqualHighsLowsInput       */ ShowEqualHighsLowsInput,
                    /* 40 EqualHighsLowsLengthInput     */ EqualHighsLowsLengthInput,
                    /* 41 EqualHighsLowsThresholdInput  */ EqualHighsLowsThresholdInput,
                    /* 42 ShowFairValueGapsInput        */ ShowFairValueGapsInput || TradeFvg,
                    /* 43 FairValueGapsThresholdInput   */ FairValueGapsThresholdInput,
                    /* 44 FairValueGapsBullColorInput   */ FairValueGapsBullColorInput,
                    /* 45 FairValueGapsBearColorInput   */ FairValueGapsBearColorInput,
                    /* 46 FairValueGapsExtendInput      */ FairValueGapsExtendInput,
                    /* 47 ShowDailyLevelsInput          */ ShowDailyLevelsInput,
                    /* 48 DailyLevelsStyleInput         */ DailyLevelsStyleInput,
                    /* 49 DailyLevelsColorInput         */ DailyLevelsColorInput,
                    /* 50 ShowWeeklyLevelsInput         */ ShowWeeklyLevelsInput,
                    /* 51 WeeklyLevelsStyleInput        */ WeeklyLevelsStyleInput,
                    /* 52 WeeklyLevelsColorInput        */ WeeklyLevelsColorInput,
                    /* 53 ShowMonthlyLevelsInput        */ ShowMonthlyLevelsInput,
                    /* 54 MonthlyLevelsStyleInput       */ MonthlyLevelsStyleInput,
                    /* 55 MonthlyLevelsColorInput       */ MonthlyLevelsColorInput,
                    /* 56 ShowPremiumDiscountZonesInput */ ShowPremiumDiscountZonesInput,
                    /* 57 PremiumZoneColorInput         */ PremiumZoneColorInput,
                    /* 58 EquilibriumZoneColorInput     */ EquilibriumZoneColorInput,
                    /* 59 DiscountZoneColorInput        */ DiscountZoneColorInput,
                    /* 60 LineWidthLiquidated           */ LineWidthLiquidated,
                    /* 61 ShowSignalDots                */ ShowSignalDots,
                    /* 62 ShowSignalsOb                 */ false,
                    /* 63 ShowSignalsFvg                */ ShowSignalsFvg,
                    /* 64 SignalOffsetPips              */ SignalOffsetPips,
                    /* 65 MinDist                       */ MinDist,
                    /* 66 MinDistFvg                    */ MinDistFvg,
                    /* 67 UseHeikinAshi                 */ UseHeikinAshi,
                    /* 68 MinBarsAfterStructureBreak    */ MinBarsAfterStructureBreak
                );
            }
        }

        // ════════════════════════════════════════════════════════════════════════
        //  BAR CLOSE EVENT — check signals on the just-completed bar
        // ════════════════════════════════════════════════════════════════════════

        protected override void OnBar()
        {
            // bar = index of the just-closed (completed) bar
            int bar = Bars.Count - 2;

            // Require enough bars for the indicator warm-up
            if (bar < _minBarsWarmup)
                return;

            // ── OB Signals ────────────────────────────────────────────────────────
            // Each type uses its own output series so they fire independently.
            // LongInternalObBottom / ShortInternalObTop  → Internal OB signal (value = OB level for SL)
            // LongSwingObBottom    / ShortSwingObTop     → Swing   OB signal (value = OB level for SL)
            if (_obIndicator != null)
            {
                if (TradeInternalOb)
                {
                    double longSl  = _obIndicator.LongInternalObBottom[bar];
                    double shortSl = _obIndicator.ShortInternalObTop[bar];

                    if (!double.IsNaN(longSl))
                        PlaceTrade(TradeType.Buy,  longSl  - SlBufferPips * Symbol.PipSize, "SMC_IntOB_Long");

                    if (!double.IsNaN(shortSl))
                        PlaceTrade(TradeType.Sell, shortSl + SlBufferPips * Symbol.PipSize, "SMC_IntOB_Short");
                }

                if (TradeSwingOb)
                {
                    double longSl  = _obIndicator.LongSwingObBottom[bar];
                    double shortSl = _obIndicator.ShortSwingObTop[bar];

                    if (!double.IsNaN(longSl))
                        PlaceTrade(TradeType.Buy,  longSl  - SlBufferPips * Symbol.PipSize, "SMC_SwingOB_Long");

                    if (!double.IsNaN(shortSl))
                        PlaceTrade(TradeType.Sell, shortSl + SlBufferPips * Symbol.PipSize, "SMC_SwingOB_Short");
                }
            }

            // ── FVG Signals ───────────────────────────────────────────────────────
            // LongObBottom[bar]  → non-NaN : bullish FVG long  signal fired
            // ShortObTop[bar]    → non-NaN : bearish FVG short signal fired
            // SL is derived from the trigger bar's low/high, NOT from the FVG level
            // stored in the output series.
            if (_fvgIndicator != null)
            {
                double fvgLongSignal  = _fvgIndicator.LongObBottom[bar];
                double fvgShortSignal = _fvgIndicator.ShortObTop[bar];

                if (!double.IsNaN(fvgLongSignal))
                {
                    // Long FVG trade: SL = trigger bar low − SlBufferPips
                    double slAbsolute = Bars.LowPrices[bar] - SlBufferPips * Symbol.PipSize;
                    PlaceTrade(TradeType.Buy, slAbsolute, "SMC_FVG_Long");
                }

                if (!double.IsNaN(fvgShortSignal))
                {
                    // Short FVG trade: SL = trigger bar high + SlBufferPips
                    double slAbsolute = Bars.HighPrices[bar] + SlBufferPips * Symbol.PipSize;
                    PlaceTrade(TradeType.Sell, slAbsolute, "SMC_FVG_Short");
                }
            }
        }

        // ════════════════════════════════════════════════════════════════════════
        //  TRADE EXECUTION HELPER
        // ════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Sizes a market order to risk exactly 0.5 % of Account.Balance, then
        /// executes it with the supplied stop-loss expressed as an absolute price.
        /// </summary>
        private void PlaceTrade(TradeType tradeType, double stopLossAbsolutePrice, string label)
        {
            // Use live ask/bid as the expected fill price
            double entryPrice = tradeType == TradeType.Buy ? Symbol.Ask : Symbol.Bid;

            // Guard: SL must be on the correct side of entry
            if (tradeType == TradeType.Buy  && stopLossAbsolutePrice >= entryPrice)
                return;
            if (tradeType == TradeType.Sell && stopLossAbsolutePrice <= entryPrice)
                return;

            // Convert absolute SL price → pips distance (always positive)
            double slDistancePips = Math.Abs(entryPrice - stopLossAbsolutePrice) / Symbol.PipSize;
            if (slDistancePips < 0.001)
                return;

            // Position size for 0.5 % risk
            // riskAmount = balance × 0.5 %
            // volume     = riskAmount / (slPips × pipValue)
            double riskAmount     = Account.Balance * 0.005;
            double volumeInUnits  = riskAmount / (slDistancePips * Symbol.PipValue);

            // Normalise to broker lot constraints
            volumeInUnits = Symbol.NormalizeVolumeInUnits(volumeInUnits, RoundingMode.Down);
            volumeInUnits = Math.Max(volumeInUnits, Symbol.VolumeInUnitsMin);
            volumeInUnits = Math.Min(volumeInUnits, Symbol.VolumeInUnitsMax);

            // Calculate TP pips from R:R ratio (null = no TP)
            double? tpDistancePips = null;
            if (TakeProfitEnabled)
            {
                double rrMultiplier = RiskReward switch
                {
                    RiskRewardOption.OneToOne     => 1.0,
                    RiskRewardOption.OneToOneHalf => 1.5,
                    RiskRewardOption.OneToTwo     => 2.0,
                    RiskRewardOption.OneToThree   => 3.0,
                    RiskRewardOption.OneToFour    => 4.0,
                    RiskRewardOption.OneToFive    => 5.0,
                    _                             => 2.0
                };
                tpDistancePips = slDistancePips * rrMultiplier;
            }

            // Execute with SL in pips (cTrader's ExecuteMarketOrder takes pips, not abs price)
            ExecuteMarketOrder(tradeType, SymbolName, volumeInUnits, label, slDistancePips, tpDistancePips);
        }
    }
}
