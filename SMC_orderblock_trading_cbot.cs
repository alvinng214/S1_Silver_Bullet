using System;
using cAlgo.API;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class SMC_OrderBlock_Trading_CBot : Robot
    {
        public enum DisplayMode { Historical, Present }
        public enum ThemeStyle { Colored, Monochrome }
        public enum StructureFilter { All, BOS, CHOCH }
        public enum LabelSizeOpt { Tiny, Small, Normal }
        public enum ObFilter { Atr, CumulativeMeanRange }
        public enum MitigationMode { Close, HighLow }
        public enum LineStyleOpt { Solid, Dashed, Dotted }
        public enum StructureSource { Close, HighLow }

        [Parameter("Mode", DefaultValue = DisplayMode.Historical, Group = "Smart Money Concepts")]
        public DisplayMode ModeInput { get; set; }
        [Parameter("Style", DefaultValue = ThemeStyle.Colored, Group = "Smart Money Concepts")]
        public ThemeStyle StyleInput { get; set; }
        [Parameter("Color Candles", DefaultValue = false, Group = "Smart Money Concepts")]
        public bool ShowTrendInput { get; set; }
        [Parameter("Plot Signal Series", DefaultValue = false, Group = "Smart Money Concepts")]
        public bool PlotSignalSeriesInput { get; set; }

        [Parameter("Show Internal Structure", DefaultValue = true, Group = "Real Time Internal Structure")]
        public bool ShowInternalsInput { get; set; }
        [Parameter("Bullish Structure", DefaultValue = StructureFilter.All, Group = "Real Time Internal Structure")]
        public StructureFilter ShowInternalBullInput { get; set; }
        [Parameter("Internal Bull Color", DefaultValue = "#089981", Group = "Real Time Internal Structure")]
        public Color InternalBullColorInput { get; set; }
        [Parameter("Bearish Structure", DefaultValue = StructureFilter.All, Group = "Real Time Internal Structure")]
        public StructureFilter ShowInternalBearInput { get; set; }
        [Parameter("Internal Bear Color", DefaultValue = "#F23645", Group = "Real Time Internal Structure")]
        public Color InternalBearColorInput { get; set; }
        [Parameter("Confluence Filter", DefaultValue = false, Group = "Real Time Internal Structure")]
        public bool InternalFilterConfluenceInput { get; set; }
        [Parameter("Internal Label Size", DefaultValue = LabelSizeOpt.Tiny, Group = "Real Time Internal Structure")]
        public LabelSizeOpt InternalStructureSize { get; set; }
        [Parameter("iBOS/iCHoCH Source", DefaultValue = StructureSource.Close, Group = "Real Time Internal Structure")]
        public StructureSource InternalStructureSourceInput { get; set; }
        [Parameter("Internal Pivot Length", DefaultValue = 5, MinValue = 1, MaxValue = 50, Group = "Real Time Internal Structure")]
        public int InternalPivotLength { get; set; }

        [Parameter("Show Swing Structure", DefaultValue = true, Group = "Real Time Swing Structure")]
        public bool ShowStructureInput { get; set; }
        [Parameter("Bullish Structure", DefaultValue = StructureFilter.All, Group = "Real Time Swing Structure")]
        public StructureFilter ShowSwingBullInput { get; set; }
        [Parameter("Swing Bull Color", DefaultValue = "#089981", Group = "Real Time Swing Structure")]
        public Color SwingBullishColorInput { get; set; }
        [Parameter("Bearish Structure", DefaultValue = StructureFilter.All, Group = "Real Time Swing Structure")]
        public StructureFilter ShowSwingBearInput { get; set; }
        [Parameter("Swing Bear Color", DefaultValue = "#F23645", Group = "Real Time Swing Structure")]
        public Color SwingBearishColorInput { get; set; }
        [Parameter("Swing Label Size", DefaultValue = LabelSizeOpt.Small, Group = "Real Time Swing Structure")]
        public LabelSizeOpt SwingStructureSize { get; set; }
        [Parameter("Show Swings Points", DefaultValue = false, Group = "Real Time Swing Structure")]
        public bool ShowSwingsInput { get; set; }
        [Parameter("Swings Length", DefaultValue = 50, MinValue = 10, Group = "Real Time Swing Structure")]
        public int SwingsLengthInput { get; set; }
        [Parameter("Show Strong/Weak High/Low", DefaultValue = true, Group = "Real Time Swing Structure")]
        public bool ShowHighLowSwingsInput { get; set; }
        [Parameter("BOS/CHoCH Source", DefaultValue = StructureSource.Close, Group = "Real Time Swing Structure")]
        public StructureSource SwingStructureSourceInput { get; set; }

        [Parameter("Internal Order Blocks", DefaultValue = true, Group = "Order Blocks")]
        public bool ShowInternalOrderBlocksInput { get; set; }
        [Parameter("Internal OB Size", DefaultValue = 5, MinValue = 1, MaxValue = 20, Group = "Order Blocks")]
        public int InternalOrderBlocksSizeInput { get; set; }
        [Parameter("Swing Order Blocks", DefaultValue = true, Group = "Order Blocks")]
        public bool ShowSwingOrderBlocksInput { get; set; }
        [Parameter("Swing OB Size", DefaultValue = 5, MinValue = 1, MaxValue = 20, Group = "Order Blocks")]
        public int SwingOrderBlocksSizeInput { get; set; }
        [Parameter("Order Block Filter", DefaultValue = ObFilter.Atr, Group = "Order Blocks")]
        public ObFilter OrderBlockFilterInput { get; set; }
        [Parameter("OB Filter ATR Period", DefaultValue = 200, MinValue = 1, MaxValue = 500, Group = "Order Blocks")]
        public int ObFilterAtrPeriod { get; set; }
        [Parameter("OB Filter CMR Period", DefaultValue = 0, MinValue = 0, MaxValue = 500, Group = "Order Blocks")]
        public int ObFilterCmrPeriod { get; set; }
        [Parameter("Order Block Mitigation", DefaultValue = MitigationMode.HighLow, Group = "Order Blocks")]
        public MitigationMode OrderBlockMitigationInput { get; set; }
        [Parameter("Internal Bullish OB", DefaultValue = "#CC3179F5", Group = "Order Blocks")]
        public Color InternalBullishOrderBlockColor { get; set; }
        [Parameter("Internal Bearish OB", DefaultValue = "#CCF77C80", Group = "Order Blocks")]
        public Color InternalBearishOrderBlockColor { get; set; }
        [Parameter("Bullish OB", DefaultValue = "#CC1848CC", Group = "Order Blocks")]
        public Color SwingBullishOrderBlockColor { get; set; }
        [Parameter("Bearish OB", DefaultValue = "#CCB22833", Group = "Order Blocks")]
        public Color SwingBearishOrderBlockColor { get; set; }
        [Parameter("Show All Historical OBs", DefaultValue = true, Group = "Order Blocks")]
        public bool ShowAllHistoricalObs { get; set; }
        [Parameter("Show Mitigated OBs", DefaultValue = true, Group = "Order Blocks")]
        public bool ShowMitigatedObs { get; set; }
        [Parameter("Mitigated OB Opacity (%)", DefaultValue = 30, MinValue = 1, MaxValue = 99, Group = "Order Blocks")]
        public int MitigatedObOpacity { get; set; }

        [Parameter("Equal High/Low", DefaultValue = true, Group = "EQH/EQL")]
        public bool ShowEqualHighsLowsInput { get; set; }
        [Parameter("Bars Confirmation", DefaultValue = 3, MinValue = 1, Group = "EQH/EQL")]
        public int EqualHighsLowsLengthInput { get; set; }
        [Parameter("Threshold", DefaultValue = 0.1, MinValue = 0, MaxValue = 0.5, Group = "EQH/EQL")]
        public double EqualHighsLowsThresholdInput { get; set; }

        [Parameter("Fair Value Gaps", DefaultValue = true, Group = "Fair Value Gaps")]
        public bool ShowFairValueGapsInput { get; set; }
        [Parameter("Auto Threshold", DefaultValue = true, Group = "Fair Value Gaps")]
        public bool FairValueGapsThresholdInput { get; set; }
        [Parameter("Bullish FVG", DefaultValue = "#7000FF68", Group = "Fair Value Gaps")]
        public Color FairValueGapsBullColorInput { get; set; }
        [Parameter("Bearish FVG", DefaultValue = "#70FF0008", Group = "Fair Value Gaps")]
        public Color FairValueGapsBearColorInput { get; set; }
        [Parameter("Extend FVG", DefaultValue = 1, MinValue = 0, Group = "Fair Value Gaps")]
        public int FairValueGapsExtendInput { get; set; }

        [Parameter("Daily", DefaultValue = false, Group = "Highs & Lows MTF")]
        public bool ShowDailyLevelsInput { get; set; }
        [Parameter("Daily Style", DefaultValue = LineStyleOpt.Solid, Group = "Highs & Lows MTF")]
        public LineStyleOpt DailyLevelsStyleInput { get; set; }
        [Parameter("Daily Color", DefaultValue = "#2157F3", Group = "Highs & Lows MTF")]
        public Color DailyLevelsColorInput { get; set; }
        [Parameter("Weekly", DefaultValue = false, Group = "Highs & Lows MTF")]
        public bool ShowWeeklyLevelsInput { get; set; }
        [Parameter("Weekly Style", DefaultValue = LineStyleOpt.Solid, Group = "Highs & Lows MTF")]
        public LineStyleOpt WeeklyLevelsStyleInput { get; set; }
        [Parameter("Weekly Color", DefaultValue = "#2157F3", Group = "Highs & Lows MTF")]
        public Color WeeklyLevelsColorInput { get; set; }
        [Parameter("Monthly", DefaultValue = false, Group = "Highs & Lows MTF")]
        public bool ShowMonthlyLevelsInput { get; set; }
        [Parameter("Monthly Style", DefaultValue = LineStyleOpt.Solid, Group = "Highs & Lows MTF")]
        public LineStyleOpt MonthlyLevelsStyleInput { get; set; }
        [Parameter("Monthly Color", DefaultValue = "#2157F3", Group = "Highs & Lows MTF")]
        public Color MonthlyLevelsColorInput { get; set; }

        [Parameter("Premium/Discount Zones", DefaultValue = false, Group = "Premium & Discount Zones")]
        public bool ShowPremiumDiscountZonesInput { get; set; }
        [Parameter("Premium Zone", DefaultValue = "#F23645", Group = "Premium & Discount Zones")]
        public Color PremiumZoneColorInput { get; set; }
        [Parameter("Equilibrium Zone", DefaultValue = "#878B94", Group = "Premium & Discount Zones")]
        public Color EquilibriumZoneColorInput { get; set; }
        [Parameter("Discount Zone", DefaultValue = "#089981", Group = "Premium & Discount Zones")]
        public Color DiscountZoneColorInput { get; set; }

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
        [Parameter("Min Dist OB (bars)", DefaultValue = 1, MinValue = 1, Group = "Signals")]
        public int MinDist { get; set; }
        [Parameter("Min Dist FVG (bars)", DefaultValue = 1, MinValue = 1, Group = "Signals")]
        public int MinDistFvg { get; set; }
        [Parameter("Use Heikin-Ashi", DefaultValue = false, Group = "Signals")]
        public bool UseHeikinAshi { get; set; }
        [Parameter("Min Bars After Structure Break", DefaultValue = 0, MinValue = 0, MaxValue = 200, Group = "Signals")]
        public int MinBarsAfterStructureBreak { get; set; }

        [Parameter("Trigger Internal Order Blocks", DefaultValue = true, Group = "Trade Triggers")]
        public bool TriggerInternalOrderBlocks { get; set; }
        [Parameter("Trigger Swing Order Blocks", DefaultValue = true, Group = "Trade Triggers")]
        public bool TriggerSwingOrderBlocks { get; set; }
        [Parameter("Trigger Fair Value Gaps", DefaultValue = true, Group = "Trade Triggers")]
        public bool TriggerFairValueGaps { get; set; }
        [Parameter("Risk Per Trade (%)", DefaultValue = 0.5, MinValue = 0.01, Step = 0.01, Group = "Risk")]
        public double RiskPerTradePercent { get; set; }
        [Parameter("SL Buffer (pips)", DefaultValue = 5.0, MinValue = 0.0, Step = 0.1, Group = "Risk")]
        public double StopLossBufferPips { get; set; }
        [Parameter("Label", DefaultValue = "SMC_OB_FVG_BOT", Group = "General")]
        public string InstanceLabel { get; set; }

        private SMC_OrderBlock_Detector _internalObIndicator;
        private SMC_OrderBlock_Detector _swingObIndicator;
        private SMC_OrderBlock_Detector _fvgIndicator;

        private int _lastInternalLongBar = -1;
        private int _lastInternalShortBar = -1;
        private int _lastSwingLongBar = -1;
        private int _lastSwingShortBar = -1;
        private int _lastFvgLongBar = -1;
        private int _lastFvgShortBar = -1;

        protected override void OnStart()
        {
            _internalObIndicator = BuildIndicator(true, false, false);
            _swingObIndicator = BuildIndicator(false, true, false);
            _fvgIndicator = BuildIndicator(false, false, true);
        }

        protected override void OnBar()
        {
            var index = Bars.Count - 2;
            if (index < 1)
                return;

            if (TriggerInternalOrderBlocks)
                HandleSignals(index, _internalObIndicator, SignalSource.InternalOb, ref _lastInternalLongBar, ref _lastInternalShortBar);

            if (TriggerSwingOrderBlocks)
                HandleSignals(index, _swingObIndicator, SignalSource.SwingOb, ref _lastSwingLongBar, ref _lastSwingShortBar);

            if (TriggerFairValueGaps)
                HandleSignals(index, _fvgIndicator, SignalSource.Fvg, ref _lastFvgLongBar, ref _lastFvgShortBar);
        }

        private enum SignalSource { InternalOb, SwingOb, Fvg }

        private void HandleSignals(int index, SMC_OrderBlock_Detector indicator, SignalSource source, ref int lastLongBar, ref int lastShortBar)
        {
            var longSlAnchor = indicator.LongObBottom[index];
            if (!double.IsNaN(longSlAnchor) && lastLongBar != index)
            {
                lastLongBar = index;
                ExecuteRiskManagedOrder(TradeType.Buy, longSlAnchor - StopLossBufferPips * Symbol.PipSize, source, index);
            }

            var shortSlAnchor = indicator.ShortObTop[index];
            if (!double.IsNaN(shortSlAnchor) && lastShortBar != index)
            {
                lastShortBar = index;
                ExecuteRiskManagedOrder(TradeType.Sell, shortSlAnchor + StopLossBufferPips * Symbol.PipSize, source, index);
            }
        }

        private void ExecuteRiskManagedOrder(TradeType tradeType, double stopLossPrice, SignalSource source, int signalBarIndex)
        {
            var entryPrice = tradeType == TradeType.Buy ? Symbol.Ask : Symbol.Bid;
            var stopLossPips = tradeType == TradeType.Buy
                ? (entryPrice - stopLossPrice) / Symbol.PipSize
                : (stopLossPrice - entryPrice) / Symbol.PipSize;

            if (stopLossPips <= 0)
            {
                Print("Skipped {0} {1} signal at bar {2}: invalid stop-loss distance.", source, tradeType, signalBarIndex);
                return;
            }

            var riskAmount = Account.Equity * (RiskPerTradePercent / 100.0);
            var volumeInUnits = riskAmount / (stopLossPips * Symbol.PipValue);
            var normalizedVolume = Symbol.NormalizeVolumeInUnits(volumeInUnits, RoundingMode.Down);

            if (normalizedVolume < Symbol.VolumeInUnitsMin)
            {
                Print("Skipped {0} {1} signal at bar {2}: volume below minimum after risk sizing.", source, tradeType, signalBarIndex);
                return;
            }

            if (normalizedVolume > Symbol.VolumeInUnitsMax)
                normalizedVolume = Symbol.VolumeInUnitsMax;

            var result = ExecuteMarketOrder(tradeType, SymbolName, normalizedVolume, InstanceLabel, stopLossPips, null);
            if (!result.IsSuccessful)
                Print("Order failed ({0} {1}) at bar {2}: {3}", source, tradeType, signalBarIndex, result.Error);
        }

        private SMC_OrderBlock_Detector BuildIndicator(bool useInternalOb, bool useSwingOb, bool useFvg)
        {
            return Indicators.GetIndicator<SMC_OrderBlock_Detector>(
                ModeInput, StyleInput, ShowTrendInput, PlotSignalSeriesInput,
                ShowInternalsInput, (SMC_OrderBlock_Detector.StructureFilter)ShowInternalBullInput, InternalBullColorInput,
                (SMC_OrderBlock_Detector.StructureFilter)ShowInternalBearInput, InternalBearColorInput, InternalFilterConfluenceInput,
                (SMC_OrderBlock_Detector.LabelSizeOpt)InternalStructureSize, (SMC_OrderBlock_Detector.StructureSource)InternalStructureSourceInput, InternalPivotLength,
                ShowStructureInput, (SMC_OrderBlock_Detector.StructureFilter)ShowSwingBullInput, SwingBullishColorInput,
                (SMC_OrderBlock_Detector.StructureFilter)ShowSwingBearInput, SwingBearishColorInput, (SMC_OrderBlock_Detector.LabelSizeOpt)SwingStructureSize,
                ShowSwingsInput, SwingsLengthInput, ShowHighLowSwingsInput, (SMC_OrderBlock_Detector.StructureSource)SwingStructureSourceInput,
                useInternalOb && ShowInternalOrderBlocksInput, InternalOrderBlocksSizeInput,
                useSwingOb && ShowSwingOrderBlocksInput, SwingOrderBlocksSizeInput,
                (SMC_OrderBlock_Detector.ObFilter)OrderBlockFilterInput, ObFilterAtrPeriod, ObFilterCmrPeriod, (SMC_OrderBlock_Detector.MitigationMode)OrderBlockMitigationInput,
                InternalBullishOrderBlockColor, InternalBearishOrderBlockColor, SwingBullishOrderBlockColor, SwingBearishOrderBlockColor,
                ShowAllHistoricalObs, ShowMitigatedObs, MitigatedObOpacity,
                ShowEqualHighsLowsInput, EqualHighsLowsLengthInput, EqualHighsLowsThresholdInput,
                useFvg && ShowFairValueGapsInput, FairValueGapsThresholdInput, FairValueGapsBullColorInput, FairValueGapsBearColorInput, FairValueGapsExtendInput,
                ShowDailyLevelsInput, (SMC_OrderBlock_Detector.LineStyleOpt)DailyLevelsStyleInput, DailyLevelsColorInput,
                ShowWeeklyLevelsInput, (SMC_OrderBlock_Detector.LineStyleOpt)WeeklyLevelsStyleInput, WeeklyLevelsColorInput,
                ShowMonthlyLevelsInput, (SMC_OrderBlock_Detector.LineStyleOpt)MonthlyLevelsStyleInput, MonthlyLevelsColorInput,
                ShowPremiumDiscountZonesInput, PremiumZoneColorInput, EquilibriumZoneColorInput, DiscountZoneColorInput,
                LineWidthLiquidated, ShowSignalDots, ShowSignalsOb, ShowSignalsFvg, SignalOffsetPips,
                MinDist, MinDistFvg, UseHeikinAshi, MinBarsAfterStructureBreak
            );
        }
    }
}
