using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;
using cAlgo.API.Indicators;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class LiquidityInducements : Indicator
    {
        private sealed class Pivot
        {
            public double Price;
            public int BarIndex;
            public int Type; // 1 high, -1 low
            public bool BreakOfStructureBroken;
            public bool LiquidityBroken;
            public bool ChangeOfCharacterBroken;
        }

        private sealed class StructureBreak
        {
            public int X1;
            public double Y1;
            public int X2;
            public double Y2;
        }

        private sealed class LiquidityGrab
        {
            public Pivot Pivot;
            public bool Taken;
            public bool Invalidated;
            public ChartIcon Icon;
            public ChartTrendLine Limit;
            public ChartTrendLine Break;
            public ChartRectangle FillBox;
        }

        private sealed class EqualPivotInducement
        {
            public double StopLosses;
            public Pivot FirstPivot;
            public Pivot SecondPivot;
            public bool LiquidityTaken;
            public ChartText Label;
            public ChartTrendLine Line;
        }

        private sealed class RetracementInducement
        {
            public Pivot Pivot;
            public bool Taken;
            public bool Invalidated;
            public int? StopIndex;
            public ChartTrendLine Line;
            public ChartText Label;
        }

        private sealed class ExternalLiquidity
        {
            public double Price;
            public Pivot Pivot;
            public bool Hidden;
            public ChartTrendLine Line;
            public ChartText Label;
        }

        private sealed class TurtleSoup
        {
            public int Start;
            public int End;
            public Pivot Pivot;
            public double Deepest;
            public ChartRectangle Box;
            public ChartTrendLine Line;
        }

        [Parameter("Pivot L", Group = "Market structure", DefaultValue = 5, MinValue = 1)]
        public int MarketLeft { get; set; }
        [Parameter("Pivot R", Group = "Market structure", DefaultValue = 5, MinValue = 1)]
        public int MarketRight { get; set; }
        [Parameter("Show pivots", Group = "Market structure", DefaultValue = false)]
        public bool MarketShowPivots { get; set; }

        [Parameter("Grabs", Group = "Liquidity ($$$)", DefaultValue = true)]
        public bool GrabsEnabled { get; set; }
        [Parameter("Big grabs", Group = "Liquidity ($$$)", DefaultValue = true)]
        public bool BigGrabsEnabled { get; set; }
        [Parameter("Sweeps", Group = "Liquidity ($$$)", DefaultValue = true)]
        public bool SweepsEnabled { get; set; }
        [Parameter("Turtle soups", Group = "Liquidity ($$$)", DefaultValue = true)]
        public bool TurtleSoupsEnabled { get; set; }
        [Parameter("Equal highs/lows", Group = "Liquidity ($$$)", DefaultValue = true)]
        public bool EqualPivotsEnabled { get; set; }
        [Parameter("BSL & SSL", Group = "Liquidity ($$$)", DefaultValue = true)]
        public bool ExternalLiquidityEnabled { get; set; }

        [Parameter("Retracement", Group = "Inducement (IDM)", DefaultValue = true)]
        public bool RetracementEnabled { get; set; }

        [Parameter("Grabs L", Group = "Grabs", DefaultValue = 3, MinValue = 1)]
        public int GrabsLeft { get; set; }
        [Parameter("Grabs R", Group = "Grabs", DefaultValue = 3, MinValue = 1)]
        public int GrabsRight { get; set; }
        [Parameter("Grabs Lookback", Group = "Grabs", DefaultValue = 5, MinValue = 1)]
        public int GrabsLookback { get; set; }
        [Parameter("Grabs Timeframe", Group = "Grabs", DefaultValue = "Minute")]
        public TimeFrame GrabsTimeframe { get; set; }
        [Parameter("Grabs Color", Group = "Grabs", DefaultValue = "#FFA500")]
        public Color GrabsColor { get; set; }

        [Parameter("Big Grabs L", Group = "Big grabs", DefaultValue = 10, MinValue = 1)]
        public int BigGrabsLeft { get; set; }
        [Parameter("Big Grabs R", Group = "Big grabs", DefaultValue = 10, MinValue = 1)]
        public int BigGrabsRight { get; set; }
        [Parameter("Big Grabs Lookback", Group = "Big grabs", DefaultValue = 5, MinValue = 1)]
        public int BigGrabsLookback { get; set; }
        [Parameter("Big Grabs Timeframe", Group = "Big grabs", DefaultValue = "Minute")]
        public TimeFrame BigGrabsTimeframe { get; set; }
        [Parameter("Big Grabs Color", Group = "Big grabs", DefaultValue = "#00FFFF")]
        public Color BigGrabsColor { get; set; }

        [Parameter("Sweeps L", Group = "Sweeps", DefaultValue = 3, MinValue = 1)]
        public int SweepsLeft { get; set; }
        [Parameter("Sweeps R", Group = "Sweeps", DefaultValue = 3, MinValue = 1)]
        public int SweepsRight { get; set; }
        [Parameter("Sweeps Lookback", Group = "Sweeps", DefaultValue = 5, MinValue = 1)]
        public int SweepsLookback { get; set; }
        [Parameter("Sweeps Timeframe", Group = "Sweeps", DefaultValue = "Minute")]
        public TimeFrame SweepsTimeframe { get; set; }
        [Parameter("Sweeps Bull", Group = "Sweeps", DefaultValue = "#008080")]
        public Color SweepsBullishColor { get; set; }
        [Parameter("Sweeps Bear", Group = "Sweeps", DefaultValue = "#FF0000")]
        public Color SweepsBearishColor { get; set; }

        [Parameter("Turtle L", Group = "Turtle soups", DefaultValue = 1, MinValue = 1)]
        public int TurtleLeft { get; set; }
        [Parameter("Turtle R", Group = "Turtle soups", DefaultValue = 1, MinValue = 1)]
        public int TurtleRight { get; set; }
        [Parameter("Turtle Lookback", Group = "Turtle soups", DefaultValue = 5, MinValue = 1)]
        public int TurtleLookback { get; set; }
        [Parameter("Turtle Timeframe", Group = "Turtle soups", DefaultValue = "Minute")]
        public TimeFrame TurtleTimeframe { get; set; }
        [Parameter("Turtle Color", Group = "Turtle soups", DefaultValue = "#B3FFA500")]
        public Color TurtleColor { get; set; }
        [Parameter("Turtle confirmation", Group = "Turtle soups", DefaultValue = true)]
        public bool TurtleConfirmation { get; set; }

        [Parameter("Equal L", Group = "Equal highs/lows", DefaultValue = 1, MinValue = 1)]
        public int EqualLeft { get; set; }
        [Parameter("Equal R", Group = "Equal highs/lows", DefaultValue = 1, MinValue = 1)]
        public int EqualRight { get; set; }
        [Parameter("Equal ATR factor", Group = "Equal highs/lows", DefaultValue = 0.5, MinValue = 0)]
        public double EqualAtrFactor { get; set; }
        [Parameter("Equal Lookback", Group = "Equal highs/lows", DefaultValue = 3, MinValue = 1)]
        public int EqualLookback { get; set; }
        [Parameter("Equal Timeframe", Group = "Equal highs/lows", DefaultValue = "Minute")]
        public TimeFrame EqualTimeframe { get; set; }
        [Parameter("Equal Liquidity Color", Group = "Equal highs/lows", DefaultValue = "#FFA500")]
        public Color EqualLiquidityColor { get; set; }
        [Parameter("Equal Bull IDM", Group = "Equal highs/lows", DefaultValue = "#008080")]
        public Color EqualBullInducementColor { get; set; }
        [Parameter("Equal Bear IDM", Group = "Equal highs/lows", DefaultValue = "#FF0000")]
        public Color EqualBearInducementColor { get; set; }

        [Parameter("Retr L", Group = "Retracement", DefaultValue = 1, MinValue = 1)]
        public int RetrLeft { get; set; }
        [Parameter("Retr R", Group = "Retracement", DefaultValue = 1, MinValue = 1)]
        public int RetrRight { get; set; }
        [Parameter("Retr Lookback", Group = "Retracement", DefaultValue = 5, MinValue = 1)]
        public int RetrLookback { get; set; }
        [Parameter("Retr Timeframe", Group = "Retracement", DefaultValue = "Minute")]
        public TimeFrame RetrTimeframe { get; set; }
        [Parameter("Retr Bull", Group = "Retracement", DefaultValue = "#008080")]
        public Color RetrBullishColor { get; set; }
        [Parameter("Retr Bear", Group = "Retracement", DefaultValue = "#FF0000")]
        public Color RetrBearishColor { get; set; }
        [Parameter("Keep invalidated", Group = "Retracement", DefaultValue = false)]
        public bool RetrKeepInvalidated { get; set; }

        [Parameter("Show external levels", Group = "External Liquidity", DefaultValue = 1, MinValue = 1)]
        public int ExternalShow { get; set; }
        [Parameter("External Bull", Group = "External Liquidity", DefaultValue = "#008080")]
        public Color ExternalBullishColor { get; set; }
        [Parameter("External Bear", Group = "External Liquidity", DefaultValue = "#FF0000")]
        public Color ExternalBearishColor { get; set; }

        [Parameter("MS Font", Group = "Display", DefaultValue = 7, MinValue = 6, MaxValue = 20)]
        public int MarketStructureFontSize { get; set; }
        [Parameter("Liquidity Font", Group = "Display", DefaultValue = 7, MinValue = 6, MaxValue = 20)]
        public int LiquidityFontSize { get; set; }
        [Parameter("Line Style", Group = "Display", DefaultValue = "Dotted")]
        public string LineStyleInput { get; set; }

        [Output("LiqBuysideTarget", LineColor = "#00FF00", PlotType = PlotType.Points, Thickness = 3)]
        public IndicatorDataSeries LiqBuysideTarget { get; set; }
        [Output("LiqSellsideTarget", LineColor = "#FF0000", PlotType = PlotType.Points, Thickness = 3)]
        public IndicatorDataSeries LiqSellsideTarget { get; set; }
        [Output("DebugTrend", LineColor = "#FFFFFF", PlotType = PlotType.Line, Thickness = 1)]
        public IndicatorDataSeries DebugTrend { get; set; }
        [Output("DebugChoch", LineColor = "#00FFFF", PlotType = PlotType.Histogram, Thickness = 2)]
        public IndicatorDataSeries DebugChoch { get; set; }
        [Output("DebugBos", LineColor = "#FFA500", PlotType = PlotType.Histogram, Thickness = 2)]
        public IndicatorDataSeries DebugBos { get; set; }

        private AverageTrueRange _atr;
        private int _structureTrend;
        private List<Pivot> _structurePivots;
        private List<StructureBreak> _structureBosList;
        private Pivot _changeOfCharacter;
        private Pivot _breakOfStructure;
        private Pivot _previousStructureBreakPivot;
        private int? _previousStructureBreakIndex;
        private int? _retracementStructureBreakIndex;

        private readonly List<LiquidityGrab> _grabsHighs = new List<LiquidityGrab>();
        private readonly List<LiquidityGrab> _grabsLows = new List<LiquidityGrab>();
        private readonly List<LiquidityGrab> _bigGrabsHighs = new List<LiquidityGrab>();
        private readonly List<LiquidityGrab> _bigGrabsLows = new List<LiquidityGrab>();
        private readonly List<LiquidityGrab> _sweepsHighs = new List<LiquidityGrab>();
        private readonly List<LiquidityGrab> _sweepsLows = new List<LiquidityGrab>();
        private readonly List<Pivot> _turtlePivotHighs = new List<Pivot>();
        private readonly List<Pivot> _turtlePivotLows = new List<Pivot>();
        private readonly List<TurtleSoup> _turtleBullish = new List<TurtleSoup>();
        private readonly List<TurtleSoup> _turtleBearish = new List<TurtleSoup>();
        private readonly List<Pivot> _eqHighs = new List<Pivot>();
        private readonly List<Pivot> _eqLows = new List<Pivot>();
        private readonly List<EqualPivotInducement> _eqBearishInducements = new List<EqualPivotInducement>();
        private readonly List<EqualPivotInducement> _eqBullishInducements = new List<EqualPivotInducement>();
        private readonly List<ExternalLiquidity> _buyside = new List<ExternalLiquidity>();
        private readonly List<ExternalLiquidity> _sellside = new List<ExternalLiquidity>();
        private readonly List<RetracementInducement> _retrHighs = new List<RetracementInducement>();
        private readonly List<RetracementInducement> _retrLows = new List<RetracementInducement>();
        private readonly List<Pivot> _retrHighPivots = new List<Pivot>();
        private readonly List<Pivot> _retrLowPivots = new List<Pivot>();

        private readonly Dictionary<string, Bars> _tfBars = new Dictionary<string, Bars>();
        private readonly Dictionary<string, int> _lastTfBarIndex = new Dictionary<string, int>();

        private LineStyle ResolvedLineStyle
        {
            get
            {
                if (string.Equals(LineStyleInput, "Solid", StringComparison.OrdinalIgnoreCase)) return LineStyle.Solid;
                if (string.Equals(LineStyleInput, "Dashed", StringComparison.OrdinalIgnoreCase)) return LineStyle.Lines;
                return LineStyle.DotsRare;
            }
        }

        protected override void Initialize()
        {
            _atr = Indicators.AverageTrueRange(14, MovingAverageType.Simple);
            _structurePivots = new List<Pivot>();
            _structureBosList = new List<StructureBreak>();
            _structureTrend = 0;

            RegisterTf("grabs", GrabsTimeframe);
            RegisterTf("big_grabs", BigGrabsTimeframe);
            RegisterTf("sweeps", SweepsTimeframe);
            RegisterTf("turtle", TurtleTimeframe);
            RegisterTf("equal", EqualTimeframe);
            RegisterTf("retr", RetrTimeframe);
        }

        public override void Calculate(int index)
        {
            if (index < 2)
                return;

            var high = Bars.HighPrices[index];
            var low = Bars.LowPrices[index];
            var close = Bars.ClosePrices[index];

            StructurePivotStep(index);
            var lastHigh = FindLatestPivot(1);
            var lastLow = FindLatestPivot(-1);

            _changeOfCharacter = ChangeOfCharacter(index);
            var structureBreakEvent = false;
            if (_changeOfCharacter != null)
            {
                _breakOfStructure = null;
                _previousStructureBreakPivot = _changeOfCharacter;
                _previousStructureBreakIndex = index;
                structureBreakEvent = true;
            }

            var bosPivot = BreakOfStructure(index);
            if (bosPivot != null)
            {
                _breakOfStructure = bosPivot;
                _previousStructureBreakPivot = bosPivot;
                _previousStructureBreakIndex = index;
                structureBreakEvent = true;
            }

            if (index > 0)
            {
                var prevHigh = Bars.HighPrices[index - 1];
                var prevLow = Bars.LowPrices[index - 1];
                ProcessGrabs(_grabsHighs, prevHigh, prevLow, close, index, GrabsColor, "grab");
                ProcessGrabs(_grabsLows, prevHigh, prevLow, close, index, GrabsColor, "grab");
                ProcessGrabs(_bigGrabsHighs, prevHigh, prevLow, close, index, BigGrabsColor, "biggrab");
                ProcessGrabs(_bigGrabsLows, prevHigh, prevLow, close, index, BigGrabsColor, "biggrab");
                ProcessSweeps(prevHigh, prevLow, close, index);
            }

            if (GrabsEnabled)
            {
                if (IsNewTfBar("grabs", index, out var tfGrabs))
                    AddPivotIfAnyTf("grabs", index, GrabsLeft, GrabsRight, GrabsLookback, _grabsHighs, _grabsLows);
            }
            if (BigGrabsEnabled)
            {
                if (IsNewTfBar("big_grabs", index, out var tfBigGrabs))
                    AddPivotIfAnyTf("big_grabs", index, BigGrabsLeft, BigGrabsRight, BigGrabsLookback, _bigGrabsHighs, _bigGrabsLows);
            }
            if (SweepsEnabled)
            {
                if (IsNewTfBar("sweeps", index, out var tfSweeps))
                    AddPivotIfAnyTf("sweeps", index, SweepsLeft, SweepsRight, SweepsLookback, _sweepsHighs, _sweepsLows);
                if (_changeOfCharacter != null && _previousStructureBreakIndex.HasValue)
                {
                    _sweepsHighs.Clear();
                    _sweepsLows.Clear();
                }
            }

            if (TurtleSoupsEnabled)
            {
                VisualizeTurtleSoups(_turtlePivotHighs, _turtleBearish, index);
                VisualizeTurtleSoups(_turtlePivotLows, _turtleBullish, index);

                if (TurtleConfirmation && _changeOfCharacter != null && _previousStructureBreakIndex.HasValue)
                {
                    ConfirmTurtle(_turtleBullish, index);
                    ConfirmTurtle(_turtleBearish, index);
                }

                if (IsNewTfBar("turtle", index, out var tfTurtle))
                    AddTurtlePivotsFromTf(index);
            }

            if (EqualPivotsEnabled)
            {
                if (IsNewTfBar("equal", index, out var tfEqual))
                    UpdateEqualPivotsFromTf(index, "equal");
                ProcessEqualPivotTriggers(index, high, low, structureBreakEvent);
            }

            if (ExternalLiquidityEnabled)
                ProcessExternalLiquidity(index, high, low, lastHigh, lastLow);

            if (RetracementEnabled)
            {
                if (IsNewTfBar("retr", index, out var tfRetr))
                    UpdateRetracementPivotsFromTf(index, "retr");
                ProcessRetracementInducements(index, high, low, structureBreakEvent);
            }

            DrawStructure(index);
            DrawExternalLiquidity(index);
            DrawRetracement(index);
            PublishOutputs(index);

            DebugTrend[index] = _structureTrend;
            DebugChoch[index] = _changeOfCharacter != null ? _changeOfCharacter.Price : 0;
            DebugBos[index] = _breakOfStructure != null ? _breakOfStructure.Price : 0;
        }


        private void RegisterTf(string key, TimeFrame tf)
        {
            _tfBars[key] = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf);
            _lastTfBarIndex[key] = -1;
        }

        private int FindBarIndexAtOrBefore(Bars bars, DateTime time)
        {
            var lo = 0;
            var hi = bars.Count - 1;
            var ans = -1;
            while (lo <= hi)
            {
                var mid = (lo + hi) / 2;
                if (bars.OpenTimes[mid] <= time)
                {
                    ans = mid;
                    lo = mid + 1;
                }
                else
                {
                    hi = mid - 1;
                }
            }
            return ans;
        }

        private bool IsNewTfBar(string key, int chartIndex, out int tfIndex)
        {
            var tfBars = _tfBars[key];
            tfIndex = FindBarIndexAtOrBefore(tfBars, Bars.OpenTimes[chartIndex]);
            if (tfIndex < 0)
                return false;
            if (_lastTfBarIndex[key] == tfIndex)
                return false;
            _lastTfBarIndex[key] = tfIndex;
            return true;
        }

        private void PublishOutputs(int index)
        {
            LiqBuysideTarget[index] = double.NaN;
            LiqSellsideTarget[index] = double.NaN;

            foreach (var b in _buyside)
            {
                if (!b.Hidden)
                {
                    LiqBuysideTarget[index] = b.Price;
                    break;
                }
            }
            foreach (var s in _sellside)
            {
                if (!s.Hidden)
                {
                    LiqSellsideTarget[index] = s.Price;
                    break;
                }
            }
        }

        private void StructurePivotStep(int index)
        {
            var left = MarketLeft;
            var right = MarketRight;
            if (index < left + right)
                return;

            var pivotIdx = index - right;
            var centerHigh = Bars.HighPrices[pivotIdx];
            var centerLow = Bars.LowPrices[pivotIdx];

            var isPh = true;
            for (var j = pivotIdx - left; j < pivotIdx; j++)
            {
                if (Bars.HighPrices[j] >= centerHigh) { isPh = false; break; }
            }
            if (isPh)
            {
                for (var j = pivotIdx + 1; j <= pivotIdx + right; j++)
                {
                    if (Bars.HighPrices[j] >= centerHigh) { isPh = false; break; }
                }
            }
            if (isPh)
            {
                if (_structurePivots.Count > 5) _structurePivots.RemoveAt(_structurePivots.Count - 1);
                _structurePivots.Insert(0, new Pivot { Price = centerHigh, BarIndex = pivotIdx, Type = 1 });
                if (MarketShowPivots)
                    Chart.DrawIcon($"ms_ph_{pivotIdx}", ChartIconType.Diamond, pivotIdx, centerHigh, Color.Gray);
            }

            var isPl = true;
            for (var j = pivotIdx - left; j < pivotIdx; j++)
            {
                if (Bars.LowPrices[j] <= centerLow) { isPl = false; break; }
            }
            if (isPl)
            {
                for (var j = pivotIdx + 1; j <= pivotIdx + right; j++)
                {
                    if (Bars.LowPrices[j] <= centerLow) { isPl = false; break; }
                }
            }
            if (isPl)
            {
                if (_structurePivots.Count > 5) _structurePivots.RemoveAt(_structurePivots.Count - 1);
                _structurePivots.Insert(0, new Pivot { Price = centerLow, BarIndex = pivotIdx, Type = -1 });
                if (MarketShowPivots)
                    Chart.DrawIcon($"ms_pl_{pivotIdx}", ChartIconType.Diamond, pivotIdx, centerLow, Color.Gray);
            }
        }

        private Pivot FindLatestPivot(int type)
        {
            foreach (var p in _structurePivots)
                if (p.Type == type) return p;
            return null;
        }

        private Pivot ChangeOfCharacter(int index)
        {
            var closeNow = Bars.ClosePrices[index];
            var closePrev = Bars.ClosePrices[index - 1];

            for (var i = 0; i < _structurePivots.Count; i++)
            {
                var pivot = _structurePivots[i];
                if (_structureTrend <= 0 && pivot.Type == 1 && closeNow > pivot.Price && closePrev < pivot.Price && !pivot.ChangeOfCharacterBroken)
                {
                    pivot.ChangeOfCharacterBroken = true;
                    _structureTrend = 1;
                    _structureBosList.Clear();
                    var remaining = new List<Pivot>();
                    foreach (var p in _structurePivots)
                    {
                        if (p.BarIndex <= pivot.BarIndex) continue;
                        p.BreakOfStructureBroken = true;
                        p.ChangeOfCharacterBroken = p.BarIndex == pivot.BarIndex;
                        remaining.Add(p);
                    }
                    _structurePivots = remaining;
                    return pivot;
                }

                if (_structureTrend >= 0 && pivot.Type == -1 && closeNow < pivot.Price && closePrev > pivot.Price && !pivot.ChangeOfCharacterBroken)
                {
                    pivot.ChangeOfCharacterBroken = true;
                    _structureTrend = -1;
                    _structureBosList.Clear();
                    var remaining = new List<Pivot>();
                    foreach (var p in _structurePivots)
                    {
                        if (p.BarIndex <= pivot.BarIndex) continue;
                        p.BreakOfStructureBroken = true;
                        p.ChangeOfCharacterBroken = p.BarIndex == pivot.BarIndex;
                        remaining.Add(p);
                    }
                    _structurePivots = remaining;
                    return pivot;
                }
            }
            return null;
        }

        private Pivot BreakOfStructure(int index)
        {
            var closeNow = Bars.ClosePrices[index];

            foreach (var pivot in _structurePivots)
            {
                if (_structureTrend == 1 && pivot.Type == 1 && closeNow > pivot.Price && !pivot.BreakOfStructureBroken)
                {
                    var create = true;
                    for (var i = _structureBosList.Count - 1; i >= 0; i--)
                    {
                        var bos = _structureBosList[i];
                        if (bos.X1 > pivot.BarIndex)
                        {
                            if (bos.Y1 < pivot.Price)
                            {
                                _structureBosList.RemoveAt(i);
                                continue;
                            }
                            create = false;
                            break;
                        }
                    }
                    if (create)
                    {
                        _structureBosList.Insert(0, new StructureBreak { X1 = pivot.BarIndex, Y1 = pivot.Price, X2 = index, Y2 = pivot.Price });
                        pivot.BreakOfStructureBroken = true;
                        return pivot;
                    }
                }

                if (_structureTrend == -1 && pivot.Type == -1 && closeNow < pivot.Price && !pivot.BreakOfStructureBroken)
                {
                    var create = true;
                    for (var i = _structureBosList.Count - 1; i >= 0; i--)
                    {
                        var bos = _structureBosList[i];
                        if (bos.X1 > pivot.BarIndex)
                        {
                            if (bos.Y1 > pivot.Price)
                            {
                                _structureBosList.RemoveAt(i);
                                continue;
                            }
                            create = false;
                            break;
                        }
                    }
                    if (create)
                    {
                        _structureBosList.Insert(0, new StructureBreak { X1 = pivot.BarIndex, Y1 = pivot.Price, X2 = index, Y2 = pivot.Price });
                        pivot.BreakOfStructureBroken = true;
                        return pivot;
                    }
                }
            }
            return null;
        }

        private Pivot DetectPivotHigh(int index, int left, int right)
        {
            if (index < left + right) return null;
            var pivotIdx = index - right;
            var center = Bars.HighPrices[pivotIdx];
            for (var j = pivotIdx - left; j < pivotIdx; j++) if (Bars.HighPrices[j] >= center) return null;
            for (var j = pivotIdx + 1; j <= pivotIdx + right; j++) if (Bars.HighPrices[j] >= center) return null;
            return new Pivot { Price = center, BarIndex = pivotIdx, Type = 1 };
        }

        private Pivot DetectPivotLow(int index, int left, int right)
        {
            if (index < left + right) return null;
            var pivotIdx = index - right;
            var center = Bars.LowPrices[pivotIdx];
            for (var j = pivotIdx - left; j < pivotIdx; j++) if (Bars.LowPrices[j] <= center) return null;
            for (var j = pivotIdx + 1; j <= pivotIdx + right; j++) if (Bars.LowPrices[j] <= center) return null;
            return new Pivot { Price = center, BarIndex = pivotIdx, Type = -1 };
        }

        private Pivot DetectPivotHighFromTf(string tfKey, int chartIndex, int left, int right)
        {
            var tfBars = _tfBars[tfKey];
            var i = FindBarIndexAtOrBefore(tfBars, Bars.OpenTimes[chartIndex]);
            if (i < left + right)
                return null;
            var pivotIdx = i - right;
            var center = tfBars.HighPrices[pivotIdx];
            for (var j = pivotIdx - left; j < pivotIdx; j++) if (tfBars.HighPrices[j] >= center) return null;
            for (var j = pivotIdx + 1; j <= pivotIdx + right; j++) if (tfBars.HighPrices[j] >= center) return null;
            var pivotTime = tfBars.OpenTimes[pivotIdx];
            var mapped = FindBarIndexAtOrBefore(Bars, pivotTime);
            if (mapped < 0) return null;
            return new Pivot { Price = center, BarIndex = mapped, Type = 1 };
        }

        private Pivot DetectPivotLowFromTf(string tfKey, int chartIndex, int left, int right)
        {
            var tfBars = _tfBars[tfKey];
            var i = FindBarIndexAtOrBefore(tfBars, Bars.OpenTimes[chartIndex]);
            if (i < left + right)
                return null;
            var pivotIdx = i - right;
            var center = tfBars.LowPrices[pivotIdx];
            for (var j = pivotIdx - left; j < pivotIdx; j++) if (tfBars.LowPrices[j] <= center) return null;
            for (var j = pivotIdx + 1; j <= pivotIdx + right; j++) if (tfBars.LowPrices[j] <= center) return null;
            var pivotTime = tfBars.OpenTimes[pivotIdx];
            var mapped = FindBarIndexAtOrBefore(Bars, pivotTime);
            if (mapped < 0) return null;
            return new Pivot { Price = center, BarIndex = mapped, Type = -1 };
        }

        private void AddPivotIfAnyTf(string tfKey, int index, int left, int right, int lookback, List<LiquidityGrab> highs, List<LiquidityGrab> lows)
        {
            var ph = DetectPivotHighFromTf(tfKey, index, left, right);
            if (ph != null)
            {
                highs.Insert(0, new LiquidityGrab { Pivot = ph });
                if (highs.Count > lookback) highs.RemoveAt(highs.Count - 1);
            }
            var pl = DetectPivotLowFromTf(tfKey, index, left, right);
            if (pl != null)
            {
                lows.Insert(0, new LiquidityGrab { Pivot = pl });
                if (lows.Count > lookback) lows.RemoveAt(lows.Count - 1);
            }
        }

        private void ProcessGrabs(List<LiquidityGrab> grabs, double prevHigh, double prevLow, double close, int index, Color c, string tag)
        {
            foreach (var grab in grabs)
            {
                if (grab.Taken || grab.Invalidated) continue;
                var grabbed = false;
                if (grab.Pivot.Type == -1)
                {
                    if (prevLow <= grab.Pivot.Price && close >= grab.Pivot.Price) grabbed = true;
                    else if (close < grab.Pivot.Price) grab.Invalidated = true;
                }
                else
                {
                    if (prevHigh >= grab.Pivot.Price && close <= grab.Pivot.Price) grabbed = true;
                    else if (close > grab.Pivot.Price) grab.Invalidated = true;
                }

                if (grabbed)
                {
                    grab.Taken = true;
                    var id = $"{tag}_{grab.Pivot.BarIndex}_{index}_{grab.Pivot.Type}";
                    var iconType = grab.Pivot.Type == -1 ? ChartIconType.UpArrow : ChartIconType.DownArrow;
                    Chart.DrawIcon(id, iconType, index, grab.Pivot.Price, c);
                    grab.Limit = Chart.DrawTrendLine(id + "_lim", grab.Pivot.BarIndex, grab.Pivot.Price, index, grab.Pivot.Price, c, 1, ResolvedLineStyle);
                    var breakPrice = grab.Pivot.Type == -1 ? Bars.LowPrices[index] : Bars.HighPrices[index];
                    grab.Break = Chart.DrawTrendLine(id + "_brk", grab.Pivot.BarIndex, breakPrice, index, breakPrice, Color.FromArgb(0, 0, 0, 0), 1, ResolvedLineStyle);
                    grab.FillBox = Chart.DrawRectangle(id + "_fill", grab.Pivot.BarIndex, Math.Max(grab.Pivot.Price, breakPrice), index, Math.Min(grab.Pivot.Price, breakPrice), Color.FromArgb(80, c.R, c.G, c.B));
                    grab.FillBox.IsFilled = true;
                    grab.FillBox.IsInteractive = false;
                    var txt = "$$$";
                    Chart.DrawText(id + "_t", txt, index, grab.Pivot.Price, c).FontSize = LiquidityFontSize;
                }
            }
        }

        private void ProcessSweeps(double prevHigh, double prevLow, double close, int index)
        {
            var all = new List<LiquidityGrab>();
            all.AddRange(_sweepsHighs);
            all.AddRange(_sweepsLows);

            foreach (var sweep in all)
            {
                if (sweep.Taken || sweep.Invalidated) continue;
                var swept = false;
                if (sweep.Pivot.Type == -1)
                {
                    if (prevLow <= sweep.Pivot.Price && close <= sweep.Pivot.Price)
                    {
                        if (_previousStructureBreakPivot != null && sweep.Pivot.BarIndex == _previousStructureBreakPivot.BarIndex)
                            sweep.Invalidated = true;
                        else
                            swept = true;
                    }
                    else if (prevLow <= sweep.Pivot.Price && close >= sweep.Pivot.Price)
                        sweep.Invalidated = true;
                }
                else
                {
                    if (prevHigh >= sweep.Pivot.Price && close >= sweep.Pivot.Price)
                    {
                        if (_previousStructureBreakPivot != null && sweep.Pivot.BarIndex == _previousStructureBreakPivot.BarIndex)
                            sweep.Invalidated = true;
                        else
                            swept = true;
                    }
                    else if (prevHigh >= sweep.Pivot.Price && close <= sweep.Pivot.Price)
                        sweep.Invalidated = true;
                }

                if (swept)
                {
                    sweep.Taken = true;
                    var c = sweep.Pivot.Type == -1 ? SweepsBullishColor : SweepsBearishColor;
                    Chart.DrawText($"sweep_{sweep.Pivot.BarIndex}_{index}", "$", index, sweep.Pivot.Price, c).FontSize = LiquidityFontSize;
                }
            }
        }

        private void AddTurtlePivotsFromTf(int index)
        {
            var ph = DetectPivotHighFromTf("turtle", index, TurtleLeft, TurtleRight);
            if (ph != null)
            {
                _turtlePivotHighs.Insert(0, ph);
                if (_turtlePivotHighs.Count > TurtleLookback) _turtlePivotHighs.RemoveAt(_turtlePivotHighs.Count - 1);
            }
            var pl = DetectPivotLowFromTf("turtle", index, TurtleLeft, TurtleRight);
            if (pl != null)
            {
                _turtlePivotLows.Insert(0, pl);
                if (_turtlePivotLows.Count > TurtleLookback) _turtlePivotLows.RemoveAt(_turtlePivotLows.Count - 1);
            }
        }

        private void VisualizeTurtleSoups(List<Pivot> pivots, List<TurtleSoup> turtleSoups, int index)
        {
            if (index < 1) return;
            foreach (var pivot in pivots)
            {
                if (pivot.LiquidityBroken) continue;
                bool confirmed;
                if (pivot.Type == -1)
                    confirmed = Bars.LowPrices[index] > pivot.Price && Bars.LowPrices[index - 1] <= pivot.Price;
                else
                    confirmed = Bars.HighPrices[index] < pivot.Price && Bars.HighPrices[index - 1] >= pivot.Price;

                if (!confirmed) continue;

                pivot.LiquidityBroken = true;
                var deepest = pivot.Type == -1 ? Bars.LowPrices[index - 1] : Bars.HighPrices[index - 1];
                var ts = new TurtleSoup { Start = index - 1, End = index, Pivot = pivot, Deepest = deepest };
                turtleSoups.Insert(0, ts);

                var id = $"ts_{pivot.BarIndex}_{index}";
                var box = Chart.DrawRectangle(id + "_b", ts.Start, Math.Max(pivot.Price, deepest), ts.End, Math.Min(pivot.Price, deepest), TurtleColor);
                box.IsFilled = true;
                box.IsInteractive = false;
                ts.Box = box;
                ts.Line = Chart.DrawTrendLine(id + "_l", ts.Start, pivot.Price, ts.End, pivot.Price, TurtleColor, 1, ResolvedLineStyle);
                Chart.DrawText(id + "_t", "$$$", index, pivot.Price, TurtleColor).FontSize = LiquidityFontSize;
                break;
            }
        }

        private void ConfirmTurtle(List<TurtleSoup> turtleSoups, int index)
        {
            for (var i = turtleSoups.Count - 1; i >= 0; i--)
            {
                var t = turtleSoups[i];
                var ok = t.Pivot.Type == -1 ? _structureTrend == 1 : _structureTrend == -1;
                if (!ok)
                {
                    if (t.Box != null) Chart.RemoveObject(t.Box.Name);
                    if (t.Line != null) Chart.RemoveObject(t.Line.Name);
                    turtleSoups.RemoveAt(i);
                }
            }
        }

        private void UpdateEqualPivotsFromTf(int index, string tfKey)
        {
            var ph = DetectPivotHighFromTf(tfKey, index, EqualLeft, EqualRight);
            var pl = DetectPivotLowFromTf(tfKey, index, EqualLeft, EqualRight);
            if (ph != null)
            {
                _eqHighs.Insert(0, ph);
                if (_eqHighs.Count > EqualLookback) _eqHighs.RemoveAt(_eqHighs.Count - 1);
            }
            if (pl != null)
            {
                _eqLows.Insert(0, pl);
                if (_eqLows.Count > EqualLookback) _eqLows.RemoveAt(_eqLows.Count - 1);
            }
        }

        private void ProcessEqualPivotTriggers(int index, double high, double low, bool structureBreakEvent)
        {

            var atr = _atr.Result[index];
            if (_eqHighs.Count > 1)
                CheckEqualPair(_eqHighs[0], _eqHighs[1], 1, atr, index);
            if (_eqLows.Count > 1)
                CheckEqualPair(_eqLows[0], _eqLows[1], -1, atr, index);

            foreach (var ind in _eqBearishInducements)
            {
                if (_structureTrend == -1 && !ind.LiquidityTaken && high >= ind.StopLosses)
                    ind.LiquidityTaken = true;
            }
            foreach (var ind in _eqBullishInducements)
            {
                if (_structureTrend == 1 && !ind.LiquidityTaken && low <= ind.StopLosses)
                    ind.LiquidityTaken = true;
            }

            if (structureBreakEvent)
            {
                _eqBullishInducements.Clear();
                _eqBearishInducements.Clear();
            }
        }

        private void CheckEqualPair(Pivot latest, Pivot prior, int type, double atr, int index)
        {
            if (double.IsNaN(atr) || atr <= 0) return;
            var tol = atr * EqualAtrFactor;
            if (Math.Abs(latest.Price - prior.Price) > tol) return;

            var trendInducement = (type == 1 && _structureTrend == -1) || (type == -1 && _structureTrend == 1);
            var text = trendInducement ? "IDM" : "$$$";
            var c = trendInducement
                ? (_structureTrend == 1 ? EqualBullInducementColor : EqualBearInducementColor)
                : EqualLiquidityColor;
            var midIndex = latest.BarIndex - ((latest.BarIndex - prior.BarIndex) / 2);
            var midPrice = latest.Price + ((prior.Price - latest.Price) / 2.0);
            Chart.DrawText($"eq_{type}_{latest.BarIndex}_{prior.BarIndex}", text, midIndex, midPrice, c).FontSize = LiquidityFontSize;
            Chart.DrawTrendLine($"eq_l_{type}_{latest.BarIndex}_{prior.BarIndex}", latest.BarIndex, latest.Price, prior.BarIndex, prior.Price, c, 1, ResolvedLineStyle);

            if (trendInducement)
            {
                var sl = type == 1 ? prior.Price + (atr * 0.1) : prior.Price - (atr * 0.1);
                var ind = new EqualPivotInducement
                {
                    StopLosses = sl,
                    FirstPivot = prior,
                    SecondPivot = latest
                };
                if (type == 1) _eqBearishInducements.Insert(0, ind); else _eqBullishInducements.Insert(0, ind);
            }
        }

        private void ProcessExternalLiquidity(int index, double high, double low, Pivot lastHigh, Pivot lastLow)
        {
            if (lastHigh != null && lastHigh.BarIndex == index - MarketRight)
            {
                foreach (var pool in _buyside) if (!pool.Hidden) pool.Hidden = true;
                _buyside.Insert(0, new ExternalLiquidity { Price = lastHigh.Price, Pivot = lastHigh, Hidden = true });
            }

            if (lastLow != null && lastLow.BarIndex == index - MarketRight)
            {
                foreach (var pool in _sellside) if (!pool.Hidden) pool.Hidden = true;
                _sellside.Insert(0, new ExternalLiquidity { Price = lastLow.Price, Pivot = lastLow, Hidden = true });
            }

            _sellside.RemoveAll(pool => low <= pool.Price);
            _buyside.RemoveAll(pool => high >= pool.Price);

            for (var i = 0; i < _buyside.Count; i++) if (i < ExternalShow) _buyside[i].Hidden = false;
            for (var i = 0; i < _sellside.Count; i++) if (i < ExternalShow) _sellside[i].Hidden = false;
        }

        private void UpdateRetracementPivotsFromTf(int index, string tfKey)
        {
            var rh = DetectPivotHighFromTf(tfKey, index, RetrLeft, RetrRight);
            var rl = DetectPivotLowFromTf(tfKey, index, RetrLeft, RetrRight);
            if (rh != null)
            {
                _retrHighPivots.Insert(0, rh);
                if (_retrHighPivots.Count > RetrLookback) _retrHighPivots.RemoveAt(_retrHighPivots.Count - 1);
            }
            if (rl != null)
            {
                _retrLowPivots.Insert(0, rl);
                if (_retrLowPivots.Count > RetrLookback) _retrLowPivots.RemoveAt(_retrLowPivots.Count - 1);
            }
        }

        private void ProcessRetracementInducements(int index, double high, double low, bool structureBreakEvent)
        {

            if (_structureTrend != 0)
            {
                var pivots = _structureTrend == -1 ? _retrHighPivots : _retrLowPivots;
                if (pivots.Count > 1)
                {
                    var latest = pivots[0];
                    var nextLatest = pivots[1];
                    if (_retracementStructureBreakIndex.HasValue)
                    {
                        var latestAfterBreak = latest.BarIndex > _retracementStructureBreakIndex.Value;
                        if (latest.BarIndex == index - RetrRight && latestAfterBreak && nextLatest.BarIndex < _retracementStructureBreakIndex.Value)
                        {
                            var target = _structureTrend == -1 ? _retrHighs : _retrLows;
                            target.Insert(0, new RetracementInducement { Pivot = latest });
                        }
                    }
                }
            }

            StopRetracementInducements(high, low, index, "take");
            if (structureBreakEvent)
            {
                StopRetracementInducements(high, low, index, "invalidate");
                _retracementStructureBreakIndex = index;
            }
        }

        private void StopRetracementInducements(double high, double low, int barIndex, string stopReason)
        {
            for (var i = _retrHighs.Count - 1; i >= 0; i--)
            {
                var ind = _retrHighs[i];
                var stop = stopReason == "invalidate" || high >= ind.Pivot.Price;
                if (!stop) continue;
                ind.StopIndex = barIndex;
                if (stopReason == "take")
                {
                    ind.Taken = true;
                    _retrHighs.RemoveAt(i);
                    continue;
                }

                ind.Invalidated = true;
                if (!RetrKeepInvalidated)
                    _retrHighs.RemoveAt(i);
            }

            for (var i = _retrLows.Count - 1; i >= 0; i--)
            {
                var ind = _retrLows[i];
                var stop = stopReason == "invalidate" || low <= ind.Pivot.Price;
                if (!stop) continue;
                ind.StopIndex = barIndex;
                if (stopReason == "take")
                {
                    ind.Taken = true;
                    _retrLows.RemoveAt(i);
                    continue;
                }

                ind.Invalidated = true;
                if (!RetrKeepInvalidated)
                    _retrLows.RemoveAt(i);
            }
        }

        private void DrawStructure(int index)
        {
            if (_changeOfCharacter != null)
            {
                var c = _structureTrend == 1 ? ExternalBullishColor : ExternalBearishColor;
                Chart.DrawText($"choch_{index}", "CHoCH", index, _changeOfCharacter.Price, c).FontSize = MarketStructureFontSize;
            }
            if (_breakOfStructure != null)
            {
                var c = _structureTrend == 1 ? ExternalBullishColor : ExternalBearishColor;
                Chart.DrawText($"bos_{index}", "BOS", index, _breakOfStructure.Price, c).FontSize = MarketStructureFontSize;
            }
        }

        private void DrawExternalLiquidity(int index)
        {
            foreach (var pool in _buyside)
            {
                var id = $"bsl_{pool.Pivot.BarIndex}";
                if (pool.Hidden)
                {
                    Chart.RemoveObject(id);
                    continue;
                }
                Chart.DrawTrendLine(id, pool.Pivot.BarIndex, pool.Price, index, pool.Price, ExternalBullishColor, 1, ResolvedLineStyle);
                Chart.DrawText(id + "_t", "BSL", index, pool.Price, ExternalBullishColor).FontSize = LiquidityFontSize;
            }
            foreach (var pool in _sellside)
            {
                var id = $"ssl_{pool.Pivot.BarIndex}";
                if (pool.Hidden)
                {
                    Chart.RemoveObject(id);
                    continue;
                }
                Chart.DrawTrendLine(id, pool.Pivot.BarIndex, pool.Price, index, pool.Price, ExternalBearishColor, 1, ResolvedLineStyle);
                Chart.DrawText(id + "_t", "SSL", index, pool.Price, ExternalBearishColor).FontSize = LiquidityFontSize;
            }
        }

        private void DrawRetracement(int index)
        {
            foreach (var ind in _retrHighs)
            {
                var id = $"retr_h_{ind.Pivot.BarIndex}";
                Chart.DrawTrendLine(id, ind.Pivot.BarIndex, ind.Pivot.Price, index, ind.Pivot.Price, RetrBearishColor, 1, ResolvedLineStyle);
                Chart.DrawText(id + "_t", "IDM", index, ind.Pivot.Price, RetrBearishColor).FontSize = LiquidityFontSize;
            }
            foreach (var ind in _retrLows)
            {
                var id = $"retr_l_{ind.Pivot.BarIndex}";
                Chart.DrawTrendLine(id, ind.Pivot.BarIndex, ind.Pivot.Price, index, ind.Pivot.Price, RetrBullishColor, 1, ResolvedLineStyle);
                Chart.DrawText(id + "_t", "IDM", index, ind.Pivot.Price, RetrBullishColor).FontSize = LiquidityFontSize;
            }
        }
    }
}
