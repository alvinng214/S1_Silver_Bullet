using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    // ── Timeframe dropdown options ────────────────────────────────────────────
    public enum TfOption { M1, M3, M15, M30, H1, H4, D1 }

    // ── Line style toggle options ─────────────────────────────────────────────
    public enum LineStyleOption { Solid, Dotted }

    [Indicator(IsOverlay = false, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class MarketStructureMtfTrendPt : Indicator
    {
        private sealed class TfState
        {
            public string Key;
            public string TfInput;
            public TimeFrame TimeFrame;
            public Bars TfBars;
            public int PivotLen;
            public bool IsLowerTf;

            public bool ShowChoch;
            public bool ShowBos;
            public Color ChochBull;
            public Color ChochBear;
            public Color BosBull;
            public Color BosBear;

            // ── Line style per signal type ────────────────────────────────────
            public LineStyle BosLineStyle;
            public LineStyle ChochLineStyle;

            public int LastProcessedTfBar = -1;
            public bool CurrentTrend;

            public double LastPivotHigh = double.NaN;
            public double LastPivotLow = double.NaN;
            public double LastBrokenHigh = double.NaN;
            public double LastBrokenLow = double.NaN;
            public DateTime PivotHighTime = DateTime.MinValue;
            public DateTime PivotLowTime = DateTime.MinValue;

            public Color CurrentColor = Color.Transparent;

            public readonly Dictionary<int, CalcPoint> CalcByTfBar = new Dictionary<int, CalcPoint>();
            public readonly Dictionary<int, int> BosBarssinceByChartBar = new Dictionary<int, int>();

            public bool PrevTrendAtChart;
            public bool PrevBosAtChart;
            public int LastMappedTfBarAtChart = -1;
            public int LastBullAlertChartBar = -1;
            public int LastBearAlertChartBar = -1;
        }

        private sealed class CalcPoint
        {
            public bool Trend;
            public bool Bos;
            public DateTime PivotHighTime;
            public DateTime PivotLowTime;
            public double PrevPivotHigh;
            public double PrevPivotLow;
        }

        private sealed class ChartPoint
        {
            public TfState State;
            public bool Trend;
            public bool Bos;
            public bool TrendChanged;
            public bool BosEdge;
            public DateTime PivotHighTime;
            public DateTime PivotLowTime;
            public double PrevPivotHigh;
            public double PrevPivotLow;
            public int BosBarsSince;
            public string TfLabel;
            public double PanelY;
        }

        // ── TF1 ───────────────────────────────────────────────────────────────
        [Parameter("Timeframe 1", Group = "TF1", DefaultValue = TfOption.M15)]
        public TfOption Timeframe1 { get; set; }
        [Parameter("Pivot Strength", Group = "TF1", DefaultValue = 15, MinValue = 1)]
        public int PivotStrength1 { get; set; }
        [Parameter("Lower than chart TF?", Group = "TF1", DefaultValue = false)]
        public bool IsTf1LowerTf { get; set; }
        [Parameter("Show CHoCH", Group = "TF1", DefaultValue = true)]
        public bool ShowChoch1 { get; set; }
        [Parameter("CHoCH Line Style", Group = "TF1", DefaultValue = LineStyleOption.Dotted)]
        public LineStyleOption ChochLineStyle1 { get; set; }
        [Parameter("CHoCH Bull", Group = "TF1", DefaultValue = "#2E6830")]
        public Color ChochBull1 { get; set; }
        [Parameter("CHoCH Bear", Group = "TF1", DefaultValue = "#802929")]
        public Color ChochBear1 { get; set; }
        [Parameter("Show BoS", Group = "TF1", DefaultValue = true)]
        public bool ShowBos1 { get; set; }
        [Parameter("BoS Line Style", Group = "TF1", DefaultValue = LineStyleOption.Dotted)]
        public LineStyleOption BosLineStyle1 { get; set; }
        [Parameter("BoS Bull", Group = "TF1", DefaultValue = "Green")]
        public Color BosBull1 { get; set; }
        [Parameter("BoS Bear", Group = "TF1", DefaultValue = "Red")]
        public Color BosBear1 { get; set; }

        // ── TF2 ───────────────────────────────────────────────────────────────
        [Parameter("Timeframe 2", Group = "TF2", DefaultValue = TfOption.M30)]
        public TfOption Timeframe2 { get; set; }
        [Parameter("Pivot Strength", Group = "TF2", DefaultValue = 15, MinValue = 1)]
        public int PivotStrength2 { get; set; }
        [Parameter("Lower than chart TF?", Group = "TF2", DefaultValue = false)]
        public bool IsTf2LowerTf { get; set; }
        [Parameter("Show CHoCH", Group = "TF2", DefaultValue = false)]
        public bool ShowChoch2 { get; set; }
        [Parameter("CHoCH Line Style", Group = "TF2", DefaultValue = LineStyleOption.Dotted)]
        public LineStyleOption ChochLineStyle2 { get; set; }
        [Parameter("CHoCH Bull", Group = "TF2", DefaultValue = "#2E6830")]
        public Color ChochBull2 { get; set; }
        [Parameter("CHoCH Bear", Group = "TF2", DefaultValue = "#802929")]
        public Color ChochBear2 { get; set; }
        [Parameter("Show BoS", Group = "TF2", DefaultValue = false)]
        public bool ShowBos2 { get; set; }
        [Parameter("BoS Line Style", Group = "TF2", DefaultValue = LineStyleOption.Dotted)]
        public LineStyleOption BosLineStyle2 { get; set; }
        [Parameter("BoS Bull", Group = "TF2", DefaultValue = "Green")]
        public Color BosBull2 { get; set; }
        [Parameter("BoS Bear", Group = "TF2", DefaultValue = "Red")]
        public Color BosBear2 { get; set; }

        // ── TF3 ───────────────────────────────────────────────────────────────
        [Parameter("Timeframe 3", Group = "TF3", DefaultValue = TfOption.H1)]
        public TfOption Timeframe3 { get; set; }
        [Parameter("Pivot Strength", Group = "TF3", DefaultValue = 15, MinValue = 1)]
        public int PivotStrength3 { get; set; }
        [Parameter("Lower than chart TF?", Group = "TF3", DefaultValue = false)]
        public bool IsTf3LowerTf { get; set; }
        [Parameter("Show CHoCH", Group = "TF3", DefaultValue = false)]
        public bool ShowChoch3 { get; set; }
        [Parameter("CHoCH Line Style", Group = "TF3", DefaultValue = LineStyleOption.Dotted)]
        public LineStyleOption ChochLineStyle3 { get; set; }
        [Parameter("CHoCH Bull", Group = "TF3", DefaultValue = "#2E6830")]
        public Color ChochBull3 { get; set; }
        [Parameter("CHoCH Bear", Group = "TF3", DefaultValue = "#802929")]
        public Color ChochBear3 { get; set; }
        [Parameter("Show BoS", Group = "TF3", DefaultValue = false)]
        public bool ShowBos3 { get; set; }
        [Parameter("BoS Line Style", Group = "TF3", DefaultValue = LineStyleOption.Dotted)]
        public LineStyleOption BosLineStyle3 { get; set; }
        [Parameter("BoS Bull", Group = "TF3", DefaultValue = "Green")]
        public Color BosBull3 { get; set; }
        [Parameter("BoS Bear", Group = "TF3", DefaultValue = "Red")]
        public Color BosBear3 { get; set; }

        // ── TF4 ───────────────────────────────────────────────────────────────
        [Parameter("Timeframe 4", Group = "TF4", DefaultValue = TfOption.H4)]
        public TfOption Timeframe4 { get; set; }
        [Parameter("Pivot Strength", Group = "TF4", DefaultValue = 15, MinValue = 1)]
        public int PivotStrength4 { get; set; }
        [Parameter("Lower than chart TF?", Group = "TF4", DefaultValue = false)]
        public bool IsTf4LowerTf { get; set; }
        [Parameter("Show CHoCH", Group = "TF4", DefaultValue = false)]
        public bool ShowChoch4 { get; set; }
        [Parameter("CHoCH Line Style", Group = "TF4", DefaultValue = LineStyleOption.Dotted)]
        public LineStyleOption ChochLineStyle4 { get; set; }
        [Parameter("CHoCH Bull", Group = "TF4", DefaultValue = "#2E6830")]
        public Color ChochBull4 { get; set; }
        [Parameter("CHoCH Bear", Group = "TF4", DefaultValue = "#802929")]
        public Color ChochBear4 { get; set; }
        [Parameter("Show BoS", Group = "TF4", DefaultValue = false)]
        public bool ShowBos4 { get; set; }
        [Parameter("BoS Line Style", Group = "TF4", DefaultValue = LineStyleOption.Dotted)]
        public LineStyleOption BosLineStyle4 { get; set; }
        [Parameter("BoS Bull", Group = "TF4", DefaultValue = "Green")]
        public Color BosBull4 { get; set; }
        [Parameter("BoS Bear", Group = "TF4", DefaultValue = "Red")]
        public Color BosBear4 { get; set; }

        [Parameter("Enable CHoCH Alerts", Group = "Alerts", DefaultValue = false)]
        public bool EnableChochAlerts { get; set; }

        // ── Signal outputs — readable by a cBot, invisible in the panel ──────
        [Output("TF1 BoS Bull",   LineColor = "Transparent", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries Tf1BosBull { get; set; }
        [Output("TF1 BoS Bear",   LineColor = "Transparent", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries Tf1BosBear { get; set; }
        [Output("TF1 CHoCH Bull", LineColor = "Transparent", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries Tf1ChochBull { get; set; }
        [Output("TF1 CHoCH Bear", LineColor = "Transparent", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries Tf1ChochBear { get; set; }

        [Output("TF2 BoS Bull",   LineColor = "Transparent", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries Tf2BosBull { get; set; }
        [Output("TF2 BoS Bear",   LineColor = "Transparent", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries Tf2BosBear { get; set; }
        [Output("TF2 CHoCH Bull", LineColor = "Transparent", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries Tf2ChochBull { get; set; }
        [Output("TF2 CHoCH Bear", LineColor = "Transparent", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries Tf2ChochBear { get; set; }

        [Output("TF3 BoS Bull",   LineColor = "Transparent", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries Tf3BosBull { get; set; }
        [Output("TF3 BoS Bear",   LineColor = "Transparent", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries Tf3BosBear { get; set; }
        [Output("TF3 CHoCH Bull", LineColor = "Transparent", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries Tf3ChochBull { get; set; }
        [Output("TF3 CHoCH Bear", LineColor = "Transparent", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries Tf3ChochBear { get; set; }

        [Output("TF4 BoS Bull",   LineColor = "Transparent", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries Tf4BosBull { get; set; }
        [Output("TF4 BoS Bear",   LineColor = "Transparent", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries Tf4BosBear { get; set; }
        [Output("TF4 CHoCH Bull", LineColor = "Transparent", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries Tf4ChochBull { get; set; }
        [Output("TF4 CHoCH Bear", LineColor = "Transparent", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries Tf4ChochBear { get; set; }

        private readonly List<TfState> _states = new List<TfState>();

        protected override void Initialize()
        {
            _states.Clear();
            AddTfState("TF1", Timeframe1, PivotStrength1, IsTf1LowerTf, ShowChoch1, ShowBos1, ChochBull1, ChochBear1, BosBull1, BosBear1, BosLineStyle1, ChochLineStyle1);
            AddTfState("TF2", Timeframe2, PivotStrength2, IsTf2LowerTf, ShowChoch2, ShowBos2, ChochBull2, ChochBear2, BosBull2, BosBear2, BosLineStyle2, ChochLineStyle2);
            AddTfState("TF3", Timeframe3, PivotStrength3, IsTf3LowerTf, ShowChoch3, ShowBos3, ChochBull3, ChochBear3, BosBull3, BosBear3, BosLineStyle3, ChochLineStyle3);
            AddTfState("TF4", Timeframe4, PivotStrength4, IsTf4LowerTf, ShowChoch4, ShowBos4, ChochBull4, ChochBear4, BosBull4, BosBear4, BosLineStyle4, ChochLineStyle4);

            ValidateTimeframeFlags();
        }

        public override void Calculate(int index)
        {
            var chartTime = Bars.OpenTimes[index];
            var chartPoints = new List<ChartPoint>(_states.Count);

            var panelYValues = new[] { 1.00, 0.66, 0.33, 0.00 };
            for (var i = 0; i < _states.Count; i++)
            {
                var panelY = panelYValues[i];
                var point = BuildChartPoint(_states[i], index, chartTime, panelY);
                chartPoints.Add(point);

                DrawPanelStripe(point.State, index, panelY);

                if (index == Bars.Count - 1)
                    DrawTfLabel(point.State, panelY);
            }

            DrawTrendChanges(index, chartPoints);

            SetSignalOutputs(index, chartPoints, 0, Tf1BosBull, Tf1BosBear, Tf1ChochBull, Tf1ChochBear);
            SetSignalOutputs(index, chartPoints, 1, Tf2BosBull, Tf2BosBear, Tf2ChochBull, Tf2ChochBear);
            SetSignalOutputs(index, chartPoints, 2, Tf3BosBull, Tf3BosBear, Tf3ChochBull, Tf3ChochBear);
            SetSignalOutputs(index, chartPoints, 3, Tf4BosBull, Tf4BosBear, Tf4ChochBull, Tf4ChochBear);
        }

        private static void SetSignalOutputs(
            int index, IList<ChartPoint> points, int tfIndex,
            IndicatorDataSeries bosBull, IndicatorDataSeries bosBear,
            IndicatorDataSeries chochBull, IndicatorDataSeries chochBear)
        {
            if (tfIndex >= points.Count)
            {
                bosBull[index] = bosBear[index] = chochBull[index] = chochBear[index] = 0.0;
                return;
            }
            var p = points[tfIndex];
            bosBull[index]   = p.BosEdge     &&  p.Trend ? 1.0 : 0.0;
            bosBear[index]   = p.BosEdge     && !p.Trend ? 1.0 : 0.0;
            chochBull[index] = p.TrendChanged &&  p.Trend ? 1.0 : 0.0;
            chochBear[index] = p.TrendChanged && !p.Trend ? 1.0 : 0.0;
        }

        private void AddTfState(
            string key, TfOption tfOption, int pivotLen, bool isLowerTf,
            bool showChoch, bool showBos,
            Color chochBull, Color chochBear, Color bosBull, Color bosBear,
            LineStyleOption bosLineStyleOpt, LineStyleOption chochLineStyleOpt)
        {
            var tfInput = TfOptionToString(tfOption);
            var tf = ParseTimeFrame(tfInput);
            var tfBars = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf);
            _states.Add(new TfState
            {
                Key = key,
                TfInput = tfInput,
                TimeFrame = tf,
                TfBars = tfBars,
                PivotLen = Math.Max(1, pivotLen),
                IsLowerTf = isLowerTf,
                ShowChoch = showChoch,
                ShowBos = showBos,
                ChochBull = chochBull,
                ChochBear = chochBear,
                BosBull = bosBull,
                BosBear = bosBear,
                BosLineStyle = ToLineStyle(bosLineStyleOpt),
                ChochLineStyle = ToLineStyle(chochLineStyleOpt)
            });
        }

        private ChartPoint BuildChartPoint(TfState s, int chartBarIndex, DateTime chartTime, double panelY)
        {
            var tfBarIndex = ResolveTfBarForChartBar(s, chartTime);
            if (tfBarIndex < 0)
            {
                return new ChartPoint
                {
                    State = s,
                    Trend = s.PrevTrendAtChart,
                    Bos = false,
                    TrendChanged = false,
                    BosEdge = false,
                    PivotHighTime = DateTime.MinValue,
                    PivotLowTime = DateTime.MinValue,
                    PrevPivotHigh = double.NaN,
                    PrevPivotLow = double.NaN,
                    BosBarsSince = int.MaxValue,
                    TfLabel = TimeframeLabel(s.TfInput),
                    PanelY = panelY
                };
            }

            for (var i = s.LastProcessedTfBar + 1; i <= tfBarIndex; i++)
                ProcessTfCalcBar(s, i);
            s.LastProcessedTfBar = Math.Max(s.LastProcessedTfBar, tfBarIndex);

            CalcPoint calc;
            if (!s.CalcByTfBar.TryGetValue(tfBarIndex, out calc))
                calc = new CalcPoint { Trend = s.CurrentTrend, Bos = false, PivotHighTime = s.PivotHighTime, PivotLowTime = s.PivotLowTime, PrevPivotHigh = s.LastPivotHigh, PrevPivotLow = s.LastPivotLow };

            var trendChanged = calc.Trend != s.PrevTrendAtChart;
            var bosEdge = calc.Bos && !s.PrevBosAtChart;

            if (calc.Bos)
                s.CurrentColor = calc.Trend ? s.BosBull : s.BosBear;
            else if (trendChanged)
                s.CurrentColor = calc.Trend ? s.ChochBull : s.ChochBear;

            var prevBarsSince = int.MaxValue;
            if (chartBarIndex > 0 && s.BosBarssinceByChartBar.TryGetValue(chartBarIndex - 1, out var prev))
                prevBarsSince = prev;
            var barsSince = calc.Bos ? 0 : (prevBarsSince == int.MaxValue ? int.MaxValue : prevBarsSince + 1);
            s.BosBarssinceByChartBar[chartBarIndex] = barsSince;

            s.PrevTrendAtChart = calc.Trend;
            s.PrevBosAtChart = calc.Bos;
            s.LastMappedTfBarAtChart = tfBarIndex;

            return new ChartPoint
            {
                State = s,
                Trend = calc.Trend,
                Bos = calc.Bos,
                TrendChanged = trendChanged,
                BosEdge = bosEdge,
                PivotHighTime = calc.PivotHighTime,
                PivotLowTime = calc.PivotLowTime,
                PrevPivotHigh = calc.PrevPivotHigh,
                PrevPivotLow = calc.PrevPivotLow,
                BosBarsSince = barsSince,
                TfLabel = TimeframeLabel(s.TfInput),
                PanelY = panelY
            };
        }

        private void DrawTrendChanges(int chartBarIndex, IList<ChartPoint> points)
        {
            var trendChg = new[] { false, false, false, false };
            var spacing = new[] { string.Empty, string.Empty, string.Empty };

            for (var i = 0; i < points.Count; i++)
                trendChg[i] = points[i].TrendChanged || points[i].BosBarsSince < 5;

            var count = 0;
            for (var i = 0; i < trendChg.Length; i++)
            {
                if (!trendChg[i])
                    continue;

                count++;
                if (i < spacing.Length)
                {
                    for (var y = 0; y < count; y++)
                        spacing[i] += " \n \n ";
                }
            }

            for (var i = 0; i < points.Count; i++)
            {
                var point = points[i];
                var space = i == 0 ? string.Empty : spacing[i - 1];

                if (point.State.ShowChoch && point.TrendChanged)
                    DrawTrendChangeLabelAndLine(chartBarIndex, point, "CHoCH\n" + point.TfLabel, space, point.State.CurrentColor, point.State.ChochLineStyle);

                if (point.State.ShowBos && point.BosEdge)
                    DrawTrendChangeLabelAndLine(chartBarIndex, point, "BoS\n" + point.TfLabel, space, point.State.CurrentColor, point.State.BosLineStyle);

                EmitChochAlerts(point, chartBarIndex);
            }
        }

        private void EmitChochAlerts(ChartPoint point, int chartBarIndex)
        {
            if (!EnableChochAlerts || !point.TrendChanged)
                return;

            if (point.Trend)
            {
                if (point.State.LastBullAlertChartBar == chartBarIndex)
                    return;
                point.State.LastBullAlertChartBar = chartBarIndex;
                Print("Bullish CHoCH on {0}", point.TfLabel);
                Chart.DrawStaticText("ms_alert_" + point.State.Key, "Bullish CHoCH on " + point.TfLabel, VerticalAlignment.Top, HorizontalAlignment.Right, point.State.ChochBull);
            }
            else
            {
                if (point.State.LastBearAlertChartBar == chartBarIndex)
                    return;
                point.State.LastBearAlertChartBar = chartBarIndex;
                Print("Bearish CHoCH on {0}", point.TfLabel);
                Chart.DrawStaticText("ms_alert_" + point.State.Key, "Bearish CHoCH on " + point.TfLabel, VerticalAlignment.Top, HorizontalAlignment.Right, point.State.ChochBear);
            }
        }

        private void DrawTrendChangeLabelAndLine(int chartBarIndex, ChartPoint point, string labelText, string spacing, Color color, LineStyle lineStyle)
        {
            var endTime = Bars.OpenTimes[chartBarIndex];

            if (point.Trend)
            {
                if (point.PivotHighTime == DateTime.MinValue || double.IsNaN(point.PrevPivotHigh))
                    return;

                var lineId = $"ms_{point.State.Key}_{chartBarIndex}_{labelText.GetHashCode()}_B_line";
                Chart.DrawTrendLine(lineId, point.PivotHighTime, point.PrevPivotHigh, endTime, point.PrevPivotHigh, color, 1, lineStyle);

                Chart.DrawText($"ms_{point.State.Key}_{chartBarIndex}_{labelText.GetHashCode()}_B_txt", labelText + spacing, endTime, point.PrevPivotHigh, color);
            }
            else
            {
                if (point.PivotLowTime == DateTime.MinValue || double.IsNaN(point.PrevPivotLow))
                    return;

                var lineId = $"ms_{point.State.Key}_{chartBarIndex}_{labelText.GetHashCode()}_S_line";
                Chart.DrawTrendLine(lineId, point.PivotLowTime, point.PrevPivotLow, endTime, point.PrevPivotLow, color, 1, lineStyle);

                Chart.DrawText($"ms_{point.State.Key}_{chartBarIndex}_{labelText.GetHashCode()}_S_txt", spacing + labelText, endTime, point.PrevPivotLow, color);
            }
        }

        private void ProcessTfCalcBar(TfState s, int tfBarIndex)
        {
            var bars = s.TfBars;
            var prevLastPivotHigh = s.LastPivotHigh;
            var prevLastPivotLow = s.LastPivotLow;

            if (tfBarIndex >= s.PivotLen * 2)
            {
                var pivotIdx = tfBarIndex - s.PivotLen;

                if (IsPivotHigh(bars, pivotIdx, s.PivotLen))
                {
                    var pivotPrice = bars.HighPrices[pivotIdx];
                    s.LastPivotHigh = s.CurrentTrend
                        ? (double.IsNaN(s.LastPivotHigh) ? pivotPrice : Math.Max(pivotPrice, s.LastPivotHigh))
                        : pivotPrice;

                    if (s.LastPivotHigh != prevLastPivotHigh)
                        s.PivotHighTime = bars.OpenTimes[pivotIdx];
                }

                if (IsPivotLow(bars, pivotIdx, s.PivotLen))
                {
                    var pivotPrice = bars.LowPrices[pivotIdx];
                    s.LastPivotLow = !s.CurrentTrend
                        ? (double.IsNaN(s.LastPivotLow) ? pivotPrice : Math.Min(pivotPrice, s.LastPivotLow))
                        : pivotPrice;

                    if (s.LastPivotLow != prevLastPivotLow)
                        s.PivotLowTime = bars.OpenTimes[pivotIdx];
                }
            }

            var breakOfStructure = false;
            var close = bars.ClosePrices[tfBarIndex];
            var prevClose = tfBarIndex > 0 ? bars.ClosePrices[tfBarIndex - 1] : close;

            if (!double.IsNaN(s.LastPivotHigh) && !double.IsNaN(prevLastPivotHigh))
            {
                if (prevClose <= prevLastPivotHigh && close > s.LastPivotHigh)
                {
                    breakOfStructure = s.CurrentTrend && s.LastPivotHigh != s.LastBrokenHigh;
                    s.CurrentTrend = true;
                    s.LastBrokenHigh = s.LastPivotHigh;
                    s.LastBrokenLow = double.NaN;
                }
            }

            if (!double.IsNaN(s.LastPivotLow) && !double.IsNaN(prevLastPivotLow))
            {
                if (prevClose >= prevLastPivotLow && close < s.LastPivotLow)
                {
                    breakOfStructure = !s.CurrentTrend && s.LastPivotLow != s.LastBrokenLow;
                    s.CurrentTrend = false;
                    s.LastBrokenLow = s.LastPivotLow;
                    s.LastBrokenHigh = double.NaN;
                }
            }

            s.CalcByTfBar[tfBarIndex] = new CalcPoint
            {
                Trend = s.CurrentTrend,
                Bos = breakOfStructure,
                PivotHighTime = s.PivotHighTime,
                PivotLowTime = s.PivotLowTime,
                PrevPivotHigh = prevLastPivotHigh,
                PrevPivotLow = prevLastPivotLow
            };
        }

        private int ResolveTfBarForChartBar(TfState s, DateTime chartTime)
        {
            if (!s.IsLowerTf)
                return FindBarIndexAtOrBefore(s.TfBars, chartTime);

            var chartBarIndex = FindBarIndexAtOrBefore(Bars, chartTime);
            if (chartBarIndex < 0)
                return -1;

            var chartOpen = Bars.OpenTimes[chartBarIndex];
            DateTime chartNextOpen;
            if (chartBarIndex + 1 < Bars.Count)
            {
                chartNextOpen = Bars.OpenTimes[chartBarIndex + 1];
            }
            else
            {
                var chartMinutes = TimeFrameToMinutes(Bars.TimeFrame);
                chartNextOpen = chartOpen.AddMinutes(chartMinutes > 0 ? chartMinutes : 1);
            }

            var firstInWindow = FindBarIndexAtOrAfter(s.TfBars, chartOpen);
            if (firstInWindow < 0)
                return FindBarIndexAtOrBefore(s.TfBars, chartTime);

            if (s.TfBars.OpenTimes[firstInWindow] >= chartNextOpen)
                return FindBarIndexAtOrBefore(s.TfBars, chartTime);

            return firstInWindow;
        }

        private void DrawPanelStripe(TfState s, int chartBarIndex, double y)
        {
            var startTime = Bars.OpenTimes[Math.Max(0, chartBarIndex - 1)];
            var endTime = Bars.OpenTimes[chartBarIndex];
            var line = IndicatorArea.DrawTrendLine($"ms_panel_{s.Key}_{chartBarIndex}", startTime, y, endTime, y, s.CurrentColor, 10, LineStyle.Solid);
            line.IsInteractive = false;
        }

        private void DrawTfLabel(TfState s, double y)
        {
            var text = IndicatorArea.DrawText($"ms_tf_{s.Key}", TimeframeLabel(s.TfInput), Bars.Count - 1, y, Color.Gray);
            text.HorizontalAlignment = HorizontalAlignment.Right;
            text.VerticalAlignment = VerticalAlignment.Center;
        }

        private void ValidateTimeframeFlags()
        {
            var chartMinutes = TimeFrameToMinutes(Bars.TimeFrame);
            var checkTf1 = false;
            var checkTf2 = false;

            for (var i = 0; i < _states.Count; i++)
            {
                var state = _states[i];
                var selectedMinutes = TimeFrameToMinutes(state.TimeFrame);

                if (!checkTf1)
                    checkTf1 = !state.IsLowerTf && chartMinutes > selectedMinutes;
                if (!checkTf2)
                    checkTf2 = state.IsLowerTf && chartMinutes < selectedMinutes;
            }

            if (checkTf1)
                Chart.DrawStaticText("ms_tf_error_1", "Please check 'Lower than chart TF?' for one or more of your selected timeframe(s) for more accurate results", VerticalAlignment.Top, HorizontalAlignment.Center, Color.Red);
            if (checkTf2)
                Chart.DrawStaticText("ms_tf_error_2", "Please uncheck 'Lower than chart TF?' for one or more of your selected timeframe(s) for more accurate results", VerticalAlignment.Top, HorizontalAlignment.Left, Color.Red);
        }

        private static bool IsPivotHigh(Bars bars, int idx, int len)
        {
            var left = idx - len;
            var right = idx + len;
            if (left < 0 || right >= bars.Count)
                return false;

            var pivot = bars.HighPrices[idx];
            for (var i = left; i <= right; i++)
            {
                if (i == idx)
                    continue;
                if (bars.HighPrices[i] >= pivot)
                    return false;
            }

            return true;
        }

        private static bool IsPivotLow(Bars bars, int idx, int len)
        {
            var left = idx - len;
            var right = idx + len;
            if (left < 0 || right >= bars.Count)
                return false;

            var pivot = bars.LowPrices[idx];
            for (var i = left; i <= right; i++)
            {
                if (i == idx)
                    continue;
                if (bars.LowPrices[i] <= pivot)
                    return false;
            }

            return true;
        }

        private static int FindBarIndexAtOrBefore(Bars bars, DateTime time)
        {
            var lo = 0;
            var hi = bars.Count - 1;

            while (lo <= hi)
            {
                var mid = (lo + hi) / 2;
                var midTime = bars.OpenTimes[mid];

                if (midTime == time)
                    return mid;
                if (midTime < time)
                    lo = mid + 1;
                else
                    hi = mid - 1;
            }

            return hi;
        }

        private static int FindBarIndexAtOrAfter(Bars bars, DateTime time)
        {
            var lo = 0;
            var hi = bars.Count - 1;
            var answer = -1;

            while (lo <= hi)
            {
                var mid = (lo + hi) / 2;
                var midTime = bars.OpenTimes[mid];

                if (midTime >= time)
                {
                    answer = mid;
                    hi = mid - 1;
                }
                else
                {
                    lo = mid + 1;
                }
            }

            return answer;
        }

        private static DateTime MidTime(DateTime a, DateTime b)
        {
            if (b <= a)
                return a;
            var halfTicks = (b - a).Ticks / 2;
            return a.AddTicks(halfTicks);
        }

        // ── Converts TfOption enum → string used by ParseTimeFrame/Label ─────
        private static string TfOptionToString(TfOption opt)
        {
            switch (opt)
            {
                case TfOption.M1:  return "1";
                case TfOption.M3:  return "3";
                case TfOption.M15: return "15";
                case TfOption.M30: return "30";
                case TfOption.H1:  return "60";
                case TfOption.H4:  return "240";
                case TfOption.D1:  return "1D";
                default:           return "15";
            }
        }

        // ── Converts LineStyleOption enum → cAlgo LineStyle ──────────────────
        private static LineStyle ToLineStyle(LineStyleOption opt)
        {
            return opt == LineStyleOption.Solid ? LineStyle.Solid : LineStyle.DotsRare;
        }

        private static TimeFrame ParseTimeFrame(string text)
        {
            switch ((text ?? string.Empty).Trim().ToUpperInvariant())
            {
                case "1":           return TimeFrame.Minute;
                case "2":           return TimeFrame.Minute2;
                case "3":           return TimeFrame.Minute3;
                case "4":           return TimeFrame.Minute4;
                case "5":           return TimeFrame.Minute5;
                case "10":          return TimeFrame.Minute10;
                case "15":          return TimeFrame.Minute15;
                case "30":          return TimeFrame.Minute30;
                case "45":          return TimeFrame.Minute45;
                case "60":
                case "1H":          return TimeFrame.Hour;
                case "120":
                case "2H":          return TimeFrame.Hour2;
                case "240":
                case "4H":          return TimeFrame.Hour4;
                case "480":
                case "8H":          return TimeFrame.Hour8;
                case "720":
                case "12H":         return TimeFrame.Hour12;
                case "D":
                case "1D":          return TimeFrame.Daily;
                case "W":
                case "1W":          return TimeFrame.Weekly;
                case "M":
                case "1M":          return TimeFrame.Monthly;
                default:            return TimeFrame.Minute15;
            }
        }

        private static int TimeFrameToMinutes(TimeFrame tf)
        {
            if (tf == TimeFrame.Minute)   return 1;
            if (tf == TimeFrame.Minute2)  return 2;
            if (tf == TimeFrame.Minute3)  return 3;
            if (tf == TimeFrame.Minute4)  return 4;
            if (tf == TimeFrame.Minute5)  return 5;
            if (tf == TimeFrame.Minute10) return 10;
            if (tf == TimeFrame.Minute15) return 15;
            if (tf == TimeFrame.Minute30) return 30;
            if (tf == TimeFrame.Minute45) return 45;
            if (tf == TimeFrame.Hour)     return 60;
            if (tf == TimeFrame.Hour2)    return 120;
            if (tf == TimeFrame.Hour4)    return 240;
            if (tf == TimeFrame.Hour8)    return 480;
            if (tf == TimeFrame.Hour12)   return 720;
            if (tf == TimeFrame.Daily)    return 1440;
            if (tf == TimeFrame.Weekly)   return 10080;
            if (tf == TimeFrame.Monthly)  return 43200;
            return 0;
        }

        private static string TimeframeLabel(string input)
        {
            var value = (input ?? string.Empty).Trim().ToUpperInvariant();
            switch (value)
            {
                case "60":  return "1H";
                case "120": return "2H";
                case "240": return "4H";
                case "480": return "8H";
                case "720": return "12H";
                case "D":
                case "1D":  return "1D";
                case "W":
                case "1W":  return "1W";
                case "M":
                case "1M":  return "1M";
            }

            var numeric = true;
            for (var i = 0; i < value.Length; i++)
            {
                if (!char.IsDigit(value[i]))
                {
                    numeric = false;
                    break;
                }
            }

            return numeric ? value + "m" : value;
        }
    }
}
