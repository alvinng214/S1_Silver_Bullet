using System;
using cAlgo.API;
using cAlgo.API.Indicators;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class FVGInstantaneousMitigationSignalsLuxAlgo : Indicator
    {
        public enum TrailingStopResetMode
        {
            EverySignals,
            InverseSignals
        }

        [Parameter("FVG Width Filter", DefaultValue = 0.0, MinValue = 0.0, Step = 0.1)]
        public double FilterWidth { get; set; }

        [Parameter("TP Area", DefaultValue = false, Group = "TP/SL")]
        public bool ShowTp { get; set; }

        [Parameter("TP Multiplier", DefaultValue = 4.0, MinValue = 0.0, Group = "TP/SL")]
        public double TpMult { get; set; }

        [Parameter("TP Color", DefaultValue = "#CC5B9CF6", Group = "TP/SL")]
        public Color TpColor { get; set; }

        [Parameter("SL Area", DefaultValue = false, Group = "TP/SL")]
        public bool ShowSl { get; set; }

        [Parameter("SL Multiplier", DefaultValue = 2.0, MinValue = 0.0, Group = "TP/SL")]
        public double SlMult { get; set; }

        [Parameter("SL Color", DefaultValue = "#CC808080", Group = "TP/SL")]
        public Color SlColor { get; set; }

        [Parameter("Reset Trailing Stop", DefaultValue = TrailingStopResetMode.EverySignals, Group = "Trailing Stop")]
        public TrailingStopResetMode TsReset { get; set; }

        [Parameter("TS Multiplier", DefaultValue = 3.0, Group = "Trailing Stop")]
        public double TsMult { get; set; }

        [Parameter("Bullish IMFVG", DefaultValue = true, Group = "Style")]
        public bool ShowBull { get; set; }

        [Parameter("Bull Color", DefaultValue = "Teal", Group = "Style")]
        public Color BullColor { get; set; }

        [Parameter("Bull Average", DefaultValue = true, Group = "Style")]
        public bool BullAvg { get; set; }

        [Parameter("Bearish IMFVG", DefaultValue = true, Group = "Style")]
        public bool ShowBear { get; set; }

        [Parameter("Bear Color", DefaultValue = "#F23645", Group = "Style")]
        public Color BearColor { get; set; }

        [Parameter("Bear Average", DefaultValue = true, Group = "Style")]
        public bool BearAvg { get; set; }

        [Output("Trailing Stop", Thickness = 2, LineColor = "#5B9CF6")]
        public IndicatorDataSeries TrailingStopPlot { get; set; }

        private AverageTrueRange _atr;

        private sealed class AreaState
        {
            public ChartRectangle TpArea;
            public ChartRectangle SlArea;
            public bool Reached;
        }

        private sealed class TrailState
        {
            public double? Ts;
            public bool Reached;
        }

        private AreaState _bullTpsl;
        private AreaState _bearTpsl;
        private TrailState _trail;

        private ChartTrendLine _bullLine;
        private ChartTrendLine _bearLine;
        private bool? _bullLevelReached;
        private bool? _bearLevelReached;

        private int _os;
        private int _prevOs;

        protected override void Initialize()
        {
            _atr = Indicators.AverageTrueRange(200, MovingAverageType.WilderSmoothing);
            _bullTpsl = new AreaState();
            _bearTpsl = new AreaState();
            _trail = new TrailState();
            _os = 0;
            _prevOs = 0;
        }

        public override void Calculate(int index)
        {
            if (index < 3)
            {
                TrailingStopPlot[index] = double.NaN;
                return;
            }

            var atr = Nz(_atr.Result[index]);
            bool bull = Bars.LowPrices[index - 3] > Bars.HighPrices[index - 1]
                        && Bars.ClosePrices[index - 2] < Bars.LowPrices[index - 3]
                        && Bars.ClosePrices[index] > Bars.LowPrices[index - 3]
                        && Filter(Bars.LowPrices[index - 3], Bars.HighPrices[index - 1], atr)
                        && ShowBull;

            bool bear = Bars.LowPrices[index - 1] > Bars.HighPrices[index - 3]
                        && Bars.ClosePrices[index - 2] > Bars.HighPrices[index - 3]
                        && Bars.ClosePrices[index] < Bars.HighPrices[index - 3]
                        && Filter(Bars.LowPrices[index - 1], Bars.HighPrices[index - 3], atr)
                        && ShowBear;

            if (bull)
            {
                DrawImbalanceBox($"bull_imb_{index}", index - 3, index, Bars.LowPrices[index - 3], Bars.HighPrices[index - 1], Color.FromArgb(128, BullColor.R, BullColor.G, BullColor.B));
                var avg = (Bars.LowPrices[index - 3] + Bars.HighPrices[index - 1]) / 2.0;

                if (BullAvg)
                {
                    _bullLine = Chart.DrawTrendLine($"bull_avg_{index}", index, avg, index, avg, BullColor, 1, LineStyle.Dots);
                }

                Chart.DrawIcon($"bull_label_{index}", ChartIconType.UpTriangle, index, Bars.LowPrices[index], BullColor);

                _os = 1;
                _bullLevelReached = false;
            }

            if (bear)
            {
                DrawImbalanceBox($"bear_imb_{index}", index - 3, index, Bars.LowPrices[index - 1], Bars.HighPrices[index - 3], Color.FromArgb(128, BearColor.R, BearColor.G, BearColor.B));
                var avg = (Bars.LowPrices[index - 1] + Bars.HighPrices[index - 3]) / 2.0;

                if (BearAvg)
                {
                    _bearLine = Chart.DrawTrendLine($"bear_avg_{index}", index, avg, index, avg, BearColor, 1, LineStyle.Dots);
                }

                Chart.DrawIcon($"bear_label_{index}", ChartIconType.DownTriangle, index, Bars.HighPrices[index], BearColor);

                _os = 0;
                _bearLevelReached = false;
            }

            if (_bullLevelReached == false && _bullLine != null)
            {
                _bullLine.Time2 = Bars.OpenTimes[index];
            }

            if (_bearLevelReached == false && _bearLine != null)
            {
                _bearLine.Time2 = Bars.OpenTimes[index];
            }

            if (_bullLine != null && Bars.ClosePrices[index] < _bullLine.Y2)
                _bullLevelReached = true;

            if (_bearLine != null && Bars.ClosePrices[index] > _bearLine.Y2)
                _bearLevelReached = true;

            var bullLevel = (Bars.LowPrices[index - 3] + Bars.HighPrices[index - 1]) / 2.0;
            var bearLevel = (Bars.LowPrices[index - 1] + Bars.HighPrices[index - 3]) / 2.0;

            bool bullReached = Tpsl(_bullTpsl, bull, bear, bullLevel, true, index, atr);
            bool bearReached = Tpsl(_bearTpsl, bear, bull, bearLevel, false, index, atr);

            bool tsReset = TsReset == TrailingStopResetMode.EverySignals ? (bull || bear) : (_os != _prevOs);
            TrailingStop(tsReset, _os, TsMult, index, atr);

            Color barColor;
            if (TryGetBarColor(_trail.Reached, _os, bullReached, bearReached, out barColor))
                Chart.SetBarColor(index, barColor);

            if (_trail.Reached || bull || bear || !_trail.Ts.HasValue)
            {
                TrailingStopPlot[index] = double.NaN;
            }
            else
            {
                TrailingStopPlot[index] = _trail.Ts.Value;
            }

            _prevOs = _os;
        }

        private void DrawImbalanceBox(string id, int leftIndex, int rightIndex, double top, double bottom, Color color)
        {
            var rect = Chart.DrawRectangle(id, leftIndex, top, rightIndex, bottom, color);
            rect.IsFilled = true;
            rect.Color = color;
        }

        private bool Tpsl(AreaState state, bool condition, bool oppositeCondition, double level, bool isLong, int index, double atr)
        {
            if (condition)
            {
                if (isLong)
                {
                    if (ShowTp)
                    {
                        state.TpArea = Chart.DrawRectangle($"tp_long_{index}", index, level + atr * TpMult, index, level, TpColor);
                        state.TpArea.IsFilled = true;
                    }
                    if (ShowSl)
                    {
                        state.SlArea = Chart.DrawRectangle($"sl_long_{index}", index, level, index, level - atr * SlMult, SlColor);
                        state.SlArea.IsFilled = true;
                    }
                }
                else
                {
                    if (ShowTp)
                    {
                        state.TpArea = Chart.DrawRectangle($"tp_short_{index}", index, level, index, level - atr * TpMult, TpColor);
                        state.TpArea.IsFilled = true;
                    }
                    if (ShowSl)
                    {
                        state.SlArea = Chart.DrawRectangle($"sl_short_{index}", index, level + atr * SlMult, index, level, SlColor);
                        state.SlArea.IsFilled = true;
                    }
                }

                state.Reached = false;
            }
            else if (oppositeCondition)
            {
                state.Reached = true;
            }

            if (!state.Reached)
            {
                if (state.TpArea != null)
                    state.TpArea.Time2 = Bars.OpenTimes[index];

                if (state.SlArea != null)
                    state.SlArea.Time2 = Bars.OpenTimes[index];

                if (isLong)
                {
                    bool tpHit = state.TpArea != null && Bars.HighPrices[index] > Math.Max(state.TpArea.Y1, state.TpArea.Y2);
                    bool slHit = state.SlArea != null && Bars.LowPrices[index] < Math.Min(state.SlArea.Y1, state.SlArea.Y2);
                    if (tpHit || slHit)
                        state.Reached = true;
                }
                else
                {
                    bool tpHit = state.TpArea != null && Bars.LowPrices[index] < Math.Min(state.TpArea.Y1, state.TpArea.Y2);
                    bool slHit = state.SlArea != null && Bars.HighPrices[index] > Math.Max(state.SlArea.Y1, state.SlArea.Y2);
                    if (tpHit || slHit)
                        state.Reached = true;
                }
            }

            return state.Reached;
        }

        private void TrailingStop(bool trigger, int trend, double mult, int index, double atr)
        {
            var close = Bars.ClosePrices[index];

            if (trigger)
            {
                _trail.Ts = trend == 1 ? close - atr * mult : close + atr * mult;
                _trail.Reached = false;
                return;
            }

            if (!_trail.Ts.HasValue)
                return;

            var ts = _trail.Ts.Value;

            if (trend == 1)
            {
                if (close - ts > atr * mult)
                    ts = close - atr * mult;

                if (close < ts)
                    _trail.Reached = true;
            }
            else
            {
                if (ts - close > atr * mult)
                    ts = close + atr * mult;

                if (close > ts)
                    _trail.Reached = true;
            }

            _trail.Ts = ts;
        }

        private bool Filter(double a, double b, double atr)
        {
            return (a - b) > atr * FilterWidth;
        }

        private static double Nz(double value)
        {
            return double.IsNaN(value) || double.IsInfinity(value) ? 0.0 : value;
        }

        private bool TryGetBarColor(bool tsReached, int os, bool bullReached, bool bearReached, out Color color)
        {
            if (tsReached)
            {
                color = default(Color);
                return false;
            }

            if (os == 1 && !bullReached)
            {
                color = BullColor;
                return true;
            }

            if (os == 0 && !bearReached)
            {
                color = BearColor;
                return true;
            }

            color = default(Color);
            return false;
        }
    }
}
