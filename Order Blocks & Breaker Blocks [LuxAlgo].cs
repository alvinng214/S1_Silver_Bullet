using System;
using System.Collections.Generic;
using cAlgo.API;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class OrderBlocksBreakerBlocksLuxAlgo : Indicator
    {
        [Parameter("Swing Lookback", DefaultValue = 10, MinValue = 3)]
        public int Length { get; set; }

        [Parameter("Show Last Bullish OB", DefaultValue = 3, MinValue = 0)]
        public int ShowBull { get; set; }

        [Parameter("Show Last Bearish OB", DefaultValue = 3, MinValue = 0)]
        public int ShowBear { get; set; }

        [Parameter("Use Candle Body", DefaultValue = false)]
        public bool UseBody { get; set; }

        [Parameter("Bullish OB", Group = "Style", DefaultValue = "#CC2157F3")]
        public Color BullCss { get; set; }

        [Parameter("Bullish Break", Group = "Style", DefaultValue = "#CCFF1100")]
        public Color BullBreakCss { get; set; }

        [Parameter("Bearish OB", Group = "Style", DefaultValue = "#CCFF5D00")]
        public Color BearCss { get; set; }

        [Parameter("Bearish Break", Group = "Style", DefaultValue = "#CC0CB51A")]
        public Color BearBreakCss { get; set; }

        [Parameter("Show Historical Polarity Changes", DefaultValue = false)]
        public bool ShowLabels { get; set; }

        private sealed class Swing
        {
            public double Y = double.NaN;
            public int X = -1;
            public bool Crossed;
        }

        private sealed class Ob
        {
            public double Top;
            public double Bottom;
            public int LocIndex;
            public bool Breaker;
            public int BreakIndex = -1;
        }

        private readonly List<Ob> _bullishObs = new List<Ob>();
        private readonly List<Ob> _bearishObs = new List<Ob>();
        private readonly Swing _top = new Swing();
        private readonly Swing _btm = new Swing();

        private int _os;
        private int _lastProcessed = -1;
        private int _prevBullBreakConf;
        private int _prevBearBreakConf;

        private const string Prefix = "OBBB_LUX_";

        public override void Calculate(int index)
        {
            if (index < Length + 2)
                return;

            if (index <= _lastProcessed)
            {
                if (index == Bars.Count - 1)
                    DrawVisible(index);
                return;
            }

            ProcessBar(index);
            _lastProcessed = index;

            if (index == Bars.Count - 1)
                DrawVisible(index);
        }

        private void ProcessBar(int index)
        {
            DetectSwings(index);

            var bullBreakConf = 0;
            var bearBreakConf = 0;

            if (!double.IsNaN(_top.Y) && !_top.Crossed && Bars.ClosePrices[index] > _top.Y)
            {
                _top.Crossed = true;

                var minima = GetMaxValue(index - 1);
                var maxima = GetMinValue(index - 1);
                var loc = index - 1;

                var distance = index - _top.X;
                for (var offset = 1; offset <= distance - 1; offset++)
                {
                    var i = index - offset;
                    if (i < 0)
                        break;

                    var currMin = GetMinValue(i);
                    if (currMin <= minima)
                    {
                        minima = currMin;
                        maxima = GetMaxValue(i);
                        loc = i;
                    }
                }

                _bullishObs.Insert(0, new Ob
                {
                    Top = maxima,
                    Bottom = minima,
                    LocIndex = loc
                });
            }

            for (var i = _bullishObs.Count - 1; i >= 0; i--)
            {
                var element = _bullishObs[i];
                if (!element.Breaker)
                {
                    if (Math.Min(Bars.ClosePrices[index], Bars.OpenPrices[index]) < element.Bottom)
                    {
                        element.Breaker = true;
                        element.BreakIndex = index;
                    }
                }
                else
                {
                    if (Bars.ClosePrices[index] > element.Top)
                    {
                        _bullishObs.RemoveAt(i);
                    }
                    else if (i < ShowBull && !double.IsNaN(_top.Y) && _top.Y < element.Top && _top.Y > element.Bottom)
                    {
                        bullBreakConf = 1;
                    }
                }
            }

            if (ShowLabels && bullBreakConf > _prevBullBreakConf && _top.X >= 0)
            {
                var label = Chart.DrawText(Prefix + "lbl_bear_" + index, "▼", _top.X, _top.Y, ToOpaque(BearCss));
                label.FontSize = 10;
            }

            if (!double.IsNaN(_btm.Y) && !_btm.Crossed && Bars.ClosePrices[index] < _btm.Y)
            {
                _btm.Crossed = true;

                var minima = GetMinValue(index - 1);
                var maxima = GetMaxValue(index - 1);
                var loc = index - 1;

                var distance = index - _btm.X;
                for (var offset = 1; offset <= distance - 1; offset++)
                {
                    var i = index - offset;
                    if (i < 0)
                        break;

                    var currMax = GetMaxValue(i);
                    if (currMax >= maxima)
                    {
                        maxima = currMax;
                        minima = GetMinValue(i);
                        loc = i;
                    }
                }

                _bearishObs.Insert(0, new Ob
                {
                    Top = maxima,
                    Bottom = minima,
                    LocIndex = loc
                });
            }

            for (var i = _bearishObs.Count - 1; i >= 0; i--)
            {
                var element = _bearishObs[i];
                if (!element.Breaker)
                {
                    if (Math.Max(Bars.ClosePrices[index], Bars.OpenPrices[index]) > element.Top)
                    {
                        element.Breaker = true;
                        element.BreakIndex = index;
                    }
                }
                else
                {
                    if (Bars.ClosePrices[index] < element.Bottom)
                    {
                        _bearishObs.RemoveAt(i);
                    }
                    else if (i < ShowBear && !double.IsNaN(_btm.Y) && _btm.Y > element.Bottom && _btm.Y < element.Top)
                    {
                        bearBreakConf = 1;
                    }
                }
            }

            if (ShowLabels && bearBreakConf > _prevBearBreakConf && _btm.X >= 0)
            {
                var label = Chart.DrawText(Prefix + "lbl_bull_" + index, "▲", _btm.X, _btm.Y, ToOpaque(BullCss));
                label.FontSize = 10;
            }

            _prevBullBreakConf = bullBreakConf;
            _prevBearBreakConf = bearBreakConf;
        }

        private void DetectSwings(int index)
        {
            var upper = Highest(index, Length);
            var lower = Lowest(index, Length);
            var pivot = index - Length;
            if (pivot < 0)
                return;

            var prevOs = _os;

            if (Bars.HighPrices[pivot] > upper)
                _os = 0;
            else if (Bars.LowPrices[pivot] < lower)
                _os = 1;

            if (_os == 0 && prevOs != 0)
            {
                _top.Y = Bars.HighPrices[pivot];
                _top.X = pivot;
                _top.Crossed = false;
            }

            if (_os == 1 && prevOs != 1)
            {
                _btm.Y = Bars.LowPrices[pivot];
                _btm.X = pivot;
                _btm.Crossed = false;
            }
        }

        private void DrawVisible(int index)
        {
            ClearDrawings();

            if (ShowBull > 0)
            {
                var maxCount = Math.Min(ShowBull, _bullishObs.Count);
                for (var i = 0; i < maxCount; i++)
                    DisplayOb(_bullishObs[i], i, true, index);
            }

            if (ShowBear > 0)
            {
                var maxCount = Math.Min(ShowBear, _bearishObs.Count);
                for (var i = 0; i < maxCount; i++)
                    DisplayOb(_bearishObs[i], i, false, index);
            }
        }

        private void DisplayOb(Ob id, int idx, bool bull, int current)
        {
            var css = bull ? BullCss : BearCss;
            var breakCss = bull ? BullBreakCss : BearBreakCss;
            var opaqueCss = ToOpaque(css);
            var opaqueBreakCss = ToOpaque(breakCss);
            var side = bull ? "bull" : "bear";

            if (id.Breaker && id.BreakIndex >= 0)
            {
                var pre = Prefix + side + "_pre_" + idx;
                var post = Prefix + side + "_post_" + idx;

                var preRect = Chart.DrawRectangle(pre + "_box", id.LocIndex, id.Top, id.BreakIndex, id.Bottom, opaqueCss, 1, LineStyle.Solid);
                preRect.IsFilled = true;
                preRect.Color = css;

                var postRect = Chart.DrawRectangle(post + "_box", id.BreakIndex, id.Top, current + 1, id.Bottom, opaqueBreakCss, 1, LineStyle.DotsRare);
                postRect.IsFilled = true;
                postRect.Color = breakCss;

                Chart.DrawTrendLine(pre + "_top", id.LocIndex, id.Top, id.BreakIndex, id.Top, opaqueCss, 1, LineStyle.Solid);
                Chart.DrawTrendLine(pre + "_btm", id.LocIndex, id.Bottom, id.BreakIndex, id.Bottom, opaqueCss, 1, LineStyle.Solid);
                Chart.DrawTrendLine(post + "_top", id.BreakIndex, id.Top, current + 1, id.Top, opaqueBreakCss, 1, LineStyle.DotsRare);
                Chart.DrawTrendLine(post + "_btm", id.BreakIndex, id.Bottom, current + 1, id.Bottom, opaqueBreakCss, 1, LineStyle.DotsRare);
            }
            else
            {
                var name = Prefix + side + "_act_" + idx;
                var rect = Chart.DrawRectangle(name + "_box", id.LocIndex, id.Top, current, id.Bottom, opaqueCss, 1, LineStyle.Solid);
                rect.IsFilled = true;
                rect.Color = css;

                Chart.DrawTrendLine(name + "_top", id.LocIndex, id.Top, current, id.Top, opaqueCss, 1, LineStyle.Solid);
                Chart.DrawTrendLine(name + "_btm", id.LocIndex, id.Bottom, current, id.Bottom, opaqueCss, 1, LineStyle.Solid);
            }
        }

        private void ClearDrawings()
        {
            var remove = new List<string>();
            foreach (var obj in Chart.Objects)
            {
                if (obj.Name.StartsWith(Prefix))
                    remove.Add(obj.Name);
            }

            foreach (var name in remove)
                Chart.RemoveObject(name);
        }

        private double GetMaxValue(int i)
        {
            return UseBody ? Math.Max(Bars.ClosePrices[i], Bars.OpenPrices[i]) : Bars.HighPrices[i];
        }

        private double GetMinValue(int i)
        {
            return UseBody ? Math.Min(Bars.ClosePrices[i], Bars.OpenPrices[i]) : Bars.LowPrices[i];
        }

        private double Highest(int index, int len)
        {
            var start = Math.Max(0, index - len + 1);
            var result = double.MinValue;
            for (var i = start; i <= index; i++)
                result = Math.Max(result, Bars.HighPrices[i]);
            return result;
        }

        private double Lowest(int index, int len)
        {
            var start = Math.Max(0, index - len + 1);
            var result = double.MaxValue;
            for (var i = start; i <= index; i++)
                result = Math.Min(result, Bars.LowPrices[i]);
            return result;
        }

        private static Color ToOpaque(Color c)
        {
            return Color.FromArgb(255, c.R, c.G, c.B);
        }
    }
}
