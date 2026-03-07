using System;
using System.Collections.Generic;
using cAlgo.API;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class MarketStructureBreakOrderBlock : Indicator
    {
        [Parameter("ZigZag Length", Group = "Settings", DefaultValue = 9, MinValue = 1)]
        public int ZigZagLength { get; set; }

        [Parameter("Show Zigzag", Group = "Settings", DefaultValue = true)]
        public bool ShowZigzag { get; set; }

        [Parameter("Fib Factor for breakout confirmation", Group = "Settings", DefaultValue = 0.33, MinValue = 0.0, MaxValue = 1.0, Step = 0.01)]
        public double FibFactor { get; set; }

        [Parameter("Show Last Bullish Blocks", Group = "Settings", DefaultValue = 3, MinValue = 0)]
        public int ShowBullishBlocks { get; set; }

        [Parameter("Show Last Bearish Blocks", Group = "Settings", DefaultValue = 3, MinValue = 0)]
        public int ShowBearishBlocks { get; set; }

        [Parameter("Show Last Bullish Breaker/Mitigation Blocks", Group = "Settings", DefaultValue = 3, MinValue = 0)]
        public int ShowBullishBreakerBlocks { get; set; }

        [Parameter("Show Last Bearish Breaker/Mitigation Blocks", Group = "Settings", DefaultValue = 3, MinValue = 0)]
        public int ShowBearishBreakerBlocks { get; set; }

        public enum LabelSizeOpt { Tiny, Small, Normal, Large, Huge }

        [Parameter("Text Size", Group = "Settings", DefaultValue = LabelSizeOpt.Tiny)]
        public LabelSizeOpt TextSize { get; set; }

        [Parameter("Delete Old/Broken Boxes", Group = "Settings", DefaultValue = true)]
        public bool DeleteBoxes { get; set; }

        [Parameter("Color", Group = "Bu-OB Display Settings", DefaultValue = "#4D00FF00")]
        public Color BuObColor { get; set; }

        [Parameter("Border Color", Group = "Bu-OB Display Settings", DefaultValue = "#00FF00")]
        public Color BuObBorderColor { get; set; }

        [Parameter("Text Color", Group = "Bu-OB Display Settings", DefaultValue = "#00FF00")]
        public Color BuObTextColor { get; set; }

        [Parameter("Color", Group = "Be-OB Display Settings", DefaultValue = "#4DFF0000")]
        public Color BeObColor { get; set; }

        [Parameter("Border Color", Group = "Be-OB Display Settings", DefaultValue = "#FF0000")]
        public Color BeObBorderColor { get; set; }

        [Parameter("Text Color", Group = "Be-OB Display Settings", DefaultValue = "#FF0000")]
        public Color BeObTextColor { get; set; }

        [Parameter("Color", Group = "Bu-BB & Bu-MB Display Settings", DefaultValue = "#4D00FF00")]
        public Color BuBbColor { get; set; }

        [Parameter("Border Color", Group = "Bu-BB & Bu-MB Display Settings", DefaultValue = "#00FF00")]
        public Color BuBbBorderColor { get; set; }

        [Parameter("Text Color", Group = "Bu-BB & Bu-MB Display Settings", DefaultValue = "#00FF00")]
        public Color BuBbTextColor { get; set; }

        [Parameter("Color", Group = "Be-BB & Be-MB Display Settings", DefaultValue = "#4DFF0000")]
        public Color BeBbColor { get; set; }

        [Parameter("Border Color", Group = "Be-BB & Be-MB Display Settings", DefaultValue = "#FF0000")]
        public Color BeBbBorderColor { get; set; }

        [Parameter("Text Color", Group = "Be-BB & Be-MB Display Settings", DefaultValue = "#FF0000")]
        public Color BeBbTextColor { get; set; }

        private sealed class Zone
        {
            public string Id;
            public string LabelId;
            public int Left;
            public int Right;
            public double Top;
            public double Bottom;
            public string Text;
            public Color Fill;
            public Color Border;
            public Color TextColor;
        }

        private readonly List<double> _highPoints = new List<double>();
        private readonly List<int> _highIndices = new List<int>();
        private readonly List<double> _lowPoints = new List<double>();
        private readonly List<int> _lowIndices = new List<int>();

        private readonly List<Zone> _buObBoxes = new List<Zone>();
        private readonly List<Zone> _beObBoxes = new List<Zone>();
        private readonly List<Zone> _buBbBoxes = new List<Zone>();
        private readonly List<Zone> _beBbBoxes = new List<Zone>();

        private readonly List<bool> _toUp = new List<bool>();
        private readonly List<bool> _toDown = new List<bool>();
        private readonly List<int> _trendSeries = new List<int>();
        private readonly List<int> _marketSeries = new List<int>();
        private readonly List<int> _l0iSeries = new List<int>();
        private readonly List<int> _h0iSeries = new List<int>();
        private readonly List<int> _buObIndexSeries = new List<int>();
        private readonly List<int> _beObIndexSeries = new List<int>();
        private readonly List<int> _buBbIndexSeries = new List<int>();
        private readonly List<int> _beBbIndexSeries = new List<int>();

        private int _trend = 1;
        private int _market = 1;
        private double _lastMarketChangeL0 = double.NaN;
        private double _lastMarketChangeH0 = double.NaN;
        private int _id;

        private const string Prefix = "MSBOB_";

        public override void Calculate(int index)
        {
            EnsureSeeded();

            var toUp = Bars.HighPrices[index] >= Highest(index, ZigZagLength);
            var toDown = Bars.LowPrices[index] <= Lowest(index, ZigZagLength);
            _toUp.Add(toUp);
            _toDown.Add(toDown);

            var prevTrend = index > 0 ? _trendSeries[index - 1] : 1;
            _trend = prevTrend == 1 && toDown ? -1 : prevTrend == -1 && toUp ? 1 : prevTrend;
            _trendSeries.Add(_trend);

            var lastTrendUpSince = BarsSince(_toUp, index - 1);
            var lowLen = lastTrendUpSince > 0 ? lastTrendUpSince : 1;
            var lowVal = Lowest(index, lowLen);
            var lowIdx = MostRecentIndexOfLow(index, lowVal);

            var lastTrendDownSince = BarsSince(_toDown, index - 1);
            var highLen = lastTrendDownSince > 0 ? lastTrendDownSince : 1;
            var highVal = Highest(index, highLen);
            var highIdx = MostRecentIndexOfHigh(index, highVal);

            _l0iSeries.Add(lowIdx);
            _h0iSeries.Add(highIdx);

            var trendChanged = index > 0 && _trendSeries[index] != _trendSeries[index - 1];
            if (trendChanged)
            {
                if (_trend == 1)
                {
                    _lowPoints.Add(lowVal);
                    _lowIndices.Add(lowIdx);
                }
                else if (_trend == -1)
                {
                    _highPoints.Add(highVal);
                    _highIndices.Add(highIdx);
                }
            }

            if (!TryGetHigh(0, out var h0, out var h0i) ||
                !TryGetHigh(1, out var h1, out var h1i) ||
                !TryGetLow(0, out var l0, out var l0i) ||
                !TryGetLow(1, out var l1, out var l1i))
            {
                _marketSeries.Add(_market);
                return;
            }

            if (trendChanged && ShowZigzag)
            {
                var zid = Prefix + "zig_" + index;
                if (_trend == 1)
                    Chart.DrawTrendLine(zid, h0i, h0, l0i, l0, Color.Gray, 1, LineStyle.Solid);
                else if (_trend == -1)
                    Chart.DrawTrendLine(zid, l0i, l0, h0i, h0, Color.Gray, 1, LineStyle.Solid);
            }

            var prevMarket = index > 0 ? _marketSeries[index - 1] : 1;
            _market = prevMarket;

            if (!NearlyEqual(_lastMarketChangeL0, l0) && !NearlyEqual(_lastMarketChangeH0, h0))
            {
                if (_market == 1 && l0 < l1 && l0 < l1 - Math.Abs(h0 - l1) * FibFactor)
                    _market = -1;
                else if (_market == -1 && h0 > h1 && h0 > h1 + Math.Abs(h1 - l0) * FibFactor)
                    _market = 1;
            }

            var marketChanged = _market != prevMarket;
            if (marketChanged)
            {
                _lastMarketChangeL0 = l0;
                _lastMarketChangeH0 = h0;
            }
            _marketSeries.Add(_market);

            var buObIndex = index > 0 ? _buObIndexSeries[index - 1] : index;
            var buObRangeEnd = GetSeriesValue(_l0iSeries, index - ZigZagLength, l0i);
            ScanPineRange(h1i, buObRangeEnd, i =>
            {
                var source = index - i;
                if (source >= 0 && source <= index && Bars.OpenPrices[source] > Bars.ClosePrices[source])
                    buObIndex = source;
            });
            _buObIndexSeries.Add(buObIndex);

            var beObIndex = index > 0 ? _beObIndexSeries[index - 1] : index;
            var beObRangeEnd = GetSeriesValue(_h0iSeries, index - ZigZagLength, h0i);
            ScanPineRange(l1i, beObRangeEnd, i =>
            {
                var source = index - i;
                if (source >= 0 && source <= index && Bars.OpenPrices[source] < Bars.ClosePrices[source])
                    beObIndex = source;
            });
            _beObIndexSeries.Add(beObIndex);

            var beBbIndex = index > 0 ? _beBbIndexSeries[index - 1] : index;
            ScanPineRange(h1i - ZigZagLength, l1i, i =>
            {
                var source = index - i;
                if (source >= 0 && source <= index && Bars.OpenPrices[source] > Bars.ClosePrices[source])
                    beBbIndex = source;
            });
            _beBbIndexSeries.Add(beBbIndex);

            var buBbIndex = index > 0 ? _buBbIndexSeries[index - 1] : index;
            ScanPineRange(l1i - ZigZagLength, h1i, i =>
            {
                var source = index - i;
                if (source >= 0 && source <= index && Bars.OpenPrices[source] < Bars.ClosePrices[source])
                    buBbIndex = source;
            });
            _buBbIndexSeries.Add(buBbIndex);

            if (marketChanged)
            {
                if (_market == 1)
                {
                    Chart.DrawTrendLine(Prefix + "msb_line_" + index, h1i, h1, h0i, h1, Color.Green, 2, LineStyle.Solid);
                    DrawMbsLabel(index, (h1i + l0i) / 2, h1, Color.Green, true);

                    CreateZone(_buObBoxes, buObIndex, index, "Bu-OB", BuObColor, BuObBorderColor, BuObTextColor);
                    CreateZone(_buBbBoxes, buBbIndex, index, l0 < l1 ? "Bu-BB" : "Bu-MB", BuBbColor, BuBbBorderColor, BuBbTextColor);
                }
                else if (_market == -1)
                {
                    Chart.DrawTrendLine(Prefix + "msb_line_" + index, l1i, l1, l0i, l1, Color.Red, 2, LineStyle.Solid);
                    DrawMbsLabel(index, (l1i + h0i) / 2, l1, Color.Red, false);

                    CreateZone(_beObBoxes, beObIndex, index, "Be-OB", BeObColor, BeObBorderColor, BeObTextColor);
                    CreateZone(_beBbBoxes, beBbIndex, index, h0 > h1 ? "Be-BB" : "Be-MB", BeBbColor, BeBbBorderColor, BeBbTextColor);
                }
            }

            ApplyVisibilityLimit(_buObBoxes, ShowBullishBlocks);
            ApplyVisibilityLimit(_buBbBoxes, ShowBullishBreakerBlocks);
            ApplyVisibilityLimit(_beObBoxes, ShowBearishBlocks);
            ApplyVisibilityLimit(_beBbBoxes, ShowBearishBreakerBlocks);

            ProcessBullZones(_buObBoxes, index, "Price in the BU-OB zone");
            ProcessBearZones(_beObBoxes, index, "Price in the BE-OB zone");
            ProcessBearZones(_beBbBoxes, index, "Price in the BE-BB zone");
            ProcessBullZones(_buBbBoxes, index, "Price in the BU-BB zone");
        }

        private void EnsureSeeded()
        {
            if (_highPoints.Count == 0)
            {
                for (var i = 0; i < 5; i++)
                {
                    _highPoints.Add(double.NaN);
                    _highIndices.Add(-1);
                    _lowPoints.Add(double.NaN);
                    _lowIndices.Add(-1);
                }
            }
        }

        private bool TryGetHigh(int ind, out double value, out int idx)
        {
            var p = _highPoints.Count - 1 - ind;
            if (p < 0)
            {
                value = double.NaN;
                idx = -1;
                return false;
            }

            value = _highPoints[p];
            idx = _highIndices[p];
            return !double.IsNaN(value) && idx >= 0;
        }

        private bool TryGetLow(int ind, out double value, out int idx)
        {
            var p = _lowPoints.Count - 1 - ind;
            if (p < 0)
            {
                value = double.NaN;
                idx = -1;
                return false;
            }

            value = _lowPoints[p];
            idx = _lowIndices[p];
            return !double.IsNaN(value) && idx >= 0;
        }

        private static void ScanPineRange(int start, int end, Action<int> visitor)
        {
            if (start > end)
                return;

            for (var i = start; i <= end; i++)
                visitor(i);
        }

        private void CreateZone(List<Zone> target, int left, int index, string text, Color fill, Color border, Color textColor)
        {
            left = Math.Max(0, Math.Min(left, index));
            var right = Math.Min(index + 10, Bars.Count - 1);
            var top = Bars.HighPrices[left];
            var bottom = Bars.LowPrices[left];

            var zone = new Zone
            {
                Id = Prefix + "zone_" + (++_id),
                LabelId = Prefix + "zone_lbl_" + _id,
                Left = left,
                Right = right,
                Top = top,
                Bottom = bottom,
                Text = text,
                Fill = fill,
                Border = border,
                TextColor = textColor
            };

            target.Add(zone);
            RedrawZone(zone);
        }

        private void ApplyVisibilityLimit(List<Zone> zones, int maxVisible)
        {
            var cap = Math.Max(0, maxVisible);
            while (zones.Count > cap)
            {
                var old = zones[0];
                zones.RemoveAt(0);
                Chart.RemoveObject(old.Id);
                Chart.RemoveObject(old.LabelId);
            }
        }

        private void ProcessBullZones(List<Zone> zones, int index, string enterMessage)
        {
            for (var i = zones.Count - 1; i >= 0; i--)
            {
                var z = zones[i];
                if (Bars.ClosePrices[index] < z.Bottom)
                {
                    DeleteOrShift(zones, i);
                }
                else if (Bars.ClosePrices[index] < z.Top)
                {
                    Print(enterMessage);
                }
                else
                {
                    z.Right = Math.Min(index + 10, Bars.Count - 1);
                    RedrawZone(z);
                }
            }
        }

        private void ProcessBearZones(List<Zone> zones, int index, string enterMessage)
        {
            for (var i = zones.Count - 1; i >= 0; i--)
            {
                var z = zones[i];
                if (Bars.ClosePrices[index] > z.Top)
                {
                    DeleteOrShift(zones, i);
                }
                else if (Bars.ClosePrices[index] > z.Bottom)
                {
                    Print(enterMessage);
                }
                else
                {
                    z.Right = Math.Min(index + 10, Bars.Count - 1);
                    RedrawZone(z);
                }
            }
        }

        private void DeleteOrShift(List<Zone> zones, int index)
        {
            if (zones.Count == 0)
                return;

            var shifted = zones[0];
            zones.RemoveAt(0);

            if (DeleteBoxes)
            {
                Chart.RemoveObject(shifted.Id);
                Chart.RemoveObject(shifted.LabelId);
            }
        }

        private void RedrawZone(Zone zone)
        {
            var rect = Chart.DrawRectangle(zone.Id, zone.Left, zone.Top, zone.Right, zone.Bottom, zone.Border, 1, LineStyle.Solid);
            rect.IsFilled = true;
            rect.Color = zone.Fill;

            var textX = Math.Max(zone.Left, zone.Right - 1);
            var textY = (zone.Top + zone.Bottom) * 0.5;
            var label = Chart.DrawText(zone.LabelId, zone.Text, textX, textY, zone.TextColor);
            label.FontSize = MapFontSize(TextSize);
        }

        private void DrawMbsLabel(int index, int x, double y, Color color, bool down)
        {
            var label = Chart.DrawText(Prefix + "msb_lbl_" + index, "MSB", x, y, color);
            label.FontSize = 11;
        }

        private static int MapFontSize(LabelSizeOpt s)
        {
            switch (s)
            {
                case LabelSizeOpt.Tiny: return 9;
                case LabelSizeOpt.Small: return 11;
                case LabelSizeOpt.Normal: return 13;
                case LabelSizeOpt.Large: return 15;
                case LabelSizeOpt.Huge: return 18;
                default: return 9;
            }
        }

        private int GetSeriesValue(List<int> series, int idx, int fallback)
        {
            if (idx < 0 || idx >= series.Count)
                return fallback;
            return series[idx];
        }

        private static int BarsSince(List<bool> cond, int idx)
        {
            if (idx < 0)
                return 0;

            for (var i = idx; i >= 0; i--)
            {
                if (cond[i])
                    return idx - i;
            }

            return 0;
        }

        private int MostRecentIndexOfLow(int index, double value)
        {
            for (var i = index; i >= 0; i--)
            {
                if (NearlyEqual(Bars.LowPrices[i], value))
                    return i;
            }
            return index;
        }

        private int MostRecentIndexOfHigh(int index, double value)
        {
            for (var i = index; i >= 0; i--)
            {
                if (NearlyEqual(Bars.HighPrices[i], value))
                    return i;
            }
            return index;
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

        private static bool NearlyEqual(double a, double b)
        {
            if (double.IsNaN(a) || double.IsNaN(b))
                return false;
            return Math.Abs(a - b) <= 1e-10;
        }
    }
}
