using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Indicators;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class SmartMoneyZonesFvgObMtfTrendPanel : Indicator
    {
        [Parameter("Show Fair Value Gaps", Group = "Display", DefaultValue = true)]
        public bool ShowFvg { get; set; }

        [Parameter("Show Order Blocks", Group = "Display", DefaultValue = true)]
        public bool ShowOb { get; set; }

        [Parameter("Max zones per TYPE", Group = "Display", DefaultValue = 20, MinValue = 5, MaxValue = 200)]
        public int MaxZones { get; set; }

        [Parameter("Remove when mitigated", Group = "Display", DefaultValue = false)]
        public bool RemoveOnMitigated { get; set; }

        [Parameter("Min impulse body %", Group = "Detection", DefaultValue = 0.5, MinValue = 0.3, MaxValue = 1.0, Step = 0.1)]
        public double MinBodyPct { get; set; }

        [Parameter("Order Block lookback", Group = "Detection", DefaultValue = 10, MinValue = 5, MaxValue = 50)]
        public int ObLookback { get; set; }

        [Parameter("ATR length", Group = "Detection", DefaultValue = 14, MinValue = 5, MaxValue = 50)]
        public int AtrLength { get; set; }

        [Parameter("Use trend filter", Group = "Trend Filter", DefaultValue = true)]
        public bool UseTrendFilter { get; set; }

        [Parameter("Trend MA period", Group = "Trend Filter", DefaultValue = 50, MinValue = 10, MaxValue = 200)]
        public int TrendMAPeriod { get; set; }

        [Parameter("Show MTF Trend Panel", Group = "Multi-Timeframe Panel", DefaultValue = true)]
        public bool ShowPanel { get; set; }

        [Parameter("Panel Position", Group = "Multi-Timeframe Panel", DefaultValue = "Top Right")]
        public string PanelPosition { get; set; }

        [Parameter("Panel Size", Group = "Multi-Timeframe Panel", DefaultValue = "Normal")]
        public string PanelSize { get; set; }

        [Parameter("MTF Trend MA Period", Group = "Multi-Timeframe Panel", DefaultValue = 50, MinValue = 10, MaxValue = 200)]
        public int MtfTrendMAPeriod { get; set; }

        [Output("Trend MA", Thickness = 2)]
        public IndicatorDataSeries TrendMAOutput { get; set; }

        private readonly List<Zone> _bullFvgZones = new List<Zone>();
        private readonly List<Zone> _bearFvgZones = new List<Zone>();
        private readonly List<Zone> _bullObZones = new List<Zone>();
        private readonly List<Zone> _bearObZones = new List<Zone>();

        private SimpleMovingAverage _trendSma;
        private AverageTrueRange _atr;

        private Bars _bars1m;
        private Bars _bars5m;
        private Bars _bars15m;
        private Bars _bars30m;
        private Bars _bars1h;
        private Bars _bars4h;
        private Bars _bars1d;

        private SimpleMovingAverage _sma1m;
        private SimpleMovingAverage _sma5m;
        private SimpleMovingAverage _sma15m;
        private SimpleMovingAverage _sma30m;
        private SimpleMovingAverage _sma1h;
        private SimpleMovingAverage _sma4h;
        private SimpleMovingAverage _sma1d;

        private const string PanelId = "smz_mtf_panel";

        private readonly Color _bullFvgColor = Color.FromArgb(38, 0, 255, 0);
        private readonly Color _bearFvgColor = Color.FromArgb(38, 255, 0, 0);
        private readonly Color _bullObColor = Color.FromArgb(51, 0, 136, 255);
        private readonly Color _bearObColor = Color.FromArgb(51, 255, 102, 0);
        private readonly Color _mitigatedColor = Color.FromArgb(13, 128, 128, 128);

        protected override void Initialize()
        {
            _trendSma = Indicators.SimpleMovingAverage(Bars.ClosePrices, TrendMAPeriod);
            _atr = Indicators.AverageTrueRange(AtrLength, MovingAverageType.Simple);

            _bars1m = MarketData.GetBars(TimeFrame.Minute);
            _bars5m = MarketData.GetBars(TimeFrame.Minute5);
            _bars15m = MarketData.GetBars(TimeFrame.Minute15);
            _bars30m = MarketData.GetBars(TimeFrame.Minute30);
            _bars1h = MarketData.GetBars(TimeFrame.Hour);
            _bars4h = MarketData.GetBars(TimeFrame.Hour4);
            _bars1d = MarketData.GetBars(TimeFrame.Daily);

            _sma1m = Indicators.SimpleMovingAverage(_bars1m.ClosePrices, MtfTrendMAPeriod);
            _sma5m = Indicators.SimpleMovingAverage(_bars5m.ClosePrices, MtfTrendMAPeriod);
            _sma15m = Indicators.SimpleMovingAverage(_bars15m.ClosePrices, MtfTrendMAPeriod);
            _sma30m = Indicators.SimpleMovingAverage(_bars30m.ClosePrices, MtfTrendMAPeriod);
            _sma1h = Indicators.SimpleMovingAverage(_bars1h.ClosePrices, MtfTrendMAPeriod);
            _sma4h = Indicators.SimpleMovingAverage(_bars4h.ClosePrices, MtfTrendMAPeriod);
            _sma1d = Indicators.SimpleMovingAverage(_bars1d.ClosePrices, MtfTrendMAPeriod);
        }

        public override void Calculate(int index)
        {
            if (index < 2)
                return;

            var trendMa = _trendSma.Result[index];
            var isUptrend = Bars.ClosePrices[index] > trendMa;
            var isDntrend = Bars.ClosePrices[index] < trendMa;

            TrendMAOutput[index] = UseTrendFilter ? trendMa : double.NaN;

            var atrValue = Math.Max(_atr.Result[index], Symbol.TickSize);

            if (ShowFvg)
                DetectFvg(index, atrValue, isUptrend, isDntrend);

            if (ShowOb && index > ObLookback)
                DetectOrderBlock(index, atrValue, isUptrend, isDntrend);

            UpdateZones(_bullFvgZones, index);
            UpdateZones(_bearFvgZones, index);
            UpdateZones(_bullObZones, index);
            UpdateZones(_bearObZones, index);

            DrawMtfPanel();
        }

        private void DetectFvg(int index, double atrValue, bool isUptrend, bool isDntrend)
        {
            if (Bars.LowPrices[index] > Bars.HighPrices[index - 2])
            {
                var middleBull = Bars.ClosePrices[index - 1] > Bars.OpenPrices[index - 1];
                var middleBody = CalcBodyPct(Bars.OpenPrices[index - 1], Bars.HighPrices[index - 1], Bars.LowPrices[index - 1], Bars.ClosePrices[index - 1]);

                if (middleBull && middleBody >= MinBodyPct && (!UseTrendFilter || isUptrend))
                {
                    var top = Bars.LowPrices[index];
                    var bottom = Bars.HighPrices[index - 2];
                    var size = top - bottom;
                    var strength = GetStrength(size, atrValue);

                    CreateZone(_bullFvgZones, true, true, index - 2, top, bottom, size, strength, _bullFvgColor, "🟢 FVG", true);
                }
            }

            if (Bars.HighPrices[index] < Bars.LowPrices[index - 2])
            {
                var middleBear = Bars.ClosePrices[index - 1] < Bars.OpenPrices[index - 1];
                var middleBody = CalcBodyPct(Bars.OpenPrices[index - 1], Bars.HighPrices[index - 1], Bars.LowPrices[index - 1], Bars.ClosePrices[index - 1]);

                if (middleBear && middleBody >= MinBodyPct && (!UseTrendFilter || isDntrend))
                {
                    var top = Bars.LowPrices[index - 2];
                    var bottom = Bars.HighPrices[index];
                    var size = top - bottom;
                    var strength = GetStrength(size, atrValue);

                    CreateZone(_bearFvgZones, false, true, index - 2, top, bottom, size, strength, _bearFvgColor, "🔴 FVG", false);
                }
            }
        }

        private void DetectOrderBlock(int index, double atrValue, bool isUptrend, bool isDntrend)
        {
            var isBullBreak = Bars.ClosePrices[index] > Bars.ClosePrices[index - 1] && (Bars.HighPrices[index] - Bars.LowPrices[index]) > atrValue * 1.2;
            var previousBear = Bars.ClosePrices[index - 1] < Bars.OpenPrices[index - 1];
            var strongUp = Bars.ClosePrices[index] > Bars.OpenPrices[index] && CalcBodyPct(Bars.OpenPrices[index], Bars.HighPrices[index], Bars.LowPrices[index], Bars.ClosePrices[index]) >= MinBodyPct;

            if (isBullBreak && previousBear && strongUp && (!UseTrendFilter || isUptrend))
            {
                var top = Math.Max(Bars.OpenPrices[index - 1], Bars.ClosePrices[index - 1]);
                var bottom = Bars.LowPrices[index - 1];
                var size = top - bottom;
                var strength = GetStrength(size, atrValue);

                CreateZone(_bullObZones, true, false, index - 1, top, bottom, size, strength, _bullObColor, "🔵 OB", true);
            }

            var isBearBreak = Bars.ClosePrices[index] < Bars.ClosePrices[index - 1] && (Bars.HighPrices[index] - Bars.LowPrices[index]) > atrValue * 1.2;
            var previousBull = Bars.ClosePrices[index - 1] > Bars.OpenPrices[index - 1];
            var strongDown = Bars.ClosePrices[index] < Bars.OpenPrices[index] && CalcBodyPct(Bars.OpenPrices[index], Bars.HighPrices[index], Bars.LowPrices[index], Bars.ClosePrices[index]) >= MinBodyPct;

            if (isBearBreak && previousBull && strongDown && (!UseTrendFilter || isDntrend))
            {
                var top = Bars.HighPrices[index - 1];
                var bottom = Math.Min(Bars.OpenPrices[index - 1], Bars.ClosePrices[index - 1]);
                var size = top - bottom;
                var strength = GetStrength(size, atrValue);

                CreateZone(_bearObZones, false, false, index - 1, top, bottom, size, strength, _bearObColor, "🟠 OB", false);
            }
        }

        private void CreateZone(List<Zone> zones, bool isBullish, bool isFvg, int leftIndex, double top, double bottom, double size, string strength, Color zoneColor, string markerText, bool markerAtTop)
        {
            var id = $"smz_{(isFvg ? "fvg" : "ob")}_{(isBullish ? "bull" : "bear")}_{leftIndex}_{Server.Time.Ticks}";

            var rectangle = Chart.DrawRectangle(id, leftIndex, top, Bars.Count - 1, bottom, zoneColor);
            rectangle.IsFilled = true;
            rectangle.IsInteractive = false;

            var markerPrice = markerAtTop ? top : bottom;
            var markerType = markerAtTop ? ChartIconType.DownTriangle : ChartIconType.UpTriangle;
            Chart.DrawIcon(id + "_icon", markerType, leftIndex + 1, markerPrice, Color.White);

            Chart.DrawText(id + "_txt", markerText + " " + strength, leftIndex + 1, markerPrice, Color.White);

            zones.Add(new Zone
            {
                Id = id,
                Top = top,
                Bottom = bottom,
                Size = Math.Max(size, Symbol.TickSize),
                IsLive = true,
                IsFvg = isFvg,
                LeftIndex = leftIndex,
                Strength = strength,
                MitigatedAmount = 0.0,
                IsBullish = isBullish
            });

            EnforceCap(zones);
        }

        private void EnforceCap(List<Zone> zones)
        {
            while (zones.Count > MaxZones)
            {
                var z = zones[0];
                DeleteZone(z);
                zones.RemoveAt(0);
            }
        }

        private void UpdateZones(List<Zone> zones, int rightIndex)
        {
            for (var i = zones.Count - 1; i >= 0; i--)
            {
                var zone = zones[i];

                var currentColor = zone.IsLive ? GetZoneColor(zone) : _mitigatedColor;
                var rectangle = Chart.DrawRectangle(zone.Id, zone.LeftIndex, zone.Top, rightIndex, zone.Bottom, currentColor);
                rectangle.IsFilled = true;
                rectangle.IsInteractive = false;

                if (!zone.IsLive)
                    continue;

                var isMitigated = false;

                if (zone.IsBullish)
                {
                    if (Bars.LowPrices[rightIndex] <= zone.Top && Bars.LowPrices[rightIndex] >= zone.Bottom)
                    {
                        var penetration = zone.Top - Bars.LowPrices[rightIndex];
                        zone.MitigatedAmount = (penetration / zone.Size) * 100.0;
                    }

                    isMitigated = zone.MitigatedAmount >= 50.0 || Bars.ClosePrices[rightIndex] <= zone.Bottom;
                }
                else
                {
                    if (Bars.HighPrices[rightIndex] >= zone.Bottom && Bars.HighPrices[rightIndex] <= zone.Top)
                    {
                        var penetration = Bars.HighPrices[rightIndex] - zone.Bottom;
                        zone.MitigatedAmount = (penetration / zone.Size) * 100.0;
                    }

                    isMitigated = zone.MitigatedAmount >= 50.0 || Bars.ClosePrices[rightIndex] >= zone.Top;
                }

                if (!isMitigated)
                {
                    zones[i] = zone;
                    continue;
                }

                if (RemoveOnMitigated)
                {
                    DeleteZone(zone);
                    zones.RemoveAt(i);
                }
                else
                {
                    zone.IsLive = false;
                    var mitigatedRectangle = Chart.DrawRectangle(zone.Id, zone.LeftIndex, zone.Top, rightIndex, zone.Bottom, _mitigatedColor);
                    mitigatedRectangle.IsFilled = true;
                    mitigatedRectangle.IsInteractive = false;
                    zones[i] = zone;
                }
            }
        }

        private Color GetZoneColor(Zone zone)
        {
            if (zone.IsFvg)
                return zone.IsBullish ? _bullFvgColor : _bearFvgColor;

            return zone.IsBullish ? _bullObColor : _bearObColor;
        }

        private void DeleteZone(Zone zone)
        {
            Chart.RemoveObject(zone.Id);
            Chart.RemoveObject(zone.Id + "_icon");
            Chart.RemoveObject(zone.Id + "_txt");
        }

        private void DrawMtfPanel()
        {
            if (!ShowPanel)
            {
                Chart.RemoveObject(PanelId);
                return;
            }

            var trend1m = GetTrend(_bars1m, _sma1m);
            var trend5m = GetTrend(_bars5m, _sma5m);
            var trend15m = GetTrend(_bars15m, _sma15m);
            var trend30m = GetTrend(_bars30m, _sma30m);
            var trend1h = GetTrend(_bars1h, _sma1h);
            var trend4h = GetTrend(_bars4h, _sma4h);
            var trend1d = GetTrend(_bars1d, _sma1d);

            var panelText =
                "⏱️ TIMEFRAME | TREND | STATUS\n" +
                BuildRow("1m", trend1m) + "\n" +
                BuildRow("5m", trend5m) + "\n" +
                BuildRow("15m", trend15m) + "\n" +
                BuildRow("30m", trend30m) + "\n" +
                BuildRow("1H", trend1h) + "\n" +
                BuildRow("4H", trend4h) + "\n" +
                BuildRow("1D", trend1d);

            var _ = ParsePanelSize(PanelSize);
            var horizontal = ParseHorizontal(PanelPosition);
            var vertical = ParseVertical(PanelPosition);

            Chart.DrawStaticText(PanelId, panelText, vertical, horizontal, Color.White);
        }

        private bool GetTrend(Bars bars, SimpleMovingAverage sma)
        {
            if (bars == null || sma == null)
                return false;

            var idx = bars.Count - 1;
            if (idx < 0 || idx < MtfTrendMAPeriod - 1)
                return false;

            return bars.ClosePrices[idx] > sma.Result[idx];
        }

        private static string BuildRow(string tf, bool isBullish)
        {
            return $"{tf,-4} | {(isBullish ? "BULLISH" : "BEARISH"),-7} | {(isBullish ? "🟢" : "🔴")}";
        }

        private static string ParsePanelSize(string value)
        {
            if (string.Equals(value, "Small", StringComparison.OrdinalIgnoreCase))
                return "Small";
            if (string.Equals(value, "Large", StringComparison.OrdinalIgnoreCase))
                return "Large";
            return "Normal";
        }

        private static HorizontalAlignment ParseHorizontal(string value)
        {
            if (!string.IsNullOrWhiteSpace(value) && value.IndexOf("Left", StringComparison.OrdinalIgnoreCase) >= 0)
                return HorizontalAlignment.Left;

            return HorizontalAlignment.Right;
        }

        private static VerticalAlignment ParseVertical(string value)
        {
            if (!string.IsNullOrWhiteSpace(value) && value.IndexOf("Bottom", StringComparison.OrdinalIgnoreCase) >= 0)
                return VerticalAlignment.Bottom;

            return VerticalAlignment.Top;
        }

        private static double CalcBodyPct(double open, double high, double low, double close)
        {
            var range = high - low;
            if (range <= 0)
                return 0.0;

            var body = Math.Abs(close - open);
            return body / range;
        }

        private string GetStrength(double size, double atrValue)
        {
            var denom = Math.Max(atrValue, Symbol.TickSize);
            var ratio = size / denom;

            if (ratio >= 2.0)
                return "VERY STRONG";
            if (ratio >= 1.5)
                return "STRONG";
            if (ratio >= 1.0)
                return "MEDIUM";
            return "WEAK";
        }

        private struct Zone
        {
            public string Id;
            public double Top;
            public double Bottom;
            public double Size;
            public bool IsLive;
            public bool IsFvg;
            public string Strength;
            public double MitigatedAmount;
            public bool IsBullish;
            public int LeftIndex;
        }
    }
}
