using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Indicators;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class OrderBlocksImbalanceMtf : Indicator
    {
        public enum MitigationMethod
        {
            Close,
            Wick
        }

        public enum ZoneState
        {
            Active,
            Mitigated,
            Invalidated
        }

        private sealed class ObZone
        {
            public string Id;
            public string LabelId;

            public ChartRectangle Box;
            public ChartText Label;

            public string TfKey;
            public string TfLabel;

            public bool IsBullish;
            public ZoneState State;

            public double Top;
            public double Bottom;

            public DateTime CreatedTfTime;
            public int CreatedChartIndex;
            public DateTime FrozenRightTime;
        }

        private sealed class TfState
        {
            public string Key;
            public string Label;
            public bool Enabled;
            public TimeFrame Tf;
            public Bars Bars;
            public AverageTrueRange Atr;
            public DateTime LastCreatedSeedTime = DateTime.MinValue;
        }

        [Parameter("Mitigation Method", DefaultValue = MitigationMethod.Wick, Group = "Logic")]
        public MitigationMethod MitigationTypeInput { get; set; }

        [Parameter("Min FVG Size (ATR Mult)", DefaultValue = 0.5, Step = 0.1, Group = "Logic")]
        public double FvgThreshold { get; set; }

        [Parameter("Show Demand (Bullish)", DefaultValue = true, Group = "Logic")]
        public bool ShowBull { get; set; }

        [Parameter("Show Supply (Bearish)", DefaultValue = true, Group = "Logic")]
        public bool ShowBear { get; set; }

        [Parameter("Show Historical OBs", DefaultValue = true, Group = "Display")]
        public bool ShowHistoricalOBs { get; set; }

        [Parameter("Show Mitigated OBs", DefaultValue = false, Group = "Display")]
        public bool ShowMitigatedOBs { get; set; }

        [Parameter("Show Invalidated OBs", DefaultValue = false, Group = "Display")]
        public bool ShowInvalidatedOBs { get; set; }

        [Parameter("Show Mitigated Text", DefaultValue = false, Group = "Display")]
        public bool ShowMitigatedText { get; set; }

        [Parameter("Show Invalidated Text", DefaultValue = false, Group = "Display")]
        public bool ShowInvalidatedText { get; set; }

        [Parameter("Enable Smart Visibility", DefaultValue = true, Group = "Display")]
        public bool UseSmartView { get; set; }

        [Parameter("Max Active Zones Per Side", DefaultValue = 10, MinValue = 1, MaxValue = 50, Group = "Display")]
        public int VisibleLimit { get; set; }

        [Parameter("Auto Extend Active Zones", DefaultValue = true, Group = "Display")]
        public bool ExtendActive { get; set; }

        [Parameter("Bull Active Color", DefaultValue = "#4600FF00", Group = "Colors")]
        public Color BullHistoricalColor { get; set; }

        [Parameter("Bear Active Color", DefaultValue = "#46FF0000", Group = "Colors")]
        public Color BearHistoricalColor { get; set; }

        [Parameter("Bull Mitigated Color", DefaultValue = "#D9FFFF00", Group = "Colors")]
        public Color MitigatedBullColor { get; set; }

        [Parameter("Bear Mitigated Color", DefaultValue = "#D9FF6600", Group = "Colors")]
        public Color MitigatedBearColor { get; set; }

        [Parameter("Bull Invalidated Color", DefaultValue = "#80999999", Group = "Colors")]
        public Color InvalidatedBullColor { get; set; }

        [Parameter("Bear Invalidated Color", DefaultValue = "#80999999", Group = "Colors")]
        public Color InvalidatedBearColor { get; set; }

        [Parameter("Label Color", DefaultValue = "White", Group = "Colors")]
        public Color LabelColor { get; set; }

        [Parameter("Enable TF1", DefaultValue = true, Group = "Timeframes")]
        public bool EnableTf1 { get; set; }

        [Parameter("TF1", DefaultValue = "Minute15", Group = "Timeframes")]
        public TimeFrame Tf1 { get; set; }

        [Parameter("Enable TF2", DefaultValue = true, Group = "Timeframes")]
        public bool EnableTf2 { get; set; }

        [Parameter("TF2", DefaultValue = "Minute30", Group = "Timeframes")]
        public TimeFrame Tf2 { get; set; }

        [Parameter("Enable TF3", DefaultValue = true, Group = "Timeframes")]
        public bool EnableTf3 { get; set; }

        [Parameter("TF3", DefaultValue = "Hour", Group = "Timeframes")]
        public TimeFrame Tf3 { get; set; }

        [Parameter("Enable TF4", DefaultValue = true, Group = "Timeframes")]
        public bool EnableTf4 { get; set; }

        [Parameter("TF4", DefaultValue = "Hour4", Group = "Timeframes")]
        public TimeFrame Tf4 { get; set; }

        private readonly List<TfState> _tfStates = new List<TfState>();
        private readonly List<ObZone> _zones = new List<ObZone>();
        private int _idCounter;

        protected override void Initialize()
        {
            _tfStates.Clear();

            RegisterTf("tf1", Tf1, EnableTf1);
            RegisterTf("tf2", Tf2, EnableTf2);
            RegisterTf("tf3", Tf3, EnableTf3);
            RegisterTf("tf4", Tf4, EnableTf4);
        }

        public override void Calculate(int index)
        {
            if (index < 2)
                return;

            var now = Bars.OpenTimes[index];

            for (int i = 0; i < _tfStates.Count; i++)
            {
                var tf = _tfStates[i];
                if (!tf.Enabled || tf.Bars == null || tf.Bars.Count < 3)
                    continue;

                int htfIndex = FindBarIndexAtOrBefore(tf.Bars, now);
                if (htfIndex < 2)
                    continue;

                double high2 = tf.Bars.HighPrices[htfIndex - 2];
                double low2 = tf.Bars.LowPrices[htfIndex - 2];
                double low0 = tf.Bars.LowPrices[htfIndex];
                double high0 = tf.Bars.HighPrices[htfIndex];
                DateTime seedTime = tf.Bars.OpenTimes[htfIndex - 2];

                double atrRef = tf.Atr.Result[Math.Max(0, htfIndex - 1)];
                double bullFvgSize = low0 - high2;
                double bearFvgSize = low2 - high0;

                bool isBullSeed = bullFvgSize > atrRef * FvgThreshold;
                bool isBearSeed = bearFvgSize > atrRef * FvgThreshold;

                if (seedTime != tf.LastCreatedSeedTime)
                {
                    if (ShowBull && isBullSeed)
                    {
                        CreateZone(tf, true, high2, low2, seedTime, index);
                        tf.LastCreatedSeedTime = seedTime;
                    }
                    else if (ShowBear && isBearSeed)
                    {
                        CreateZone(tf, false, high2, low2, seedTime, index);
                        tf.LastCreatedSeedTime = seedTime;
                    }
                }
            }

            UpdateZoneStates(index);

            if (_zones.Count > 1000)
            {
                var oldest = _zones[0];
                RemoveZoneGraphics(oldest);
                _zones.RemoveAt(0);
            }

            if (index != Bars.Count - 1)
                return;

            var dt = index > 0 ? (Bars.OpenTimes[index] - Bars.OpenTimes[index - 1]) : TimeSpan.FromMinutes(1);
            var futureTime = Bars.OpenTimes[index].AddTicks(dt.Ticks * 10);

            if (UseSmartView)
                ApplySmartVisibility(index, futureTime);
            else
                DrawAllVisible(index, futureTime);
        }

        private void RegisterTf(string key, TimeFrame tf, bool enabled)
        {
            if (!enabled)
                return;

            var bars = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf);
            var atr = Indicators.AverageTrueRange(bars, 14, MovingAverageType.Simple);

            _tfStates.Add(new TfState
            {
                Key = key,
                Label = GetTimeFrameLabel(tf),
                Enabled = enabled,
                Tf = tf,
                Bars = bars,
                Atr = atr
            });
        }

        private void CreateZone(TfState tf, bool isBull, double top, double bottom, DateTime seedTime, int chartIndex)
        {
            double zoneTop = Math.Max(top, bottom);
            double zoneBottom = Math.Min(top, bottom);

            if (double.IsNaN(zoneTop) || double.IsNaN(zoneBottom))
                return;

            var zone = new ObZone
            {
                Id = "ob_" + tf.Key + "_" + _idCounter++,
                TfKey = tf.Key,
                TfLabel = tf.Label,
                IsBullish = isBull,
                State = ZoneState.Active,
                Top = zoneTop,
                Bottom = zoneBottom,
                CreatedTfTime = seedTime,
                CreatedChartIndex = chartIndex,
                FrozenRightTime = Bars.OpenTimes[Math.Min(chartIndex + 1, Bars.Count - 1)]
            };
            zone.LabelId = zone.Id + "_txt";

            _zones.Add(zone);
        }

        private void UpdateZoneStates(int index)
        {
            double close = Bars.ClosePrices[index];
            double low = Bars.LowPrices[index];
            double high = Bars.HighPrices[index];
            DateTime freezeTime = Bars.OpenTimes[Math.Min(index + 1, Bars.Count - 1)];

            for (int i = 0; i < _zones.Count; i++)
            {
                var z = _zones[i];
                if (z.State == ZoneState.Invalidated)
                    continue;

                if (z.IsBullish)
                {
                    bool invalidated = close < z.Bottom;
                    bool mitigated = (MitigationTypeInput == MitigationMethod.Wick ? low : close) <= z.Top;

                    if (invalidated)
                    {
                        z.State = ZoneState.Invalidated;
                        z.FrozenRightTime = freezeTime;
                    }
                    else if (mitigated && z.State == ZoneState.Active)
                    {
                        z.State = ZoneState.Mitigated;
                        z.FrozenRightTime = freezeTime;
                    }
                }
                else
                {
                    bool invalidated = close > z.Top;
                    bool mitigated = (MitigationTypeInput == MitigationMethod.Wick ? high : close) >= z.Bottom;

                    if (invalidated)
                    {
                        z.State = ZoneState.Invalidated;
                        z.FrozenRightTime = freezeTime;
                    }
                    else if (mitigated && z.State == ZoneState.Active)
                    {
                        z.State = ZoneState.Mitigated;
                        z.FrozenRightTime = freezeTime;
                    }
                }
            }
        }

        private void ApplySmartVisibility(int index, DateTime futureTime)
        {
            var visibleBull = new List<int>();
            var visibleBear = new List<int>();

            for (int i = 0; i < _zones.Count; i++)
            {
                var z = _zones[i];

                if (z.State == ZoneState.Active)
                {
                    if (z.IsBullish)
                        visibleBull.Add(i);
                    else
                        visibleBear.Add(i);
                }
                else
                {
                    DrawZoneForState(z, index, z.FrozenRightTime);
                }
            }

            visibleBull.Sort((a, b) => _zones[b].Top.CompareTo(_zones[a].Top));
            visibleBear.Sort((a, b) => _zones[a].Bottom.CompareTo(_zones[b].Bottom));

            int bullCount = Math.Min(VisibleLimit, visibleBull.Count);
            int bearCount = Math.Min(VisibleLimit, visibleBear.Count);

            var showActive = new HashSet<int>();
            for (int i = 0; i < bullCount; i++) showActive.Add(visibleBull[i]);
            for (int i = 0; i < bearCount; i++) showActive.Add(visibleBear[i]);

            for (int i = 0; i < _zones.Count; i++)
            {
                var z = _zones[i];

                if (z.State != ZoneState.Active)
                    continue;

                if (showActive.Contains(i))
                    DrawZoneForState(z, index, ExtendActive ? futureTime : Bars.OpenTimes[Math.Min(index + 1, Bars.Count - 1)]);
                else
                    HideZone(z);
            }
        }

        private void DrawAllVisible(int index, DateTime futureTime)
        {
            for (int i = 0; i < _zones.Count; i++)
            {
                var z = _zones[i];

                if (z.State == ZoneState.Active)
                    DrawZoneForState(z, index, ExtendActive ? futureTime : Bars.OpenTimes[Math.Min(index + 1, Bars.Count - 1)]);
                else
                    DrawZoneForState(z, index, z.FrozenRightTime);
            }
        }

        private void DrawZoneForState(ObZone z, int index, DateTime rightTime)
        {
            bool visible = false;
            Color fillColor = Color.Transparent;
            LineStyle style = LineStyle.Solid;
            string text = z.TfLabel + " " + (z.IsBullish ? "OB Demand" : "OB Supply");

            if (z.State == ZoneState.Active && ShowHistoricalOBs)
            {
                visible = true;
                fillColor = z.IsBullish ? BullHistoricalColor : BearHistoricalColor;
                style = LineStyle.Solid;
            }
            else if (z.State == ZoneState.Mitigated && ShowMitigatedOBs)
            {
                visible = true;
                fillColor = z.IsBullish ? MitigatedBullColor : MitigatedBearColor;
                style = LineStyle.DotsRare;
                if (ShowMitigatedText)
                    text += " Mitigated";
            }
            else if (z.State == ZoneState.Invalidated && ShowInvalidatedOBs)
            {
                visible = true;
                fillColor = z.IsBullish ? InvalidatedBullColor : InvalidatedBearColor;
                style = LineStyle.DotsRare;
                if (ShowInvalidatedText)
                    text += " Invalidated";
            }

            if (!visible)
            {
                HideZone(z);
                return;
            }

            EnsureZoneGraphics(z, index);

            z.Box.Time1 = Bars.OpenTimes[Math.Max(0, Math.Min(z.CreatedChartIndex, Bars.Count - 1))];
            z.Box.Time2 = rightTime;
            z.Box.Y1 = z.Top;
            z.Box.Y2 = z.Bottom;
            z.Box.Color = fillColor;
            z.Box.LineStyle = style;
            z.Box.IsFilled = true;

            z.Label.Time = rightTime;
            z.Label.Y = (z.Top + z.Bottom) / 2.0;
            z.Label.Text = text;
            z.Label.Color = LabelColor;
        }

        private void EnsureZoneGraphics(ObZone z, int index)
        {
            if (z.Box == null)
            {
                var border = z.IsBullish ? Color.Green : Color.Red;
                z.Box = Chart.DrawRectangle(
                    z.Id,
                    Bars.OpenTimes[Math.Max(0, Math.Min(z.CreatedChartIndex, Bars.Count - 1))],
                    z.Top,
                    Bars.OpenTimes[Math.Min(index + 1, Bars.Count - 1)],
                    z.Bottom,
                    border,
                    1,
                    LineStyle.Solid);
                z.Box.IsFilled = true;
            }

            if (z.Label == null)
            {
                z.Label = Chart.DrawText(
                    z.LabelId,
                    z.TfLabel + " " + (z.IsBullish ? "OB Demand" : "OB Supply"),
                    Bars.OpenTimes[Math.Min(index + 1, Bars.Count - 1)],
                    (z.Top + z.Bottom) / 2.0,
                    LabelColor);
            }
        }

        private void HideZone(ObZone z)
        {
            if (z.Box != null)
                z.Box.Color = Color.FromArgb(0, Color.White);

            if (z.Label != null)
                z.Label.Color = Color.FromArgb(0, Color.White);
        }

        private void RemoveZoneGraphics(ObZone z)
        {
            Chart.RemoveObject(z.Id);
            Chart.RemoveObject(z.LabelId);
        }

        private int FindBarIndexAtOrBefore(Bars bars, DateTime time)
        {
            int idx = bars.OpenTimes.GetIndexByTime(time);
            if (idx >= 0)
                return idx;

            for (int i = bars.Count - 1; i >= 0; i--)
            {
                if (bars.OpenTimes[i] <= time)
                    return i;
            }

            return -1;
        }

        private string GetTimeFrameLabel(TimeFrame tf)
        {
            if (tf == TimeFrame.Minute) return "1m";
            if (tf == TimeFrame.Minute2) return "2m";
            if (tf == TimeFrame.Minute3) return "3m";
            if (tf == TimeFrame.Minute4) return "4m";
            if (tf == TimeFrame.Minute5) return "5m";
            if (tf == TimeFrame.Minute10) return "10m";
            if (tf == TimeFrame.Minute15) return "15m";
            if (tf == TimeFrame.Minute30) return "30m";
            if (tf == TimeFrame.Minute45) return "45m";
            if (tf == TimeFrame.Hour) return "1H";
            if (tf == TimeFrame.Hour2) return "2H";
            if (tf == TimeFrame.Hour4) return "4H";
            if (tf == TimeFrame.Hour8) return "8H";
            if (tf == TimeFrame.Hour12) return "12H";
            if (tf == TimeFrame.Daily) return "1D";
            if (tf == TimeFrame.Weekly) return "1W";
            if (tf == TimeFrame.Monthly) return "1M";
            return tf.ToString();
        }
    }
}
