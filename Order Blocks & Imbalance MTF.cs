using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;
using cAlgo.API.Indicators;

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

        private sealed class ObZone
        {
            public string Id;
            public ChartRectangle Box;
            public ChartText Label;
            public string LabelId;
            public double Top;
            public double Bottom;
            public bool IsBullish;
            public bool Mitigated;
            public DateTime CreatedTime;
        }

        [Parameter("Use Current Timeframe", DefaultValue = true, Group = "Logic Settings")]
        public bool UseCurrentTimeframe { get; set; }

        [Parameter("Timeframe", DefaultValue = "Hour", Group = "Logic Settings")]
        public TimeFrame TfInput { get; set; }

        [Parameter("Min FVG Size (ATR Multiplier)", DefaultValue = 0.5, Step = 0.1, Group = "Logic Settings")]
        public double FvgThreshold { get; set; }

        [Parameter("Mitigation Method", DefaultValue = MitigationMethod.Wick, Group = "Logic Settings")]
        public MitigationMethod MitigationTypeInput { get; set; }

        [Parameter("Show Demand (Bullish)", DefaultValue = true, Group = "Logic Settings")]
        public bool ShowBull { get; set; }

        [Parameter("Show Supply (Bearish)", DefaultValue = true, Group = "Logic Settings")]
        public bool ShowBear { get; set; }

        [Parameter("Enable Smart Visibility", DefaultValue = true, Group = "Smart Visibility")]
        public bool UseSmartView { get; set; }

        [Parameter("Max Zones to Show (Per Side)", DefaultValue = 10, MinValue = 1, MaxValue = 20, Group = "Smart Visibility")]
        public int VisibleLimit { get; set; }

        [Parameter("Auto-Extend Visible Zones", DefaultValue = true, Group = "Smart Visibility")]
        public bool ExtendActive { get; set; }

        private readonly List<ObZone> _obArray = new List<ObZone>();
        private DateTime _lastCreatedTime = DateTime.MinValue;
        private int _idCounter;

        private Bars _htfBars;
        private AverageTrueRange _htfAtr;

        protected override void Initialize()
        {
            var tf = UseCurrentTimeframe ? Bars.TimeFrame : TfInput;
            _htfBars = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf);
            _htfAtr = Indicators.AverageTrueRange(_htfBars, 14, MovingAverageType.Simple);
        }

        public override void Calculate(int index)
        {
            if (index < 2)
                return;

            var now = Bars.OpenTimes[index];
            var htfIndex = FindBarIndexAtOrBefore(_htfBars, now);
            if (htfIndex >= 2)
            {
                var htfHigh2 = _htfBars.HighPrices[htfIndex - 2];
                var htfLow2 = _htfBars.LowPrices[htfIndex - 2];
                var htfLow0 = _htfBars.LowPrices[htfIndex];
                var htfHigh0 = _htfBars.HighPrices[htfIndex];
                var htfTime2 = _htfBars.OpenTimes[htfIndex - 2];

                var atrRef = _htfAtr.Result[htfIndex - 1];
                var bullFvgSize = htfLow0 - htfHigh2;
                var bearFvgSize = htfLow2 - htfHigh0;

                var htfIsBull = bullFvgSize > (atrRef * FvgThreshold);
                var htfIsBear = bearFvgSize > (atrRef * FvgThreshold);

                if (htfTime2 != _lastCreatedTime)
                {
                    if (ShowBull && htfIsBull)
                    {
                        CreateZone(true, htfHigh2, htfLow2, htfTime2, now, index);
                        _lastCreatedTime = htfTime2;
                    }
                    else if (ShowBear && htfIsBear)
                    {
                        CreateZone(false, htfHigh2, htfLow2, htfTime2, now, index);
                        _lastCreatedTime = htfTime2;
                    }
                }
            }

            UpdateMitigationAndRemoval(index);

            if (_obArray.Count > 450)
            {
                var oldest = _obArray[0];
                DeleteZone(oldest);
                _obArray.RemoveAt(0);
            }

            if (index != Bars.Count - 1)
                return;

            var dt = index > 0 ? (Bars.OpenTimes[index] - Bars.OpenTimes[index - 1]) : TimeSpan.FromMinutes(1);
            var future10 = Bars.OpenTimes[index].AddTicks(dt.Ticks * 10);
            var future5 = Bars.OpenTimes[index].AddTicks(dt.Ticks * 5);

            if (UseSmartView)
                ApplySmartVisibility(future10, index);
            else
                ExtendAllVisible(future5);
        }

        private void CreateZone(bool isBull, double top, double bottom, DateTime leftTime, DateTime now, int index)
        {
            var id = $"ob_mtf_{_idCounter++}";
            var rightTime = now.AddMinutes(1);
            var baseColor = isBull ? Color.Green : Color.Red;

            var rect = Chart.DrawRectangle(id, leftTime, top, rightTime, bottom, baseColor, 1, LineStyle.Solid);
            rect.IsFilled = true;
            rect.Color = Color.FromArgb(70, baseColor);

            var labelId = id + "_txt";
            var label = Chart.DrawText(labelId, isBull ? "OB Demand" : "OB Supply", rightTime, (top + bottom) / 2.0, Color.White);

            var zone = new ObZone
            {
                Id = id,
                Box = rect,
                Label = label,
                LabelId = labelId,
                Top = top,
                Bottom = bottom,
                IsBullish = isBull,
                Mitigated = false,
                CreatedTime = leftTime
            };

            if (UseSmartView)
                ToggleZone(zone, false);

            _obArray.Add(zone);
        }

        private void ToggleZone(ObZone zone, bool visible)
        {
            var baseCol = zone.IsBullish ? Color.Green : Color.Red;
            if (visible)
            {
                if (zone.Mitigated)
                {
                    zone.Box.Color = Color.FromArgb(90, baseCol);
                    zone.Box.BorderColor = baseCol;
                    zone.Box.LineStyle = LineStyle.DotsRare;
                    if (zone.Label != null)
                        zone.Label.Color = Color.White;
                }
                else
                {
                    zone.Box.Color = Color.FromArgb(70, baseCol);
                    zone.Box.BorderColor = baseCol;
                    zone.Box.LineStyle = LineStyle.Solid;
                    if (zone.Label != null)
                        zone.Label.Color = Color.White;
                }
            }
            else
            {
                var hide = Color.FromArgb(0, Color.White);
                zone.Box.Color = hide;
                zone.Box.BorderColor = hide;
                if (zone.Label != null)
                    zone.Label.Color = hide;
            }
        }

        private void UpdateMitigationAndRemoval(int index)
        {
            var close = Bars.ClosePrices[index];
            var low = Bars.LowPrices[index];
            var high = Bars.HighPrices[index];

            for (int i = _obArray.Count - 1; i >= 0; i--)
            {
                var zone = _obArray[i];
                var remove = false;

                if (zone.IsBullish)
                {
                    if (close < zone.Bottom)
                    {
                        remove = true;
                    }
                    else
                    {
                        var mitVal = MitigationTypeInput == MitigationMethod.Wick ? low : close;
                        if (mitVal <= zone.Top && !zone.Mitigated)
                        {
                            zone.Mitigated = true;
                            if (!UseSmartView)
                            {
                                zone.Box.Color = Color.FromArgb(90, Color.Green);
                                zone.Box.LineStyle = LineStyle.DotsRare;
                            }
                            if (zone.Label != null)
                                zone.Label.Text = "Mitigated";
                        }
                    }
                }
                else
                {
                    if (close > zone.Top)
                    {
                        remove = true;
                    }
                    else
                    {
                        var mitVal = MitigationTypeInput == MitigationMethod.Wick ? high : close;
                        if (mitVal >= zone.Bottom && !zone.Mitigated)
                        {
                            zone.Mitigated = true;
                            if (!UseSmartView)
                            {
                                zone.Box.Color = Color.FromArgb(90, Color.Red);
                                zone.Box.LineStyle = LineStyle.DotsRare;
                            }
                            if (zone.Label != null)
                                zone.Label.Text = "Mitigated";
                        }
                    }
                }

                if (remove)
                {
                    DeleteZone(zone);
                    _obArray.RemoveAt(i);
                }
            }
        }

        private void ApplySmartVisibility(DateTime futureTime, int index)
        {
            var bulls = new List<(double key, int idx)>();
            var bears = new List<(double key, int idx)>();

            for (int i = 0; i < _obArray.Count; i++)
            {
                var z = _obArray[i];
                ToggleZone(z, false);
                if (z.IsBullish)
                    bulls.Add((z.Top, i));
                else
                    bears.Add((z.Bottom, i));
            }

            bulls.Sort((a, b) => b.key.CompareTo(a.key));
            bears.Sort((a, b) => a.key.CompareTo(b.key));

            var limitBull = Math.Min(bulls.Count, VisibleLimit);
            for (int r = 0; r < limitBull; r++)
            {
                var z = _obArray[bulls[r].idx];
                ToggleZone(z, true);
                if (ExtendActive)
                {
                    z.Box.Time2 = futureTime;
                    if (z.Label != null)
                        z.Label.Time = futureTime;
                }
            }

            var limitBear = Math.Min(bears.Count, VisibleLimit);
            for (int r = 0; r < limitBear; r++)
            {
                var z = _obArray[bears[r].idx];
                ToggleZone(z, true);
                if (ExtendActive)
                {
                    z.Box.Time2 = futureTime;
                    if (z.Label != null)
                        z.Label.Time = futureTime;
                }
            }
        }

        private void ExtendAllVisible(DateTime futureTime)
        {
            for (int i = 0; i < _obArray.Count; i++)
            {
                var z = _obArray[i];
                z.Box.Time2 = futureTime;
                if (z.Label != null)
                    z.Label.Time = futureTime;
            }
        }

        private void DeleteZone(ObZone zone)
        {
            Chart.RemoveObject(zone.Id);
            if (zone.Label != null)
                Chart.RemoveObject(zone.LabelId);
        }

        private int FindBarIndexAtOrBefore(Bars bars, DateTime time)
        {
            var idx = bars.OpenTimes.GetIndexByTime(time);
            if (idx >= 0)
                return idx;

            for (int i = bars.Count - 1; i >= 0; i--)
            {
                if (bars.OpenTimes[i] <= time)
                    return i;
            }

            return -1;
        }
    }
}
