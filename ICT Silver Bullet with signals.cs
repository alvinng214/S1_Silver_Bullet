using System;
using System.Collections.Generic;
using cAlgo.API;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.EasternStandardTime, AccessRights = AccessRights.None)]
    public class ICTSilverBulletWithSignals : Indicator
    {
        public enum HtfMinutes
        {
            M1 = 1,
            M3 = 3,
            M5 = 5,
            M10 = 10,
            M15 = 15
        }

        [Parameter("only last x bars", DefaultValue = false, Group = "Settings")]
        public bool LastBarsOnly { get; set; }

        [Parameter("Last Bars", DefaultValue = 3000, MinValue = 10, Group = "Settings")]
        public int LastBarsCount { get; set; }

        [Parameter("Left", DefaultValue = 10, MinValue = 1, MaxValue = 20, Group = "Swings settings (left-right)")]
        public int Left { get; set; }

        [Parameter("Right", DefaultValue = 1, MinValue = 1, MaxValue = 5, Group = "Swings settings (left-right)")]
        public int Right { get; set; }

        [Parameter("HTF (minutes)", DefaultValue = HtfMinutes.M15, Group = "FVG")]
        public HtfMinutes OHTF { get; set; }

        [Parameter("remove broken FVG's", DefaultValue = true, Group = "FVG")]
        public bool RemoveBrokenFvg { get; set; }

        [Parameter("~ trend", DefaultValue = true, Group = "FVG")]
        public bool FilterByTrend { get; set; }

        [Parameter("extend", DefaultValue = true, Group = "FVG")]
        public bool ExtendFvg { get; set; }

        [Parameter("Bull FVG Color", DefaultValue = "#4DD0E145", Group = "FVG")]
        public Color BullFvgColor { get; set; }

        [Parameter("Bear FVG Color", DefaultValue = "#FFC1B140", Group = "FVG")]
        public Color BearFvgColor { get; set; }

        [Parameter("Extend Target-lines to their source", DefaultValue = false, Group = "Targets - Support/Resistance")]
        public bool ExtendLeft { get; set; }

        [Parameter("Support Line Color", DefaultValue = "#B22833", Group = "Targets - Support/Resistance")]
        public Color SupportColor { get; set; }

        [Parameter("Resistance Line Color", DefaultValue = "#3E89FA", Group = "Targets - Support/Resistance")]
        public Color ResistanceColor { get; set; }

        [Parameter("SB session", DefaultValue = true, Group = "Show")]
        public bool ShowSb { get; set; }

        [Parameter("SB session Color", DefaultValue = "#B2B5BE50", Group = "Show")]
        public Color SessionColor { get; set; }

        [Parameter("Trend", DefaultValue = false, Group = "Show")]
        public bool ShowZz { get; set; }

        [Parameter("HTF Candles", DefaultValue = false, Group = "Show")]
        public bool ShowHtfCandles { get; set; }

        [Parameter("Minimum Trade Framework", DefaultValue = false, Group = "Show")]
        public bool ShowMinFramework { get; set; }

        [Output("Bull FVG Formed", LineColor = "Lime")]
        public IndicatorDataSeries BullFvgFormed { get; set; }

        [Output("Bull FVG Cancel", LineColor = "Red")]
        public IndicatorDataSeries BullFvgCancel { get; set; }

        [Output("Bull FVG Retrace", LineColor = "Aqua")]
        public IndicatorDataSeries BullFvgRetrace { get; set; }

        [Output("Bull Target Reached", LineColor = "Yellow")]
        public IndicatorDataSeries BullTargetReached { get; set; }

        [Output("Bear FVG Formed", LineColor = "Lime")]
        public IndicatorDataSeries BearFvgFormed { get; set; }

        [Output("Bear FVG Cancel", LineColor = "Red")]
        public IndicatorDataSeries BearFvgCancel { get; set; }

        [Output("Bear FVG Retrace", LineColor = "Aqua")]
        public IndicatorDataSeries BearFvgRetrace { get; set; }

        [Output("Bear Target Reached", LineColor = "Yellow")]
        public IndicatorDataSeries BearTargetReached { get; set; }

        private const int MaxSize = 100;

        private readonly List<int> _zzD = new List<int>();
        private readonly List<int> _zzX = new List<int>();
        private readonly List<double> _zzY = new List<double>();

        private readonly List<PivotPoint> _pivH = new List<PivotPoint>();
        private readonly List<PivotPoint> _pivL = new List<PivotPoint>();
        private readonly List<PivotPoint> _pivW = new List<PivotPoint>();
        private readonly List<PivotPoint> _pivD = new List<PivotPoint>();

        private readonly List<FvgZone> _bullFvgs = new List<FvgZone>();
        private readonly List<FvgZone> _bearFvgs = new List<FvgZone>();

        private readonly List<BarPoint> _bpH = new List<BarPoint>();
        private readonly List<BarPoint> _bpL = new List<BarPoint>();

        private readonly List<ChartTrendLine> _sessionLines = new List<ChartTrendLine>();

        private int _trend;
        private bool _prevIsInSb;
        private bool _prevIsPreSb;
        private bool _prevEndSb;

        private double _fridayClose = double.NaN;
        private int _fridayIndex = -1;

        private string _lastSessionName = string.Empty;

        private double _minTf;
        private int _htfMinutes;

        protected override void Initialize()
        {
            _htfMinutes = (int)OHTF;

            for (var i = 0; i < MaxSize; i++)
            {
                _zzD.Add(0);
                _zzX.Add(0);
                _zzY.Add(Bars.HighPrices.Count > 0 ? Bars.HighPrices[0] : 0.0);
            }

            _bpH.Add(new BarPoint { Index = 0, Price = double.NaN });
            _bpL.Add(new BarPoint { Index = 0, Price = double.NaN });
            _bpH.Add(new BarPoint { Index = 0, Price = double.NaN });
            _bpL.Add(new BarPoint { Index = 0, Price = double.NaN });
            _bpH.Add(new BarPoint { Index = 0, Price = double.NaN });
            _bpL.Add(new BarPoint { Index = 0, Price = double.NaN });

            _minTf = Symbol.TickSize * 150;
            if (!SymbolName.ToLowerInvariant().Contains("usd") && !SymbolName.ToLowerInvariant().Contains("eur"))
                _minTf = Symbol.TickSize * 40;
        }

        public override void Calculate(int index)
        {
            ResetOutputs(index);

            if (index < Left + Right + 2)
                return;

            var now = Bars.OpenTimes[index];
            var prev = index > 0 ? Bars.OpenTimes[index - 1] : now;
            var lastBarsOk = !LastBarsOnly || (Bars.Count - 1 - index < LastBarsCount);

            var sbLn = InSession(now, 3, 0, 4, 0);
            var sbAm = InSession(now, 10, 0, 11, 0);
            var sbPm = InSession(now, 14, 0, 15, 0);

            var preLn = InSession(now, 2, 30, 3, 0);
            var preAm = InSession(now, 9, 30, 10, 0);
            var prePm = InSession(now, 13, 30, 14, 0);

            var isInSb = sbLn || sbAm || sbPm;
            var isPreSb = preLn || preAm || prePm;
            var strSbPre = isPreSb && !_prevIsPreSb;
            var strSb = isInSb && !_prevIsInSb;
            var endSb = !isInSb && _prevIsInSb;

            if (strSb && ShowSb)
                DrawSessionMarker(index, sbLn ? "3-4 AM NY" : (sbAm ? "10-11 AM NY" : "2-3 PM NY"));
            if (endSb && ShowSb)
                DrawSessionBoundary(index, "SB End");

            UpdateDailyWeeklyPivots(index, now, prev);
            UpdateSwingAndTrend(index);

            var barMinutes = Bars.TimeFrame.ToString().Contains("Minute") ? Bars.TimeFrame.ToString() : string.Empty;
            var isHtf = _htfMinutes > 1;

            if (strSbPre && isHtf)
            {
                PushBarPoint(_bpH, new BarPoint { Index = index, Price = Bars.HighPrices[index] });
                PushBarPoint(_bpL, new BarPoint { Index = index, Price = Bars.LowPrices[index] });
            }

            if ((isPreSb || isInSb) && lastBarsOk && isHtf)
            {
                _bpH[0] = new BarPoint { Index = _bpH[0].Index, Price = Math.Max(_bpH[0].Price, Bars.HighPrices[index]) };
                _bpL[0] = new BarPoint { Index = _bpL[0].Index, Price = Math.Min(_bpL[0].Price, Bars.LowPrices[index]) };
            }

            if (isInSb && lastBarsOk)
            {
                var allowFvg = true;
                if (isHtf)
                    allowFvg = _bpH.Count > 2 && now.Minute % _htfMinutes == 0;

                if (allowFvg)
                    TryCreateFvgs(index, isHtf);

                UpdateFvgState(index, isHtf);
            }

            UpdateTargets(index);

            if ((isPreSb || isInSb) && lastBarsOk && isHtf && now.Minute % _htfMinutes == 0)
            {
                PushBarPoint(_bpH, new BarPoint { Index = index, Price = Bars.HighPrices[index] });
                PushBarPoint(_bpL, new BarPoint { Index = index, Price = Bars.LowPrices[index] });
            }

            if (endSb && lastBarsOk)
                CleanupEndOfSession(index);

            if (_prevEndSb && lastBarsOk)
            {
                foreach (var f in _bullFvgs)
                {
                    f.Active = false;
                    f.Current = false;
                }

                foreach (var f in _bearFvgs)
                {
                    f.Active = false;
                    f.Current = false;
                }
            }

            if (Bars.TimeFrame.ToString().Contains("Minute") && ExtractMinutes(Bars.TimeFrame.ToString()) > 15)
                Chart.DrawStaticText("sb_tf_warn", "Please use a timeframe <= 15 minutes", VerticalAlignment.Top, HorizontalAlignment.Right, Color.Red);

            _prevEndSb = endSb;
            _prevIsInSb = isInSb;
            _prevIsPreSb = isPreSb;
        }

        private void TryCreateFvgs(int index, bool isHtf)
        {
            var hi = isHtf ? _bpH[0].Price : Bars.HighPrices[index];
            var lo = isHtf ? _bpL[0].Price : Bars.LowPrices[index];
            var hi2 = isHtf ? _bpH[2].Price : Bars.HighPrices[index - 2];
            var lo2 = isHtf ? _bpL[2].Price : Bars.LowPrices[index - 2];
            var ix = isHtf ? _bpH[0].Index : index;
            var ix2 = isHtf ? _bpH[2].Index : index - 2;

            if (double.IsNaN(hi) || double.IsNaN(lo) || double.IsNaN(hi2) || double.IsNaN(lo2))
                return;

            if (hi < lo2 && (!FilterByTrend || _trend == -1))
            {
                var fvg = CreateFvg(ix2, ix, lo2, hi, false, index);
                _bearFvgs.Insert(0, fvg);
                BearFvgFormed[index] = 1;
            }

            if (lo > hi2 && (!FilterByTrend || _trend == 1))
            {
                var fvg = CreateFvg(ix2, ix, hi2, lo, true, index);
                _bullFvgs.Insert(0, fvg);
                BullFvgFormed[index] = 1;
            }
        }

        private FvgZone CreateFvg(int startIndex, int endIndex, double top, double bottom, bool isBull, int index)
        {
            var id = (isBull ? "bull_fvg_" : "bear_fvg_") + startIndex + "_" + endIndex;
            var rect = Chart.DrawRectangle(id, startIndex, top, endIndex, bottom, isBull ? BullFvgColor : BearFvgColor);
            rect.IsFilled = true;
            rect.Color = isBull ? BullFvgColor : BearFvgColor;

            return new FvgZone
            {
                StartIndex = startIndex,
                EndIndex = endIndex,
                Top = top,
                Bottom = bottom,
                IsBull = isBull,
                Active = false,
                Broken = false,
                Current = true,
                Box = rect,
                Id = id
            };
        }

        private void UpdateFvgState(int index, bool isHtf)
        {
            for (var i = _bullFvgs.Count - 1; i >= 0; i--)
            {
                var fvg = _bullFvgs[i];
                if (!fvg.Current)
                    continue;

                fvg.EndIndex = index;
                if (ExtendFvg && fvg.Box != null)
                    fvg.Box.Time2 = Bars.OpenTimes[index];

                if (Bars.ClosePrices[index] < fvg.Bottom)
                {
                    fvg.Broken = true;
                    BullFvgCancel[index] = 1;
                    if (RemoveBrokenFvg)
                    {
                        RemoveFvgVisuals(fvg);
                        _bullFvgs.RemoveAt(i);
                    }
                }
                else if (!fvg.Active && Bars.LowPrices[index] < fvg.Top)
                {
                    fvg.Active = true;
                    BullFvgRetrace[index] = 1;
                    var diff = Bars.ClosePrices[index] + _minTf;
                    CreateTargets(fvg, index, true, diff);
                    if (ShowMinFramework)
                        fvg.FrameworkLine = Chart.DrawTrendLine("mtfw_b_" + index, index - 1, diff, index, diff, Color.Yellow);
                }
            }

            for (var i = _bearFvgs.Count - 1; i >= 0; i--)
            {
                var fvg = _bearFvgs[i];
                if (!fvg.Current)
                    continue;

                fvg.EndIndex = index;
                if (ExtendFvg && fvg.Box != null)
                    fvg.Box.Time2 = Bars.OpenTimes[index];

                if (Bars.ClosePrices[index] > fvg.Top)
                {
                    fvg.Broken = true;
                    BearFvgCancel[index] = 1;
                    if (RemoveBrokenFvg)
                    {
                        RemoveFvgVisuals(fvg);
                        _bearFvgs.RemoveAt(i);
                    }
                }
                else if (!fvg.Active && Bars.HighPrices[index] > fvg.Bottom)
                {
                    fvg.Active = true;
                    BearFvgRetrace[index] = 1;
                    var diff = Bars.ClosePrices[index] - _minTf;
                    CreateTargets(fvg, index, false, diff);
                    if (ShowMinFramework)
                        fvg.FrameworkLine = Chart.DrawTrendLine("mtfw_s_" + index, index - 1, diff, index, diff, Color.Yellow);
                }
            }
        }

        private void UpdateTargets(int index)
        {
            foreach (var fvg in _bullFvgs)
            {
                for (var t = 0; t < fvg.Targets.Count; t++)
                {
                    var target = fvg.Targets[t];
                    if (!target.Active)
                        continue;

                    if (target.Line != null)
                        target.Line.Time2 = Bars.OpenTimes[index];

                    if (Bars.HighPrices[index] > target.Price)
                    {
                        target.Active = false;
                        BullTargetReached[index] = 1;
                    }
                }
            }

            foreach (var fvg in _bearFvgs)
            {
                for (var t = 0; t < fvg.Targets.Count; t++)
                {
                    var target = fvg.Targets[t];
                    if (!target.Active)
                        continue;

                    if (target.Line != null)
                        target.Line.Time2 = Bars.OpenTimes[index];

                    if (Bars.LowPrices[index] < target.Price)
                    {
                        target.Active = false;
                        BearTargetReached[index] = 1;
                    }
                }
            }
        }

        private void CreateTargets(FvgZone fvg, int index, bool bullish, double diff)
        {
            if (bullish)
            {
                AddTargetsFromPivots(fvg, _pivH, index, diff, ResistanceColor);
                AddTargetsFromPivots(fvg, _pivW, index, diff, ResistanceColor);
                AddTargetsFromPivots(fvg, _pivD, index, diff, ResistanceColor);
            }
            else
            {
                AddTargetsFromPivots(fvg, _pivL, index, diff, SupportColor);
                AddTargetsFromPivots(fvg, _pivW, index, diff, SupportColor);
                AddTargetsFromPivots(fvg, _pivD, index, diff, SupportColor);
            }
        }

        private void AddTargetsFromPivots(FvgZone fvg, List<PivotPoint> pivots, int index, double diff, Color color)
        {
            foreach (var p in pivots)
            {
                if (p.Index >= fvg.StartIndex)
                    continue;

                if (fvg.IsBull && p.Source == "swing" && p.Kind != "high")
                    continue;
                if (!fvg.IsBull && p.Source == "swing" && p.Kind != "low")
                    continue;

                if (fvg.IsBull && p.Price <= diff)
                    continue;
                if (!fvg.IsBull && p.Price >= diff)
                    continue;

                if (index - p.Index >= 4500)
                    continue;

                var broken = false;
                for (var i = Math.Max(p.Index, 1); i < index; i++)
                {
                    var bodyHigh = Math.Max(Bars.OpenPrices[i], Bars.ClosePrices[i]);
                    var bodyLow = Math.Min(Bars.OpenPrices[i], Bars.ClosePrices[i]);
                    if (bodyHigh > p.Price && bodyLow < p.Price)
                    {
                        broken = true;
                        break;
                    }
                }

                if (broken)
                    continue;

                var x1 = ExtendLeft ? p.Index : index;
                var line = Chart.DrawTrendLine($"tar_{fvg.Id}_{p.Index}", x1, p.Price, index, p.Price, color, 1, LineStyle.DotsRare);
                fvg.Targets.Add(new TargetLevel { Active = true, Price = p.Price, Line = line });
            }
        }

        private void CleanupEndOfSession(int index)
        {
            for (var i = _bullFvgs.Count - 1; i >= 0; i--)
            {
                var fvg = _bullFvgs[i];
                if (!fvg.Current)
                    continue;

                if (!fvg.Active || (RemoveBrokenFvg && fvg.Broken) || Bars.ClosePrices[index] < fvg.Bottom)
                {
                    BullFvgCancel[index] = 1;
                    RemoveFvgVisuals(fvg);
                    _bullFvgs.RemoveAt(i);
                }
            }

            for (var i = _bearFvgs.Count - 1; i >= 0; i--)
            {
                var fvg = _bearFvgs[i];
                if (!fvg.Current)
                    continue;

                if (!fvg.Active || (RemoveBrokenFvg && fvg.Broken) || Bars.ClosePrices[index] > fvg.Top)
                {
                    BearFvgCancel[index] = 1;
                    RemoveFvgVisuals(fvg);
                    _bearFvgs.RemoveAt(i);
                }
            }

            _bpH.Clear();
            _bpL.Clear();
            _bpH.Add(new BarPoint { Index = index, Price = double.NaN });
            _bpL.Add(new BarPoint { Index = index, Price = double.NaN });
            _bpH.Add(new BarPoint { Index = index, Price = double.NaN });
            _bpL.Add(new BarPoint { Index = index, Price = double.NaN });
            _bpH.Add(new BarPoint { Index = index, Price = double.NaN });
            _bpL.Add(new BarPoint { Index = index, Price = double.NaN });
        }

        private void UpdateSwingAndTrend(int index)
        {
            var pivotIndex = index - Right;
            if (pivotIndex <= Left)
                return;

            var ph = IsPivotHigh(pivotIndex);
            var pl = IsPivotLow(pivotIndex);

            if (ph)
            {
                PushPivot(_pivH, new PivotPoint { Index = pivotIndex, Price = Bars.HighPrices[pivotIndex], Kind = "high", Source = "swing" });
                var dir = _zzD[0];
                var x1 = _zzX[0];
                var y1 = _zzY[0];
                var y2 = Bars.HighPrices[pivotIndex];

                if (y2 > y1)
                {
                    if (dir < 1)
                        ZzInOut(1, x1, y1, pivotIndex, y2);
                    else if (dir == 1)
                    {
                        _zzX[0] = pivotIndex;
                        _zzY[0] = y2;
                    }
                }
            }

            if (pl)
            {
                PushPivot(_pivL, new PivotPoint { Index = pivotIndex, Price = Bars.LowPrices[pivotIndex], Kind = "low", Source = "swing" });
                var dir = _zzD[0];
                var x1 = _zzX[0];
                var y1 = _zzY[0];
                var y2 = Bars.LowPrices[pivotIndex];

                if (y2 < y1)
                {
                    if (dir > -1)
                        ZzInOut(-1, x1, y1, pivotIndex, y2);
                    else if (dir == -1)
                    {
                        _zzX[0] = pivotIndex;
                        _zzY[0] = y2;
                    }
                }
            }

            var iH = _zzD[2] == 1 ? 2 : 1;
            var iL = _zzD[2] == -1 ? 2 : 1;

            if (Bars.ClosePrices[index] > _zzY[iH] && _zzD[iH] == 1 && _trend < 1)
                _trend = 1;

            if (Bars.ClosePrices[index] < _zzY[iL] && _zzD[iL] == -1 && _trend > -1)
                _trend = -1;
        }

        private void UpdateDailyWeeklyPivots(int index, DateTime now, DateTime prev)
        {
            if (now.DayOfWeek == DayOfWeek.Friday)
            {
                _fridayClose = Bars.ClosePrices[index];
                _fridayIndex = index;
            }

            if (index > 0 && prev.Date != now.Date)
            {
                if (now.DayOfWeek == DayOfWeek.Monday && !double.IsNaN(_fridayClose) && _fridayIndex >= 0)
                {
                    PushPivot(_pivW, new PivotPoint { Index = index, Price = Bars.OpenPrices[index], Kind = "any", Source = "weekly" });
                    PushPivot(_pivW, new PivotPoint { Index = _fridayIndex, Price = _fridayClose, Kind = "any", Source = "weekly" });
                }

                PushPivot(_pivD, new PivotPoint { Index = index, Price = Bars.OpenPrices[index], Kind = "any", Source = "daily" });
                PushPivot(_pivD, new PivotPoint { Index = index - 1, Price = Bars.ClosePrices[index - 1], Kind = "any", Source = "daily" });
            }
        }

        private bool IsPivotHigh(int i)
        {
            var p = Bars.HighPrices[i];
            for (var j = i - Left; j <= i + Right; j++)
            {
                if (j == i || j < 0 || j >= Bars.Count)
                    continue;
                if (Bars.HighPrices[j] >= p)
                    return false;
            }
            return true;
        }

        private bool IsPivotLow(int i)
        {
            var p = Bars.LowPrices[i];
            for (var j = i - Left; j <= i + Right; j++)
            {
                if (j == i || j < 0 || j >= Bars.Count)
                    continue;
                if (Bars.LowPrices[j] <= p)
                    return false;
            }
            return true;
        }

        private void ZzInOut(int d, int x1, double y1, int x2, double y2)
        {
            _zzD.Insert(0, d);
            _zzX.Insert(0, x2);
            _zzY.Insert(0, y2);
            if (_zzD.Count > MaxSize) _zzD.RemoveAt(_zzD.Count - 1);
            if (_zzX.Count > MaxSize) _zzX.RemoveAt(_zzX.Count - 1);
            if (_zzY.Count > MaxSize) _zzY.RemoveAt(_zzY.Count - 1);

            if (ShowZz)
                Chart.DrawTrendLine("zz_" + x2, x1, y1, x2, y2, Color.FromArgb(120, Color.Blue));
        }

        private void DrawSessionMarker(int index, string label)
        {
            DrawSessionBoundary(index, label);
            Chart.DrawText("sb_label_" + index, label, index, Bars.HighPrices[index] + Symbol.TickSize * 20, SessionColor);
        }

        private void DrawSessionBoundary(int index, string text)
        {
            var line = Chart.DrawTrendLine("sb_line_" + index, index, Bars.LowPrices[index], index, Bars.HighPrices[index], SessionColor, 2, LineStyle.Solid);
            _sessionLines.Add(line);
            _lastSessionName = text;
        }

        private static bool InSession(DateTime time, int sh, int sm, int eh, int em)
        {
            var t = time.TimeOfDay;
            var start = new TimeSpan(sh, sm, 0);
            var end = new TimeSpan(eh, em, 0);
            return t >= start && t < end;
        }

        private static int ExtractMinutes(string tf)
        {
            var digits = string.Empty;
            foreach (var c in tf)
                if (char.IsDigit(c)) digits += c;
            return int.TryParse(digits, out var m) ? m : 1;
        }

        private static void PushPivot(List<PivotPoint> list, PivotPoint p)
        {
            list.Insert(0, p);
            while (list.Count > MaxSize)
                list.RemoveAt(list.Count - 1);
        }

        private static void PushBarPoint(List<BarPoint> list, BarPoint bp)
        {
            list.Insert(0, bp);
            while (list.Count > 3)
                list.RemoveAt(list.Count - 1);
        }

        private void RemoveFvgVisuals(FvgZone fvg)
        {
            if (fvg.Box != null)
                Chart.RemoveObject(fvg.Id);
            if (fvg.FrameworkLine != null)
                Chart.RemoveObject(fvg.FrameworkLine.Name);

            foreach (var t in fvg.Targets)
            {
                if (t.Line != null)
                    Chart.RemoveObject(t.Line.Name);
            }
        }

        private void ResetOutputs(int index)
        {
            BullFvgFormed[index] = 0;
            BullFvgCancel[index] = 0;
            BullFvgRetrace[index] = 0;
            BullTargetReached[index] = 0;
            BearFvgFormed[index] = 0;
            BearFvgCancel[index] = 0;
            BearFvgRetrace[index] = 0;
            BearTargetReached[index] = 0;
        }

        private sealed class PivotPoint
        {
            public int Index;
            public double Price;
            public string Kind;
            public string Source;
        }

        private sealed class BarPoint
        {
            public int Index;
            public double Price;
        }

        private sealed class TargetLevel
        {
            public double Price;
            public bool Active;
            public ChartTrendLine Line;
        }

        private sealed class FvgZone
        {
            public string Id;
            public int StartIndex;
            public int EndIndex;
            public double Top;
            public double Bottom;
            public bool IsBull;
            public bool Active;
            public bool Broken;
            public bool Current;
            public ChartRectangle Box;
            public ChartTrendLine FrameworkLine;
            public readonly List<TargetLevel> Targets = new List<TargetLevel>();
        }
    }
}
