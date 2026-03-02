using System;
using System.Collections.Generic;
using System.Linq;
using cAlgo.API;
using cAlgo.API.Indicators;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class ICTUnicornFluxCharts : Indicator
    {
        private const string Prefix = "UNICORN_FLUX_";
        private const int MaxDistanceToLastBar = 3000;
        private const int MaxOrderBlocks = 40;
        private const int ShowLastXFvgs = 20;
        private const int ExtendLastXFvgsCount = 20;

        private sealed class FvgZone
        {
            public int StartIndex;
            public int EndIndex;
            public double Min;
            public double Max;
            public bool IsBull;
            public bool IsInverse;
            public bool Disabled;
            public int LastTouched = -1;
            public int InverseEndIndex = -1;
            public double TotalVolume;
            public double InverseVolume;
        }

        private sealed class OrderBlockZone
        {
            public int StartIndex;
            public int BreakIndex;
            public int EndIndex;
            public double Top;
            public double Bottom;
            public bool IsBull;
            public bool IsBreaker;
            public bool Disabled;
            public double ObVolume;
            public double BbVolume;
        }

        private sealed class UnicornTrade
        {
            public string OverlapDirection;
            public int CreateIndex;
            public int? EntryIndex;
            public int? ExitIndex;
            public double? EntryPrice;
            public double? ExitPrice;
            public double SlTarget;
            public double TpTarget;
            public string EntryType;
            public double RetraceTo;
            public string State = "Awaiting Entry";
            public FvgZone Fvg;
            public OrderBlockZone Ob;
        }

        [Parameter("FVG Detection Sensitivity", Group = "General Configuration", DefaultValue = "Normal")]
        public string FvgSensitivityText { get; set; }

        [Parameter("Swing Length", Group = "General Configuration", DefaultValue = 10, MinValue = 3)]
        public int SwingLength { get; set; }

        [Parameter("Require Retracement", Group = "General Configuration", DefaultValue = false)]
        public bool RequireRetracement { get; set; }

        [Parameter("Show Breaker Blocks", Group = "General Configuration", DefaultValue = true)]
        public bool ShowBB { get; set; }

        [Parameter("FVGs", Group = "General Configuration", DefaultValue = true)]
        public bool ShowFvg { get; set; }

        [Parameter("Enabled", Group = "TP / SL", DefaultValue = true)]
        public bool ShowTpSl { get; set; }

        [Parameter("TP / SL Method", Group = "TP / SL", DefaultValue = "Unicorn")]
        public string TpSlMethod { get; set; }

        [Parameter("Dynamic Risk", Group = "TP / SL", DefaultValue = "Normal")]
        public string RiskAmount { get; set; }

        [Parameter("Fixed Take Profit %", Group = "TP / SL", DefaultValue = 0.3)]
        public double TpPercent { get; set; }

        [Parameter("Fixed Stop Loss %", Group = "TP / SL", DefaultValue = 0.4)]
        public double SlPercent { get; set; }

        [Parameter("Backtesting Dashboard Enabled", Group = "Backtesting Dashboard", DefaultValue = true)]
        public bool DashboardEnabled { get; set; }

        [Parameter("Buy Signal", Group = "Alerts", DefaultValue = true)]
        public bool BuyAlertEnabled { get; set; }

        [Parameter("Sell Signal", Group = "Alerts", DefaultValue = true)]
        public bool SellAlertEnabled { get; set; }

        [Parameter("Take-Profit Signal", Group = "Alerts", DefaultValue = true)]
        public bool TpAlertEnabled { get; set; }

        [Parameter("Stop-Loss Signal", Group = "Alerts", DefaultValue = true)]
        public bool SlAlertEnabled { get; set; }

        [Parameter("Bullish Breaker", Group = "Visuals", DefaultValue = "#BF2962FF")]
        public Color BullishBreakerBlockColor { get; set; }

        [Parameter("Bearish Breaker", Group = "Visuals", DefaultValue = "#BFFFEB3B")]
        public Color BearishBreakerBlockColor { get; set; }

        [Parameter("Buy", Group = "Visuals", DefaultValue = "#80089981")]
        public Color HighColor { get; set; }

        [Parameter("Sell", Group = "Visuals", DefaultValue = "#80F23646")]
        public Color LowColor { get; set; }

        [Parameter("Text", Group = "Visuals", DefaultValue = "#FFFFFFFF")]
        public Color TextColor { get; set; }

        [Parameter("Show Historic Zones", Group = "General Configuration", DefaultValue = true)]
        public bool ShowInvalidated { get; set; }

        [Parameter("IFVG Enabled", Group = "Inversion Fair Value Gaps", DefaultValue = false)]
        public bool IfvgEnabled { get; set; }

        [Parameter("IFVG Zone Invalidation", Group = "Inversion Fair Value Gaps", DefaultValue = "Wick")]
        public string IfvgEndMethod { get; set; }

        [Parameter("Bullish IFVG", Group = "Inversion Fair Value Gaps", DefaultValue = "#80089981")]
        public Color BullishInverseColor { get; set; }

        [Parameter("Bearish IFVG", Group = "Inversion Fair Value Gaps", DefaultValue = "#80F23645")]
        public Color BearishInverseColor { get; set; }

        private readonly List<FvgZone> _fvgs = new List<FvgZone>();
        private readonly List<OrderBlockZone> _obs = new List<OrderBlockZone>();
        private readonly List<UnicornTrade> _unicorns = new List<UnicornTrade>();
        private AverageTrueRange _atr10;
        private AverageTrueRange _atr50;

        protected override void Initialize()
        {
            _atr10 = Indicators.AverageTrueRange(10, MovingAverageType.Simple);
            _atr50 = Indicators.AverageTrueRange(50, MovingAverageType.Simple);
        }

        public override void Calculate(int index)
        {
            if (index != Bars.Count - 1)
                return;

            ClearObjects();
            _fvgs.Clear();
            _obs.Clear();
            _unicorns.Clear();

            int start = Math.Max(2, Bars.Count - MaxDistanceToLastBar);
            for (int i = start; i < Bars.Count - 2; i++)
            {
                DetectFvg(i);
                DetectOrderBlock(i);
                UpdateObBreakers(i);
                UpdateFvgInversions(i);
                DetectUnicornOverlap(i);
                UpdateUnicornState(i);
            }

            RenderLastZonesAndTrades();
            RenderDashboard();
        }

        private void DetectFvg(int i)
        {
            if (!ShowFvg)
                return;

            double sensMult = GetSensitivityMultiplier();
            double minSize = _atr10.Result[i] * sensMult;

            bool bullGap = Bars.LowPrices[i] > Bars.HighPrices[i - 2];
            if (bullGap)
            {
                double min = Bars.HighPrices[i - 2];
                double max = Bars.LowPrices[i];
                if (max - min >= minSize)
                {
                    _fvgs.Add(new FvgZone
                    {
                        StartIndex = i - 2,
                        EndIndex = i,
                        Min = min,
                        Max = max,
                        IsBull = true,
                        TotalVolume = Bars.TickVolumes[i] + Bars.TickVolumes[i - 1] + Bars.TickVolumes[i - 2]
                    });
                }
            }

            bool bearGap = Bars.HighPrices[i] < Bars.LowPrices[i - 2];
            if (bearGap)
            {
                double min = Bars.HighPrices[i];
                double max = Bars.LowPrices[i - 2];
                if (max - min >= minSize)
                {
                    _fvgs.Add(new FvgZone
                    {
                        StartIndex = i - 2,
                        EndIndex = i,
                        Min = min,
                        Max = max,
                        IsBull = false,
                        TotalVolume = Bars.TickVolumes[i] + Bars.TickVolumes[i - 1] + Bars.TickVolumes[i - 2]
                    });
                }
            }
        }

        private void DetectOrderBlock(int i)
        {
            bool pivotHigh = IsPivotHigh(i, SwingLength);
            bool pivotLow = IsPivotLow(i, SwingLength);

            if (pivotLow)
            {
                int trigger = FindBreakAbove(i, Math.Min(Bars.Count - 1, i + SwingLength * 2), Bars.HighPrices[i]);
                if (trigger > i)
                {
                    _obs.Add(new OrderBlockZone
                    {
                        StartIndex = i,
                        EndIndex = i + 100,
                        Top = Math.Max(Bars.OpenPrices[i], Bars.ClosePrices[i]),
                        Bottom = Bars.LowPrices[i],
                        IsBull = true,
                        ObVolume = Bars.TickVolumes[i]
                    });
                }
            }

            if (pivotHigh)
            {
                int trigger = FindBreakBelow(i, Math.Min(Bars.Count - 1, i + SwingLength * 2), Bars.LowPrices[i]);
                if (trigger > i)
                {
                    _obs.Add(new OrderBlockZone
                    {
                        StartIndex = i,
                        EndIndex = i + 100,
                        Top = Bars.HighPrices[i],
                        Bottom = Math.Min(Bars.OpenPrices[i], Bars.ClosePrices[i]),
                        IsBull = false,
                        ObVolume = Bars.TickVolumes[i]
                    });
                }
            }

            if (_obs.Count > MaxOrderBlocks)
                _obs.RemoveRange(0, _obs.Count - MaxOrderBlocks);
        }

        private void UpdateObBreakers(int i)
        {
            foreach (var ob in _obs)
            {
                if (ob.Disabled)
                    continue;
                if (!ob.IsBreaker)
                {
                    if (ob.IsBull && Bars.ClosePrices[i] < ob.Bottom)
                    {
                        ob.IsBreaker = true;
                        ob.BreakIndex = i;
                        ob.BbVolume = Bars.TickVolumes[i];
                    }
                    else if (!ob.IsBull && Bars.ClosePrices[i] > ob.Top)
                    {
                        ob.IsBreaker = true;
                        ob.BreakIndex = i;
                        ob.BbVolume = Bars.TickVolumes[i];
                    }
                }
            }
        }

        private void UpdateFvgInversions(int i)
        {
            foreach (var fvg in _fvgs)
            {
                if (fvg.Disabled)
                    continue;

                bool touched = Bars.HighPrices[i] >= fvg.Min && Bars.LowPrices[i] <= fvg.Max;
                if (touched)
                    fvg.LastTouched = i;

                if (!fvg.IsInverse && IfvgEnabled)
                {
                    if (fvg.IsBull)
                    {
                        bool inv = IfvgEndMethod == "Wick" ? Bars.LowPrices[i] < fvg.Min : Bars.ClosePrices[i] < fvg.Min;
                        if (inv)
                        {
                            fvg.IsInverse = true;
                            fvg.InverseEndIndex = i;
                            fvg.InverseVolume = Bars.TickVolumes[i];
                        }
                    }
                    else
                    {
                        bool inv = IfvgEndMethod == "Wick" ? Bars.HighPrices[i] > fvg.Max : Bars.ClosePrices[i] > fvg.Max;
                        if (inv)
                        {
                            fvg.IsInverse = true;
                            fvg.InverseEndIndex = i;
                            fvg.InverseVolume = Bars.TickVolumes[i];
                        }
                    }
                }
            }
        }

        private void DetectUnicornOverlap(int i)
        {
            var fvgCandidates = _fvgs.Where(f => !f.Disabled && (!f.IsInverse || IfvgEnabled)).ToList();
            var obCandidates = _obs.Where(o => !o.Disabled && (o.IsBreaker || !ShowBB)).ToList();

            foreach (var ob in obCandidates)
            {
                foreach (var fvg in fvgCandidates)
                {
                    string direction = GetOverlapDirection(ob, fvg);
                    if (direction == null)
                        continue;

                    bool exists = _unicorns.Any(u => u.Ob == ob && u.Fvg == fvg);
                    if (exists)
                        continue;

                    double retraceTo = direction == "Bull" ? fvg.Min : fvg.Max;
                    _unicorns.Add(new UnicornTrade
                    {
                        OverlapDirection = direction,
                        CreateIndex = i,
                        RetraceTo = retraceTo,
                        Ob = ob,
                        Fvg = fvg
                    });
                }
            }
        }

        private void UpdateUnicornState(int i)
        {
            foreach (var trade in _unicorns)
            {
                if (trade.State == "Awaiting Entry")
                {
                    if (i <= trade.CreateIndex + 1)
                        continue;

                    bool retraceCond = !RequireRetracement || (trade.OverlapDirection == "Bull" ? Bars.LowPrices[i] <= trade.RetraceTo : Bars.HighPrices[i] >= trade.RetraceTo);
                    if (!retraceCond)
                        continue;

                    trade.EntryIndex = i;
                    trade.EntryPrice = Bars.ClosePrices[i];
                    trade.EntryType = trade.OverlapDirection == "Bull" ? "Long" : "Short";
                    ConfigureTargets(trade, i);
                    trade.State = "Entry Taken";

                    if (trade.EntryType == "Long" && BuyAlertEnabled)
                        Print("Buy Signal");
                    if (trade.EntryType == "Short" && SellAlertEnabled)
                        Print("Sell Signal");
                }
                else if (trade.State == "Entry Taken" && trade.EntryIndex.HasValue && i > trade.EntryIndex.Value)
                {
                    if (trade.EntryType == "Long")
                    {
                        if (Bars.HighPrices[i] >= trade.TpTarget)
                        {
                            trade.ExitIndex = i;
                            trade.ExitPrice = trade.TpTarget;
                            trade.State = "TP Hit";
                            if (TpAlertEnabled) Print("Take-Profit Signal");
                        }
                        else if (Bars.LowPrices[i] <= trade.SlTarget)
                        {
                            trade.ExitIndex = i;
                            trade.ExitPrice = trade.SlTarget;
                            trade.State = "SL Hit";
                            if (SlAlertEnabled) Print("Stop-Loss Signal");
                        }
                    }
                    else
                    {
                        if (Bars.LowPrices[i] <= trade.TpTarget)
                        {
                            trade.ExitIndex = i;
                            trade.ExitPrice = trade.TpTarget;
                            trade.State = "TP Hit";
                            if (TpAlertEnabled) Print("Take-Profit Signal");
                        }
                        else if (Bars.HighPrices[i] >= trade.SlTarget)
                        {
                            trade.ExitIndex = i;
                            trade.ExitPrice = trade.SlTarget;
                            trade.State = "SL Hit";
                            if (SlAlertEnabled) Print("Stop-Loss Signal");
                        }
                    }
                }
            }
        }

        private void ConfigureTargets(UnicornTrade trade, int i)
        {
            double entry = trade.EntryPrice ?? Bars.ClosePrices[i];
            double atr = _atr50.Result[i];
            double slAtrMult = ResolveRiskAtrMultiplier();
            const double unicornRR = 0.57;
            const double dynamicRR = 0.86;
            const double unicornSlOffset = 4.75;

            if (TpSlMethod == "Fixed")
            {
                if (trade.EntryType == "Long")
                {
                    trade.SlTarget = entry * (1 - SlPercent / 100.0);
                    trade.TpTarget = entry * (1 + TpPercent / 100.0);
                }
                else
                {
                    trade.SlTarget = entry * (1 + SlPercent / 100.0);
                    trade.TpTarget = entry * (1 - TpPercent / 100.0);
                }
            }
            else if (TpSlMethod == "Dynamic")
            {
                if (trade.EntryType == "Long")
                {
                    trade.SlTarget = entry - atr * slAtrMult;
                    trade.TpTarget = entry + Math.Abs(entry - trade.SlTarget) * dynamicRR;
                }
                else
                {
                    trade.SlTarget = entry + atr * slAtrMult;
                    trade.TpTarget = entry - Math.Abs(entry - trade.SlTarget) * dynamicRR;
                }
            }
            else
            {
                double localLow = Bars.LowPrices[Math.Max(0, i - 5)];
                double localHigh = Bars.HighPrices[Math.Max(0, i - 5)];
                if (trade.EntryType == "Long")
                {
                    trade.SlTarget = localLow - atr * unicornSlOffset;
                    trade.TpTarget = entry + Math.Abs(entry - trade.SlTarget) * unicornRR;
                }
                else
                {
                    trade.SlTarget = localHigh + atr * unicornSlOffset;
                    trade.TpTarget = entry - Math.Abs(entry - trade.SlTarget) * unicornRR;
                }
            }
        }

        private void RenderLastZonesAndTrades()
        {
            int fvgStart = Math.Max(0, _fvgs.Count - ShowLastXFvgs);
            for (int i = fvgStart; i < _fvgs.Count; i++)
            {
                var fvg = _fvgs[i];
                RenderFvg(fvg, i >= _fvgs.Count - ExtendLastXFvgsCount);
            }

            foreach (var ob in _obs)
                RenderOrderBlock(ob);

            foreach (var u in _unicorns)
                RenderUnicorn(u);
        }

        private void RenderFvg(FvgZone fvg, bool extend)
        {
            if (!ShowInvalidated && fvg.Disabled)
                return;

            int endIndex = extend ? Bars.Count - 1 : Math.Min(Bars.Count - 1, fvg.EndIndex + 30);
            Color c = fvg.IsBull ? HighColor : LowColor;
            string name = Prefix + "FVG_" + fvg.StartIndex;
            var rect = Chart.DrawRectangle(name, Bars.OpenTimes[fvg.StartIndex], fvg.Max, Bars.OpenTimes[endIndex], fvg.Min, c);
            rect.IsFilled = true;

            if (fvg.IsInverse && IfvgEnabled)
            {
                Color ic = fvg.IsBull ? BearishInverseColor : BullishInverseColor;
                string iName = Prefix + "IFVG_" + fvg.StartIndex;
                var iRect = Chart.DrawRectangle(iName, Bars.OpenTimes[Math.Max(fvg.InverseEndIndex, fvg.StartIndex)], fvg.Max, Bars.OpenTimes[endIndex], fvg.Min, ic);
                iRect.IsFilled = true;
            }
        }

        private void RenderOrderBlock(OrderBlockZone ob)
        {
            if (!ShowInvalidated && ob.Disabled)
                return;

            int endIndex = Math.Min(Bars.Count - 1, ob.EndIndex);
            var color = ob.IsBreaker ? (ob.IsBull ? BearishBreakerBlockColor : BullishBreakerBlockColor) : (ob.IsBull ? HighColor : LowColor);
            string nm = Prefix + "OB_" + ob.StartIndex + (ob.IsBreaker ? "_BR" : "");
            var rect = Chart.DrawRectangle(nm, Bars.OpenTimes[ob.StartIndex], ob.Top, Bars.OpenTimes[endIndex], ob.Bottom, color);
            rect.IsFilled = true;
        }

        private void RenderUnicorn(UnicornTrade u)
        {
            if (!u.EntryIndex.HasValue || !u.EntryPrice.HasValue)
                return;

            int ei = u.EntryIndex.Value;
            string type = u.EntryType == "Long" ? "Buy" : "Sell";
            Chart.DrawText(Prefix + "ENTRY_" + ei, type, Bars.OpenTimes[ei], u.EntryType == "Long" ? Bars.LowPrices[ei] : Bars.HighPrices[ei], TextColor);

            if (ShowTpSl)
            {
                int end = u.ExitIndex ?? Math.Min(Bars.Count - 1, ei + 100);
                var tpColor = HighColor;
                var slColor = LowColor;
                Chart.DrawTrendLine(Prefix + "TP_" + ei, Bars.OpenTimes[ei], u.TpTarget, Bars.OpenTimes[end], u.TpTarget, tpColor, 1, LineStyle.DotsRare);
                Chart.DrawTrendLine(Prefix + "SL_" + ei, Bars.OpenTimes[ei], u.SlTarget, Bars.OpenTimes[end], u.SlTarget, slColor, 1, LineStyle.DotsRare);
                Chart.DrawText(Prefix + "TP_LBL_" + ei, "TP", Bars.OpenTimes[end], u.TpTarget, TextColor);
                Chart.DrawText(Prefix + "SL_LBL_" + ei, "SL", Bars.OpenTimes[end], u.SlTarget, TextColor);
            }
        }

        private void RenderDashboard()
        {
            if (!DashboardEnabled)
                return;

            int total = _unicorns.Count(u => u.EntryPrice.HasValue && u.ExitPrice.HasValue);
            int win = _unicorns.Count(u => u.EntryPrice.HasValue && u.ExitPrice.HasValue &&
                                          ((u.EntryType == "Long" && u.ExitPrice > u.EntryPrice) || (u.EntryType == "Short" && u.ExitPrice < u.EntryPrice)));
            double wr = total == 0 ? 0 : (double)win / total * 100.0;

            double pnl = 0;
            foreach (var u in _unicorns.Where(x => x.EntryPrice.HasValue && x.ExitPrice.HasValue))
            {
                double diffPct = (u.ExitPrice.Value - u.EntryPrice.Value) / u.EntryPrice.Value * 100.0;
                if (u.EntryType == "Short")
                    diffPct = -diffPct;
                pnl += diffPct;
            }

            string txt = $"ICT Unicorn | Flux\nTrades: {total}\nWins: {win}\nWin Rate: {wr:F1}%\nTotal PnL%: {pnl:F2}";
            Chart.DrawStaticText(Prefix + "DASH", txt, VerticalAlignment.Top, HorizontalAlignment.Center, TextColor);
        }

        private string GetOverlapDirection(OrderBlockZone ob, FvgZone fvg)
        {
            double top = Math.Min(ob.Top, fvg.Max);
            double bottom = Math.Max(ob.Bottom, fvg.Min);
            if (top <= bottom)
                return null;

            double intersection = top - bottom;
            double union = Math.Max(ob.Top, fvg.Max) - Math.Min(ob.Bottom, fvg.Min);
            double overlapPct = union <= 0 ? 0 : (intersection / union) * 100.0;
            if (overlapPct <= 0)
                return null;

            if (ob.IsBull && fvg.IsBull)
                return "Bull";
            if (!ob.IsBull && !fvg.IsBull)
                return "Bear";
            return null;
        }

        private double GetSensitivityMultiplier()
        {
            switch (FvgSensitivityText)
            {
                case "Extreme": return 0.02;
                case "High": return 0.05;
                case "Normal": return 0.08;
                case "Low": return 0.12;
                default: return 0.08;
            }
        }

        private double ResolveRiskAtrMultiplier()
        {
            switch (RiskAmount)
            {
                case "Highest": return 9.5;
                case "High": return 6;
                case "Normal": return 5;
                case "Low": return 4;
                case "Lowest": return 1.5;
                default: return 5;
            }
        }

        private bool IsPivotHigh(int i, int len)
        {
            if (i - len < 0 || i + len >= Bars.Count)
                return false;
            double pivot = Bars.HighPrices[i];
            for (int j = i - len; j <= i + len; j++)
            {
                if (j == i)
                    continue;
                if (Bars.HighPrices[j] >= pivot)
                    return false;
            }
            return true;
        }

        private bool IsPivotLow(int i, int len)
        {
            if (i - len < 0 || i + len >= Bars.Count)
                return false;
            double pivot = Bars.LowPrices[i];
            for (int j = i - len; j <= i + len; j++)
            {
                if (j == i)
                    continue;
                if (Bars.LowPrices[j] <= pivot)
                    return false;
            }
            return true;
        }

        private int FindBreakAbove(int from, int to, double level)
        {
            for (int i = from + 1; i <= to; i++)
                if (Bars.ClosePrices[i] > level)
                    return i;
            return -1;
        }

        private int FindBreakBelow(int from, int to, double level)
        {
            for (int i = from + 1; i <= to; i++)
                if (Bars.ClosePrices[i] < level)
                    return i;
            return -1;
        }

        private void ClearObjects()
        {
            var names = new List<string>();
            foreach (var o in Chart.Objects)
            {
                if (o.Name.StartsWith(Prefix, StringComparison.Ordinal))
                    names.Add(o.Name);
            }
            foreach (var n in names)
                Chart.RemoveObject(n);
        }
    }
}
