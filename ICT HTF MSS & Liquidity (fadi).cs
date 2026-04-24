using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;

// ─────────────────────────────────────────────────────────────────────────────
// ICT HTF MSS & Liquidity (fadi) — cTrader / cAlgo port
//
// Faithful C# port of the Pine v6 indicator by @fadizeidan:
//   ICT HTF MSS & Liquidity (fadi).txt
//
// Direct mapping of every user-visible behaviour:
//   - HTF pivot detection using a 3-bar middle-bar rule with equal-highs /
//     equal-lows skip-back (Pine: SkipEQHigh / SkipEQLow).
//   - Market-structure labelling: STH / STL, promoted to ITH / ITL, then to
//     LTH / LTL via the "first > middle > third" check on the most-recent three
//     pivots of each tier (Pine: FindIT / FindLT, SkipEQPivot preserved
//     bug-for-bug — see notes on SkipEqPivot below).
//   - Liquidity lines drawn ONLY for pivots whose label prefix matches the
//     user's chosen 'Use' level (Pine: AddLiquidity gating).
//   - Sweep ("claimed") + reclaim state on every LTF bar (Pine: CheckClaimed).
//   - Line extension beyond current time by N LTF-bar periods (Pine: extend).
//   - Max-lines cap with FIFO eviction (Pine: lines array + max_lines).
//
// Port decisions (discussed with the user before implementation):
//   Q1 — The Pine's CheckSetup() is defined but never called; omitted here.
//   Q2 — Pine hard-codes red on reclaim; exposed as `Reclaimed Color` input.
//   Q3 — Pine's unicode style labels (⎯⎯⎯/----/····) renamed Solid/Dashed/Dotted.
//   Q4 — File placed alongside `Market Structure MTF Trend [Pt].cs`.
//
// HTF alignment:
//   Pine uses `request.security(..., lookahead_on)` + `high[1]..high[6]`, which
//   at the LTF bar where a new HTF bar has JUST OPENED publishes the
//   just-closed HTF bar as [1]. In cTrader we reproduce this by processing
//   one HTF bar at a time as it closes:
//     * on each LTF `Calculate(index)`, binary-search `_htfBars.OpenTimes`
//       for the HTF bar currently CONTAINING the LTF bar's time;
//     * every HTF bar strictly BEFORE that index is closed and eligible
//       to be processed; we advance `_lastProcessedHtfIndex` through them
//       in chronological order, one FindST pass per closed HTF bar.
//   This avoids the fragile "OpenTimes.LastValue change detection" pattern
//   and matches the behaviour of Market Structure MTF Trend [Pt].cs.
// ─────────────────────────────────────────────────────────────────────────────
namespace cAlgo
{
    // ── UI enums ──────────────────────────────────────────────────────────────
    public enum FadiLineStyleOption { Solid, Dashed, Dotted }
    public enum FadiLevelOption { ShortTerm, IntermediateTerm, LongTerm }

    // HTF dropdown — replaces Pine's free-text ``input.timeframe('15', ...)``
    // with a typed picker so users can't accidentally set an HTF lower than
    // the chart's timeframe (which would silently disable the whole indicator).
    public enum FadiHtfOption
    {
        M1, M2, M3, M5, M10, M15, M30, M45,
        H1, H2, H4, H8, H12,
        D1, W1, MN1
    }

    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class IctHtfMssLiquidityFadi : Indicator
    {
        // ── Liquidity (HTF + tier selector) ───────────────────────────────────
        // Default H1 — a safe higher-TF for the typical M1–M30 chart this
        // indicator is used on. Choose a value STRICTLY GREATER than the
        // chart timeframe, otherwise the indicator shows an on-chart warning
        // and draws nothing.
        [Parameter("Higher Timeframe", Group = "Liquidity", DefaultValue = FadiHtfOption.H1)]
        public FadiHtfOption HigherTimeframeOption { get; set; }

        [Parameter("Use", Group = "Liquidity", DefaultValue = FadiLevelOption.ShortTerm)]
        public FadiLevelOption UseLevel { get; set; }

        // ── Liquidity Style — Open ────────────────────────────────────────────
        [Parameter("Open Style", Group = "Liquidity Style", DefaultValue = FadiLineStyleOption.Solid)]
        public FadiLineStyleOption OpenStyle { get; set; }

        [Parameter("Open Size", Group = "Liquidity Style", DefaultValue = 1, MinValue = 1, MaxValue = 4)]
        public int OpenSize { get; set; }

        // Pine default: color.new(color.purple, 50)  →  #80800080 (purple @ 50% alpha)
        [Parameter("Open Color", Group = "Liquidity Style", DefaultValue = "#80800080")]
        public Color OpenColor { get; set; }

        // ── Liquidity Style — Claimed ─────────────────────────────────────────
        [Parameter("Claimed Style", Group = "Liquidity Style", DefaultValue = FadiLineStyleOption.Dotted)]
        public FadiLineStyleOption ClaimedStyle { get; set; }

        [Parameter("Claimed Size", Group = "Liquidity Style", DefaultValue = 1, MinValue = 1, MaxValue = 4)]
        public int ClaimedSize { get; set; }

        // Pine default: color.new(color.black, 50)  →  #80000000 (black @ 50% alpha)
        [Parameter("Claimed Color", Group = "Liquidity Style", DefaultValue = "#80000000")]
        public Color ClaimedColor { get; set; }

        // ── Liquidity Style — Reclaimed (Q2) ──────────────────────────────────
        [Parameter("Reclaimed Color", Group = "Liquidity Style", DefaultValue = "Red")]
        public Color ReclaimedColor { get; set; }

        // ── Liquidity Style — Extension + capacity ────────────────────────────
        [Parameter("Extend (bars)", Group = "Liquidity Style", DefaultValue = 10, MinValue = 1)]
        public int ExtendBars { get; set; }

        [Parameter("Number of lines", Group = "Liquidity Style", DefaultValue = 50, MinValue = 1, MaxValue = 250)]
        public int MaxLinesCfg { get; set; }

        // ─────────────────────────────────────────────────────────────────────
        // Internal state
        // ─────────────────────────────────────────────────────────────────────
        private const int MAX_BUFFER = 500;

        private sealed class Pivot
        {
            public DateTime Time;
            public double Price;
            public DateTime TimeLast;
            public int ClaimedIndex;
            public bool Claimed;
            public bool Reclaimed;
            public bool IsHigh;
            public bool IsLow;
            public string LblText = "";
            public ChartTrendLine Line;     // null until AddLiquidity draws one
        }

        private TimeFrame _htfTimeFrame;
        private Bars _htfBars;
        private bool _htfValid;
        private bool _tfErrorShown;

        // HTF boundary tracking — we process one HTF bar at a time, each time
        // it closes. `_lastProcessedHtfIndex` is the absolute HTF bar index
        // (into `_htfBars`) of the most recently processed CLOSED HTF bar.
        // Mirrors the per-TF bar accounting used by ``Market Structure MTF
        // Trend [Pt].cs`` and avoids the fragile ``OpenTimes.LastValue``
        // change-detection pattern.
        private int _lastProcessedHtfIndex = -1;

        // Newest-first mirror of Pine's array unshift order: arrays[0] = latest.
        private readonly List<double> _highs = new List<double>();
        private readonly List<double> _lows = new List<double>();
        private readonly List<DateTime> _times = new List<DateTime>();

        // Pine: MS.ST / STH / STL / ITH / ITL / LT  — all newest-first.
        private readonly List<Pivot> _st = new List<Pivot>();
        private readonly List<Pivot> _stH = new List<Pivot>();
        private readonly List<Pivot> _stL = new List<Pivot>();
        private readonly List<Pivot> _itH = new List<Pivot>();
        private readonly List<Pivot> _itL = new List<Pivot>();
        private readonly List<Pivot> _lt = new List<Pivot>();

        // Pine: var array<line> lines — for the max_lines FIFO cap.
        private readonly List<Pivot> _activeLinePivots = new List<Pivot>();

        private int _pivotSeq;

        // ─────────────────────────────────────────────────────────────────────
        // Initialize
        // ─────────────────────────────────────────────────────────────────────
        protected override void Initialize()
        {
            _htfTimeFrame = HtfOptionToTimeFrame(HigherTimeframeOption);
            _htfBars = MarketData.GetBars(_htfTimeFrame);

            int chartSec = TfToSeconds(Bars.TimeFrame);
            int htfSec = TfToSeconds(_htfTimeFrame);

            // Pine: if helper.Validtimeframe(HTF)  → only runs when chart TF < HTF.
            _htfValid = chartSec > 0 && htfSec > 0 && chartSec < htfSec;

            if (!_htfValid && !_tfErrorShown)
            {
                var msg = "ICT HTF MSS & Liquidity: 'Higher Timeframe' ("
                          + HtfOptionLabel(HigherTimeframeOption)
                          + ") must be STRICTLY GREATER than the chart timeframe ("
                          + Bars.TimeFrame + "). No lines will be drawn.";
                Chart.DrawStaticText(
                    "fadi_tf_error", msg,
                    VerticalAlignment.Top, HorizontalAlignment.Center, Color.Red);
                Print(msg);
                _tfErrorShown = true;
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Calculate (Pine main section)
        //
        // Strategy per LTF bar:
        //   1. Locate the HTF bar that currently CONTAINS the chart bar time
        //      (binary search over `_htfBars.OpenTimes`).
        //   2. Every HTF bar strictly BEFORE that index is closed / confirmed
        //      and hasn't been published to the state machine yet → process
        //      each one in chronological order.
        //   3. After catching up, run CheckClaimed on every LTF bar.
        //
        // This pattern is identical to the one used by Market Structure MTF
        // Trend [Pt].cs — it is robust against cTrader's historical iteration
        // timing (which does not guarantee one Calculate call per HTF boundary).
        // ─────────────────────────────────────────────────────────────────────
        public override void Calculate(int index)
        {
            if (!_htfValid) return;

            var chartTime = Bars.OpenTimes[index];
            var htfIdxAtChart = FindBarIndexAtOrBefore(_htfBars, chartTime);

            // Process every HTF bar that has CLOSED but not yet been processed.
            // "Closed" = any HTF bar whose absolute index < htfIdxAtChart,
            // because htfIdxAtChart is the still-forming HTF bar at chart time.
            // We need >= 6 prior HTF bars buffered to evaluate the pivot rule,
            // so we only fire once history is deep enough.
            while (_lastProcessedHtfIndex + 1 < htfIdxAtChart)
            {
                int newClosedHtfIdx = _lastProcessedHtfIndex + 1;
                if (newClosedHtfIdx >= 5)
                {
                    RefreshHtfWindowAtIndex(newClosedHtfIdx);
                    FindSt();
                }
                _lastProcessedHtfIndex = newClosedHtfIdx;
            }

            // Pine: Term.CheckClaimed()  — runs on every LTF bar.
            CheckClaimed(index);
        }

        // ─────────────────────────────────────────────────────────────────────
        // RefreshHtfWindowAtIndex — build the 6-element sliding window, where
        // `closedHtfIdx` is the absolute index of the HTF bar that JUST closed
        // (= Pine's `high[1]` with lookahead_on at the HTF-boundary LTF bar).
        //
        // Pine's newest-first ordering:
        //   highs[0] = h      = just-closed HTF bar's high
        //   highs[1] = h1     = one bar before that
        //   ...
        //   highs[5] = h5     = five bars before that
        // times[0]  = t       = open time of just-closed HTF bar
        // times[1]  = t1      = open time of the one before that
        // ─────────────────────────────────────────────────────────────────────
        private void RefreshHtfWindowAtIndex(int closedHtfIdx)
        {
            _highs.Clear();
            _lows.Clear();
            _times.Clear();

            for (int k = 0; k < 6; k++)
            {
                int src = closedHtfIdx - k;
                if (src < 0) return;
                _highs.Add(_htfBars.HighPrices[src]);
                _lows.Add(_htfBars.LowPrices[src]);
            }
            _times.Add(_htfBars.OpenTimes[closedHtfIdx]);
            if (closedHtfIdx - 1 >= 0)
                _times.Add(_htfBars.OpenTimes[closedHtfIdx - 1]);
        }

        // ─────────────────────────────────────────────────────────────────────
        // SkipEQ* — Pine source preserved 1:1
        // ─────────────────────────────────────────────────────────────────────
        private int SkipEqHigh(int idx)
        {
            int i = idx;
            while (i < 5 && i - 1 >= 0 && i < _highs.Count && _highs[i] == _highs[i - 1])
                i++;
            return i;
        }

        private int SkipEqLow(int idx)
        {
            int i = idx;
            while (i < 5 && i - 1 >= 0 && i < _lows.Count && _lows[i] == _lows[i - 1])
                i++;
            return i;
        }

        // Pine's SkipEQPivot contains the condition `p.size() < i` (written with
        // `<`, not `>`). With any normal caller (array size >= 3, idx == 2) this
        // evaluates false and the loop never iterates, so in practice
        // SkipEQPivot(2) always returns 2. Reproduced verbatim so behaviour
        // stays identical — promotion tier checks compare the top-three pivots
        // directly without skipping equal-price / equal-label chains.
        private int SkipEqPivot(List<Pivot> p, int idx)
        {
            int i = idx;
            while (i - 1 >= 0 && i < p.Count
                   && p[i].Price == p[i - 1].Price
                   && p[i].LblText == p[i - 1].LblText
                   && p.Count < i)
                i++;
            return i;
        }

        // ─────────────────────────────────────────────────────────────────────
        // findST — middle-bar pivot rule on (h[2], h[1], h[0])
        //
        // Pine:
        //   _h = h.get(1) > h.get(SkipEQHigh(2)) and h.get(1) > h.first()
        //   _l = l.get(1) < l.get(SkipEQLow(2))  and l.get(1) < l.first()
        //
        // With arrays newest-first (h[0] = just-closed bar), a pivot is
        // confirmed on the middle bar h[1], whose time is times[1] (= t1).
        // ─────────────────────────────────────────────────────────────────────
        private void FindSt()
        {
            if (_highs.Count < 3 || _lows.Count < 3 || _times.Count < 2) return;

            int kh = SkipEqHigh(2);
            int kl = SkipEqLow(2);
            if (kh >= _highs.Count || kl >= _lows.Count) return;

            bool isH = _highs[1] > _highs[kh] && _highs[1] > _highs[0];
            bool isL = _lows[1] < _lows[kl] && _lows[1] < _lows[0];

            if (isH)
                AddPivot(_highs[1], _times[1], true, false, "STH");
            if (isL)
                AddPivot(_lows[1], _times[1], false, true, "STL");
        }

        // ─────────────────────────────────────────────────────────────────────
        // Add — drops dupes (same pivot re-emitted on a later HTF-change),
        // pushes to the ST stream, runs promotion cascade, enforces
        // MAX_BUFFER on the ST stream.
        // ─────────────────────────────────────────────────────────────────────
        private void AddPivot(double price, DateTime time, bool isHigh, bool isLow, string lbl)
        {
            bool isNew = true;
            if (_st.Count > 0)
                isNew = _st[0].Time < time;
            if (!isNew) return;

            var pivot = new Pivot
            {
                Price = price,
                Time = time,
                IsHigh = isHigh,
                IsLow = isLow,
                LblText = lbl,
            };

            _st.Insert(0, pivot);
            if (isHigh) _stH.Insert(0, pivot); else _stL.Insert(0, pivot);

            AddLiquidity(pivot);
            FindIt();
            FindLt();

            // Pine: MS.ST.pop() when size exceeds MAX_BUFFER.
            if (_st.Count > MAX_BUFFER)
            {
                var dropped = _st[_st.Count - 1];
                _st.RemoveAt(_st.Count - 1);
                RemovePivotLine(dropped);
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // FindIT / FindLT — promotion passes
        // ─────────────────────────────────────────────────────────────────────
        private void FindIt()
        {
            if (_stH.Count > 2)
            {
                var h1 = _stH[0];
                var h2 = _stH[1];
                int h3Idx = SkipEqPivot(_stH, 2);
                if (h3Idx < _stH.Count)
                {
                    var h3 = _stH[h3Idx];
                    if (h2.Price > h3.Price && h2.Price > h1.Price && h2.LblText == "STH")
                    {
                        h2.LblText = "ITH";
                        _itH.Insert(0, h2);
                        AddLiquidity(h2);
                    }
                }
            }
            if (_stL.Count > 2)
            {
                var l1 = _stL[0];
                var l2 = _stL[1];
                int l3Idx = SkipEqPivot(_stL, 2);
                if (l3Idx < _stL.Count)
                {
                    var l3 = _stL[l3Idx];
                    if (l2.Price < l3.Price && l2.Price < l1.Price && l2.LblText == "STL")
                    {
                        l2.LblText = "ITL";
                        _itL.Insert(0, l2);
                        AddLiquidity(l2);
                    }
                }
            }
        }

        private void FindLt()
        {
            if (_itH.Count > 2)
            {
                var h1 = _itH[0];
                var h2 = _itH[1];
                int h3Idx = SkipEqPivot(_itH, 2);
                if (h3Idx < _itH.Count)
                {
                    var h3 = _itH[h3Idx];
                    if (h2.Price > h3.Price && h2.Price > h1.Price && h2.LblText == "ITH")
                    {
                        h2.LblText = "LTH";
                        _lt.Insert(0, h2);
                        AddLiquidity(h2);
                    }
                }
            }
            if (_itL.Count > 2)
            {
                var l1 = _itL[0];
                var l2 = _itL[1];
                int l3Idx = SkipEqPivot(_itL, 2);
                if (l3Idx < _itL.Count)
                {
                    var l3 = _itL[l3Idx];
                    if (l2.Price < l3.Price && l2.Price < l1.Price && l2.LblText == "ITL")
                    {
                        l2.LblText = "LTL";
                        _lt.Insert(0, l2);
                        AddLiquidity(l2);
                    }
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // AddLiquidity — create line if tier matches 'Use', otherwise update.
        // Pine precedence (and > or): (level==ST AND starts(ST))
        //                          OR (level==IT AND starts(IT))
        //                          OR (level==LT AND starts(LT))
        // ─────────────────────────────────────────────────────────────────────
        private void AddLiquidity(Pivot pivot)
        {
            if (!MatchesLevel(pivot.LblText)) return;

            if (pivot.Line == null)
            {
                string name = "fadi_px_" + (++_pivotSeq).ToString();
                pivot.Line = Chart.DrawTrendLine(
                    name, pivot.Time, pivot.Price,
                    ExtendedEndTime(), pivot.Price,
                    OpenColor, OpenSize, ToLineStyle(OpenStyle));

                _activeLinePivots.Insert(0, pivot);

                // Pine: if lines.size() > max_lines → pop() + line.delete()
                if (_activeLinePivots.Count > MaxLinesCfg)
                {
                    var evicted = _activeLinePivots[_activeLinePivots.Count - 1];
                    _activeLinePivots.RemoveAt(_activeLinePivots.Count - 1);
                    RemovePivotLine(evicted);
                }
            }
            else
            {
                // Promotion path: pivot already has a line. Refresh its style
                // to the current claimed/open state (mirrors Pine's else-branch
                // in AddLiquidity that re-applies colour/width/x2/style).
                UpdatePivotLine(pivot);
            }
        }

        private bool MatchesLevel(string lbl)
        {
            if (string.IsNullOrEmpty(lbl)) return false;
            switch (UseLevel)
            {
                case FadiLevelOption.ShortTerm:        return lbl.StartsWith("ST");
                case FadiLevelOption.IntermediateTerm: return lbl.StartsWith("IT");
                case FadiLevelOption.LongTerm:         return lbl.StartsWith("LT");
            }
            return false;
        }

        // ─────────────────────────────────────────────────────────────────────
        // CheckClaimed — sweep + reclaim on every LTF bar
        // ─────────────────────────────────────────────────────────────────────
        private void CheckClaimed(int index)
        {
            if (_st.Count == 0) return;

            double close = Bars.ClosePrices[index];
            DateTime nowTime = Bars.OpenTimes[index];
            int chartBar = index;

            for (int i = 0; i < _st.Count; i++)
            {
                var pivot = _st[i];
                if (pivot.Line == null) continue;

                // ── not yet claimed ─────────────────────────────────────────
                if (!pivot.Claimed)
                {
                    if (pivot.IsHigh && close > pivot.Price) pivot.Claimed = true;
                    else if (pivot.IsLow && close < pivot.Price) pivot.Claimed = true;

                    if (pivot.Claimed)
                    {
                        pivot.TimeLast = nowTime;
                        pivot.ClaimedIndex = chartBar;
                    }
                    UpdatePivotLine(pivot);
                }

                // ── claimed but not yet reclaimed ───────────────────────────
                if (pivot.Claimed && !pivot.Reclaimed)
                {
                    if (pivot.IsHigh && close < pivot.Price) pivot.Reclaimed = true;
                    else if (pivot.IsLow && close > pivot.Price) pivot.Reclaimed = true;

                    if (pivot.Reclaimed)
                    {
                        // Pine hard-codes color.red here; we expose it as
                        // `ReclaimedColor` (Q2 decision).
                        if (pivot.Line != null)
                            pivot.Line.Color = ReclaimedColor;

                        // Pine: same-direction cascade — every OTHER same-side
                        // claimed-but-not-reclaimed pivot becomes reclaimed.
                        for (int j = 0; j < _st.Count; j++)
                        {
                            var p = _st[j];
                            if (((pivot.IsHigh && p.IsHigh) || (pivot.IsLow && p.IsLow))
                                && p.Claimed && !p.Reclaimed)
                            {
                                p.Reclaimed = true;
                                if (p.Line != null)
                                    p.Line.Color = ReclaimedColor;
                            }
                        }
                    }
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Line maintenance
        // ─────────────────────────────────────────────────────────────────────
        private void UpdatePivotLine(Pivot pivot)
        {
            var ln = pivot.Line;
            if (ln == null) return;

            if (pivot.Reclaimed)
            {
                // x2 stays frozen at claim time; colour already = ReclaimedColor.
                ln.Color = ReclaimedColor;
                ln.Time2 = pivot.TimeLast;
                return;
            }

            if (pivot.Claimed)
            {
                ln.Color = ClaimedColor;
                ln.Thickness = ClaimedSize;
                ln.LineStyle = ToLineStyle(ClaimedStyle);
                ln.Time2 = pivot.TimeLast;
            }
            else
            {
                ln.Color = OpenColor;
                ln.Thickness = OpenSize;
                ln.LineStyle = ToLineStyle(OpenStyle);
                ln.Time2 = ExtendedEndTime();
            }
        }

        private void RemovePivotLine(Pivot pivot)
        {
            if (pivot.Line == null) return;
            Chart.RemoveObject(pivot.Line.Name);
            pivot.Line = null;
            _activeLinePivots.Remove(pivot);
        }

        // Pine: time + (time - time[1]) * MS.settings.extend
        private DateTime ExtendedEndTime()
        {
            int n = Bars.Count;
            if (n < 2) return Bars.OpenTimes.LastValue;
            TimeSpan span = Bars.OpenTimes.LastValue - Bars.OpenTimes.Last(1);
            if (span <= TimeSpan.Zero) span = TimeSpan.FromMinutes(1);
            return Bars.OpenTimes.LastValue + TimeSpan.FromTicks(span.Ticks * ExtendBars);
        }

        private static LineStyle ToLineStyle(FadiLineStyleOption opt)
        {
            switch (opt)
            {
                case FadiLineStyleOption.Dashed: return LineStyle.Lines;
                case FadiLineStyleOption.Dotted: return LineStyle.Dots;
                default:                         return LineStyle.Solid;
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Binary-search helper — identical contract to the one used by
        // Market Structure MTF Trend [Pt].cs: returns the index of the last
        // bar whose OpenTime is <= `time`, or -1 if no such bar exists.
        // ─────────────────────────────────────────────────────────────────────
        private static int FindBarIndexAtOrBefore(Bars bars, DateTime time)
        {
            int lo = 0;
            int hi = bars.Count - 1;
            while (lo <= hi)
            {
                int mid = (lo + hi) / 2;
                var midTime = bars.OpenTimes[mid];
                if (midTime == time) return mid;
                if (midTime < time) lo = mid + 1;
                else hi = mid - 1;
            }
            return hi;
        }

        // ─────────────────────────────────────────────────────────────────────
        // Timeframe helpers
        // ─────────────────────────────────────────────────────────────────────
        private static TimeFrame HtfOptionToTimeFrame(FadiHtfOption opt)
        {
            switch (opt)
            {
                case FadiHtfOption.M1:  return TimeFrame.Minute;
                case FadiHtfOption.M2:  return TimeFrame.Minute2;
                case FadiHtfOption.M3:  return TimeFrame.Minute3;
                case FadiHtfOption.M5:  return TimeFrame.Minute5;
                case FadiHtfOption.M10: return TimeFrame.Minute10;
                case FadiHtfOption.M15: return TimeFrame.Minute15;
                case FadiHtfOption.M30: return TimeFrame.Minute30;
                case FadiHtfOption.M45: return TimeFrame.Minute45;
                case FadiHtfOption.H1:  return TimeFrame.Hour;
                case FadiHtfOption.H2:  return TimeFrame.Hour2;
                case FadiHtfOption.H4:  return TimeFrame.Hour4;
                case FadiHtfOption.H8:  return TimeFrame.Hour8;
                case FadiHtfOption.H12: return TimeFrame.Hour12;
                case FadiHtfOption.D1:  return TimeFrame.Daily;
                case FadiHtfOption.W1:  return TimeFrame.Weekly;
                case FadiHtfOption.MN1: return TimeFrame.Monthly;
                default:                return TimeFrame.Hour;
            }
        }

        private static string HtfOptionLabel(FadiHtfOption opt)
        {
            switch (opt)
            {
                case FadiHtfOption.M1:  return "M1";
                case FadiHtfOption.M2:  return "M2";
                case FadiHtfOption.M3:  return "M3";
                case FadiHtfOption.M5:  return "M5";
                case FadiHtfOption.M10: return "M10";
                case FadiHtfOption.M15: return "M15";
                case FadiHtfOption.M30: return "M30";
                case FadiHtfOption.M45: return "M45";
                case FadiHtfOption.H1:  return "H1";
                case FadiHtfOption.H2:  return "H2";
                case FadiHtfOption.H4:  return "H4";
                case FadiHtfOption.H8:  return "H8";
                case FadiHtfOption.H12: return "H12";
                case FadiHtfOption.D1:  return "D1";
                case FadiHtfOption.W1:  return "W1";
                case FadiHtfOption.MN1: return "MN1";
                default:                return opt.ToString();
            }
        }

        private static int TfToSeconds(TimeFrame tf)
        {
            if (tf == TimeFrame.Minute)   return 60;
            if (tf == TimeFrame.Minute2)  return 120;
            if (tf == TimeFrame.Minute3)  return 180;
            if (tf == TimeFrame.Minute4)  return 240;
            if (tf == TimeFrame.Minute5)  return 300;
            if (tf == TimeFrame.Minute10) return 600;
            if (tf == TimeFrame.Minute15) return 900;
            if (tf == TimeFrame.Minute30) return 1800;
            if (tf == TimeFrame.Minute45) return 2700;
            if (tf == TimeFrame.Hour)     return 3600;
            if (tf == TimeFrame.Hour2)    return 7200;
            if (tf == TimeFrame.Hour4)    return 14400;
            if (tf == TimeFrame.Hour8)    return 28800;
            if (tf == TimeFrame.Hour12)   return 43200;
            if (tf == TimeFrame.Daily)    return 86400;
            if (tf == TimeFrame.Weekly)   return 604800;
            if (tf == TimeFrame.Monthly)  return 2592000;
            return 0;
        }
    }
}
