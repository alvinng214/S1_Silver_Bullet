// Translated from Pine v5: cd_sweep&cisd_Cx (© cdikici71)
// Mozilla Public License 2.0 — https://mozilla.org/MPL/2.0/
// Subsystems implemented: A (CISD detection), B (CISD pending lines + triggered cisd+/cisd- boxes),
//                         C (HTF candle box overlay), D (HTF sweep boxes),
//                         I (HTF bias plot), K (Key levels for 3 TFs).
// Excluded by user: HTF mini-candle display, sweep marker lines on mini candles,
//                   cross-asset screener, HTF Swept screener table, SMT divergence,
//                   remaining-time countdown, alerts, CISD detection table.
// HTF semantics: option (b) — mirror Pine state machine on LTF (no MarketData.GetBars).

using System;
using System.Collections.Generic;
using cAlgo.API;

namespace cAlgo
{
    public enum HtfMode { Auto, Fixed }
    public enum HtfBoxStyle { Solid, Dotted, Dashed }

    [Indicator(IsOverlay = true, AccessRights = AccessRights.None, AutoRescale = false)]
    public class CdSweepCisdCx : Indicator
    {
        // ───────────────────────────── HTF ─────────────────────────────
        [Parameter("HTF Selection method:", Group = "HTF", DefaultValue = HtfMode.Auto)]
        public HtfMode HtfOpt { get; set; }

        [Parameter("If Fixed HTF:", Group = "HTF", DefaultValue = "Hour")]
        public TimeFrame HtfFixed { get; set; }

        [Parameter("Show HTF boxes", Group = "HTF", DefaultValue = true)]
        public bool ShowHtf { get; set; }

        [Parameter("Border Width", Group = "HTF", DefaultValue = 1, MinValue = 1, MaxValue = 5)]
        public int WidthBox { get; set; }

        [Parameter("Style", Group = "HTF", DefaultValue = HtfBoxStyle.Dotted)]
        public HtfBoxStyle StyleBox { get; set; }

        // ───────────────────────────── BIAS ────────────────────────────
        [Parameter("HTF for bias", Group = "BIAS", DefaultValue = "Hour")]
        public TimeFrame HtfBias { get; set; }

        [Parameter("Plot Htf Bias", Group = "BIAS", DefaultValue = false)]
        public bool PlotBias { get; set; }

        // ──────────────────────────── Colors ───────────────────────────
        [Parameter("Bull Color", Group = "Colors", DefaultValue = "Teal")]
        public Color BullColor { get; set; }

        [Parameter("Bear Color", Group = "Colors", DefaultValue = "Red")]
        public Color BearColor { get; set; }

        [Parameter("Range Color", Group = "Colors", DefaultValue = "Navy")]
        public Color RangeColor { get; set; }

        [Parameter("Sweep Box Color", Group = "Colors", DefaultValue = "#32FFEB3B")]
        public Color SweptColor { get; set; }

        [Parameter("Key Levels Text Color", Group = "Colors", DefaultValue = "Red")]
        public Color KeyTxtColor { get; set; }

        [Parameter("CISD Line Color", Group = "Colors", DefaultValue = "Navy")]
        public Color CsdColor { get; set; }

        // ─────────────────────── Time Frames => Key Levels ───────────────────────
        [Parameter("Show Key Levels", Group = "Key Levels", DefaultValue = true)]
        public bool ShowKeyLvl { get; set; }

        [Parameter("Start from ahead (bars)", Group = "Key Levels", DefaultValue = 20)]
        public int Ky { get; set; }

        [Parameter("Show Tf1", Group = "Key Levels", DefaultValue = true)]
        public bool ShowKeyTf1 { get; set; }

        [Parameter("Tf1", Group = "Key Levels", DefaultValue = "Monthly")]
        public TimeFrame KeyTf1 { get; set; }

        [Parameter("Tf1 Color", Group = "Key Levels", DefaultValue = "Gray")]
        public Color KeyTf1Clr { get; set; }

        [Parameter("Show Tf2", Group = "Key Levels", DefaultValue = true)]
        public bool ShowKeyTf2 { get; set; }

        [Parameter("Tf2", Group = "Key Levels", DefaultValue = "Weekly")]
        public TimeFrame KeyTf2 { get; set; }

        [Parameter("Tf2 Color", Group = "Key Levels", DefaultValue = "Gray")]
        public Color KeyTf2Clr { get; set; }

        [Parameter("Show Tf3", Group = "Key Levels", DefaultValue = true)]
        public bool ShowKeyTf3 { get; set; }

        [Parameter("Tf3", Group = "Key Levels", DefaultValue = "Daily")]
        public TimeFrame KeyTf3 { get; set; }

        [Parameter("Tf3 Color", Group = "Key Levels", DefaultValue = "Gray")]
        public Color KeyTf3Clr { get; set; }

        // ─────────────────── runtime state (HTF state machine) ───────────────────
        // current HTF candle
        private double _o0, _h0, _l0, _c0;
        private int _h0bar, _l0bar;
        private DateTime _t0;
        // previous HTF candle (1 ago)
        private double _o1, _h1, _l1, _c1;
        private int _h1bar, _l1bar;
        private DateTime _t1;
        // 2 ago
        private double _o2, _h2, _l2, _c2;
        // 3 ago
        private double _o3, _h3, _l3, _c3;
        private bool _hSwept1, _lSwept1;

        // bias HTF state machine
        private double _bo0, _bh0, _bl0, _bc0;
        private DateTime _bt0;
        private double _bo1, _bh1, _bl1, _bc1;
        private DateTime _bt1;
        private double _bo2, _bh2, _bl2, _bc2;
        private int _bias;

        // CISD state
        private double _bullLevel = double.NaN;
        private double _bearLevel = double.NaN;
        private int _bullIndex;
        private int _bearIndex;
        private bool _xCisd;
        private bool _yCisd;
        private readonly List<string> _bucisdg = new List<string>();
        private readonly List<string> _becisdg = new List<string>();

        // mini-state machines for the three key-levels timeframes
        private MiniHtfState _key1, _key2, _key3;

        // sweep-box dedupe (Pine: lasthighswept/lastlowswept)
        private DateTime _lastHighSwept = DateTime.MinValue;
        private DateTime _lastLowSwept = DateTime.MinValue;
        private string _lastHighSweptName, _lastLowSweptName;

        // current HTF live-box names (for delete-and-redraw)
        private string _liveHtfBoxName;
        private readonly List<string> _allDrawNames = new List<string>();

        // initialised flag — defer first init until index 0
        private bool _initDone;

        protected override void Initialize()
        {
            _key1 = new MiniHtfState();
            _key2 = new MiniHtfState();
            _key3 = new MiniHtfState();
        }

        // ─────────────────────────── helpers ───────────────────────────
        private TimeFrame Htf => HtfOpt == HtfMode.Auto ? AutoHtf(TimeFrame) : HtfFixed;

        private static TimeFrame AutoHtf(TimeFrame tf)
        {
            // Mirrors Pine f_look_tf with cTrader-equivalent timeframes.
            // Mapping approved by user. Anything not in the table falls back to current TF.
            if (tf == TimeFrame.Minute)    return TimeFrame.Minute15;
            if (tf == TimeFrame.Minute2)   return TimeFrame.Hour;
            if (tf == TimeFrame.Minute3)   return TimeFrame.Minute30;
            if (tf == TimeFrame.Minute5)   return TimeFrame.Hour;
            if (tf == TimeFrame.Minute15)  return TimeFrame.Hour4;
            if (tf == TimeFrame.Minute30)  return TimeFrame.Hour12;
            if (tf == TimeFrame.Hour)      return TimeFrame.Daily;
            if (tf == TimeFrame.Hour4)     return TimeFrame.Weekly;
            if (tf == TimeFrame.Daily)     return TimeFrame.Monthly;
            if (tf == TimeFrame.Weekly)    return TimeFrame.Monthly;
            return tf;
        }

        // Pine's timeframe.change(htf) detector: did the HTF "bucket" change between the
        // previous and current LTF bars? We bucket each LTF bar's open time into the HTF.
        private bool HtfChanged(int index, TimeFrame htf)
        {
            if (index <= 0) return true;
            return BucketStart(Bars.OpenTimes[index], htf) != BucketStart(Bars.OpenTimes[index - 1], htf);
        }

        private static DateTime BucketStart(DateTime t, TimeFrame tf)
        {
            // Map well-known cTrader timeframes to bucket starts. Falls back to Date when unknown.
            if (tf == TimeFrame.Minute)    return new DateTime(t.Year, t.Month, t.Day, t.Hour, t.Minute, 0, DateTimeKind.Utc);
            if (tf == TimeFrame.Minute2)   return BucketAtMinute(t, 2);
            if (tf == TimeFrame.Minute3)   return BucketAtMinute(t, 3);
            if (tf == TimeFrame.Minute5)   return BucketAtMinute(t, 5);
            if (tf == TimeFrame.Minute10)  return BucketAtMinute(t, 10);
            if (tf == TimeFrame.Minute15)  return BucketAtMinute(t, 15);
            if (tf == TimeFrame.Minute20)  return BucketAtMinute(t, 20);
            if (tf == TimeFrame.Minute30)  return BucketAtMinute(t, 30);
            if (tf == TimeFrame.Minute45)  return BucketAtMinute(t, 45);
            if (tf == TimeFrame.Hour)      return new DateTime(t.Year, t.Month, t.Day, t.Hour, 0, 0, DateTimeKind.Utc);
            if (tf == TimeFrame.Hour2)     return BucketAtHour(t, 2);
            if (tf == TimeFrame.Hour3)     return BucketAtHour(t, 3);
            if (tf == TimeFrame.Hour4)     return BucketAtHour(t, 4);
            if (tf == TimeFrame.Hour6)     return BucketAtHour(t, 6);
            if (tf == TimeFrame.Hour8)     return BucketAtHour(t, 8);
            if (tf == TimeFrame.Hour12)    return BucketAtHour(t, 12);
            if (tf == TimeFrame.Daily)     return new DateTime(t.Year, t.Month, t.Day, 0, 0, 0, DateTimeKind.Utc);
            if (tf == TimeFrame.Weekly)    return WeekStart(t);
            if (tf == TimeFrame.Monthly)   return new DateTime(t.Year, t.Month, 1, 0, 0, 0, DateTimeKind.Utc);
            return new DateTime(t.Year, t.Month, t.Day, 0, 0, 0, DateTimeKind.Utc);
        }

        private static DateTime BucketAtMinute(DateTime t, int n)
        {
            int m = (t.Minute / n) * n;
            return new DateTime(t.Year, t.Month, t.Day, t.Hour, m, 0, DateTimeKind.Utc);
        }

        private static DateTime BucketAtHour(DateTime t, int n)
        {
            int h = (t.Hour / n) * n;
            return new DateTime(t.Year, t.Month, t.Day, h, 0, 0, DateTimeKind.Utc);
        }

        private static DateTime WeekStart(DateTime t)
        {
            var d = new DateTime(t.Year, t.Month, t.Day, 0, 0, 0, DateTimeKind.Utc);
            int diff = ((int)d.DayOfWeek + 6) % 7; // Monday-anchored
            return d.AddDays(-diff);
        }

        private static int InSeconds(TimeFrame tf)
        {
            // Approximate seconds-per-bar for tf_ok checks. Monthly counted as 30d.
            if (tf == TimeFrame.Minute)    return 60;
            if (tf == TimeFrame.Minute2)   return 120;
            if (tf == TimeFrame.Minute3)   return 180;
            if (tf == TimeFrame.Minute5)   return 300;
            if (tf == TimeFrame.Minute10)  return 600;
            if (tf == TimeFrame.Minute15)  return 900;
            if (tf == TimeFrame.Minute20)  return 1200;
            if (tf == TimeFrame.Minute30)  return 1800;
            if (tf == TimeFrame.Minute45)  return 2700;
            if (tf == TimeFrame.Hour)      return 3600;
            if (tf == TimeFrame.Hour2)     return 7200;
            if (tf == TimeFrame.Hour3)     return 10800;
            if (tf == TimeFrame.Hour4)     return 14400;
            if (tf == TimeFrame.Hour6)     return 21600;
            if (tf == TimeFrame.Hour8)     return 28800;
            if (tf == TimeFrame.Hour12)    return 43200;
            if (tf == TimeFrame.Daily)     return 86400;
            if (tf == TimeFrame.Weekly)    return 604800;
            if (tf == TimeFrame.Monthly)   return 2592000;
            return 60;
        }

        private bool TfCon(TimeFrame htf) => InSeconds(htf) >= InSeconds(TimeFrame);

        private bool TfOkForKey(TimeFrame htf) => InSeconds(htf) >= InSeconds(TimeFrame);

        private LineStyle BoxLineStyle()
        {
            switch (StyleBox)
            {
                case HtfBoxStyle.Solid:  return LineStyle.Solid;
                case HtfBoxStyle.Dashed: return LineStyle.Lines;
                default: return LineStyle.Dots;
            }
        }

        private static string F_Text(TimeFrame tf)
        {
            if (tf == TimeFrame.Daily)   return "D";
            if (tf == TimeFrame.Weekly)  return "W";
            if (tf == TimeFrame.Monthly) return "M";
            int s = InSeconds(tf);
            int m = s / 60;
            if (m < 60) return m + "m";
            return "h" + (m / 60);
        }

        // ───────────────────────────── main ─────────────────────────────
        public override void Calculate(int index)
        {
            if (index < 1) return;

            if (!_initDone)
            {
                InitState(index);
                _initDone = true;
            }

            // Snapshot rolling HTF extremes as they stood on the prior bar.
            // _h0/_l0 are not yet updated for the current bar at this point.
            _h0PrevBar = _h0;
            _l0PrevBar = _l0;

            UpdateMainHtfState(index);
            UpdateBiasState(index);
            _key1.Update(index, KeyTf1, this);
            _key2.Update(index, KeyTf2, this);
            _key3.Update(index, KeyTf3, this);

            // Subsystem D (sweep) booleans — must come before Subsystem A which references them
            bool hSwept = _h0 > _h1 && Math.Max(_o0, _c0) < _h1;
            bool lSwept = _l0 < _l1 && Math.Min(_o0, _c0) > _l1;

            // Pine: at the bar where htf rolls, h_swept1/l_swept1 := h_swept[1]/l_swept[1]
            if (HtfChanged(index, Htf))
            {
                _hSwept1 = _hSweptPrevBar;
                _lSwept1 = _lSweptPrevBar;
            }

            // Subsystems C and D — HTF box overlays + sweep boxes
            DrawHtfBoxes(index);
            DrawSweepBoxes(index, hSwept, lSwept);

            // Subsystem A + B — CISD detection (uses _hSweptPrevBar / _lSweptPrevBar — prior bar)
            UpdateCisdState(index, hSwept, lSwept);

            // Subsystem I — HTF bias plot
            DrawBias(index);

            // Subsystem K — key levels
            if (IsLastBar)
                DrawKeyLevels(index);

            // Roll prior-bar trackers AFTER all logic for this bar has run.
            _hSweptPrevBar = hSwept;
            _lSweptPrevBar = lSwept;
        }

        private void InitState(int i)
        {
            _o0 = Bars.OpenPrices[i]; _h0 = Bars.HighPrices[i]; _l0 = Bars.LowPrices[i]; _c0 = Bars.ClosePrices[i];
            _o1 = _o0; _h1 = _h0; _l1 = _l0; _c1 = _c0;
            _o2 = _o0; _h2 = _h0; _l2 = _l0; _c2 = _c0;
            _o3 = _o0; _h3 = _h0; _l3 = _l0; _c3 = _c0;
            _h0bar = i; _l0bar = i; _h1bar = i; _l1bar = i;
            _t0 = Bars.OpenTimes[i]; _t1 = _t0;
            _bullLevel = Bars.HighPrices[i];
            _bearLevel = Bars.LowPrices[i];
            _bullIndex = i; _bearIndex = i;

            _bo0 = _o0; _bh0 = _h0; _bl0 = _l0; _bc0 = _c0;
            _bo1 = _o0; _bh1 = _h0; _bl1 = _l0; _bc1 = _c0;
            _bo2 = _o0; _bh2 = _h0; _bl2 = _l0; _bc2 = _c0;
            _bt0 = _t0; _bt1 = _t0;
        }

        private void UpdateMainHtfState(int i)
        {
            double open  = Bars.OpenPrices[i];
            double high  = Bars.HighPrices[i];
            double low   = Bars.LowPrices[i];
            double close = Bars.ClosePrices[i];
            DateTime t   = Bars.OpenTimes[i];

            if (HtfChanged(i, Htf))
            {
                // Roll: 1←0, 2←1, 3←2 (using the *just-finished* HTF candle's values)
                _o3 = _o2; _h3 = _h2; _l3 = _l2; _c3 = _c2;
                _o2 = _o1; _h2 = _h1; _l2 = _l1; _c2 = _c1;
                _o1 = _o0;
                _h1 = _h0;
                _l1 = _l0;
                _t1 = _t0;
                _c1 = i >= 1 ? Bars.ClosePrices[i - 1] : _c0; // Pine: c1 := close[1]
                _h1bar = _h0bar;
                _l1bar = _l0bar;

                // Reset 0
                _t0 = t;
                _o0 = open;
                _h0 = high;
                _l0 = low;
                _h0bar = i;
                _l0bar = i;
            }

            if (high >= _h0)
            {
                _h0 = high;
                _h0bar = i;
            }
            if (low <= _l0)
            {
                _l0 = low;
                _l0bar = i;
            }
            _c0 = close;
        }

        private void UpdateBiasState(int i)
        {
            double open  = Bars.OpenPrices[i];
            double high  = Bars.HighPrices[i];
            double low   = Bars.LowPrices[i];
            double close = Bars.ClosePrices[i];
            DateTime t   = Bars.OpenTimes[i];

            bool changed = HtfChanged(i, HtfBias);

            if (changed)
            {
                _bo2 = _bo1; _bh2 = _bh1; _bl2 = _bl1; _bc2 = _bc1;
                _bo1 = _bo0; _bh1 = _bh0; _bl1 = _bl0; _bt1 = _bt0;
                _bc1 = i >= 1 ? Bars.ClosePrices[i - 1] : _bc0;

                _bt0 = t;
                _bo0 = open;
                _bh0 = high;
                _bl0 = low;
            }
            if (high >= _bh0) _bh0 = high;
            if (low  <= _bl0) _bl0 = low;
            _bc0 = close;

            if (changed)
            {
                _bias = 0;
                if (_bc1 > _bh2) _bias = 1;
                if (_bc1 < _bl2) _bias = -1;
                if (_bc1 < _bh2 && _bc1 > _bl2 && _bh1 > _bh2 && _bl1 > _bl2) _bias = -1;
                if (_bc1 > _bl2 && _bc1 < _bh2 && _bh1 < _bh2 && _bl1 < _bl2) _bias = 1;
                if (_bh1 <= _bh2 && _bl1 >= _bl2)
                    _bias = _bc2 > _bo2 ? 1 : -1;
            }
        }

        // ───────────────────── Subsystem C — HTF candle box overlay ─────────────────────
        private void DrawHtfBoxes(int i)
        {
            if (!ShowHtf) return;
            if (!TfCon(Htf)) return;
            if (TimeFrame == Htf) return;

            // Static box for the just-closed previous HTF candle, drawn once when a new HTF bucket starts
            if (HtfChanged(i, Htf) && i >= 1)
            {
                Color clr = _c1 > _o1 ? BullColor : _c1 < _o1 ? BearColor : RangeColor;
                string name = "cdc_htf_static_" + _t1.Ticks;
                var rect = Chart.DrawRectangle(name, _t1, _h1, _t0, _l1, clr, WidthBox, BoxLineStyle());
                rect.IsFilled = false;
                _allDrawNames.Add(name);
            }

            // Live box for the in-progress HTF candle — redrawn each tick using the same name
            Color liveClr = _c0 > _o0 ? BullColor : _c0 < _o0 ? BearColor : RangeColor;
            double top = Bars.HighPrices[i] > _h0 ? Bars.HighPrices[i] : _h0;
            double bot = Bars.LowPrices[i]  < _l0 ? Bars.LowPrices[i]  : _l0;
            DateTime right = Bars.OpenTimes[i].AddSeconds(InSeconds(TimeFrame));
            _liveHtfBoxName = "cdc_htf_live";
            var live = Chart.DrawRectangle(_liveHtfBoxName, _t0, top, right, bot, liveClr, WidthBox, BoxLineStyle());
            live.IsFilled = false;
        }

        // ──────────────────── Subsystem D — HTF sweep boxes ────────────────────
        private void DrawSweepBoxes(int i, bool hSwept, bool lSwept)
        {
            if (!TfCon(Htf)) return;
            if (TimeFrame == Htf) return;

            double close = Bars.ClosePrices[i];

            // High-side swept zone: HTF wicked above prior HTF high then closed back inside
            if (_h0 > _h1 && close < _h1 && _o0 < _h1)
            {
                if (_lastHighSwept == _t0 && _lastHighSweptName != null)
                    Chart.RemoveObject(_lastHighSweptName);
                string name = "cdc_swept_h_" + _t0.Ticks;
                DateTime right = Bars.OpenTimes[i];
                var rect = Chart.DrawRectangle(name, _t0, _h0, right, _h1, SweptColor, 1, LineStyle.Dots);
                rect.IsFilled = true;
                _lastHighSwept = _t0;
                _lastHighSweptName = name;
                _allDrawNames.Add(name);
            }

            // Low-side swept zone
            if (_l0 < _l1 && close > _l1 && _o0 > _l1)
            {
                if (_lastLowSwept == _t0 && _lastLowSweptName != null)
                    Chart.RemoveObject(_lastLowSweptName);
                string name = "cdc_swept_l_" + _t0.Ticks;
                DateTime right = Bars.OpenTimes[i];
                var rect = Chart.DrawRectangle(name, _t0, _l1, right, _l0, SweptColor, 1, LineStyle.Dots);
                rect.IsFilled = true;
                _lastLowSwept = _t0;
                _lastLowSweptName = name;
                _allDrawNames.Add(name);
            }
        }

        // ──────────── Subsystem A + B — CISD detection, lines, triggered boxes ────────────
        private void UpdateCisdState(int i, bool hSwept, bool lSwept)
        {
            double open  = Bars.OpenPrices[i];
            double high  = Bars.HighPrices[i];
            double low   = Bars.LowPrices[i];
            double close = Bars.ClosePrices[i];

            bool up = close > open;
            bool dw = close < open;
            bool eq = close == open;

            bool up1 = i >= 1 && Bars.ClosePrices[i - 1] > Bars.OpenPrices[i - 1];
            bool dw1 = i >= 1 && Bars.ClosePrices[i - 1] < Bars.OpenPrices[i - 1];
            bool eq1 = i >= 1 && Bars.ClosePrices[i - 1] == Bars.OpenPrices[i - 1];

            // ───── bull CISD pending-line generation ─────
            if (low == _l0 && low < _l1)
            {
                if ((dw || eq) && (up1 || eq1) && !(eq && eq1))
                {
                    _bullLevel = open;
                    _bullIndex = i;
                    AddPendingLine(_bucisdg, _bullIndex, _bullLevel);
                }
                else
                {
                    for (int k = 2; k <= 10; k++)
                    {
                        if (i - k < 0) break;
                        if (Bars.LowPrices[i - k] < low) break;
                        bool upK   = Bars.ClosePrices[i - k] > Bars.OpenPrices[i - k];
                        bool eqK   = Bars.ClosePrices[i - k] == Bars.OpenPrices[i - k];
                        bool dwKm1 = Bars.ClosePrices[i - (k - 1)] < Bars.OpenPrices[i - (k - 1)];
                        if ((upK || eqK) && dwKm1)
                        {
                            int bar = k - 1;
                            _bullLevel = Bars.OpenPrices[i - bar];
                            _bullIndex = i - bar;
                            for (int j = bar; j >= 0; j--)
                            {
                                bool dwJ = Bars.ClosePrices[i - j] < Bars.OpenPrices[i - j];
                                if (Bars.OpenPrices[i - j] > _bullLevel && dwJ)
                                {
                                    _bullLevel = Bars.OpenPrices[i - j];
                                    _bullIndex = i - j;
                                }
                            }
                            if (_bullLevel < open && !(close > open))
                            {
                                _bullLevel = open;
                                _bullIndex = i;
                            }
                            if (_bullLevel < open && (close > open))
                            {
                                _bullLevel = high;
                                _bullIndex = i;
                            }
                            AddPendingLine(_bucisdg, _bullIndex, _bullLevel);
                            break;
                        }
                    }
                }
            }

            // ───── bear CISD pending-line generation ─────
            if (high == _h0 && high > _h1)
            {
                if ((up || eq) && (dw1 || eq1) && !(eq && eq1))
                {
                    _bearLevel = open;
                    _bearIndex = i;
                    AddPendingLine(_becisdg, _bearIndex, _bearLevel);
                }
                else
                {
                    for (int k = 2; k <= 10; k++)
                    {
                        if (i - k < 0) break;
                        if (Bars.HighPrices[i - k] > high) break;
                        bool dwK   = Bars.ClosePrices[i - k] < Bars.OpenPrices[i - k];
                        bool eqK   = Bars.ClosePrices[i - k] == Bars.OpenPrices[i - k];
                        bool upKm1 = Bars.ClosePrices[i - (k - 1)] > Bars.OpenPrices[i - (k - 1)];
                        if ((dwK || eqK) && upKm1)
                        {
                            int ybar = k - 1;
                            _bearLevel = Bars.OpenPrices[i - ybar];
                            _bearIndex = i - ybar;
                            for (int j = ybar; j >= 0; j--)
                            {
                                bool upJ = Bars.ClosePrices[i - j] > Bars.OpenPrices[i - j];
                                if (Bars.OpenPrices[i - j] < _bearLevel && upJ)
                                {
                                    _bearLevel = Bars.OpenPrices[i - j];
                                    _bearIndex = i - j;
                                }
                            }
                            if (_bearLevel > open && !(close < open))
                            {
                                _bearLevel = open;
                                _bearIndex = i;
                            }
                            if (_bearLevel > open && (close < open))
                            {
                                _bearLevel = low;
                                _bearIndex = i;
                            }
                            AddPendingLine(_becisdg, _bearIndex, _bearLevel);
                            break;
                        }
                    }
                }
            }

            // Pine: if high >= h0[1] → ycisd:=false ; if low <= l0[1] → xcisd:=false
            // Note: _h0PrevBar / _l0PrevBar hold the rolling HTF extreme as of the previous bar.
            if (i >= 1)
            {
                if (Bars.HighPrices[i] >= _h0PrevBar) _yCisd = false;
                if (Bars.LowPrices[i]  <= _l0PrevBar) _xCisd = false;
            }

            // ───── trigger conditions ─────
            double closePrev = i >= 1 ? Bars.ClosePrices[i - 1] : close;
            bool xbull = closePrev > _bullLevel
                         && (lSweptPrev() || (_l1 <= _l0 && _lSwept1))
                         && !_xCisd
                         && (i - 1) >= _bullIndex;

            if (xbull)
            {
                string name = "cdc_cisdplus_solid_" + i;
                Chart.DrawRectangle(name, _bullIndex, _bullLevel, i - 1, _bullLevel, CsdColor, 1, LineStyle.Solid);
                Chart.DrawText("cdc_cisdplus_txt_" + i, "cisd+", i - 1, _bullLevel, CsdColor);
                _allDrawNames.Add(name);
                _bullLevel = 1000000.0;
                _xCisd = true;
                if (_bucisdg.Count > 0)
                {
                    Chart.RemoveObject(_bucisdg[0]);
                    _bucisdg.RemoveAt(0);
                }
            }
            // Pine: unconfirmed cisd+ box (no liquidity sweep) — half-faded border
            else if (closePrev > _bullLevel && !(lSweptPrev() || (_l1 <= _l0 && _lSwept1)) && !_xCisd && (i - 1) >= _bullIndex)
            {
                string name = "cdc_cisdplus_faded_" + i;
                Color faded = Color.FromArgb(127, CsdColor.R, CsdColor.G, CsdColor.B);
                Chart.DrawRectangle(name, _bullIndex, _bullLevel, i - 1, _bullLevel, faded, 1, LineStyle.Solid);
                Chart.DrawText("cdc_cisdplus_txt_" + i, "cisd+", i - 1, _bullLevel, faded);
                _allDrawNames.Add(name);
                _bullLevel = 1000000.0;
            }

            // Trim bucisdg to keep only newest pending line
            while (_bucisdg.Count > 1)
            {
                int last = _bucisdg.Count - 1;
                Chart.RemoveObject(_bucisdg[last]);
                _bucisdg.RemoveAt(last);
            }

            bool xbear = closePrev < _bearLevel
                         && (hSweptPrev() || (_h1 >= _h0 && _hSwept1))
                         && !_yCisd
                         && (i - 1) >= _bearIndex;

            if (xbear)
            {
                string name = "cdc_cisdminus_solid_" + i;
                Chart.DrawRectangle(name, _bearIndex, _bearLevel, i - 1, _bearLevel, CsdColor, 1, LineStyle.Solid);
                Chart.DrawText("cdc_cisdminus_txt_" + i, "cisd-", i - 1, _bearLevel, CsdColor);
                _allDrawNames.Add(name);
                _bearLevel = 0.0;
                _yCisd = true;
                if (_becisdg.Count > 0)
                {
                    Chart.RemoveObject(_becisdg[0]);
                    _becisdg.RemoveAt(0);
                }
            }
            else if (closePrev < _bearLevel && !(hSweptPrev() || (_h1 >= _h0 && _hSwept1)) && !_yCisd && (i - 1) >= _bearIndex)
            {
                string name = "cdc_cisdminus_faded_" + i;
                Color faded = Color.FromArgb(127, CsdColor.R, CsdColor.G, CsdColor.B);
                Chart.DrawRectangle(name, _bearIndex, _bearLevel, i - 1, _bearLevel, faded, 1, LineStyle.Solid);
                Chart.DrawText("cdc_cisdminus_txt_" + i, "cisd-", i - 1, _bearLevel, faded);
                _allDrawNames.Add(name);
                _bearLevel = 0.0;
            }

            while (_becisdg.Count > 1)
            {
                int last = _becisdg.Count - 1;
                Chart.RemoveObject(_becisdg[last]);
                _becisdg.RemoveAt(last);
            }
        }

        // Snapshot of _h0 / _l0 as of the previous bar, captured at end of UpdateMainHtfState.
        private double _h0PrevBar, _l0PrevBar;

        // Pine: l_swept[1] / h_swept[1] — state just before the current bar's update.
        // We capture pre-update via UpdateCisdState being called *after* state update; so the
        // booleans we computed in Calculate() reflect the *current* bar. To reproduce [1] semantics,
        // we cache one-bar-back values explicitly.
        private bool _lSweptPrevBar, _hSweptPrevBar;
        private bool lSweptPrev() => _lSweptPrevBar;
        private bool hSweptPrev() => _hSweptPrevBar;

        private void AddPendingLine(List<string> arr, int x, double y)
        {
            string name = "cdc_pending_" + (arr == _bucisdg ? "bull_" : "bear_") + x + "_" + DateTime.UtcNow.Ticks;
            Chart.DrawTrendLine(name, x, y, x + 4, y, CsdColor, 2, LineStyle.Dots);
            arr.Insert(0, name);
            // Cap at 100 (Pine: array.pop when size > 100)
            while (arr.Count > 100)
            {
                int last = arr.Count - 1;
                Chart.RemoveObject(arr[last]);
                arr.RemoveAt(last);
            }
        }

        // ─────────────────── Subsystem I — HTF bias plot ───────────────────
        private void DrawBias(int i)
        {
            if (!PlotBias) return;
            // Square at chart bottom — use a small dot via DrawIcon at the bar's low minus offset
            string name = "cdc_bias_" + i;
            Color c = _bias == 1 ? BullColor : _bias == -1 ? BearColor : Color.Gray;
            double y = Bars.LowPrices[i] - (Bars.HighPrices[i] - Bars.LowPrices[i]) * 0.15;
            Chart.DrawIcon(name, ChartIconType.Square, i, y, c);
        }

        // ───────────────────── Subsystem K — key levels ─────────────────────
        private void DrawKeyLevels(int i)
        {
            if (!ShowKeyLvl) return;

            int rightIdx = i + Ky;

            if (TfOkForKey(KeyTf1) && ShowKeyTf1)
                DrawKeyTriple(i, rightIdx, _key1, KeyTf1, KeyTf1Clr, "k1");

            if (TfOkForKey(KeyTf2) && ShowKeyTf2)
                DrawKeyTriple(i, rightIdx, _key2, KeyTf2, KeyTf2Clr, "k2");

            if (TfOkForKey(KeyTf3) && ShowKeyTf3)
                DrawKeyTriple(i, rightIdx, _key3, KeyTf3, KeyTf3Clr, "k3");
        }

        private void DrawKeyTriple(int i, int rightIdx, MiniHtfState s, TimeFrame tf, Color clr, string tag)
        {
            string addH = s.H0 >= s.H1 ? "(-)" : "";
            string addL = s.L0 <= s.L1 ? "(-)" : "";
            string label = F_Text(tf);

            string nh = "cdc_key_" + tag + "_h";
            string nl = "cdc_key_" + tag + "_l";
            string no = "cdc_key_" + tag + "_o";
            int leftIdx = s.T0Index >= 0 ? s.T0Index : i;

            var lh = Chart.DrawTrendLine(nh, leftIdx, s.H1, rightIdx, s.H1, clr, 1, LineStyle.Dots);
            Chart.DrawText(nh + "_t", "p " + label + " h " + addH, rightIdx, s.H1, KeyTxtColor);

            var ll = Chart.DrawTrendLine(nl, leftIdx, s.L1, rightIdx, s.L1, clr, 1, LineStyle.Dots);
            Chart.DrawText(nl + "_t", "p " + label + " l " + addL, rightIdx, s.L1, KeyTxtColor);

            var lo = Chart.DrawTrendLine(no, leftIdx, s.O0, rightIdx, s.O0, clr, 1, LineStyle.Dots);
            Chart.DrawText(no + "_t", label + " open", rightIdx, s.O0, KeyTxtColor);
        }

        // ───────────────────────── nested mini HTF state ─────────────────────────
        private class MiniHtfState
        {
            public double O0, H0, L0, C0;
            public double O1, H1, L1, C1;
            public DateTime T0, T1;
            public int T0Index = -1;
            private bool _init;

            public void Update(int i, TimeFrame htf, CdSweepCisdCx ind)
            {
                double open  = ind.Bars.OpenPrices[i];
                double high  = ind.Bars.HighPrices[i];
                double low   = ind.Bars.LowPrices[i];
                double close = ind.Bars.ClosePrices[i];
                DateTime t   = ind.Bars.OpenTimes[i];

                if (!_init)
                {
                    O0 = open; H0 = high; L0 = low; C0 = close;
                    O1 = O0; H1 = H0; L1 = L0; C1 = C0;
                    T0 = t; T1 = t; T0Index = i;
                    _init = true;
                }

                if (ind.HtfChanged(i, htf))
                {
                    O1 = O0; H1 = H0; L1 = L0; T1 = T0;
                    C1 = i >= 1 ? ind.Bars.ClosePrices[i - 1] : C0;
                    T0 = t; O0 = open; H0 = high; L0 = low;
                    T0Index = i;
                }
                if (high >= H0) H0 = high;
                if (low  <= L0) L0 = low;
                C0 = close;
            }
        }
    }
}
