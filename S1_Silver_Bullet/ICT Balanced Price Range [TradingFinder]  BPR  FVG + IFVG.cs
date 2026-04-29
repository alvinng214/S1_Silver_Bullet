// =============================================================================
// ICT Balanced Price Range [TradingFinder]  BPR | FVG + IFVG.cs
// C# cTrader port of TFlab's Pine v5 indicator (MPL 2.0)
// -----------------------------------------------------------------------------
// Source: ICT Balanced Price Range [TradingFinder]  BPR  FVG + IFVG.txt
// Plus the four imported TFlab Pine libraries (user-supplied this conversation):
//   * OrderBlockDrawing_TradingFinder/4
//   * FVGDetectorLibrary/3
//   * OrderBlockOverlappingDrawing/1
//   * AlertSenderLibrary_TradingFinder/1
//
// Library NOT available — Dark_Light_Theme_TradingFinder_Switching_Colors_Library/1
//   The Pine call `SC.SwitchingColorMode(color, mode)` adapts a base color to
//   light/dark theme. Without source, this port treats the result as the input
//   color regardless of mode (pass-through). The user can pick an explicit
//   colour per zone via the Bullish/Bearish IFVG/BPR Color parameters, so the
//   visual outcome is fully under their control. The Theme Mode parameter is
//   retained for UI parity but is currently a no-op.
//
// -----------------------------------------------------------------------------
// Architecture: "last-bar-only + ClearObjects + redraw" pattern
//   cTrader's Chart.Draw* APIs do NOT update an object's coords by name when
//   re-called (unlike Pine's line.set_x2). The Pine indicator extends every
//   live line every bar — re-implementing this incrementally would create
//   thousands of duplicate objects. So on every Calculate(last bar) we:
//     1. ClearObjects (only objects we created — name-prefixed)
//     2. Replay the full Pine simulation across all bars
//     3. Draw the final state in one pass
//
// -----------------------------------------------------------------------------
// Pipeline (mirrors the 91-line Pine source):
//   FVGDetector(filter, type)
//        → (DConditionFVG, DDFVG, DPFVG, BarDFVG,           ← bullish FVG
//           SConditionFVG, SDFVG, SPFVG, BarSFVG)            ← bearish FVG
//
//   OBDrawing('Demand', DCondFVG, DDFVG, DPFVG, BarDFVG, …)
//        Tracks the bullish FVG; on mitigation, watches next 4 bars for a
//        close < FVG-distal → forms a Supply Breaker Block (= Bearish IFVG).
//        → (Alert_DFVG, Alert_SIFVG, ProximalPrice_SIFVG, DistalPrice_SIFVG, Index_SIFVG)
//
//   OBDrawing('Supply', SCondFVG, SDFVG, SPFVG, BarSFVG, …)
//        Symmetric — bearish FVG mitigation → Demand Breaker Block (= Bullish IFVG).
//        → (Alert_SFVG, Alert_DIFVG, ProximalPrice_DIFVG, DistalPrice_DIFVG, Index_DIFVG)
//
//   OBOverlappingDrawing('Demand', DCondFVG, DistalPrice_DIFVG, ProximalPrice_DIFVG,
//                                  DDFVG, DPFVG, BarDFVG, …)
//        On a new bullish FVG, computes the overlap with the live bullish IFVG
//        (= Demand IFVG). The overlap zone IS the Bullish BPR.
//
//   OBOverlappingDrawing('Supply', …)  → Bearish BPR (symmetric).
//
//   AlertSender on BPR-mitigation transitions (Pine: Check[1]==true & Check==false).
//
// -----------------------------------------------------------------------------
// State simplification inherited from Pine:
//   The TFlab libraries track ONE active demand FVG / IFVG / BPR at a time
//   (var-scoped state inside the function). On a new trigger, prior state is
//   overwritten — the previously drawn lines remain on chart but are no longer
//   updated. This port preserves that behaviour: each new FVG/IFVG/BPR adds a
//   new ZoneRecord to the drawables list and freezes the previous one's EndBar.
// =============================================================================

using System;
using System.Collections.Generic;
using System.Globalization;
using cAlgo.API;
using cAlgo.API.Indicators;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class IctBalancedPriceRangeTradingFinder : Indicator
    {
        // ── enums (mirror Pine string-option inputs) ────────────────────────────

        public enum FilterTypeOption { VeryAggressive, Aggressive, Defensive, VeryDefensive }
        public enum MitigationLevelOption { Proximal, FiftyPercentOb, Distal }
        public enum ColorThemeOption { Off, Light, Dark }
        public enum AlertOnOffOption { On, Off }
        public enum AlertFrequencyOption { All, OncePerBar, OncePerBarClose }

        // ── parameters: Global Setting ──────────────────────────────────────────

        [Parameter("Show All FVG & IFVG", Group = "Global Setting", DefaultValue = true)]
        public bool ShowAllIFVG { get; set; }

        [Parameter("FVG & IFVG Validity Period (Bar)", Group = "Global Setting", DefaultValue = 500, MinValue = 10, MaxValue = 4998)]
        public int FvgValidity { get; set; }

        [Parameter("Switching Colors Theme Mode", Group = "Global Setting", DefaultValue = ColorThemeOption.Light)]
        public ColorThemeOption SCMode { get; set; }

        // ── parameters: Balanced Price Range ────────────────────────────────────

        [Parameter("Show Bullish BPR", Group = "Balanced Price Range", DefaultValue = true)]
        public bool ShowBullishBpr { get; set; }

        [Parameter("Show Bearish BPR", Group = "Balanced Price Range", DefaultValue = true)]
        public bool ShowBearishBpr { get; set; }

        // Pine: #16dcff69 (RGBA) → cTrader: #AARRGGBB = #6916DCFF
        [Parameter("Bullish BPR Color", Group = "Balanced Price Range", DefaultValue = "#6916DCFF")]
        public Color BullishBprColor { get; set; }

        // Pine: #f1a20f71 (RGBA) → #71F1A20F
        [Parameter("Bearish BPR Color", Group = "Balanced Price Range", DefaultValue = "#71F1A20F")]
        public Color BearishBprColor { get; set; }

        [Parameter("Mitigation Level BPR", Group = "Balanced Price Range", DefaultValue = MitigationLevelOption.Proximal)]
        public MitigationLevelOption MlBpr { get; set; }

        // ── parameters: IFVG & FVG ──────────────────────────────────────────────
        // Pine input names preserved verbatim; semantics: each toggle controls
        // BOTH the FVG zone and the IFVG/Breaker that forms when it is mitigated.

        [Parameter(" Show Bearish IFVG & FVG", Group = "IFVG & FVG", DefaultValue = false)]
        public bool ShowBearishIfvg { get; set; }

        [Parameter(" Show Bullish IFVG & FVG", Group = "IFVG & FVG", DefaultValue = false)]
        public bool ShowBullishIfvg { get; set; }

        // Pine: #4caf4f27 → #274CAF4F (greenish — paints the Bullish FVG zone in the Demand pipeline)
        [Parameter("Bearish IFVG/FVG Color", Group = "IFVG & FVG", DefaultValue = "#274CAF4F")]
        public Color BearishIfvgColor { get; set; }

        // Pine: #ff52522f → #2FFF5252 (reddish — paints the Bearish IFVG / breaker)
        [Parameter("Bullish IFVG/FVG Color", Group = "IFVG & FVG", DefaultValue = "#2FFF5252")]
        public Color BullishIfvgColor { get; set; }

        [Parameter("Mitigation Level  IFVG & FVG", Group = "IFVG & FVG", DefaultValue = MitigationLevelOption.Proximal)]
        public MitigationLevelOption MlIfvg { get; set; }

        // ── parameters: FVG ─────────────────────────────────────────────────────

        [Parameter("FVG Filter", Group = "FVG", DefaultValue = true)]
        public bool FvgFilterEnabled { get; set; }

        [Parameter("FVG Filter Type", Group = "FVG", DefaultValue = FilterTypeOption.Defensive)]
        public FilterTypeOption FvgFilterType { get; set; }

        // ── parameters: Alert ───────────────────────────────────────────────────

        [Parameter("Alerts Name", Group = "Alert", DefaultValue = "BPR [TradingFinder]")]
        public string AlertNameParam { get; set; }

        [Parameter("Alert BPR Mitigation", Group = "Alert", DefaultValue = AlertOnOffOption.On)]
        public AlertOnOffOption AlertBprMitigation { get; set; }

        [Parameter("Message Frequency", Group = "Alert", DefaultValue = AlertFrequencyOption.OncePerBar)]
        public AlertFrequencyOption MessageFrequency { get; set; }

        [Parameter("Show Alert time by Time Zone", Group = "Alert", DefaultValue = "UTC")]
        public string AlertTimeZoneStr { get; set; }

        // ── object naming ───────────────────────────────────────────────────────

        private const string Prefix = "BPR_TF_";
        private int _objId;

        // ── ATR (used by FVGDetector filter — Pine ta.atr(55)) ──────────────────

        private AverageTrueRange _atr;

        // ── drawable records ────────────────────────────────────────────────────

        private enum ZoneKind { FvgBull, FvgBear, ObDemand, ObSupply, IfvgDemand, IfvgSupply, BprBull, BprBear }

        private sealed class ZoneRecord
        {
            public ZoneKind  Kind;
            public int       OriginBar;
            public int       EndBar;       // updated each bar while alive; frozen on mitigation / new trigger
            public double    Distal;       // far edge of the zone (semantics depend on Kind — see drawing helper)
            public double    Proximal;     // near edge
            public bool      Mitigated;
            public Color     ZoneColor;    // fill colour (already theme-resolved)
        }

        private readonly List<ZoneRecord> _zones = new List<ZoneRecord>();

        // ── FVGDetector state (Pine library var-scoped vars) ────────────────────

        private double _ddfvg, _dpfvg, _ddfvgPrev;
        private double _sdfvg, _spfvg, _sdfvgPrev;
        private int    _barDfvg, _barSfvg;
        private bool   _dMitigated = true, _sMitigated = true;
        private ZoneRecord _liveBullFvg, _liveBearFvg;

        // ── OBDrawing state — DEMAND instance (tracks bullish FVG → Bearish IFVG) ─

        private double _demDist, _demProx;
        private int    _demIndex;
        private bool   _demCheck = true, _demCheckPrev = true;
        private double _demBbDist, _demBbProx;
        private int    _demBbIndex;
        private bool   _demBbCheck = false, _demBbCheckPrev = false;
        private bool   _demCbb0, _demCbb1, _demCbb2, _demCbb3;   // CBB lookback window (current + 3 prior)
        private bool   _demTrigBb;                               // TriggerCondition_BB this bar
        private ZoneRecord _liveDemObZone;
        private ZoneRecord _liveSupIfvgZone;                     // bearish breaker (= Supply IFVG)

        // ── OBDrawing state — SUPPLY instance (tracks bearish FVG → Bullish IFVG) ─

        private double _supDist, _supProx;
        private int    _supIndex;
        private bool   _supCheck = true, _supCheckPrev = true;
        private double _supBbDist, _supBbProx;
        private int    _supBbIndex;
        private bool   _supBbCheck = false, _supBbCheckPrev = false;
        private bool   _supCbb0, _supCbb1, _supCbb2, _supCbb3;
        private bool   _supTrigBb;
        private ZoneRecord _liveSupObZone;
        private ZoneRecord _liveDemIfvgZone;                     // bullish breaker (= Demand IFVG)

        // ── OBOverlappingDrawing state — Bullish BPR instance ───────────────────

        private double _demBprDist, _demBprProx;
        private int    _demBprIndex;
        private int    _demBprBar;                               // = Index_Curr last consumed (dedup)
        private bool   _demBprCheck = true, _demBprCheckPrev = true;
        private bool   _demBprTrigger;
        private ZoneRecord _liveDemBprZone;

        // ── OBOverlappingDrawing state — Bearish BPR instance ───────────────────

        private double _supBprDist, _supBprProx;
        private int    _supBprIndex;
        private int    _supBprBar;
        private bool   _supBprCheck = true, _supBprCheckPrev = true;
        private bool   _supBprTrigger;
        private ZoneRecord _liveSupBprZone;

        // ── alert dedup (Once-Per-Bar) ──────────────────────────────────────────

        private DateTime _lastDemBprAlertBar = DateTime.MinValue;
        private DateTime _lastSupBprAlertBar = DateTime.MinValue;

        // ─────────────────────────────────────────────────────────────────────────

        protected override void Initialize()
        {
            // Pine ta.atr(55) — Wilder/RMA. cTrader Exponential is the closest built-in.
            _atr = Indicators.AverageTrueRange(55, MovingAverageType.Exponential);
        }

        public override void Calculate(int index)
        {
            if (index != Bars.Count - 1) return;

            ClearMyObjects();
            ResetAllState();

            int n = Bars.Count;
            for (int i = 0; i < n; i++)
                ProcessBar(i, lastIndex: n - 1);

            // Any zones still live extend to the last bar
            DateTime lastUtc = Bars.OpenTimes[n - 1];
            DateTime nextUtc = Bars.OpenTimes[n - 1].AddTicks(1);

            DrawAll(lastUtc, nextUtc);
        }

        // ────────────────────────────────────────────────────────────────────────
        //  PER-BAR PIPELINE
        // ────────────────────────────────────────────────────────────────────────

        private void ProcessBar(int i, int lastIndex)
        {
            if (i < 2) return;

            // Snapshot end-of-prev-bar values (Pine `[1]`) BEFORE this bar mutates them
            double ddfvgPrevSnap = _ddfvg;
            double sdfvgPrevSnap = _sdfvg;
            bool   demCheckLast  = _demCheckPrev;
            bool   supCheckLast  = _supCheckPrev;
            bool   demBbCheckLast = _demBbCheckPrev;
            bool   supBbCheckLast = _supBbCheckPrev;
            bool   demBprCheckLast = _demBprCheckPrev;
            bool   supBprCheckLast = _supBprCheckPrev;

            // 1) FVG Detector
            (bool dCondFvg, bool sCondFvg) = RunFvgDetector(i, ddfvgPrevSnap, sdfvgPrevSnap);

            // 2) Demand OBDrawing → Supply IFVG breaker
            RunDemandObDrawing(i, dCondFvg, demCheckLast, demBbCheckLast);

            // 3) Supply OBDrawing → Demand IFVG breaker
            RunSupplyObDrawing(i, sCondFvg, supCheckLast, supBbCheckLast);

            // 4) Demand BPR (overlap of new bullish FVG with live bullish IFVG)
            RunDemandBprOverlap(i, dCondFvg, demBprCheckLast);

            // 5) Supply BPR
            RunSupplyBprOverlap(i, sCondFvg, supBprCheckLast);

            // Save end-of-bar values for next bar's `[1]` reads
            _demCheckPrev    = _demCheck;
            _supCheckPrev    = _supCheck;
            _demBbCheckPrev  = _demBbCheck;
            _supBbCheckPrev  = _supBbCheck;
            _demBprCheckPrev = _demBprCheck;
            _supBprCheckPrev = _supBprCheck;

            // Shift CBB history (CBB[3]=CBB[2], CBB[2]=CBB[1], CBB[1]=CBB[0])
            _demCbb3 = _demCbb2; _demCbb2 = _demCbb1; _demCbb1 = _demCbb0; _demCbb0 = false;
            _supCbb3 = _supCbb2; _supCbb2 = _supCbb1; _supCbb1 = _supCbb0; _supCbb0 = false;

            // Extend live zones to current bar
            ExtendLiveZones(i);
        }

        // ────────────────────────────────────────────────────────────────────────
        //  1) FVG Detector  (Pine: FVGDetectorLibrary/3)
        // ────────────────────────────────────────────────────────────────────────

        private (bool dCondFvg, bool sCondFvg) RunFvgDetector(int i, double ddfvgPrevSnap, double sdfvgPrevSnap)
        {
            double atr = (i >= 55 && !double.IsNaN(_atr.Result[i])) ? _atr.Result[i] : 0.0;

            double o0 = Bars.OpenPrices[i],   h0 = Bars.HighPrices[i],   l0 = Bars.LowPrices[i],   c0 = Bars.ClosePrices[i];
            double o1 = Bars.OpenPrices[i-1], h1 = Bars.HighPrices[i-1], l1 = Bars.LowPrices[i-1], c1 = Bars.ClosePrices[i-1];
            double o2 = Bars.OpenPrices[i-2], h2 = Bars.HighPrices[i-2], l2 = Bars.LowPrices[i-2], c2 = Bars.ClosePrices[i-2];

            bool dCondFvg, sCondFvg;

            if (!FvgFilterEnabled)
            {
                dCondFvg = l0 > h2;
                sCondFvg = h0 < l2;
            }
            else
            {
                // ── Bullish (Demand) ────
                bool baseBull = (l0 > h2) && (h0 > h1);
                switch (FvgFilterType)
                {
                    case FilterTypeOption.VeryAggressive:
                        dCondFvg = baseBull;
                        break;
                    case FilterTypeOption.Aggressive:
                        dCondFvg = baseBull && (h1 - l1) >= 1.0 * atr;
                        break;
                    case FilterTypeOption.Defensive:
                    {
                        bool size  = (h1 - l1) >= 1.5 * atr;
                        bool body  = ((c2 - o2 > 0 && c1 - o1 > 0)
                                      || Math.Abs((c1 - o1) / NonZero(h1 - l1)) > 0.7);
                        dCondFvg = (l0 > h2) && size && (h0 > h1) && body;
                        break;
                    }
                    case FilterTypeOption.VeryDefensive:
                    default:
                    {
                        bool size  = (h1 - l1) >= 1.5 * atr;
                        bool body  = ((c2 - o2 > 0 && c1 - o1 > 0)
                                      && Math.Abs((c1 - o1) / NonZero(h1 - l1)) > 0.7)
                                      && Math.Abs((c2 - o2) / NonZero(h2 - l2)) > 0.35
                                      && Math.Abs((c0 - o0) / NonZero(h0 - l0)) > 0.35;
                        dCondFvg = (l0 > h2) && size && (h0 > h1) && body;
                        break;
                    }
                }

                // ── Bearish (Supply) ────
                bool baseBear = (h0 < l2) && (l1 > l0);
                switch (FvgFilterType)
                {
                    case FilterTypeOption.VeryAggressive:
                        sCondFvg = baseBear;
                        break;
                    case FilterTypeOption.Aggressive:
                        sCondFvg = baseBear && (h1 - l1) >= 1.0 * atr;
                        break;
                    case FilterTypeOption.Defensive:
                    {
                        bool size  = (h1 - l1) >= 1.5 * atr;
                        bool body  = ((c2 - o2 < 0 && c1 - o1 < 0)
                                      || Math.Abs((c1 - o1) / NonZero(h1 - l1)) > 0.7);
                        sCondFvg = (h0 < l2) && size && (l1 > l0) && body;
                        break;
                    }
                    case FilterTypeOption.VeryDefensive:
                    default:
                    {
                        bool size  = (h1 - l1) >= 1.5 * atr;
                        bool body  = ((c2 - o2 < 0 && c1 - o1 < 0)
                                      && Math.Abs((c1 - o1) / NonZero(h1 - l1)) > 0.7)
                                      && Math.Abs((c2 - o2) / NonZero(h2 - l2)) > 0.35
                                      && Math.Abs((c0 - o0) / NonZero(h0 - l0)) > 0.35;
                        sCondFvg = (h0 < l2) && size && (l1 > l0) && body;
                        break;
                    }
                }
            }

            // ── update DDFVG / DPFVG and seed a new zone record ─────────────────
            if (dCondFvg)
            {
                FreezeIfLive(_liveBullFvg, i);
                _ddfvg   = h2;
                _dpfvg   = l0;
                _barDfvg = i;
                _dMitigated = true;
                // Note: ShowAllIFVG controls whether ALL FVG/IFVG history is shown.
                // The actual visibility flag for Bullish FVG = ShowBearishIfvg
                // (Pine `Show=PShowDeIFVG` for the Demand OBDrawing call).
                _liveBullFvg = NewZone(ZoneKind.FvgBull, i, _ddfvg, _dpfvg, ResolveColor(BearishIfvgColor));
            }
            if (sCondFvg)
            {
                FreezeIfLive(_liveBearFvg, i);
                _sdfvg   = l2;
                _spfvg   = h0;
                _barSfvg = i;
                _sMitigated = true;
                _liveBearFvg = NewZone(ZoneKind.FvgBear, i, _sdfvg, _spfvg, ResolveColor(BullishIfvgColor));
            }

            // Bullish FVG mitigation (price entered through proximal=top from above)
            if (_liveBullFvg != null && i >= 1 && _dMitigated)
            {
                double l1now = Bars.LowPrices[i-1];
                double l0now = Bars.LowPrices[i];
                if (l1now >= _dpfvg && l0now <= _dpfvg)
                {
                    _dMitigated = false;
                    _liveBullFvg.Mitigated = true;
                    _liveBullFvg.EndBar    = i;
                }
            }

            // Bearish FVG mitigation
            if (_liveBearFvg != null && i >= 1 && _sMitigated)
            {
                double h1now = Bars.HighPrices[i-1];
                double h0now = Bars.HighPrices[i];
                if (h1now <= _spfvg && h0now >= _spfvg)
                {
                    _sMitigated = false;
                    _liveBearFvg.Mitigated = true;
                    _liveBearFvg.EndBar    = i;
                }
            }

            return (dCondFvg, sCondFvg);
        }

        // ────────────────────────────────────────────────────────────────────────
        //  2) Demand OBDrawing  (tracks bullish FVG → Bearish IFVG / Supply BB)
        // ────────────────────────────────────────────────────────────────────────

        private void RunDemandObDrawing(int i, bool triggerCond, bool checkLast, bool bbCheckLast)
        {
            // Pine: TriggerCondition updates DistalPrice/ProximalPrice
            if (triggerCond)
            {
                _demDist  = _ddfvg;
                _demProx  = _dpfvg;
                _demIndex = _barDfvg;
            }

            // Mitigation level for the demand OB (bullish zone — proximal = upper edge)
            double ml = MlIfvg switch
            {
                MitigationLevelOption.Proximal       => _demProx,
                MitigationLevelOption.Distal         => _demDist,
                MitigationLevelOption.FiftyPercentOb => 0.5 * (_demProx + _demDist),
                _ => _demProx
            };

            // ── primary zone (Demand FVG / OB) ──────────────────────────────────
            if (triggerCond && ShowBearishIfvg)
            {
                // Pine: line.new on a new trigger
                FreezeIfLive(_liveDemObZone, i);
                _liveDemObZone = NewZone(ZoneKind.ObDemand, _demIndex, _demDist, _demProx,
                                         ResolveColor(BearishIfvgColor));
            }

            // Mitigation: low < ML  OR  bar count ≥ validity
            // (Pine: `if (low < ML) or (bar_index - Index) >= OBValidDis: Check := false`)
            double l0 = Bars.LowPrices[i];
            if ((l0 < ml) || (i - _demIndex) >= FvgValidity)
            {
                if (_demCheck && _liveDemObZone != null)
                {
                    _liveDemObZone.Mitigated = true;
                    _liveDemObZone.EndBar    = i;
                }
                _demCheck = false;
            }

            if (triggerCond)
                _demCheck = true;

            // ── breaker block formation (Demand FVG → Supply BB / Bearish IFVG) ─
            // Pine: CBB := Check[1] == true and Check == false
            _demCbb0 = checkLast && !_demCheck;

            // Pine: if (CBB[1] or CBB[2] or CBB[3] or CBB) and Check_BB == false:
            //         if close < DistalPrice: form Supply BB
            // CBB lookback window: positions [1], [2], [3] are PREVIOUS bars
            // (since we haven't shifted yet at this point in the bar)
            bool cbbWindow = _demCbb0 || _demCbb1 || _demCbb2 || _demCbb3;
            double c0 = Bars.ClosePrices[i];

            if (cbbWindow && !_demBbCheck)
            {
                if (c0 < _demDist)
                {
                    _demBbIndex = i;
                    _demBbCheck = true;
                    _demBbDist  = _demProx;   // BB inverts edges
                    _demBbProx  = _demDist;
                }
            }

            // BB just formed this bar (Pine: Check_BB[1] == false and Check_BB == true)
            _demTrigBb = !bbCheckLast && _demBbCheck;
            if (_demTrigBb && ShowBearishIfvg)
            {
                FreezeIfLive(_liveSupIfvgZone, i);
                _liveSupIfvgZone = NewZone(ZoneKind.IfvgSupply, _demBbIndex, _demBbDist, _demBbProx,
                                           ResolveColor(BullishIfvgColor));
            }

            // BB mitigation level (Supply zone — proximal = lower edge for incoming bull retracement)
            double mlBb = MlIfvg switch
            {
                MitigationLevelOption.Proximal       => _demBbProx,
                MitigationLevelOption.Distal         => _demBbDist,
                MitigationLevelOption.FiftyPercentOb => 0.5 * (_demBbProx + _demBbDist),
                _ => _demBbProx
            };

            // BB mitigation: high > ML_BB  OR  bar count ≥ validity  OR  primary just got mitigated this bar
            double h0 = Bars.HighPrices[i];
            if ((h0 > mlBb) || (i - _demBbIndex) >= FvgValidity || (checkLast && !_demCheck))
            {
                if (_demBbCheck && _liveSupIfvgZone != null)
                {
                    _liveSupIfvgZone.Mitigated = true;
                    _liveSupIfvgZone.EndBar    = i;
                }
                _demBbCheck = false;
            }
        }

        // ────────────────────────────────────────────────────────────────────────
        //  3) Supply OBDrawing  (tracks bearish FVG → Bullish IFVG / Demand BB)
        // ────────────────────────────────────────────────────────────────────────

        private void RunSupplyObDrawing(int i, bool triggerCond, bool checkLast, bool bbCheckLast)
        {
            if (triggerCond)
            {
                _supDist  = _sdfvg;
                _supProx  = _spfvg;
                _supIndex = _barSfvg;
            }

            // Mitigation level for the supply OB (bearish zone — proximal = lower edge)
            double ml = MlIfvg switch
            {
                MitigationLevelOption.Proximal       => _supProx,
                MitigationLevelOption.Distal         => _supDist,
                MitigationLevelOption.FiftyPercentOb => 0.5 * (_supProx + _supDist),
                _ => _supProx
            };

            if (triggerCond && ShowBullishIfvg)
            {
                FreezeIfLive(_liveSupObZone, i);
                _liveSupObZone = NewZone(ZoneKind.ObSupply, _supIndex, _supDist, _supProx,
                                         ResolveColor(BullishIfvgColor));
            }

            double h0 = Bars.HighPrices[i];
            if ((h0 > ml) || (i - _supIndex) >= FvgValidity)
            {
                if (_supCheck && _liveSupObZone != null)
                {
                    _liveSupObZone.Mitigated = true;
                    _liveSupObZone.EndBar    = i;
                }
                _supCheck = false;
            }

            if (triggerCond)
                _supCheck = true;

            // Breaker formation: Supply mitigated, then close > DistalPrice → Demand BB / Bullish IFVG
            _supCbb0 = checkLast && !_supCheck;
            bool cbbWindow = _supCbb0 || _supCbb1 || _supCbb2 || _supCbb3;
            double c0 = Bars.ClosePrices[i];

            if (cbbWindow && !_supBbCheck)
            {
                if (c0 > _supDist)
                {
                    _supBbIndex = i;
                    _supBbCheck = true;
                    _supBbDist  = _supProx;
                    _supBbProx  = _supDist;
                }
            }

            _supTrigBb = !bbCheckLast && _supBbCheck;
            if (_supTrigBb && ShowBullishIfvg)
            {
                FreezeIfLive(_liveDemIfvgZone, i);
                _liveDemIfvgZone = NewZone(ZoneKind.IfvgDemand, _supBbIndex, _supBbDist, _supBbProx,
                                           ResolveColor(BearishIfvgColor));
            }

            double mlBb = MlIfvg switch
            {
                MitigationLevelOption.Proximal       => _supBbProx,
                MitigationLevelOption.Distal         => _supBbDist,
                MitigationLevelOption.FiftyPercentOb => 0.5 * (_supBbProx + _supBbDist),
                _ => _supBbProx
            };

            double l0 = Bars.LowPrices[i];
            if ((l0 < mlBb) || (i - _supBbIndex) >= FvgValidity || (checkLast && !_supCheck))
            {
                if (_supBbCheck && _liveDemIfvgZone != null)
                {
                    _liveDemIfvgZone.Mitigated = true;
                    _liveDemIfvgZone.EndBar    = i;
                }
                _supBbCheck = false;
            }
        }

        // ────────────────────────────────────────────────────────────────────────
        //  4) Demand BPR overlap  (new bullish FVG ∩ live bullish IFVG)
        // ────────────────────────────────────────────────────────────────────────
        // Bullish-zone semantics: distal = lower edge, proximal = upper edge.
        //   Case A: new FVG ABOVE pre IFVG with overlap
        //           (proxC ≥ proxP) ∧ (distC ≤ proxP) ∧ (distC ≥ distP)
        //           → BPR proximal = proxP, distal = distC
        //   Case B: new FVG BELOW pre IFVG with overlap
        //           (proxC ≤ proxP) ∧ (distP ≤ proxC) ∧ (distC ≤ distP)
        //           → BPR proximal = proxC, distal = distP
        //   Case C: new FVG INSIDE pre IFVG (or partial inner)
        //           (proxC ≤ proxP) ∧ (distP ≤ proxC)         [distC > distP implicit]
        //           → BPR proximal = proxC, distal = distC
        // ────────────────────────────────────────────────────────────────────────

        private void RunDemandBprOverlap(int i, bool triggerOrigin, bool checkLast)
        {
            // Pine: dedup with `Bar != Index_Curr` — the `Index_Curr` is the new FVG's bar.
            // Each call passes `BarDFVG`. Trigger only fires once per BarDFVG value.
            bool fired = false;

            if (triggerOrigin && _liveDemIfvgZone != null && _demBprBar != _barDfvg)
            {
                _demBprBar = _barDfvg;

                double distP = _liveDemIfvgZone.Distal;     // pre IFVG distal (lower edge of bullish zone)
                double proxP = _liveDemIfvgZone.Proximal;   // pre IFVG proximal (upper edge)
                double distC = _ddfvg;                      // new FVG distal
                double proxC = _dpfvg;                      // new FVG proximal

                if (proxC >= proxP && distC <= proxP && distC >= distP)
                {
                    _demBprProx = proxP;
                    _demBprDist = distC;
                    _demBprTrigger = true; fired = true;
                }
                else if (proxC <= proxP && distP <= proxC && distC <= distP)
                {
                    _demBprProx = proxC;
                    _demBprDist = distP;
                    _demBprTrigger = true; fired = true;
                }
                else if (proxC <= proxP && distP <= proxC)
                {
                    _demBprProx = proxC;
                    _demBprDist = distC;
                    _demBprTrigger = true; fired = true;
                }
                else
                {
                    _demBprTrigger = false;
                }
            }

            if (fired)
            {
                _demBprIndex = _barDfvg;

                if (ShowBullishBpr)
                {
                    FreezeIfLive(_liveDemBprZone, i);
                    _liveDemBprZone = NewZone(ZoneKind.BprBull, _demBprIndex, _demBprDist, _demBprProx,
                                              ResolveColor(BullishBprColor));
                }
            }

            // Mitigation level
            double ml = MlBpr switch
            {
                MitigationLevelOption.Proximal       => _demBprProx,
                MitigationLevelOption.Distal         => _demBprDist,
                MitigationLevelOption.FiftyPercentOb => 0.5 * (_demBprProx + _demBprDist),
                _ => _demBprProx
            };

            double l0 = Bars.LowPrices[i];
            if ((l0 < ml) || (i - _demBprIndex) >= FvgValidity)
            {
                if (_demBprCheck && _liveDemBprZone != null)
                {
                    _liveDemBprZone.Mitigated = true;
                    _liveDemBprZone.EndBar    = i;
                }
                _demBprCheck = false;
            }

            if (fired)
            {
                _demBprCheck = true;
                _demBprTrigger = false;
            }

            // Alert: Check[1]==true and Check==false  (mitigation transition)
            if (checkLast && !_demBprCheck && AlertBprMitigation == AlertOnOffOption.On)
                MaybeFireAlert(i, isBullish: true);
        }

        // ────────────────────────────────────────────────────────────────────────
        //  5) Supply BPR overlap (new bearish FVG ∩ live bearish IFVG)
        // ────────────────────────────────────────────────────────────────────────
        //   Supply-zone semantics: distal = upper edge, proximal = lower edge.
        //   Case A: new FVG BELOW pre IFVG with overlap
        //           (distC ≥ distP) ∧ (distP ≥ proxC) ∧ (proxP ≤ proxC)
        //   Case B: new FVG ABOVE pre IFVG with overlap
        //           (distP ≥ distC) ∧ (proxP ≤ distC) ∧ (proxP ≥ proxC)
        //   Case C: new FVG INSIDE pre IFVG (partial outer)
        //           (distP ≥ distC) ∧ (proxC ≥ proxP)
        // ────────────────────────────────────────────────────────────────────────

        private void RunSupplyBprOverlap(int i, bool triggerOrigin, bool checkLast)
        {
            bool fired = false;

            if (triggerOrigin && _liveSupIfvgZone != null && _supBprBar != _barSfvg)
            {
                _supBprBar = _barSfvg;

                double distP = _liveSupIfvgZone.Distal;
                double proxP = _liveSupIfvgZone.Proximal;
                double distC = _sdfvg;
                double proxC = _spfvg;

                if (distC >= distP && distP >= proxC && proxP <= proxC)
                {
                    _supBprProx = proxC;
                    _supBprDist = distP;
                    _supBprTrigger = true; fired = true;
                }
                else if (distP >= distC && proxP <= distC && proxP >= proxC)
                {
                    _supBprProx = proxP;
                    _supBprDist = distC;
                    _supBprTrigger = true; fired = true;
                }
                else if (distP >= distC && proxC >= proxP)
                {
                    _supBprProx = proxC;
                    _supBprDist = distC;
                    _supBprTrigger = true; fired = true;
                }
            }

            if (fired)
            {
                _supBprIndex = _barSfvg;

                if (ShowBearishBpr)
                {
                    FreezeIfLive(_liveSupBprZone, i);
                    _liveSupBprZone = NewZone(ZoneKind.BprBear, _supBprIndex, _supBprDist, _supBprProx,
                                              ResolveColor(BearishBprColor));
                }
            }

            double ml = MlBpr switch
            {
                MitigationLevelOption.Proximal       => _supBprProx,
                MitigationLevelOption.Distal         => _supBprDist,
                MitigationLevelOption.FiftyPercentOb => 0.5 * (_supBprProx + _supBprDist),
                _ => _supBprProx
            };

            double h0 = Bars.HighPrices[i];
            if ((h0 > ml) || (i - _supBprIndex) >= FvgValidity)
            {
                if (_supBprCheck && _liveSupBprZone != null)
                {
                    _liveSupBprZone.Mitigated = true;
                    _liveSupBprZone.EndBar    = i;
                }
                _supBprCheck = false;
            }

            if (fired)
            {
                _supBprCheck = true;
                _supBprTrigger = false;
            }

            if (checkLast && !_supBprCheck && AlertBprMitigation == AlertOnOffOption.On)
                MaybeFireAlert(i, isBullish: false);
        }

        // ────────────────────────────────────────────────────────────────────────
        //  Zone bookkeeping
        // ────────────────────────────────────────────────────────────────────────

        private ZoneRecord NewZone(ZoneKind kind, int originBar, double distal, double proximal, Color zoneColor)
        {
            var z = new ZoneRecord
            {
                Kind      = kind,
                OriginBar = originBar,
                EndBar    = originBar,
                Distal    = distal,
                Proximal  = proximal,
                Mitigated = false,
                ZoneColor = zoneColor
            };
            _zones.Add(z);
            return z;
        }

        private static void FreezeIfLive(ZoneRecord z, int currentBar)
        {
            if (z != null && !z.Mitigated)
                z.EndBar = currentBar;
        }

        private void ExtendLiveZones(int i)
        {
            // While a zone is alive, its end follows the current bar (Pine: line.set_x2)
            if (_liveBullFvg     != null && !_liveBullFvg.Mitigated)     _liveBullFvg.EndBar     = i;
            if (_liveBearFvg     != null && !_liveBearFvg.Mitigated)     _liveBearFvg.EndBar     = i;
            if (_liveDemObZone   != null && !_liveDemObZone.Mitigated && _demCheck)   _liveDemObZone.EndBar   = i;
            if (_liveSupObZone   != null && !_liveSupObZone.Mitigated && _supCheck)   _liveSupObZone.EndBar   = i;
            if (_liveSupIfvgZone != null && !_liveSupIfvgZone.Mitigated && _demBbCheck) _liveSupIfvgZone.EndBar = i;
            if (_liveDemIfvgZone != null && !_liveDemIfvgZone.Mitigated && _supBbCheck) _liveDemIfvgZone.EndBar = i;
            if (_liveDemBprZone  != null && !_liveDemBprZone.Mitigated && _demBprCheck) _liveDemBprZone.EndBar  = i;
            if (_liveSupBprZone  != null && !_liveSupBprZone.Mitigated && _supBprCheck) _liveSupBprZone.EndBar  = i;
        }

        // ────────────────────────────────────────────────────────────────────────
        //  Drawing — replays the final state in one pass
        // ────────────────────────────────────────────────────────────────────────

        private void DrawAll(DateTime lastUtc, DateTime nextUtc)
        {
            // Honor ShowAll: if false, only draw the most recent zone of each kind
            if (!ShowAllIFVG)
            {
                _zones.RemoveAll(z =>
                    (z.Kind == ZoneKind.FvgBull     && z != _liveBullFvg) ||
                    (z.Kind == ZoneKind.FvgBear     && z != _liveBearFvg) ||
                    (z.Kind == ZoneKind.ObDemand    && z != _liveDemObZone) ||
                    (z.Kind == ZoneKind.ObSupply    && z != _liveSupObZone) ||
                    (z.Kind == ZoneKind.IfvgSupply  && z != _liveSupIfvgZone) ||
                    (z.Kind == ZoneKind.IfvgDemand  && z != _liveDemIfvgZone));
            }

            // BPR ShowAll is hardcoded TRUE in the Pine call OBOverlappingDrawing(..., true, ..., ...)
            // — so all BPR history is preserved regardless of ShowAllIFVG.

            foreach (var z in _zones)
            {
                if (!ShouldDraw(z)) continue;
                DrawZone(z);
            }
        }

        private bool ShouldDraw(ZoneRecord z)
        {
            return z.Kind switch
            {
                ZoneKind.FvgBull    => ShowBearishIfvg,   // Pine: Show=PShowDeIFVG on Demand call
                ZoneKind.FvgBear    => ShowBullishIfvg,
                ZoneKind.ObDemand   => ShowBearishIfvg,
                ZoneKind.ObSupply   => ShowBullishIfvg,
                ZoneKind.IfvgSupply => ShowBearishIfvg,   // Show_BB also bound to PShowDeIFVG
                ZoneKind.IfvgDemand => ShowBullishIfvg,
                ZoneKind.BprBull    => ShowBullishBpr,
                ZoneKind.BprBear    => ShowBearishBpr,
                _ => false
            };
        }

        private void DrawZone(ZoneRecord z)
        {
            DateTime t1 = Bars.OpenTimes[z.OriginBar];
            DateTime t2 = Bars.OpenTimes[Math.Max(z.OriginBar, Math.Min(z.EndBar, Bars.Count - 1))];

            double top    = Math.Max(z.Distal, z.Proximal);
            double bottom = Math.Min(z.Distal, z.Proximal);

            string id = $"{Prefix}{z.Kind}_{NextId()}";

            // Filled rectangle = the zone fill (Pine linefill.new)
            Chart.DrawRectangle(id + "_fill", t1, top, t2, bottom, z.ZoneColor, 1, LineStyle.Solid)
                 .IsFilled = true;

            // Distal line (dashed, dark — Pine color.rgb(0,0,0,45))
            Color edge = Color.FromArgb(140, 0, 0, 0);
            Chart.DrawTrendLine(id + "_dist", t1, z.Distal, t2, z.Distal, edge, 1, LineStyle.LinesDots);

            // Proximal line (dashed, dark)
            Chart.DrawTrendLine(id + "_prox", t1, z.Proximal, t2, z.Proximal, edge, 1, LineStyle.LinesDots);

            // 50% line (dotted, full black — Pine color.rgb(0,0,0))
            double mid = 0.5 * (z.Distal + z.Proximal);
            Chart.DrawTrendLine(id + "_mid", t1, mid, t2, mid,
                                Color.FromArgb(255, 0, 0, 0), 1, LineStyle.Dots);

            // FVG label (Pine FVGDetector adds a small "FVG" label at the origin)
            if (z.Kind == ZoneKind.FvgBull || z.Kind == ZoneKind.FvgBear)
            {
                int midBar = (z.OriginBar + Math.Min(z.EndBar, Bars.Count - 1)) / 2;
                DateTime tm = Bars.OpenTimes[Math.Max(0, Math.Min(midBar, Bars.Count - 1))];
                double labelY = z.Kind == ZoneKind.FvgBull ? z.Distal : z.Distal;
                var txt = Chart.DrawText(id + "_lbl", "FVG", tm, labelY, Color.Black);
                txt.HorizontalAlignment = HorizontalAlignment.Center;
                txt.VerticalAlignment   = z.Kind == ZoneKind.FvgBull
                    ? VerticalAlignment.Top
                    : VerticalAlignment.Bottom;
            }
        }

        // ────────────────────────────────────────────────────────────────────────
        //  Theme color resolver — Pine SC.SwitchingColorMode(color, mode)
        //  Source library not available; pass-through with a documented note.
        //  User-picked color parameters take precedence in all cases.
        // ────────────────────────────────────────────────────────────────────────

        private Color ResolveColor(Color baseColor)
        {
            // SCMode is preserved as a UI parameter for parity. The TFlab theme
            // library was not provided so we cannot reproduce its exact tint
            // adjustments — return the user's chosen color verbatim.
            return baseColor;
        }

        // ────────────────────────────────────────────────────────────────────────
        //  Alerts — Pine AlertSenderLibrary AlertSender(...)
        // ────────────────────────────────────────────────────────────────────────

        private void MaybeFireAlert(int barIndex, bool isBullish)
        {
            if (barIndex >= Bars.Count) return;

            DateTime barTime = Bars.OpenTimes[barIndex];

            // Frequency: All / OncePerBar / OncePerBarClose
            if (MessageFrequency == AlertFrequencyOption.OncePerBar ||
                MessageFrequency == AlertFrequencyOption.OncePerBarClose)
            {
                ref DateTime lastBar = ref (isBullish ? ref _lastDemBprAlertBar : ref _lastSupBprAlertBar);
                if (lastBar == barTime) return;
                lastBar = barTime;
            }

            // OncePerBarClose: only fire on closed bars (any bar except the forming one)
            if (MessageFrequency == AlertFrequencyOption.OncePerBarClose && barIndex == Bars.Count - 1)
                return;

            string typeStr   = isBullish ? "Long Order Block Signal"  : "Short Order Block Signal";
            string sideStr   = isBullish ? "Bullish"                  : "Bearish";
            string posStr    = isBullish ? "Long Position in Bullish BPR"
                                          : "Short Position in Bearish BPR";

            // Replicate the Pine AlertSender format (MoreInfo='Off')
            // ⏰ Alert Name / ‼️ Alert Type / 🔠 Symbol / 📈 Time Frame / 📩 Message
            string tf = TimeFrameLabel();
            string msg = string.Concat(
                "⏰Alert Name: ", AlertNameParam,
                "\n‼️Alert Type: ", typeStr, " (", sideStr, ")",
                "\n🔠Symbol: ", Symbol.Name,
                "\n📈Time Frame: ", tf,
                "\n🕘Time: ", barTime.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture), " ", AlertTimeZoneStr,
                "\n📩Message: ", posStr);

            try
            {
                Notifications.PlaySound(SoundType.Announcement);
                Print(msg);
            }
            catch
            {
                // Notifications may be unavailable in some contexts (backtest, etc.) — ignore.
            }
        }

        private string TimeFrameLabel()
        {
            try { return TimeFrame.ToString(); } catch { return ""; }
        }

        // ────────────────────────────────────────────────────────────────────────
        //  Reset / cleanup
        // ────────────────────────────────────────────────────────────────────────

        private void ResetAllState()
        {
            _zones.Clear();
            _objId = 0;

            _ddfvg = _dpfvg = _ddfvgPrev = 0;
            _sdfvg = _spfvg = _sdfvgPrev = 0;
            _barDfvg = _barSfvg = 0;
            _dMitigated = _sMitigated = true;
            _liveBullFvg = _liveBearFvg = null;

            _demDist = _demProx = 0; _demIndex = 0;
            _demCheck = _demCheckPrev = true;
            _demBbDist = _demBbProx = 0; _demBbIndex = 0;
            _demBbCheck = _demBbCheckPrev = false;
            _demCbb0 = _demCbb1 = _demCbb2 = _demCbb3 = false;
            _demTrigBb = false;
            _liveDemObZone = _liveSupIfvgZone = null;

            _supDist = _supProx = 0; _supIndex = 0;
            _supCheck = _supCheckPrev = true;
            _supBbDist = _supBbProx = 0; _supBbIndex = 0;
            _supBbCheck = _supBbCheckPrev = false;
            _supCbb0 = _supCbb1 = _supCbb2 = _supCbb3 = false;
            _supTrigBb = false;
            _liveSupObZone = _liveDemIfvgZone = null;

            _demBprDist = _demBprProx = 0; _demBprIndex = 0; _demBprBar = 0;
            _demBprCheck = _demBprCheckPrev = true; _demBprTrigger = false;
            _liveDemBprZone = null;

            _supBprDist = _supBprProx = 0; _supBprIndex = 0; _supBprBar = 0;
            _supBprCheck = _supBprCheckPrev = true; _supBprTrigger = false;
            _liveSupBprZone = null;

            _lastDemBprAlertBar = DateTime.MinValue;
            _lastSupBprAlertBar = DateTime.MinValue;
        }

        private void ClearMyObjects()
        {
            var names = new List<string>();
            foreach (var obj in Chart.Objects)
                if (obj.Name.StartsWith(Prefix, StringComparison.Ordinal))
                    names.Add(obj.Name);
            foreach (var n in names)
                Chart.RemoveObject(n);
        }

        private string NextId() => (_objId++).ToString(CultureInfo.InvariantCulture);

        private static double NonZero(double x) => x == 0 ? 1e-9 : x;
    }
}
