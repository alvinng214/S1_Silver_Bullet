// PriceAction.cs
// cTrader indicator implementation that mirrors the TradingView Pine library:
// S1_Silver_Bullet/PriceAction.txt  (© mickes, MPL 2.0)
// This C# version aims to match the library's detection/state logic and (where possible) its visuals,
// under cTrader/cAlgo limitations (no native box text, different alpha model, no label styles, etc).
//
// Source reference (Pine):
// https://raw.githubusercontent.com/alvinng214/S1_Silver_Bullet/main/S1_Silver_Bullet/PriceAction.txt
//
// Key parity notes:
// - Pine library is not an indicator UI; this indicator exposes parameters to drive the library logic.
// - Pine `ta.atr(14)` is approximated with Wilder/RMA ATR (RMA of TR), which matches TradingView closer than SMA.
// - Pine boxes/labels are approximated with ChartRectangle + ChartText (text overlay).
// - Pine extend.right is emulated by extending rectangles/lines to a far future time using bar duration.
// - Pine transparency is 0..100; mapped to cTrader alpha 0..255 via alpha = 255*(100-transp)/100.
//
// Limitations that cannot be perfectly matched:
// - cTrader ChartRectangle has no embedded text; we draw a ChartText at the box midpoint instead.
// - cTrader lacks TradingView label styles; we approximate via alignment and placement.
// - Object lifetime differs: Pine arrays can be cleared without deleting objects; we remove objects to avoid clutter.

using System;
using System.Collections.Generic;
using System.Linq;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class PriceAction : Indicator
    {
        // -----------------------------
        // Parameters (driving the library logic)
        // -----------------------------

        [Parameter("Show Liquidity ($$$)", DefaultValue = true, Group = "Liquidity")]
        public bool ShowLiquidity { get; set; }

        [Parameter("Liquidity Confirmation Bars", DefaultValue = 0, MinValue = 0, Group = "Liquidity")]
        public int LiquidityConfirmationBars { get; set; }

        [Parameter("Liquidity Pivots Lookback", DefaultValue = 25, MinValue = 1, Group = "Liquidity")]
        public int LiquidityPivotsLookback { get; set; }

        [Parameter("Liquidity Font Size", DefaultValue = 11, MinValue = 6, MaxValue = 30, Group = "Liquidity")]
        public int LiquidityFontSize { get; set; }

        [Parameter("Structure: Type", DefaultValue = StructureType.Swing, Group = "Structure")]
        public StructureType StructureMode { get; set; }

        [Parameter("Pivot Left", DefaultValue = 10, MinValue = 1, Group = "Structure")]
        public int PivotLeft { get; set; }

        [Parameter("Pivot Right", DefaultValue = 10, MinValue = 1, Group = "Structure")]
        public int PivotRight { get; set; }

        [Parameter("Show Pivot Labels", DefaultValue = true, Group = "Structure")]
        public bool ShowPivotLabels { get; set; }

        [Parameter("Show Equal High/Low Zones", DefaultValue = true, Group = "Structure")]
        public bool ShowEqualPivots { get; set; }

        [Parameter("Equal Pivots Factor (ATR%)", DefaultValue = 20.0, MinValue = 0.0, Group = "Structure")]
        public double EqualPivotsFactor { get; set; }

        [Parameter("Extend Equal Pivot Zones", DefaultValue = true, Group = "Structure")]
        public bool ExtendEqualPivotsZones { get; set; }

        [Parameter("Equal Pivot Zone Color", DefaultValue = "Gold", Group = "Structure")]
        public Color ExtendEqualPivotsColor { get; set; }

        [Parameter("Structure Font Size", DefaultValue = 11, MinValue = 6, MaxValue = 30, Group = "Structure")]
        public int StructureFontSize { get; set; }

        [Parameter("Show Current Trading Range", DefaultValue = false, Group = "Structure")]
        public bool ShowCurrentRange { get; set; }

        [Parameter("Alert CHoCH (Print)", DefaultValue = false, Group = "Alerts")]
        public bool AlertChangeOfCharacter { get; set; }

        [Parameter("Alert BOS (Print)", DefaultValue = false, Group = "Alerts")]
        public bool AlertBreakOfStructure { get; set; }

        [Parameter("Alert Equal High/Low (Print)", DefaultValue = false, Group = "Alerts")]
        public bool AlertEqualPivots { get; set; }

        // -----------------------------
        // Pine types -> C# types
        // -----------------------------

        public enum StructureType
        {
            Internal = 0,
            Swing = 1
        }

        private sealed class StructureBreak
        {
            public string LineId;
            public string LabelId;
            public int PivotBarIndex;
            public double Price;
        }

        private sealed class Pivot
        {
            public double Price;
            public int BarIndex;
            public int Type; // -1 low, 1 high
            public DateTime Time;
            public bool BreakOfStructureBroken;
            public bool LiquidityBroken;
            public bool ChangeOfCharacterBroken;
        }

        private sealed class Structure
        {
            public int LeftLength;
            public int RightLength;
            public StructureType Type;
            public int Trend; // -1 downtrend, 1 uptrend, 0 unknown
            public double EqualPivotsFactor;
            public bool ExtendEqualPivotsZones;
            public Color ExtendEqualPivotsColor;
            public readonly List<string> EqualHighs = new List<string>();
            public readonly List<string> EqualLows = new List<string>();
            public readonly List<StructureBreak> BreakOfStructures = new List<StructureBreak>();
            public readonly List<Pivot> Pivots = new List<Pivot>(); // newest first
            public int FontSize;
            public bool AlertChangeOfCharacter;
            public bool AlertBreakOfStructure;
            public bool AlertEqualPivots;
        }

        private sealed class Liquidity
        {
            public readonly List<Pivot> LiquidityPivotsHigh = new List<Pivot>(); // newest first
            public readonly List<Pivot> LiquidityPivotsLow = new List<Pivot>();  // newest first
            public int LiquidityConfirmationBars;
            public int LiquidityPivotsLookback;
            public int FontSize;
        }

        // -----------------------------
        // State
        // -----------------------------

        private Liquidity _liquidity;
        private Structure _structure;

        // Wilder/RMA ATR(14) (TradingView-like)
        private const int AtrLen = 14;
        private double[] _atrRma; // indexed by bar index
        private double[] _tr;     // true range series

        private int _lastCalculated = -1;

        // object bookkeeping
        private const int MaxObjects = 2500;
        private readonly Queue<string> _objectsFifo = new Queue<string>();

        // -----------------------------
        // Initialize
        // -----------------------------

        protected override void Initialize()
        {
            _liquidity = new Liquidity
            {
                LiquidityConfirmationBars = LiquidityConfirmationBars,
                LiquidityPivotsLookback = LiquidityPivotsLookback,
                FontSize = LiquidityFontSize
            };

            _structure = new Structure
            {
                LeftLength = PivotLeft,
                RightLength = PivotRight,
                Type = StructureMode,
                Trend = 0,
                EqualPivotsFactor = EqualPivotsFactor,
                ExtendEqualPivotsZones = ExtendEqualPivotsZones,
                ExtendEqualPivotsColor = ExtendEqualPivotsColor,
                FontSize = StructureFontSize,
                AlertChangeOfCharacter = AlertChangeOfCharacter,
                AlertBreakOfStructure = AlertBreakOfStructure,
                AlertEqualPivots = AlertEqualPivots
            };

            _atrRma = new double[Bars.Count];
            _tr = new double[Bars.Count];
        }

        public override void Calculate(int index)
        {
            // Ensure arrays fit if Bars expanded (backtesting/live)
            if (_atrRma == null || _atrRma.Length != Bars.Count)
            {
                _atrRma = new double[Bars.Count];
                _tr = new double[Bars.Count];
                _lastCalculated = -1;
            }

            if (index <= _lastCalculated)
                return;

            _lastCalculated = index;

            SyncParams();

            if (index < 2)
            {
                ComputeAtrRma(index);
                return;
            }

            ComputeAtrRma(index);

            if (ShowLiquidity)
                Liqudity(_liquidity, index);

            // Structure library functions
            PivotSet(_structure, index);

            if (ShowPivotLabels)
                PivotLabels(_structure, index);

            if (ShowEqualPivots)
                EqualHighOrLow(_structure, index);

            ChangeOfCharacter(_structure, index);
            BreakOfStructure(_structure, index);

            if (ShowCurrentRange)
                VisualizeCurrent(_structure, index);
        }

        private void SyncParams()
        {
            _liquidity.LiquidityConfirmationBars = LiquidityConfirmationBars;
            _liquidity.LiquidityPivotsLookback = LiquidityPivotsLookback;
            _liquidity.FontSize = LiquidityFontSize;

            _structure.LeftLength = PivotLeft;
            _structure.RightLength = PivotRight;
            _structure.Type = StructureMode;
            _structure.EqualPivotsFactor = EqualPivotsFactor;
            _structure.ExtendEqualPivotsZones = ExtendEqualPivotsZones;
            _structure.ExtendEqualPivotsColor = ExtendEqualPivotsColor;
            _structure.FontSize = StructureFontSize;
            _structure.AlertChangeOfCharacter = AlertChangeOfCharacter;
            _structure.AlertBreakOfStructure = AlertBreakOfStructure;
            _structure.AlertEqualPivots = AlertEqualPivots;
        }

        // -----------------------------
        // TradingView-like ATR (RMA of TR)
        // -----------------------------

        private void ComputeAtrRma(int index)
        {
            // TR = max(high-low, abs(high-prevClose), abs(low-prevClose))
            double high = Bars.HighPrices[index];
            double low = Bars.LowPrices[index];
            double prevClose = index > 0 ? Bars.ClosePrices[index - 1] : Bars.ClosePrices[index];

            double tr = Math.Max(high - low, Math.Max(Math.Abs(high - prevClose), Math.Abs(low - prevClose)));
            _tr[index] = tr;

            if (index == 0)
            {
                _atrRma[index] = tr;
                return;
            }

            if (index < AtrLen)
            {
                // seed with SMA of TR up to index (reasonable warmup)
                double sum = 0.0;
                for (int i = 0; i <= index; i++) sum += _tr[i];
                _atrRma[index] = sum / (index + 1);
                return;
            }

            // RMA: atr = (prevAtr*(len-1) + tr)/len
            double prevAtr = _atrRma[index - 1];
            _atrRma[index] = (prevAtr * (AtrLen - 1) + tr) / AtrLen;
        }

        private double Atr(int index) => _atrRma[Math.Max(0, Math.Min(index, _atrRma.Length - 1))];

        // -----------------------------
        // Utils: time/index mapping & alpha
        // -----------------------------

        private DateTime TimeAt(int barIndex)
        {
            barIndex = Math.Max(0, Math.Min(Bars.Count - 1, barIndex));
            return Bars.OpenTimes[barIndex];
        }

        private int IndexAtTime(DateTime time)
        {
            // nearest <= time
            int lo = 0, hi = Bars.Count - 1;
            while (lo <= hi)
            {
                int mid = (lo + hi) >> 1;
                var t = Bars.OpenTimes[mid];
                if (t == time) return mid;
                if (t < time) lo = mid + 1;
                else hi = mid - 1;
            }
            return Math.Max(0, Math.Min(Bars.Count - 1, hi));
        }

        private TimeSpan BarDuration(int index)
        {
            if (index <= 0) return TimeSpan.FromMinutes(1);
            return Bars.OpenTimes[index] - Bars.OpenTimes[index - 1];
        }

        private static byte AlphaFromPineTransparency(int pineTransparency0to100)
        {
            pineTransparency0to100 = Math.Max(0, Math.Min(100, pineTransparency0to100));
            // Pine: 0 opaque, 100 fully transparent
            double a = 255.0 * (100.0 - pineTransparency0to100) / 100.0;
            return (byte)Math.Max(0, Math.Min(255, (int)Math.Round(a)));
        }

        private static Color WithPineTransparency(Color baseColor, int pineTransparency0to100)
        {
            byte a = AlphaFromPineTransparency(pineTransparency0to100);
            return Color.FromArgb(a, baseColor.R, baseColor.G, baseColor.B);
        }

        private static string Id(string kind, int a, int b = 0, int c = 0) => $"PA_{kind}_{a}_{b}_{c}";

        private void Remember(string id)
        {
            _objectsFifo.Enqueue(id);
            if (_objectsFifo.Count > MaxObjects)
            {
                var old = _objectsFifo.Dequeue();
                SafeRemove(old);
            }
        }

        private void SafeRemove(string id)
        {
            try
            {
                if (Chart.FindObject(id) != null)
                    Chart.RemoveObject(id);
            }
            catch { }
        }

        private void PrintAlert(string message)
        {
            // Pine uses alert.freq_once_per_bar_close in this library
            Print(message);
        }

        // -----------------------------
        // Pine: InLimits(highLimit, lowLimit, price)
        // -----------------------------
        private static bool InLimits(double highLimit, double lowLimit, double price) => price >= lowLimit && price <= highLimit;

        // -----------------------------
        // Pine: BrokenByBar(pivot)
        // -----------------------------
        private bool BrokenByBar(Pivot pivot, int currentIndex)
        {
            bool broken = false;

            int maxI = currentIndex - 1 - pivot.BarIndex;
            if (maxI < 1) return false;

            for (int i = 1; i <= maxI; i++)
            {
                int idx = currentIndex - i;
                if (pivot.Type == 1)
                {
                    if (Bars.LowPrices[idx] > pivot.Price)
                    {
                        broken = true;
                        break;
                    }
                }
                else
                {
                    if (Bars.HighPrices[idx] < pivot.Price)
                    {
                        broken = true;
                        break;
                    }
                }
            }

            return broken;
        }

        // -----------------------------
        // Pine: BrokenByBar(box zone, type)
        // -----------------------------
        private bool BrokenByBar(string rectId, int type, int currentIndex)
        {
            var rect = Chart.FindObject(rectId) as ChartRectangle;
            if (rect == null)
                return false;

            int left = IndexAtTime(rect.Time1);

            bool broken = false;
            int maxI = currentIndex - 1 - left;
            if (maxI < 1) return false;

            double top = Math.Max(rect.Y1, rect.Y2);
            double bottom = Math.Min(rect.Y1, rect.Y2);

            for (int i = 1; i <= maxI; i++)
            {
                int idx = currentIndex - i;
                if (type == 1)
                {
                    if (Bars.LowPrices[idx] > top)
                    {
                        broken = true;
                        break;
                    }
                }
                else
                {
                    if (Bars.HighPrices[idx] < bottom)
                    {
                        broken = true;
                        break;
                    }
                }
            }

            return broken;
        }

        // -----------------------------
        // Pine: Liqudity(liquidity)
        // -----------------------------
        private void Liqudity(Liquidity liquidity, int index)
        {
            VisualizeLiquidations(liquidity.LiquidityPivotsHigh, liquidity, index, isHigh: true);
            VisualizeLiquidations(liquidity.LiquidityPivotsLow, liquidity, index, isHigh: false);

            // pivothigh(1,1), pivotlow(1,1)
            double? pivotHigh = PivotHighValue(index, 1, 1);
            double? pivotLow = PivotLowValue(index, 1, 1);

            int pivotBar = index - 1;
            if (pivotBar < 0) return;

            if (pivotHigh.HasValue)
            {
                if (liquidity.LiquidityPivotsHigh.Count >= liquidity.LiquidityPivotsLookback)
                    liquidity.LiquidityPivotsHigh.RemoveAt(liquidity.LiquidityPivotsHigh.Count - 1);

                liquidity.LiquidityPivotsHigh.Insert(0, new Pivot
                {
                    Price = pivotHigh.Value,
                    BarIndex = pivotBar,
                    Type = 1,
                    Time = TimeAt(pivotBar)
                });
            }

            if (pivotLow.HasValue)
            {
                if (liquidity.LiquidityPivotsLow.Count >= liquidity.LiquidityPivotsLookback)
                    liquidity.LiquidityPivotsLow.RemoveAt(liquidity.LiquidityPivotsLow.Count - 1);

                liquidity.LiquidityPivotsLow.Insert(0, new Pivot
                {
                    Price = pivotLow.Value,
                    BarIndex = pivotBar,
                    Type = -1,
                    Time = TimeAt(pivotBar)
                });
            }
        }

        // Pine: Liquidation(liquidity, pivot)
        private bool Liquidation(Liquidity liquidity, Pivot pivot, int index)
        {
            int c = liquidity.LiquidityConfirmationBars;
            if (index - c < 0) return false;

            if (pivot.Type == -1)
            {
                // low[c] <= pivot and close[c] >= pivot and all closes between >= pivot
                int idx = index - c;
                if (Bars.LowPrices[idx] <= pivot.Price && Bars.ClosePrices[idx] >= pivot.Price)
                {
                    for (int i = c - 1; i >= 0; i--)
                    {
                        if (Bars.ClosePrices[index - i] < pivot.Price)
                            return false;
                    }
                    return true;
                }
                return false;
            }
            else
            {
                int idx = index - c;
                if (Bars.HighPrices[idx] >= pivot.Price && Bars.ClosePrices[idx] <= pivot.Price)
                {
                    for (int i = c - 1; i >= 0; i--)
                    {
                        if (Bars.ClosePrices[index - i] > pivot.Price)
                            return false;
                    }
                    return true;
                }
                return false;
            }
        }

        // Pine: method VisualizeLiquidations(pivots, liquidity)
        private void VisualizeLiquidations(List<Pivot> pivots, Liquidity liquidity, int index, bool isHigh)
        {
            foreach (var pivot in pivots)
            {
                if (index < pivot.BarIndex + liquidity.LiquidityConfirmationBars + 1)
                    continue;

                if (pivot.LiquidityBroken)
                    continue;

                if (Liquidation(liquidity, pivot, index))
                {
                    int c = liquidity.LiquidityConfirmationBars;
                    int endIdx = index - c;

                    // line.new(..., style_dotted, color.orange)
                    var limitId = Id(isHigh ? "LQH_LIMIT" : "LQL_LIMIT", endIdx, pivot.BarIndex);
                    DrawOrUpdateLine(limitId, pivot.BarIndex, pivot.Price, endIdx, pivot.Price, Color.Orange, LineStyle.Dots);
                    Remember(limitId);

                    // linefill approximation: rectangle between pivot.Price and breakPrice
                    double breakPrice = pivot.Type == -1 ? Bars.LowPrices[endIdx] : Bars.HighPrices[endIdx];
                    var fillId = Id(isHigh ? "LQH_FILL" : "LQL_FILL", endIdx, pivot.BarIndex);
                    DrawOrUpdateRect(fillId, pivot.BarIndex, pivot.Price, endIdx, breakPrice, WithPineTransparency(Color.Orange, 80), extendRight: false);
                    Remember(fillId);

                    // label "$$$" at midpoint
                    int mid = endIdx - ((endIdx - pivot.BarIndex) / 2);
                    var lblId = Id(isHigh ? "LQH_LBL" : "LQL_LBL", endIdx, pivot.BarIndex);
                    DrawOrUpdateText(lblId, mid, pivot.Price, "$$$", WithPineTransparency(Color.Orange, 80), liquidity.FontSize,
                        VerticalAlignment.Center, HorizontalAlignment.Center);
                    Remember(lblId);

                    pivot.LiquidityBroken = true;
                }
                else
                {
                    // invalidate if close crosses wrong side (as in Pine)
                    if (pivot.Type == -1)
                    {
                        if (Bars.ClosePrices[index] < pivot.Price)
                            pivot.LiquidityBroken = true;
                    }
                    else
                    {
                        if (Bars.ClosePrices[index] > pivot.Price)
                            pivot.LiquidityBroken = true;
                    }
                }
            }
        }

        // -----------------------------
        // Pine: Pivot(structure)
        // -----------------------------
        private void PivotSet(Structure structure, int index)
        {
            double? pivotHigh = PivotHighValue(index, structure.LeftLength, structure.RightLength);
            double? pivotLow = PivotLowValue(index, structure.LeftLength, structure.RightLength);

            int pivotBar = index - structure.RightLength;
            if (pivotBar < 0)
                return;

            if (pivotHigh.HasValue)
            {
                if (structure.Pivots.Count > 5)
                    structure.Pivots.RemoveAt(structure.Pivots.Count - 1);

                structure.Pivots.Insert(0, new Pivot
                {
                    Price = pivotHigh.Value,
                    BarIndex = pivotBar,
                    Type = 1,
                    Time = TimeAt(pivotBar),
                    BreakOfStructureBroken = false,
                    LiquidityBroken = false,
                    ChangeOfCharacterBroken = false
                });
            }

            if (pivotLow.HasValue)
            {
                if (structure.Pivots.Count > 5)
                    structure.Pivots.RemoveAt(structure.Pivots.Count - 1);

                structure.Pivots.Insert(0, new Pivot
                {
                    Price = pivotLow.Value,
                    BarIndex = pivotBar,
                    Type = -1,
                    Time = TimeAt(pivotBar),
                    BreakOfStructureBroken = false,
                    LiquidityBroken = false,
                    ChangeOfCharacterBroken = false
                });
            }
        }

        // Strict pivot (no ties) like Pine's ta.pivot* returning na on ties within window
        private double? PivotHighValue(int index, int left, int right)
        {
            int p = index - right;
            if (p - left < 0 || p + right >= Bars.Count)
                return null;

            double pv = Bars.HighPrices[p];
            for (int i = p - left; i <= p + right; i++)
            {
                if (i == p) continue;
                if (Bars.HighPrices[i] >= pv) return null;
            }
            return pv;
        }

        private double? PivotLowValue(int index, int left, int right)
        {
            int p = index - right;
            if (p - left < 0 || p + right >= Bars.Count)
                return null;

            double pv = Bars.LowPrices[p];
            for (int i = p - left; i <= p + right; i++)
            {
                if (i == p) continue;
                if (Bars.LowPrices[i] <= pv) return null;
            }
            return pv;
        }

        // -----------------------------
        // Pine: PivotLabels(structure)
        // -----------------------------
        private void PivotLabels(Structure structure, int index)
        {
            foreach (var pivot in structure.Pivots)
            {
                if (pivot.BarIndex != index - structure.RightLength)
                    continue;

                string text = "";

                for (int i = 1; i <= structure.Pivots.Count - 1; i++)
                {
                    if (structure.Pivots.Count == 1) break;

                    var previous = structure.Pivots[i];
                    if (previous.Type != pivot.Type || previous.BarIndex == pivot.BarIndex)
                        continue;

                    if (Math.Abs(pivot.Price - previous.Price) < Symbol.TickSize / 2)
                        text = "EQ";
                    else if (pivot.Price > previous.Price)
                        text = "H";
                    else if (pivot.Price < previous.Price)
                        text = "L";

                    break;
                }

                int transparency = structure.Type == StructureType.Internal ? 60 : 20;
                Color baseCol = pivot.Type == -1 ? Color.Teal : Color.Red;
                Color col = WithPineTransparency(baseCol, transparency);

                if (pivot.Type == -1) text += "L";
                else text += "H";

                var id = Id("PIVOT_LBL", pivot.BarIndex, pivot.Type, (int)structure.Type);
                DrawOrUpdateText(id, pivot.BarIndex, pivot.Price, text, col, structure.FontSize,
                    pivot.Type == -1 ? VerticalAlignment.Bottom : VerticalAlignment.Top,
                    HorizontalAlignment.Center);
                Remember(id);
            }
        }

        // -----------------------------
        // Pine: EqualHighOrLow(structure)
        // -----------------------------
        private void EqualHighOrLow(Structure structure, int index)
        {
            if (index < 2) return;

            bool retestHigh = Bars.HighPrices[index] < Bars.HighPrices[index - 1] && Bars.HighPrices[index - 1] > Bars.HighPrices[index - 2];
            bool retestLow = Bars.LowPrices[index] > Bars.LowPrices[index - 1] && Bars.LowPrices[index - 1] < Bars.LowPrices[index - 2];

            double limit = Atr(index) * (structure.EqualPivotsFactor / 100.0);

            // Update existing equal highs
            if (retestHigh)
            {
                for (int k = 0; k < structure.EqualHighs.Count; k++)
                {
                    string rectId = structure.EqualHighs[k];
                    var rect = Chart.FindObject(rectId) as ChartRectangle;
                    if (rect == null) continue;

                    double price = Bars.HighPrices[index - 1];
                    double bottom = Math.Min(rect.Y1, rect.Y2);
                    double top = Math.Max(rect.Y1, rect.Y2);

                    double lowLimit = bottom - limit;
                    double highLimit = top;

                    if (InLimits(highLimit, lowLimit, price) && !BrokenByBar(rectId, 1, index))
                    {
                        // set_right (time), set_top(price); if price < bottom then set_rightbottom
                        rect.Time2 = TimeAt(index - 1);
                        // emulate set_top / set_rightbottom by updating coordinates
                        rect.Y1 = price; // top
                        if (price < bottom)
                            rect.Y2 = price; // bottom
                        rect.Color = WithPineTransparency(structure.ExtendEqualPivotsColor, 70);

                        if (structure.AlertEqualPivots)
                            PrintAlert("Added bar to existing equal high");
                    }
                }
            }

            // Update existing equal lows
            if (retestLow)
            {
                for (int k = 0; k < structure.EqualLows.Count; k++)
                {
                    string rectId = structure.EqualLows[k];
                    var rect = Chart.FindObject(rectId) as ChartRectangle;
                    if (rect == null) continue;

                    double price = Bars.LowPrices[index - 1];
                    double bottom = Math.Min(rect.Y1, rect.Y2);
                    double top = Math.Max(rect.Y1, rect.Y2);

                    double lowLimit = bottom;
                    double highLimit = top + limit;

                    if (InLimits(highLimit, lowLimit, price) && !BrokenByBar(rectId, -1, index))
                    {
                        rect.Time2 = TimeAt(index - 1);
                        rect.Y1 = price;
                        if (price < bottom)
                            rect.Y2 = price;
                        rect.Color = WithPineTransparency(structure.ExtendEqualPivotsColor, 70);

                        if (structure.AlertEqualPivots)
                            PrintAlert("Added bar to existing equal low");
                    }
                }
            }

            // Create new equal zone from pivots
            foreach (var pivot in structure.Pivots)
            {
                double price, lowLimit, highLimit;

                if (pivot.Type == -1)
                {
                    if (!retestLow) continue;
                    price = Bars.LowPrices[index - 1];
                    lowLimit = pivot.Price;
                    highLimit = pivot.Price + limit;
                }
                else
                {
                    if (!retestHigh) continue;
                    price = Bars.HighPrices[index - 1];
                    lowLimit = pivot.Price - limit;
                    highLimit = pivot.Price;
                }

                if (!InLimits(highLimit, lowLimit, price)) continue;
                if (BrokenByBar(pivot, index)) continue;

                int left = pivot.BarIndex;
                int right = index - 1;

                double top = Math.Max(price, pivot.Price);
                double bottom = Math.Min(price, pivot.Price);

                bool isExactSame = Math.Abs(pivot.Price - price) < Symbol.TickSize / 2;

                string rectId = Id(pivot.Type == -1 ? "EQL" : "EQH", right, left, (int)structure.Type);
                DrawOrUpdateRect(rectId, left, top, right, bottom, WithPineTransparency(structure.ExtendEqualPivotsColor, 70), extendRight: structure.ExtendEqualPivotsZones);
                Remember(rectId);

                // Pine adds box text "Equal low/high". We approximate by drawing text at box midpoint.
                string label = pivot.Type == -1 ? "Equal low" : "Equal high";
                int mid = right - ((right - left) / 2);
                string txtId = Id(pivot.Type == -1 ? "EQL_TXT" : "EQH_TXT", right, left, (int)structure.Type);
                DrawOrUpdateText(txtId, mid, pivot.Price, label, WithPineTransparency(structure.ExtendEqualPivotsColor, 0), structure.FontSize,
                    pivot.Type == -1 ? VerticalAlignment.Bottom : VerticalAlignment.Top,
                    HorizontalAlignment.Center);
                Remember(txtId);

                // Pine: border_width = isExactSame ? 1 : 0 (not available reliably); ignore.

                if (pivot.Type == -1)
                {
                    structure.EqualLows.Insert(0, rectId);
                    if (structure.EqualLows.Count > 50) structure.EqualLows.RemoveAt(structure.EqualLows.Count - 1);
                    if (structure.AlertEqualPivots) PrintAlert("Equal low appeared");
                }
                else
                {
                    structure.EqualHighs.Insert(0, rectId);
                    if (structure.EqualHighs.Count > 50) structure.EqualHighs.RemoveAt(structure.EqualHighs.Count - 1);
                    if (structure.AlertEqualPivots) PrintAlert("Equal high appeared");
                }
            }
        }

        // -----------------------------
        // Pine: ChangeOfCharacter(structure)
        // -----------------------------
        private void ChangeOfCharacter(Structure structure, int index)
        {
            // Pine builds CHoCH when:
            // bullish: structure.Trend <= 0 AND pivot.Type=1 AND close>pivot AND close[1]<pivot AND !pivot.CHBroken
            // bearish: structure.Trend >= 0 AND pivot.Type=-1 AND close<pivot AND close[1]>pivot AND !pivot.CHBroken

            foreach (var pivot in structure.Pivots)
            {
                if (index - 1 < 0) break;

                if (structure.Trend <= 0 && pivot.Type == 1 &&
                    Bars.ClosePrices[index] > pivot.Price && Bars.ClosePrices[index - 1] < pivot.Price &&
                    !pivot.ChangeOfCharacterBroken)
                {
                    pivot.ChangeOfCharacterBroken = true;

                    string txt = "CHoCH";
                    if (structure.Pivots.Count >= 2 && structure.Trend != 0)
                    {
                        // CHoCH+ logic from Pine: check if latest low > next latest low
                        txt = ComputeChoChPlusBull(structure) ? "CHoCH+" : "CHoCH";
                    }

                    DrawChoChLineAndLabel(structure, index, pivot, txt, bullish: true);

                    // trend flips to up
                    structure.Trend = 1;

                    // Pine clears arrays (refs). We remove visuals to reduce clutter.
                    ClearAndRemove(structure.EqualHighs);
                    ClearAndRemove(structure.EqualLows);
                    structure.BreakOfStructures.Clear();

                    // Pine: remove pivots <= pivot.BarIndex; set other pivots' BOSBroken = true
                    PrunePivotsAfterChoCh(structure, pivot);

                    // Pine: set all pivots CH broken to false except pivot
                    foreach (var p in structure.Pivots)
                        if (p.BarIndex != pivot.BarIndex)
                            p.ChangeOfCharacterBroken = false;

                    if (structure.AlertChangeOfCharacter)
                        PrintAlert($"{txt} to an uptrend on {(structure.Type == StructureType.Internal ? "internal" : "swing")} market structure");

                    break;
                }

                if (structure.Trend >= 0 && pivot.Type == -1 &&
                    Bars.ClosePrices[index] < pivot.Price && Bars.ClosePrices[index - 1] > pivot.Price &&
                    !pivot.ChangeOfCharacterBroken)
                {
                    pivot.ChangeOfCharacterBroken = true;

                    string txt = "CHoCH";
                    if (structure.Pivots.Count >= 2 && structure.Trend != 0)
                    {
                        // CHoCH+ logic from Pine: check if latest high < next latest high
                        txt = ComputeChoChPlusBear(structure) ? "CHoCH+" : "CHoCH";
                    }

                    DrawChoChLineAndLabel(structure, index, pivot, txt, bullish: false);

                    structure.Trend = -1;

                    ClearAndRemove(structure.EqualHighs);
                    ClearAndRemove(structure.EqualLows);
                    structure.BreakOfStructures.Clear();

                    PrunePivotsAfterChoCh(structure, pivot);

                    foreach (var p in structure.Pivots)
                        if (p.BarIndex != pivot.BarIndex)
                            p.ChangeOfCharacterBroken = false;

                    if (structure.AlertChangeOfCharacter)
                        PrintAlert($"{txt} to a downtrend on {(structure.Type == StructureType.Internal ? "internal" : "swing")} market structure");

                    break;
                }
            }
        }

        private bool ComputeChoChPlusBull(Structure structure)
        {
            // Find 2 lows: latest low > next latest low => CHoCH+
            for (int i = 0; i <= structure.Pivots.Count - 2; i++)
            {
                var latest = structure.Pivots[i];
                if (latest.Type != -1) continue;

                for (int j = i + 1; j <= structure.Pivots.Count - 1; j++)
                {
                    var next = structure.Pivots[j];
                    if (next.Type != -1) continue;
                    return latest.Price > next.Price;
                }
            }
            return false;
        }

        private bool ComputeChoChPlusBear(Structure structure)
        {
            // Find 2 highs: latest high < next latest high => CHoCH+
            for (int i = 0; i <= structure.Pivots.Count - 2; i++)
            {
                var latest = structure.Pivots[i];
                if (latest.Type != 1) continue;

                for (int j = i + 1; j <= structure.Pivots.Count - 1; j++)
                {
                    var next = structure.Pivots[j];
                    if (next.Type != 1) continue;
                    return latest.Price < next.Price;
                }
            }
            return false;
        }

        private void PrunePivotsAfterChoCh(Structure structure, Pivot pivot)
        {
            var removeIdx = new List<int>();
            for (int i = 0; i < structure.Pivots.Count; i++)
            {
                var p = structure.Pivots[i];
                if (p.BarIndex <= pivot.BarIndex)
                    removeIdx.Add(i);
                else
                    p.BreakOfStructureBroken = true; // disable BOS on newer pivots as Pine does
            }
            // remove from list (account for shifting indices)
            for (int i = 0; i < removeIdx.Count; i++)
                structure.Pivots.RemoveAt(removeIdx[i] - i);
        }

        private void DrawChoChLineAndLabel(Structure structure, int index, Pivot pivot, string txt, bool bullish)
        {
            // Pine style: internal -> dashed, swing -> solid (approx)
            var style = structure.Type == StructureType.Internal ? LineStyle.Lines : LineStyle.Solid;
            var lineColor = bullish ? Color.Teal : Color.Red;
            var labelColor = WithPineTransparency(lineColor, 80);

            string lineId = Id("CHOCH_L", index, pivot.BarIndex, bullish ? 1 : -1);
            DrawOrUpdateLine(lineId, pivot.BarIndex, pivot.Price, index, pivot.Price, lineColor, style);
            Remember(lineId);

            int mid = index - ((index - pivot.BarIndex) / 2);
            string lblId = Id("CHOCH_T", index, pivot.BarIndex, bullish ? 1 : -1);
            DrawOrUpdateText(lblId, mid, pivot.Price, txt, labelColor, structure.FontSize, VerticalAlignment.Center, HorizontalAlignment.Center);
            Remember(lblId);
        }

        private void ClearAndRemove(List<string> rectIds)
        {
            foreach (var id in rectIds)
                SafeRemove(id);
            rectIds.Clear();
        }

        // -----------------------------
        // Pine: BreakOfStructure(structure)
        // -----------------------------
        private void BreakOfStructure(Structure structure, int index)
        {
            // Pine line styles: swing = solid, internal = dashed
            var style = structure.Type == StructureType.Swing ? LineStyle.Solid : LineStyle.Lines;

            foreach (var pivot in structure.Pivots)
            {
                // Uptrend BOS on highs
                if (structure.Trend == 1 && pivot.Type == 1 && Bars.ClosePrices[index] > pivot.Price && !pivot.BreakOfStructureBroken)
                {
                    bool create = true;

                    // Remove/disable previous BOS lines if needed (per Pine)
                    for (int i = 0; i < structure.BreakOfStructures.Count; i++)
                    {
                        var bos = structure.BreakOfStructures[i];
                        if (bos.PivotBarIndex > pivot.BarIndex)
                        {
                            if (bos.Price < pivot.Price)
                            {
                                SafeRemove(bos.LineId);
                                SafeRemove(bos.LabelId);
                            }
                            else
                            {
                                create = false;
                                break;
                            }
                        }
                    }

                    if (!create) break;

                    string lineId = Id("BOS_L", index, pivot.BarIndex, 1);
                    DrawOrUpdateLine(lineId, pivot.BarIndex, pivot.Price, index, pivot.Price, Color.Teal, style);
                    Remember(lineId);

                    int mid = index - ((index - pivot.BarIndex) / 2);
                    string lblId = Id("BOS_T", index, pivot.BarIndex, 1);
                    DrawOrUpdateText(lblId, mid, pivot.Price, "BOS", WithPineTransparency(Color.Teal, 80), structure.FontSize,
                        VerticalAlignment.Center, HorizontalAlignment.Center);
                    Remember(lblId);

                    structure.BreakOfStructures.Insert(0, new StructureBreak
                    {
                        LineId = lineId,
                        LabelId = lblId,
                        PivotBarIndex = pivot.BarIndex,
                        Price = pivot.Price
                    });

                    pivot.BreakOfStructureBroken = true;

                    if (structure.AlertBreakOfStructure)
                        PrintAlert($"BOS on an uptrend on {(structure.Type == StructureType.Internal ? "internal" : "swing")} market structure");

                    break;
                }

                // Downtrend BOS on lows
                if (structure.Trend == -1 && pivot.Type == -1 && Bars.ClosePrices[index] < pivot.Price && !pivot.BreakOfStructureBroken)
                {
                    bool create = true;

                    for (int i = 0; i < structure.BreakOfStructures.Count; i++)
                    {
                        var bos = structure.BreakOfStructures[i];
                        if (bos.PivotBarIndex > pivot.BarIndex)
                        {
                            if (bos.Price > pivot.Price)
                            {
                                SafeRemove(bos.LineId);
                                SafeRemove(bos.LabelId);
                            }
                            else
                            {
                                create = false;
                                break;
                            }
                        }
                    }

                    if (!create) break;

                    string lineId = Id("BOS_L", index, pivot.BarIndex, -1);
                    DrawOrUpdateLine(lineId, pivot.BarIndex, pivot.Price, index, pivot.Price, Color.Red, style);
                    Remember(lineId);

                    int mid = index - ((index - pivot.BarIndex) / 2);
                    string lblId = Id("BOS_T", index, pivot.BarIndex, -1);
                    DrawOrUpdateText(lblId, mid, pivot.Price, "BOS", WithPineTransparency(Color.Red, 80), structure.FontSize,
                        VerticalAlignment.Center, HorizontalAlignment.Center);
                    Remember(lblId);

                    structure.BreakOfStructures.Insert(0, new StructureBreak
                    {
                        LineId = lineId,
                        LabelId = lblId,
                        PivotBarIndex = pivot.BarIndex,
                        Price = pivot.Price
                    });

                    pivot.BreakOfStructureBroken = true;

                    if (structure.AlertBreakOfStructure)
                        PrintAlert($"BOS on a downtrend on {(structure.Type == StructureType.Internal ? "internal" : "swing")} market structure");

                    break;
                }
            }
        }

        // -----------------------------
        // Pine: VisualizeCurrent(structure)
        // -----------------------------
        private void VisualizeCurrent(Structure structure, int index)
        {
            // Pine creates a box at current bar and then sets left/top/right/bottom when pivots complete.
            string rectId = Id("TRANGE", 0, (int)structure.Type, 0);

            // initial dummy
            DrawOrUpdateRect(rectId, index, Bars.HighPrices[index], index, Bars.HighPrices[index], WithPineTransparency(Color.Gray, 70), extendRight: true);
            Remember(rectId);

            var rect = Chart.FindObject(rectId) as ChartRectangle;
            if (rect == null) return;

            Pivot latestHigh = null, latestLow = null;
            foreach (var p in structure.Pivots)
            {
                if (p.Type == 1 && latestHigh == null) latestHigh = p;
                if (p.Type == -1 && latestLow == null) latestLow = p;
                if (latestHigh != null && latestLow != null) break;
            }

            if (latestHigh != null && structure.RightLength == index - latestHigh.BarIndex)
            {
                int offset = index - latestHigh.BarIndex;
                rect.Time1 = TimeAt(index - offset);
                rect.Y1 = Bars.HighPrices[index - offset];
            }

            if (latestLow != null && structure.RightLength == index - latestLow.BarIndex)
            {
                int offset = index - latestLow.BarIndex;
                rect.Time2 = ExtendToFutureTime(TimeAt(index - offset), index);
                rect.Y2 = Bars.LowPrices[index - offset];
            }
        }

        private DateTime ExtendToFutureTime(DateTime from, int index)
        {
            // emulate extend.right: push by N bars into the future using bar duration at current index
            var dur = BarDuration(index);
            if (dur.Ticks <= 0) dur = TimeSpan.FromMinutes(1);
            int extendBars = 5000;
            return from + TimeSpan.FromTicks(dur.Ticks * extendBars);
        }

        // -----------------------------
        // Drawing primitives
        // -----------------------------
        private void DrawOrUpdateLine(string id, int x1Index, double y1, int x2Index, double y2, Color color, LineStyle style)
        {
            DateTime t1 = TimeAt(x1Index);
            DateTime t2 = TimeAt(x2Index);

            var line = Chart.FindObject(id) as TrendLine;
            if (line == null)
            {
                line = Chart.DrawTrendLine(id, t1, y1, t2, y2, color, 1, style);
                line.IsInteractive = false;
            }
            else
            {
                line.Time1 = t1;
                line.Y1 = y1;
                line.Time2 = t2;
                line.Y2 = y2;
                line.Color = color;
                line.LineStyle = style;
            }
        }

        private void DrawOrUpdateRect(string id, int leftIndex, double top, int rightIndex, double bottom, Color color, bool extendRight)
        {
            DateTime t1 = TimeAt(leftIndex);
            DateTime t2 = TimeAt(rightIndex);

            if (extendRight)
                t2 = ExtendToFutureTime(t2, rightIndex);

            var rect = Chart.FindObject(id) as ChartRectangle;
            if (rect == null)
            {
                rect = Chart.DrawRectangle(id, t1, top, t2, bottom, color);
                rect.IsFilled = true;
                rect.IsInteractive = false;
            }
            else
            {
                rect.Time1 = t1;
                rect.Y1 = top;
                rect.Time2 = t2;
                rect.Y2 = bottom;
                rect.Color = color;
                rect.IsFilled = true;
            }
        }

        private void DrawOrUpdateText(string id, int xIndex, double y, string text, Color color, int fontSize, VerticalAlignment vAlign, HorizontalAlignment hAlign)
        {
            DateTime t = TimeAt(xIndex);

            var txt = Chart.FindObject(id) as ChartText;
            if (txt == null)
            {
                txt = Chart.DrawText(id, text ?? "", t, y, color);
                txt.FontSize = fontSize;
                txt.VerticalAlignment = vAlign;
                txt.HorizontalAlignment = hAlign;
                txt.IsInteractive = false;
            }
            else
            {
                txt.Text = text ?? "";
                txt.Time = t;
                txt.Y = y;
                txt.Color = color;
                txt.FontSize = fontSize;
                txt.VerticalAlignment = vAlign;
                txt.HorizontalAlignment = hAlign;
            }
        }
    }
}
