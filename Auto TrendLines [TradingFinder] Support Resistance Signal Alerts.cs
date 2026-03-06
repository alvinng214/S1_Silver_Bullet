using System;
using System.Collections.Generic;
using cAlgo.API;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class AutoTrendLinesTradingFinderSupportResistanceSignalAlerts : Indicator
    {
        public enum OnOff { On, Off }
        public enum MessageFrequency { All, OncePerBar, PerBarClose }
        public enum LineStyleInput { Solid, Dashed, Dotted }
        public enum ExtendInput { None, Both, Right, Left }

        [Parameter("Pivot Period", DefaultValue = 5, Group = "Zig Zag Logic")]
        public int PP { get; set; }

        [Parameter("Alert Name", DefaultValue = "Auto TrendLines Alerts [TradingFinder]", Group = "Alert")]
        public string AlertName { get; set; }

        [Parameter("Message Frequency", DefaultValue = MessageFrequency.OncePerBar, Group = "Alert")]
        public MessageFrequency Frequency { get; set; }

        [Parameter("Show Alert time by Time Zone", DefaultValue = "UTC", Group = "Alert")]
        public string UTC { get; set; }

        [Parameter("Break Major External Up TrendLine Alert", DefaultValue = OnOff.On, Group = "Alert")]
        public OnOff Alert_MjExUp_B { get; set; }
        [Parameter("React Major External Up TrendLine Alert", DefaultValue = OnOff.On, Group = "Alert")]
        public OnOff Alert_MjExUp_R { get; set; }
        [Parameter("Break Major External Down TrendLine Alert", DefaultValue = OnOff.On, Group = "Alert")]
        public OnOff Alert_MjExDown_B { get; set; }
        [Parameter("React Major External Down TrendLine Alert", DefaultValue = OnOff.On, Group = "Alert")]
        public OnOff Alert_MjExDown_R { get; set; }
        [Parameter("Break Major Internal Up TrendLine Alert", DefaultValue = OnOff.On, Group = "Alert")]
        public OnOff Alert_MjInUp_B { get; set; }
        [Parameter("React Major Internal Up TrendLine Alert", DefaultValue = OnOff.On, Group = "Alert")]
        public OnOff Alert_MjInUp_R { get; set; }
        [Parameter("Break Major Internal Down TrendLine Alert", DefaultValue = OnOff.On, Group = "Alert")]
        public OnOff Alert_MjInDown_B { get; set; }
        [Parameter("React Major Internal Down TrendLine Alert", DefaultValue = OnOff.On, Group = "Alert")]
        public OnOff Alert_MjInDown_R { get; set; }
        [Parameter("Break Minor External Up TrendLine Alert", DefaultValue = OnOff.On, Group = "Alert")]
        public OnOff Alert_MnExUp_B { get; set; }
        [Parameter("React Minor External Up TrendLine Alert", DefaultValue = OnOff.On, Group = "Alert")]
        public OnOff Alert_MnExUp_R { get; set; }
        [Parameter("Break Minor External Down TrendLine Alert", DefaultValue = OnOff.On, Group = "Alert")]
        public OnOff Alert_MnExDown_B { get; set; }
        [Parameter("React Minor External Down TrendLine Alert", DefaultValue = OnOff.On, Group = "Alert")]
        public OnOff Alert_MnExDown_R { get; set; }
        [Parameter("Break Minor Internal Up TrendLine Alert", DefaultValue = OnOff.On, Group = "Alert")]
        public OnOff Alert_MnInUp_B { get; set; }
        [Parameter("React Minor Internal Up TrendLine Alert", DefaultValue = OnOff.On, Group = "Alert")]
        public OnOff Alert_MnInUp_R { get; set; }
        [Parameter("Break Minor Internal Down TrendLine Alert", DefaultValue = OnOff.On, Group = "Alert")]
        public OnOff Alert_MnInDown_B { get; set; }
        [Parameter("React Minor Internal Down TrendLine Alert", DefaultValue = OnOff.On, Group = "Alert")]
        public OnOff Alert_MnInDown_R { get; set; }

        [Parameter("Show Major External Up   TrendLine", DefaultValue = true, Group = "Major External Up   TrendLine")]
        public bool Show_MjExUp { get; set; }
        [Parameter("Delete Previous Major External Up   TrendLine", DefaultValue = true, Group = "Major External Up   TrendLine")]
        public bool Delete_Pre_MjExUp { get; set; }
        [Parameter("Color", DefaultValue = "#016B05", Group = "Major External Up   TrendLine")]
        public Color Color_MjExUp { get; set; }
        [Parameter("Style", DefaultValue = LineStyleInput.Solid, Group = "Major External Up   TrendLine")]
        public LineStyleInput Style_MjExUp { get; set; }
        [Parameter("Extend", DefaultValue = ExtendInput.None, Group = "Major External Up   TrendLine")]
        public ExtendInput Extend_MjExUp { get; set; }
        [Parameter("Width", DefaultValue = 2, Group = "Major External Up   TrendLine")]
        public int Width_MjExUp { get; set; }

        [Parameter("Show Major External Down TrendLine", DefaultValue = true, Group = "Major External Down TrendLine")]
        public bool Show_MjExDown { get; set; }
        [Parameter("Delete Previous Major External Down   TrendLine", DefaultValue = true, Group = "Major External Down TrendLine")]
        public bool Delete_Pre_MjExDown { get; set; }
        [Parameter("Color", DefaultValue = "#AA0202", Group = "Major External Down TrendLine")]
        public Color Color_MjExDown { get; set; }
        [Parameter("Style", DefaultValue = LineStyleInput.Solid, Group = "Major External Down TrendLine")]
        public LineStyleInput Style_MjExDown { get; set; }
        [Parameter("Extend", DefaultValue = ExtendInput.None, Group = "Major External Down TrendLine")]
        public ExtendInput Extend_MjExDown { get; set; }
        [Parameter("Width", DefaultValue = 2, Group = "Major External Down TrendLine")]
        public int Width_MjExDown { get; set; }

        [Parameter("Show Major Internal Up   TrendLine", DefaultValue = true, Group = "Major Internal Up   TrendLine")]
        public bool Show_MjInUp { get; set; }
        [Parameter("Delete Previous Major Internal Up   TrendLine", DefaultValue = true, Group = "Major Internal Up   TrendLine")]
        public bool Delete_Pre_MjInUp { get; set; }
        [Parameter("Color", DefaultValue = "#016B05", Group = "Major Internal Up   TrendLine")]
        public Color Color_MjInUp { get; set; }
        [Parameter("Style", DefaultValue = LineStyleInput.Solid, Group = "Major Internal Up   TrendLine")]
        public LineStyleInput Style_MjInUp { get; set; }
        [Parameter("Extend", DefaultValue = ExtendInput.None, Group = "Major Internal Up   TrendLine")]
        public ExtendInput Extend_MjInUp { get; set; }
        [Parameter("Width", DefaultValue = 1, Group = "Major Internal Up   TrendLine")]
        public int Width_MjInUp { get; set; }

        [Parameter("Show Major Internal Down TrendLine", DefaultValue = true, Group = "Major Internal Down TrendLine")]
        public bool Show_MjInDown { get; set; }
        [Parameter("Delete Previous Major Internal Down TrendLine", DefaultValue = true, Group = "Major Internal Down TrendLine")]
        public bool Delete_Pre_MjInDown { get; set; }
        [Parameter("Color", DefaultValue = "#AA0202", Group = "Major Internal Down TrendLine")]
        public Color Color_MjInDown { get; set; }
        [Parameter("Style", DefaultValue = LineStyleInput.Solid, Group = "Major Internal Down TrendLine")]
        public LineStyleInput Style_MjInDown { get; set; }
        [Parameter("Extend", DefaultValue = ExtendInput.None, Group = "Major Internal Down TrendLine")]
        public ExtendInput Extend_MjInDown { get; set; }
        [Parameter("Width", DefaultValue = 1, Group = "Major Internal Down TrendLine")]
        public int Width_MjInDown { get; set; }

        [Parameter("Show Minor External Up   TrendLine", DefaultValue = true, Group = "Minor External Up   TrendLine")]
        public bool Show_MnExUp { get; set; }
        [Parameter("Delete Previous Minor External Up   TrendLine", DefaultValue = true, Group = "Minor External Up   TrendLine")]
        public bool Delete_Pre_MnExUp { get; set; }
        [Parameter("Color", DefaultValue = "#016B05A6", Group = "Minor External Up   TrendLine")]
        public Color Color_MnExUp { get; set; }
        [Parameter("Style", DefaultValue = LineStyleInput.Dashed, Group = "Minor External Up   TrendLine")]
        public LineStyleInput Style_MnExUp { get; set; }
        [Parameter("Extend", DefaultValue = ExtendInput.None, Group = "Minor External Up   TrendLine")]
        public ExtendInput Extend_MnExUp { get; set; }
        [Parameter("Width", DefaultValue = 1, Group = "Minor External Up   TrendLine")]
        public int Width_MnExUp { get; set; }

        [Parameter("Show Minor External Down TrendLine", DefaultValue = true, Group = "Minor External Down TrendLine")]
        public bool Show_MnExDown { get; set; }
        [Parameter("Delete Previous Minor External Down   TrendLine", DefaultValue = true, Group = "Minor External Down TrendLine")]
        public bool Delete_Pre_MnExDown { get; set; }
        [Parameter("Color", DefaultValue = "#AA0202A6", Group = "Minor External Down TrendLine")]
        public Color Color_MnExDown { get; set; }
        [Parameter("Style", DefaultValue = LineStyleInput.Dashed, Group = "Minor External Down TrendLine")]
        public LineStyleInput Style_MnExDown { get; set; }
        [Parameter("Extend", DefaultValue = ExtendInput.None, Group = "Minor External Down TrendLine")]
        public ExtendInput Extend_MnExDown { get; set; }
        [Parameter("Width", DefaultValue = 1, Group = "Minor External Down TrendLine")]
        public int Width_MnExDown { get; set; }

        [Parameter("Show Minor Internal Up   TrendLine", DefaultValue = true, Group = "Minor Internal Up   TrendLine")]
        public bool Show_MnInUp { get; set; }
        [Parameter("Delete Previous Minor Internal Up   TrendLine", DefaultValue = true, Group = "Minor Internal Up   TrendLine")]
        public bool Delete_Pre_MnInUp { get; set; }
        [Parameter("Color", DefaultValue = "#016B05A6", Group = "Minor Internal Up   TrendLine")]
        public Color Color_MnInUp { get; set; }
        [Parameter("Style", DefaultValue = LineStyleInput.Dotted, Group = "Minor Internal Up   TrendLine")]
        public LineStyleInput Style_MnInUp { get; set; }
        [Parameter("Extend", DefaultValue = ExtendInput.None, Group = "Minor Internal Up   TrendLine")]
        public ExtendInput Extend_MnInUp { get; set; }
        [Parameter("Width", DefaultValue = 1, Group = "Minor Internal Up   TrendLine")]
        public int Width_MnInUp { get; set; }

        [Parameter("Show Minor Internal Down TrendLine", DefaultValue = true, Group = "Minor Internal Down TrendLine")]
        public bool Show_MnInDown { get; set; }
        [Parameter("Delete Previous Minor Internal Down TrendLine", DefaultValue = true, Group = "Minor Internal Down TrendLine")]
        public bool Delete_Pre_MnInDown { get; set; }
        [Parameter("Color", DefaultValue = "#AA0202A6", Group = "Minor Internal Down TrendLine")]
        public Color Color_MnInDown { get; set; }
        [Parameter("Style", DefaultValue = LineStyleInput.Dotted, Group = "Minor Internal Down TrendLine")]
        public LineStyleInput Style_MnInDown { get; set; }
        [Parameter("Extend", DefaultValue = ExtendInput.None, Group = "Minor Internal Down TrendLine")]
        public ExtendInput Extend_MnInDown { get; set; }
        [Parameter("Width", DefaultValue = 1, Group = "Minor Internal Down TrendLine")]
        public int Width_MnInDown { get; set; }

        private readonly List<string> _arrayType = new List<string>();
        private readonly List<double> _arrayValue = new List<double>();
        private readonly List<int> _arrayIndex = new List<int>();
        private readonly List<string> _arrayTypeAdv = new List<string>();
        private readonly List<double> _arrayValueAdv = new List<double>();
        private readonly List<int> _arrayIndexAdv = new List<int>();

        private bool _lock0 = true;
        private bool _lock1 = true;

        private double _majorHighLevel = double.NaN, _majorLowLevel = double.NaN;
        private int _majorHighIndex = -1, _majorLowIndex = -1;

        // Sticky "valuewhen" equivalents for cross-type pivot references
        private double _lastHighValue = double.NaN;
        private int _lastHighIndex = -1;
        private double _lastLowValue = double.NaN;
        private int _lastLowIndex = -1;

        // Previous-bar snapshot of the last ZZ element, used in UpdateAdvancedArrays
        private double _prevLastZzValue = double.NaN;
        private string _prevLastZzType;

        private readonly Dictionary<string, PointerState> _pointers = new Dictionary<string, PointerState>();
        private readonly Dictionary<string, TrendLineState> _lines = new Dictionary<string, TrendLineState>();

        protected override void Initialize()
        {
            var keys = new[] { "MHL", "MLH", "MHH", "MLL", "mHL", "mLH", "mHH", "mLL" };
            foreach (var key in keys)
                _pointers[key] = new PointerState();

            _lines["MjExUp"] = new TrendLineState();
            _lines["MjExDown"] = new TrendLineState();
            _lines["MjInUp"] = new TrendLineState();
            _lines["MjInDown"] = new TrendLineState();
            _lines["MnExUp"] = new TrendLineState();
            _lines["MnExDown"] = new TrendLineState();
            _lines["MnInUp"] = new TrendLineState();
            _lines["MnInDown"] = new TrendLineState();
        }

        public override void Calculate(int index)
        {
            ResetSignals(index);
            if (index < PP * 2 + 2)
                return;

            UpdateZigZag(index);

            if (_arrayTypeAdv.Count > 2)
            {
                var x0 = _arrayIndexAdv[_arrayTypeAdv.Count - 1];
                var y0 = _arrayValueAdv[_arrayTypeAdv.Count - 1];
                var t0 = _arrayTypeAdv[_arrayTypeAdv.Count - 1];
                UpdatePointers(x0, y0, t0);
            }

            var mll = _pointers["MLL"]; var mhh = _pointers["MHH"]; var mhl = _pointers["MHL"]; var mlh = _pointers["MLH"];
            var mllm = _pointers["mLL"]; var mhhm = _pointers["mHH"]; var mhlm = _pointers["mHL"]; var mlhm = _pointers["mLH"];

            var a1 = CorrectionChecker(index, "MjExUp", mll, true, Show_MjExUp, Delete_Pre_MjExUp, Color_MjExUp, Style_MjExUp, Extend_MjExUp, Width_MjExUp, Alert_MjExUp_B, Alert_MjExUp_R);
            var a2 = CorrectionChecker(index, "MjExDown", mhh, false, Show_MjExDown, Delete_Pre_MjExDown, Color_MjExDown, Style_MjExDown, Extend_MjExDown, Width_MjExDown, Alert_MjExDown_B, Alert_MjExDown_R);
            var a3 = CorrectionChecker(index, "MjInUp", mhl, true, Show_MjInUp, Delete_Pre_MjInUp, Color_MjInUp, Style_MjInUp, Extend_MjInUp, Width_MjInUp, Alert_MjInUp_B, Alert_MjInUp_R);
            var a4 = CorrectionChecker(index, "MjInDown", mlh, false, Show_MjInDown, Delete_Pre_MjInDown, Color_MjInDown, Style_MjInDown, Extend_MjInDown, Width_MjInDown, Alert_MjInDown_B, Alert_MjInDown_R);
            var a5 = CorrectionChecker(index, "MnExUp", mllm, true, Show_MnExUp, Delete_Pre_MnExUp, Color_MnExUp, Style_MnExUp, Extend_MnExUp, Width_MnExUp, Alert_MnExUp_B, Alert_MnExUp_R);
            var a6 = CorrectionChecker(index, "MnExDown", mhhm, false, Show_MnExDown, Delete_Pre_MnExDown, Color_MnExDown, Style_MnExDown, Extend_MnExDown, Width_MnExDown, Alert_MnExDown_B, Alert_MnExDown_R);
            var a7 = CorrectionChecker(index, "MnInUp", mhlm, true, Show_MnInUp, Delete_Pre_MnInUp, Color_MnInUp, Style_MnInUp, Extend_MnInUp, Width_MnInUp, Alert_MnInUp_B, Alert_MnInUp_R);
            var a8 = CorrectionChecker(index, "MnInDown", mlhm, false, Show_MnInDown, Delete_Pre_MnInDown, Color_MnInDown, Style_MnInDown, Extend_MnInDown, Width_MnInDown, Alert_MnInDown_B, Alert_MnInDown_R);

            DrawSignals(index, "MjExUp",   true,  a1.breakAlert, a1.reactAlert);
            DrawSignals(index, "MjExDown", false, a2.breakAlert, a2.reactAlert);
            DrawSignals(index, "MjInUp",   true,  a3.breakAlert, a3.reactAlert);
            DrawSignals(index, "MjInDown", false, a4.breakAlert, a4.reactAlert);
            DrawSignals(index, "MnExUp",   true,  a5.breakAlert, a5.reactAlert);
            DrawSignals(index, "MnExDown", false, a6.breakAlert, a6.reactAlert);
            DrawSignals(index, "MnInUp",   true,  a7.breakAlert, a7.reactAlert);
            DrawSignals(index, "MnInDown", false, a8.breakAlert, a8.reactAlert);
        }

        private (bool breakAlert, bool reactAlert) CorrectionChecker(int index, string key, PointerState p, bool isUp, bool showLine, bool deletePrev, Color color, LineStyleInput style, ExtendInput extend, int width, OnOff breakOn, OnOff reactOn)
        {
            var st = _lines[key];
            if (p.X0 != 0 && p.X1 != 0 && p.X0 != st.LastSeedX0)
            {
                var lineValid = isUp ? p.Y1 > p.Y0 : p.Y1 < p.Y0;
                if (lineValid)
                {
                    var permit = true;
                    for (var i = 1; i <= index - p.X0; i++)
                    {
                        var x = p.X0 + i;
                        var lp = LinePrice(p.X0, p.Y0, p.X1, p.Y1, x);
                        var close = Bars.ClosePrices[x];
                        permit = isUp ? permit && close > lp : permit && close < lp;
                        if (!permit) break;
                    }

                    if (permit)
                    {
                        if (showLine)
                        {
                            var line = Chart.DrawTrendLine($"ATL_{key}_{index}", p.X0, p.Y0, p.X1, p.Y1, color, width, MapStyle(style));
                            line.ExtendToInfinity = extend == ExtendInput.Both || extend == ExtendInput.Right;
                            st.Line = line;
                            if (deletePrev && st.PrevLineName != null)
                                Chart.RemoveObject(st.PrevLineName);
                            st.PrevLineName = line.Name;
                        }
                        st.PermitSet = true;
                    }
                }
                st.LastSeedX0 = p.X0;
            }

            if (st.Line != null)
            {
                var nowPrice = LinePrice(st.Line.Time1, st.Line.Y1, st.Line.Time2, st.Line.Y2, Bars.OpenTimes[index]);
                var keep = isUp ? Bars.ClosePrices[index] > nowPrice : Bars.ClosePrices[index] < nowPrice;
                if (keep && st.PermitSet)
                    st.Line.Time2 = Bars.OpenTimes[Math.Min(index + 1, Bars.Count - 1)];
                else
                    st.PermitSet = false;
            }

            var breakAlert = st.PermitSetPrev && !st.PermitSet;
            var reactAlert = false;
            if (st.Line != null && st.PermitSet)
            {
                var pNow = LinePrice(st.Line.Time1, st.Line.Y1, st.Line.Time2, st.Line.Y2, Bars.OpenTimes[index]);
                reactAlert = isUp
                    ? Bars.ClosePrices[index] > pNow && Bars.LowPrices[index] < pNow
                    : Bars.ClosePrices[index] < pNow && Bars.HighPrices[index] > pNow;
            }

            st.PermitSetPrev = st.PermitSet;
            breakAlert = breakAlert && breakOn == OnOff.On;
            reactAlert = reactAlert && reactOn == OnOff.On;
            return (breakAlert, reactAlert);
        }

        private void UpdatePointers(int x0, double y0, string t0)
        {
            foreach (var kv in _pointers)
            {
                var p = kv.Value;
                if (t0 == kv.Key && t0 != p.PrevType)
                {
                    if (p.X0 == 0)
                    {
                        p.X0 = x0; p.Y0 = y0;
                    }
                    else if (p.X1 == 0)
                    {
                        p.X1 = x0; p.Y1 = y0;
                    }
                    else
                    {
                        p.X0 = p.X1; p.Y0 = p.Y1;
                        p.X1 = x0; p.Y1 = y0;
                    }
                }
                p.PrevType = t0;
            }
        }

        private void UpdateZigZag(int index)
        {
            var pivotIndex = index - PP;
            if (pivotIndex - PP < 0 || pivotIndex + PP >= Bars.Count)
                return;

            var hasHigh = IsPivotHigh(pivotIndex);
            var hasLow = IsPivotLow(pivotIndex);
            var highValue = Bars.HighPrices[pivotIndex];
            var lowValue = Bars.LowPrices[pivotIndex];
            var highIndex = pivotIndex;
            var lowIndex = pivotIndex;

            // Update sticky "valuewhen" equivalents (Pine: ta.valuewhen)
            if (hasHigh) { _lastHighValue = highValue; _lastHighIndex = highIndex; }
            if (hasLow)  { _lastLowValue  = lowValue;  _lastLowIndex  = lowIndex; }

            if (hasHigh && hasLow)
            {
                if (_arrayType.Count == 0)
                    return;

                var lastType = Last(_arrayType);
                var lastVal = Last(_arrayValue);
                if (lastType == "L" || lastType == "LL")
                {
                    if (lowValue < lastVal) ReplaceLast(PivotTypeForLow(lowValue), lowValue, lowIndex); else Push(PivotTypeForHigh(highValue), highValue, highIndex);
                }
                else if (lastType == "H" || lastType == "HH")
                {
                    if (highValue > lastVal) ReplaceLast(PivotTypeForHigh(highValue), highValue, highIndex); else Push(PivotTypeForLow(lowValue), lowValue, lowIndex);
                }
                else if (lastType == "LH")
                {
                    if (highValue < lastVal) Push(PivotTypeForLow(lowValue), lowValue, lowIndex);
                    else if (Bars.ClosePrices[index] < lastVal) ReplaceLast(PivotTypeForHigh(highValue), highValue, highIndex);
                    else Push(PivotTypeForLow(lowValue), lowValue, lowIndex);
                }
                else if (lastType == "HL")
                {
                    if (lowValue > lastVal) Push(PivotTypeForHigh(highValue), highValue, highIndex);
                    else if (Bars.ClosePrices[index] > lastVal) ReplaceLast(PivotTypeForLow(lowValue), lowValue, lowIndex);
                    else Push(PivotTypeForHigh(highValue), highValue, highIndex);
                }
            }
            else if (hasHigh)
            {
                if (_arrayType.Count == 0) Push("H", highValue, highIndex);
                else
                {
                    var lastType = Last(_arrayType);
                    var lastVal = Last(_arrayValue);
                    if (lastType == "L" || lastType == "HL" || lastType == "LL")
                    {
                        if (highValue > lastVal) Push(PivotTypeForHigh(highValue), highValue, highIndex);
                        else if (!double.IsNaN(_lastLowValue)) ReplaceLast(PivotTypeForLow(_lastLowValue), _lastLowValue, _lastLowIndex);
                    }
                    else if (lastVal < highValue)
                        ReplaceLast(PivotTypeForHigh(highValue), highValue, highIndex);
                }
            }
            else if (hasLow)
            {
                if (_arrayType.Count == 0) Push("L", lowValue, lowIndex);
                else
                {
                    var lastType = Last(_arrayType);
                    var lastVal = Last(_arrayValue);
                    if (lastType == "H" || lastType == "HH" || lastType == "LH")
                    {
                        if (lowValue < lastVal) Push(PivotTypeForLow(lowValue), lowValue, lowIndex);
                        else if (!double.IsNaN(_lastHighValue)) ReplaceLast(PivotTypeForHigh(_lastHighValue), _lastHighValue, _lastHighIndex);
                    }
                    else if (lastVal > lowValue)
                        ReplaceLast(PivotTypeForLow(lowValue), lowValue, lowIndex);
                }
            }

            UpdateAdvancedArrays(index);

            // Save current last ZZ element for next bar's "previous-bar" comparison
            if (_arrayValue.Count > 0)
            {
                _prevLastZzValue = _arrayValue[_arrayValue.Count - 1];
                _prevLastZzType  = _arrayType[_arrayType.Count - 1];
            }
        }

        private void UpdateAdvancedArrays(int index)
        {
            if (_arrayType.Count == 2)
            {
                if (_arrayType[0] == "H")
                {
                    _majorHighLevel = _arrayValue[0]; _majorLowLevel = _arrayValue[1];
                    _majorHighIndex = _arrayIndex[0]; _majorLowIndex = _arrayIndex[1];
                }
                else if (_arrayType[0] == "L")
                {
                    _majorHighLevel = _arrayValue[1]; _majorLowLevel = _arrayValue[0];
                    _majorHighIndex = _arrayIndex[1]; _majorLowIndex = _arrayIndex[0];
                }
            }

            if (_arrayValue.Count == 1 && _lock0)
            {
                _arrayTypeAdv.Insert(0, "M" + _arrayType[0]); _arrayValueAdv.Insert(0, _arrayValue[0]); _arrayIndexAdv.Insert(0, _arrayIndex[0]); _lock0 = false;
            }
            if (_arrayValue.Count == 2 && _lock1)
            {
                _arrayTypeAdv.Insert(1, "M" + _arrayType[1]); _arrayValueAdv.Insert(1, _arrayValue[1]); _arrayIndexAdv.Insert(1, _arrayIndex[1]); _lock1 = false;
            }

            // Pine: "Making Copies of Arrays" — fires whenever the last ZZ element changed
            // compared to the previous bar (Pine's [1] operator).
            if (_arrayValue.Count > 1 && !double.IsNaN(_prevLastZzValue) &&
                _prevLastZzValue != _arrayValue[_arrayValue.Count - 1])
            {
                var last        = _arrayValue[_arrayValue.Count - 1];
                var zzLastType  = _arrayType[_arrayType.Count - 1];
                // Compare last character of current vs previous-bar type to decide push vs update
                var prevSuffix = _prevLastZzType != null ? Suffix(_prevLastZzType) : string.Empty;
                var lastSuffix = Suffix(zzLastType);
                if (prevSuffix != lastSuffix)
                {
                    _arrayTypeAdv.Add("m" + zzLastType);
                    _arrayValueAdv.Add(last);
                    _arrayIndexAdv.Add(_arrayIndex[_arrayIndex.Count - 1]);
                }
                else if (_arrayValueAdv.Count > 0)
                {
                    _arrayValueAdv[_arrayValueAdv.Count - 1] = last;
                    _arrayIndexAdv[_arrayIndexAdv.Count - 1] = _arrayIndex[_arrayIndex.Count - 1];
                }
            }

            if (_arrayValueAdv.Count <= 1 || double.IsNaN(_majorHighLevel) || double.IsNaN(_majorLowLevel))
                return;

            var lastType = Last(_arrayTypeAdv);
            var lastVal = Last(_arrayValueAdv);
            var lastIdx = Last(_arrayIndexAdv);

            if (Bars.ClosePrices[index] > _majorHighLevel)
            {
                if (lastType == "mL") PromoteLast("ML", lastVal, lastIdx, false);
                else if (lastType == "mHL" || lastType == "mLL") PromoteLast("M" + Last(_arrayType), lastVal, lastIdx, false);
                else if ((lastType == "mLH" || lastType == "mHH" || lastType == "MLH" || lastType == "MHH") && _arrayTypeAdv.Count > 1)
                {
                    var t2 = _arrayTypeAdv[_arrayTypeAdv.Count - 2];
                    if (t2 == "mHL" || t2 == "mLL") PromoteIndex(_arrayTypeAdv.Count - 2, "M" + _arrayType[_arrayType.Count - 2], false);
                }
            }

            if (lastVal > _majorHighLevel)
            {
                if (lastType == "mH") PromoteLast("MH", lastVal, lastIdx, true);
                else if (lastType == "mLH" || lastType == "mHH" || lastType == "MHH") PromoteLast("M" + Last(_arrayType), lastVal, lastIdx, true);
            }

            if (Bars.ClosePrices[index] < _majorLowLevel)
            {
                if (lastType == "mH") PromoteLast("MH", lastVal, lastIdx, true);
                else if (lastType == "mLH" || lastType == "mHH") PromoteLast("M" + Last(_arrayType), lastVal, lastIdx, true);
                else if ((lastType == "mHL" || lastType == "mLL" || lastType == "MHL" || lastType == "MLL") && _arrayTypeAdv.Count > 1)
                {
                    var t2 = _arrayTypeAdv[_arrayTypeAdv.Count - 2];
                    if (t2 == "mLH" || t2 == "mHH") PromoteIndex(_arrayTypeAdv.Count - 2, "M" + _arrayType[_arrayType.Count - 2], true);
                }
            }

            if (lastVal < _majorLowLevel)
            {
                if (lastType == "mL") PromoteLast("ML", lastVal, lastIdx, false);
                else if (lastType == "mHL" || lastType == "mLL" || lastType == "MLL") PromoteLast("M" + Last(_arrayType), lastVal, lastIdx, false);
            }
        }

        private void PromoteLast(string newType, double value, int idx, bool isHigh)
        {
            _arrayTypeAdv[_arrayTypeAdv.Count - 1] = newType;
            if (isHigh) { _majorHighLevel = value; _majorHighIndex = idx; }
            else { _majorLowLevel = value; _majorLowIndex = idx; }
        }

        private void PromoteIndex(int i, string newType, bool isHigh)
        {
            _arrayTypeAdv[i] = newType;
            if (isHigh) { _majorHighLevel = _arrayValueAdv[i]; _majorHighIndex = _arrayIndexAdv[i]; }
            else { _majorLowLevel = _arrayValueAdv[i]; _majorLowIndex = _arrayIndexAdv[i]; }
        }

        private static string Suffix(string t) => t.Length <= 1 ? t : t.Substring(t.Length - 1);
        private string PivotTypeForHigh(double high) => _arrayValue.Count > 2 ? (_arrayValue[_arrayValue.Count - 2] < high ? "HH" : "LH") : "H";
        private string PivotTypeForLow(double low) => _arrayValue.Count > 2 ? (_arrayValue[_arrayValue.Count - 2] < low ? "HL" : "LL") : "L";

        private void Push(string t, double v, int i) { _arrayType.Add(t); _arrayValue.Add(v); _arrayIndex.Add(i); }
        private void ReplaceLast(string t, double v, int i)
        {
            if (_arrayType.Count == 0) { Push(t, v, i); return; }
            _arrayType.RemoveAt(_arrayType.Count - 1); _arrayValue.RemoveAt(_arrayValue.Count - 1); _arrayIndex.RemoveAt(_arrayIndex.Count - 1);
            Push(t, v, i);
        }

        private static bool IsEnumDotted(LineStyleInput s) => s == LineStyleInput.Dotted;

        private bool IsPivotHigh(int i)
        {
            var p = Bars.HighPrices[i];
            for (var j = i - PP; j <= i + PP; j++)
            {
                if (j == i || j < 0 || j >= Bars.Count) continue;
                if (Bars.HighPrices[j] >= p) return false;
            }
            return true;
        }

        private bool IsPivotLow(int i)
        {
            var p = Bars.LowPrices[i];
            for (var j = i - PP; j <= i + PP; j++)
            {
                if (j == i || j < 0 || j >= Bars.Count) continue;
                if (Bars.LowPrices[j] <= p) return false;
            }
            return true;
        }

        private static T Last<T>(List<T> list) => list[list.Count - 1];

        private static LineStyle MapStyle(LineStyleInput style)
        {
            switch (style)
            {
                case LineStyleInput.Dashed: return LineStyle.Lines;
                case LineStyleInput.Dotted: return LineStyle.DotsRare;
                default: return LineStyle.Solid;
            }
        }

        private static double LinePrice(int x1, double y1, int x2, double y2, int x)
        {
            if (x2 == x1) return y2;
            var m = (y2 - y1) / (x2 - x1);
            return y1 + m * (x - x1);
        }

        private static double LinePrice(DateTime t1, double y1, DateTime t2, double y2, DateTime t)
        {
            var dt = (t2 - t1).TotalSeconds;
            if (Math.Abs(dt) < 1e-9) return y2;
            var m = (y2 - y1) / dt;
            return y1 + m * (t - t1).TotalSeconds;
        }

        private static readonly string[] SignalKeys = { "MjExUp", "MjExDown", "MjInUp", "MjInDown", "MnExUp", "MnExDown", "MnInUp", "MnInDown" };

        private void ResetSignals(int index)
        {
            foreach (var key in SignalKeys)
            {
                Chart.RemoveObject($"ATL_B_{key}_{index}");
                Chart.RemoveObject($"ATL_R_{key}_{index}");
            }
        }

        // Break Up   → red   ▼ above bar  (support trendline broken downward)
        // Break Down → green ▲ below bar  (resistance trendline broken upward)
        // React Up   → green ▲ below bar  (price bounced off support)
        // React Down → red   ▼ above bar  (price bounced off resistance)
        private void DrawSignals(int index, string key, bool isUp, bool breakAlert, bool reactAlert)
        {
            var pip = Symbol.PipSize;
            if (breakAlert)
            {
                ChartText t;
                if (isUp)
                    t = Chart.DrawText($"ATL_B_{key}_{index}", "▼", index, Bars.HighPrices[index] + pip, Color.Red);
                else
                    t = Chart.DrawText($"ATL_B_{key}_{index}", "▲", index, Bars.LowPrices[index]  - pip, Color.Green);
                t.FontSize = 12;
            }
            if (reactAlert)
            {
                ChartText t;
                if (isUp)
                    t = Chart.DrawText($"ATL_R_{key}_{index}", "▲", index, Bars.LowPrices[index]  - pip, Color.Green);
                else
                    t = Chart.DrawText($"ATL_R_{key}_{index}", "▼", index, Bars.HighPrices[index] + pip, Color.Red);
                t.FontSize = 12;
            }
        }

        private sealed class PointerState
        {
            public int X0, X1;
            public double Y0, Y1;
            public string PrevType = string.Empty;
        }

        private sealed class TrendLineState
        {
            public int LastSeedX0;
            public bool PermitSet;
            public bool PermitSetPrev;
            public ChartTrendLine Line;
            public string PrevLineName;
        }
    }
}
