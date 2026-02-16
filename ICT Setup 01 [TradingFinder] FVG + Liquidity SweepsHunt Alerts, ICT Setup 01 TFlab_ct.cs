using System;
using cAlgo.API;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class ICTSetup01TradingFinderFVGAndLiquiditySweepsHuntAlertsICTSetup01TFlab_ct : Indicator
    {
        [Parameter("FVG Detector Multiplier Factor", DefaultValue = 1.0, MinValue = 1.0, Group = "FVGs Setting")]
        public double MATR { get; set; }

        [Parameter("FVG Validity Period", DefaultValue = 15, MinValue = 2, Group = "FVGs Setting")]
        public int ValLenFVG { get; set; }

        [Parameter("Level in Low Risk Zone", DefaultValue = false, Group = "Discount & Premium")]
        public bool DisANDPre { get; set; }

        [Parameter("Issuing Signals Method", DefaultValue = "Hunt", Group = "Signal")]
        public string IssSigM { get; set; }

        [Parameter("The number of signals allowed from a Zone", DefaultValue = 3, MinValue = 1, Group = "Signal")]
        public int NSA { get; set; }

        [Parameter("Signal after Hunts/Sweeps", DefaultValue = false, Group = "Signal")]
        public bool SaH { get; set; }

        [Parameter("How Many Hunts/Sweeps?", DefaultValue = 2, MinValue = 1, Group = "Signal")]
        public int HMH { get; set; }

        [Parameter("Show All Long Setup", DefaultValue = false, Group = "Show or Hide")]
        public bool SALS { get; set; }

        [Parameter("Show All Short Setup", DefaultValue = false, Group = "Show or Hide")]
        public bool SASS { get; set; }

        [Parameter("Alert", DefaultValue = "On", Group = "Alert")]
        public string AlertSetting { get; set; }

        [Parameter("Alert Name", DefaultValue = "ICT Setup 01 Alerts [TradingFinder]", Group = "Alert")]
        public string AlertName { get; set; }

        [Parameter("Long Position Message", DefaultValue = "Long Signal Position Based on ICT Setup 01 [FVG Hunts]", Group = "Alert")]
        public string MessageBull { get; set; }

        [Parameter("Short Position Message", DefaultValue = "Short Signal Position Based on ICT Setup 01 [FVG Hunts]", Group = "Alert")]
        public string MessageBear { get; set; }

        private const int AtrLength = 55;

        private IndicatorDataSeries _tr;
        private IndicatorDataSeries _atr;

        private IndicatorDataSeries _distalLvLBu;
        private IndicatorDataSeries _proximalLvLBu;
        private IndicatorDataSeries _distalLvLBe;
        private IndicatorDataSeries _proximalLvLBe;
        private IndicatorDataSeries _buPointSeries;
        private IndicatorDataSeries _bePointSeries;

        private double _buFVGDistal;
        private double _buFVGProximal;
        private int _buFVGPoint;
        private double _buPremium;
        private double _buDiscount;
        private double _buEquilibrium;

        private double _beFVGDistal;
        private double _beFVGProximal;
        private int _beFVGPoint;
        private double _bePremium;
        private double _beDiscount;
        private double _beEquilibrium;

        private bool _validityBuFVG = true;
        private bool _validityBeFVG = true;

        private double _lowTracker;
        private double _highTracker;

        private int _longCount;
        private int _shortCount;

        private bool _longSignal;
        private bool _shortSignal;

        private bool _bullFVG;
        private bool _bearFVG;


        protected override void Initialize()
        {
            _tr = CreateDataSeries();
            _atr = CreateDataSeries();

            _distalLvLBu = CreateDataSeries();
            _proximalLvLBu = CreateDataSeries();
            _distalLvLBe = CreateDataSeries();
            _proximalLvLBe = CreateDataSeries();
            _buPointSeries = CreateDataSeries();
            _bePointSeries = CreateDataSeries();
        }

        public override void Calculate(int index)
        {
            CalculateAtr(index);

            if (index == 0)
            {
                _distalLvLBu[index] = 0.0;
                _proximalLvLBu[index] = 0.0;
                _distalLvLBe[index] = 0.0;
                _proximalLvLBe[index] = 0.0;
                _buPointSeries[index] = 0.0;
                _bePointSeries[index] = 0.0;
                return;
            }

            _distalLvLBu[index] = _distalLvLBu[index - 1];
            _proximalLvLBu[index] = _proximalLvLBu[index - 1];
            _distalLvLBe[index] = _distalLvLBe[index - 1];
            _proximalLvLBe[index] = _proximalLvLBe[index - 1];

            var high = Bars.HighPrices[index];
            var low = Bars.LowPrices[index];
            var close = Bars.ClosePrices[index];

            _bullFVG = false;
            _bearFVG = false;

            if (index >= 2)
            {
                var high2 = Bars.HighPrices[index - 2];
                var low2 = Bars.LowPrices[index - 2];
                var high1 = Bars.HighPrices[index - 1];
                var low1 = Bars.LowPrices[index - 1];
                var atrValue = _atr[index];

                if ((high - low2) > (MATR * atrValue))
                {
                    if (low > high2 && low2 < low1 && high1 < high && (high + low2) / 2.0 >= high2)
                    {
                        _buFVGDistal = high2;
                        _buFVGProximal = low;
                        _buFVGPoint = index;
                        _buDiscount = low2;
                        _buPremium = high;
                        _buEquilibrium = (high + low2) / 2.0;
                        _bullFVG = true;
                    }
                }

                if ((high2 - low) > (MATR * atrValue))
                {
                    if (low2 > high && high2 > high1 && low1 > low && (low + high2) / 2.0 <= low2)
                    {
                        _beFVGDistal = low2;
                        _beFVGProximal = high;
                        _beFVGPoint = index;
                        _beDiscount = low;
                        _bePremium = high2;
                        _beEquilibrium = (low + high2) / 2.0;
                        _bearFVG = true;
                    }
                }
            }

            if (DisANDPre)
            {
                if (_bullFVG)
                {
                    _distalLvLBu[index] = _buFVGDistal;
                    _proximalLvLBu[index] = _buEquilibrium >= _buFVGProximal ? _buFVGProximal : _buEquilibrium;
                }

                if (_bearFVG)
                {
                    _distalLvLBe[index] = _beFVGDistal;
                    _proximalLvLBe[index] = _beEquilibrium <= _beFVGProximal ? _beFVGProximal : _beEquilibrium;
                }
            }
            else
            {
                if (_bullFVG)
                {
                    _distalLvLBu[index] = _buFVGDistal;
                    _proximalLvLBu[index] = _buFVGProximal;
                }

                if (_bearFVG)
                {
                    _distalLvLBe[index] = _beFVGDistal;
                    _proximalLvLBe[index] = _beFVGProximal;
                }
            }

            var body1 = Bars.ClosePrices[index - 1] - Bars.OpenPrices[index - 1];
            var prevDistalBu = _distalLvLBu[index - 1];
            var prevProximalBu = _proximalLvLBu[index - 1];
            var prevDistalBe = _distalLvLBe[index - 1];
            var prevProximalBe = _proximalLvLBe[index - 1];

            if (_validityBuFVG)
            {
                if (body1 > 0)
                {
                    var sweepCheck = IssSigM == "Sweeps"
                        ? Bars.OpenPrices[index - 1] < prevDistalBu
                        : Bars.LowPrices[index - 1] < prevDistalBu;

                    if (sweepCheck || index > _buFVGPoint + ValLenFVG || (!SaH && _longCount > NSA - 1))
                        _validityBuFVG = false;
                    else if (Bars.OpenPrices[index - 1] < prevProximalBu && Bars.OpenPrices[index - 1] > prevDistalBu)
                        _proximalLvLBu[index] = Bars.OpenPrices[index - 1];
                }

                if (body1 <= 0)
                {
                    var sweepCheck = IssSigM == "Sweeps"
                        ? Bars.ClosePrices[index - 1] < prevDistalBu
                        : Bars.LowPrices[index - 1] < prevDistalBu;

                    if (sweepCheck || index > _buFVGPoint + ValLenFVG || (!SaH && _longCount > NSA - 1))
                        _validityBuFVG = false;
                    else if (Bars.ClosePrices[index - 1] < prevProximalBu && Bars.ClosePrices[index - 1] > prevDistalBu)
                        _proximalLvLBu[index] = Bars.ClosePrices[index - 1];
                }
            }

            if (_validityBeFVG)
            {
                if (body1 > 0)
                {
                    var sweepCheck = IssSigM == "Sweeps"
                        ? Bars.ClosePrices[index - 1] > prevDistalBe
                        : Bars.HighPrices[index - 1] > prevDistalBe;

                    if (sweepCheck || index > _beFVGPoint + ValLenFVG || (!SaH && _shortCount > NSA - 1))
                        _validityBeFVG = false;
                    else if (Bars.ClosePrices[index - 1] > prevProximalBe && Bars.ClosePrices[index - 1] < prevDistalBe)
                        _proximalLvLBe[index] = Bars.ClosePrices[index - 1];
                }

                if (body1 <= 0)
                {
                    var sweepCheck = IssSigM == "Sweeps"
                        ? Bars.OpenPrices[index - 1] > prevDistalBe
                        : Bars.HighPrices[index - 1] > prevDistalBe;

                    if (sweepCheck || index > _beFVGPoint + ValLenFVG || (!SaH && _shortCount > NSA - 1))
                        _validityBeFVG = false;
                    else if (Bars.OpenPrices[index - 1] > prevProximalBe && Bars.OpenPrices[index - 1] < prevDistalBe)
                        _proximalLvLBe[index] = Bars.OpenPrices[index - 1];
                }
            }

            if (_buPointSeries[index - 1] != _buFVGPoint)
            {
                _validityBuFVG = true;
                _lowTracker = 0.0;
                _longCount = 0;
                _longSignal = false;
            }

            if (_bePointSeries[index - 1] != _beFVGPoint)
            {
                _validityBeFVG = true;
                _highTracker = 0.0;
                _shortCount = 0;
                _shortSignal = false;
            }

            if (_validityBuFVG)
            {
                if (_lowTracker == 0.0 && low < _proximalLvLBu[index])
                    _lowTracker = low;

                if (low < _lowTracker && _lowTracker > 0.0)
                {
                    _lowTracker = low;
                    if (close >= _proximalLvLBu[index])
                    {
                        _longCount += 1;
                        if (SaH)
                            _longSignal = _longCount == HMH;
                        else
                            _longSignal = true;
                    }
                    else
                    {
                        _longSignal = false;
                    }
                }
                else
                {
                    _longSignal = false;
                }
            }
            else
            {
                _lowTracker = 0.0;
                _longCount = 0;
                _longSignal = false;
            }

            if (_validityBeFVG)
            {
                if (_highTracker == 0.0 && high > _proximalLvLBe[index])
                    _highTracker = high;

                if (high > _highTracker && _highTracker > 0.0)
                {
                    _highTracker = high;
                    if (close <= _proximalLvLBe[index])
                    {
                        _shortCount += 1;
                        if (SaH)
                            _shortSignal = _shortCount == HMH;
                        else
                            _shortSignal = true;
                    }
                    else
                    {
                        _shortSignal = false;
                    }
                }
                else
                {
                    _shortSignal = false;
                }
            }
            else
            {
                _highTracker = 0.0;
                _shortCount = 0;
                _shortSignal = false;
            }

            _buPointSeries[index] = _buFVGPoint;
            _bePointSeries[index] = _beFVGPoint;

            DrawSignals(index);
            DrawCurrentZones(index);
            EmitAlerts(index);
        }

        private void CalculateAtr(int index)
        {
            var high = Bars.HighPrices[index];
            var low = Bars.LowPrices[index];

            if (index == 0)
            {
                _tr[index] = double.NaN;
                _atr[index] = double.NaN;
                return;
            }

            var prevClose = Bars.ClosePrices[index - 1];
            var tr1 = high - low;
            var tr2 = Math.Abs(high - prevClose);
            var tr3 = Math.Abs(low - prevClose);
            _tr[index] = Math.Max(tr1, Math.Max(tr2, tr3));

            if (index < AtrLength)
            {
                var sum = 0.0;
                var count = 0;
                for (var i = 0; i <= index; i++)
                {
                    var v = _tr[i];
                    if (double.IsNaN(v))
                        continue;
                    sum += v;
                    count++;
                }

                _atr[index] = count > 0 ? sum / count : double.NaN;
                return;
            }

            var rollingSum = 0.0;
            var rollingCount = 0;
            var start = index - AtrLength + 1;
            for (var i = start; i <= index; i++)
            {
                var v = _tr[i];
                if (double.IsNaN(v))
                    continue;
                rollingSum += v;
                rollingCount++;
            }

            _atr[index] = rollingCount == AtrLength ? rollingSum / AtrLength : double.NaN;
        }

        private void DrawSignals(int index)
        {
            if (_longSignal)
            {
                Chart.DrawIcon($"long_signal_{index}", ChartIconType.UpTriangle, index, Bars.LowPrices[index], Color.Green);
            }

            if (_shortSignal)
            {
                Chart.DrawIcon($"short_signal_{index}", ChartIconType.DownTriangle, index, Bars.HighPrices[index], Color.Red);
            }
        }

        private void DrawCurrentZones(int index)
        {
            if (_bullFVG)
            {
                var bullDistalId = $"bull_distal_{_buFVGPoint}";
                var bullProximalId = $"bull_proximal_{_buFVGPoint}";
                var bullLabelId = $"bull_lbl_{_buFVGPoint}";

                Chart.DrawTrendLine(bullDistalId, _buFVGPoint, _distalLvLBu[index], index, _distalLvLBu[index], Color.FromArgb(186, 8, 8, 8), 1, LineStyle.LinesDots);
                Chart.DrawTrendLine(bullProximalId, _buFVGPoint, _proximalLvLBu[index], index, _proximalLvLBu[index], Color.FromArgb(186, 8, 8, 8), 1, LineStyle.LinesDots);
                Chart.DrawText(bullLabelId, "FVG", _buFVGPoint + 1, _distalLvLBu[index], Color.Black);

                if (!SALS)
                {
                    var previousPoint = (int)_buPointSeries[index - 1];
                    if (previousPoint > 0 && previousPoint != _buFVGPoint)
                    {
                        Chart.RemoveObject($"bull_distal_{previousPoint}");
                        Chart.RemoveObject($"bull_proximal_{previousPoint}");
                        Chart.RemoveObject($"bull_lbl_{previousPoint}");
                    }
                }
            }

            if (_bearFVG)
            {
                var bearDistalId = $"bear_distal_{_beFVGPoint}";
                var bearProximalId = $"bear_proximal_{_beFVGPoint}";
                var bearLabelId = $"bear_lbl_{_beFVGPoint}";

                Chart.DrawTrendLine(bearDistalId, _beFVGPoint, _distalLvLBe[index], index, _distalLvLBe[index], Color.FromArgb(186, 8, 8, 8), 1, LineStyle.LinesDots);
                Chart.DrawTrendLine(bearProximalId, _beFVGPoint, _proximalLvLBe[index], index, _proximalLvLBe[index], Color.FromArgb(186, 8, 8, 8), 1, LineStyle.LinesDots);
                Chart.DrawText(bearLabelId, "FVG", _beFVGPoint + 1, _distalLvLBe[index], Color.Black);

                if (!SASS)
                {
                    var previousPoint = (int)_bePointSeries[index - 1];
                    if (previousPoint > 0 && previousPoint != _beFVGPoint)
                    {
                        Chart.RemoveObject($"bear_distal_{previousPoint}");
                        Chart.RemoveObject($"bear_proximal_{previousPoint}");
                        Chart.RemoveObject($"bear_lbl_{previousPoint}");
                    }
                }
            }
        }

        private void EmitAlerts(int index)
        {
            if (AlertSetting != "On")
                return;

            if (_longSignal)
                Print("{0} | LONG | Bar={1} | {2}", AlertName, index, MessageBull);

            if (_shortSignal)
                Print("{0} | SHORT | Bar={1} | {2}", AlertName, index, MessageBear);
        }
    }
}
