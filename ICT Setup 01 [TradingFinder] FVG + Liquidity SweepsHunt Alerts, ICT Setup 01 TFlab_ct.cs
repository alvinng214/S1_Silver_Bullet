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

        [Parameter("Message Frequency", DefaultValue = "Once Per Bar", Group = "Alert")]
        public string Frequncy { get; set; }

        [Parameter("Show Alert time by Time Zone", DefaultValue = "UTC", Group = "Alert")]
        public string UTC { get; set; }

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

        private int _lastLongAlertBar = -1;
        private int _lastShortAlertBar = -1;


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
                _tr[index] = high - low;
                _atr[index] = double.NaN;
                return;
            }

            var prevClose = Bars.ClosePrices[index - 1];
            var tr1 = high - low;
            var tr2 = Math.Abs(high - prevClose);
            var tr3 = Math.Abs(low - prevClose);
            _tr[index] = Math.Max(tr1, Math.Max(tr2, tr3));

            if (index < AtrLength - 1)
            {
                _atr[index] = double.NaN;
                return;
            }

            if (index == AtrLength - 1)
            {
                var sum = 0.0;
                for (var i = 0; i < AtrLength; i++)
                    sum += _tr[i];

                _atr[index] = sum / AtrLength;
                return;
            }

            _atr[index] = ((_atr[index - 1] * (AtrLength - 1)) + _tr[index]) / AtrLength;
        }

        private void DrawSignals(int index)
        {
            if (_longSignal)
            {
                Chart.DrawIcon($"long_signal_{index}", ChartIconType.UpTriangle, index, Bars.LowPrices[index], Color.FromHex("#008304"));
            }

            if (_shortSignal)
            {
                Chart.DrawIcon($"short_signal_{index}", ChartIconType.DownTriangle, index, Bars.HighPrices[index], Color.FromHex("#d30000"));
            }
        }

        private void DrawCurrentZones(int index)
        {
            if (_validityBuFVG && _buFVGPoint > 0 && index <= _buFVGPoint + ValLenFVG)
            {
                var bullDistalId = $"bull_distal_{_buFVGPoint}";
                var bullProximalId = $"bull_proximal_{_buFVGPoint}";
                var bullFillId = $"bull_fill_{_buFVGPoint}";
                var bullLabelId = $"bull_lbl_{_buFVGPoint}";

                Chart.DrawTrendLine(bullDistalId, _buFVGPoint, _distalLvLBu[index], index, _distalLvLBu[index], Color.FromArgb(186, 8, 8, 8), 1, LineStyle.LinesDots);
                Chart.DrawTrendLine(bullProximalId, _buFVGPoint, _proximalLvLBu[index], index, _proximalLvLBu[index], Color.FromArgb(186, 8, 8, 8), 1, LineStyle.LinesDots);
                var bullFill = Chart.DrawRectangle(bullFillId, _buFVGPoint, _distalLvLBu[index], index, _proximalLvLBu[index], Color.FromArgb(77, 76, 175, 79));
                bullFill.IsFilled = true;
                bullFill.IsInteractive = false;
                Chart.DrawText(bullLabelId, "FVG", _buFVGPoint + 1, _distalLvLBu[index], Color.Black);
            }

            if (_bullFVG)
            {
                DrawBullDiscountPremium(_buFVGPoint);
                if (!SALS)
                {
                    var previousPoint = (int)_buPointSeries[index - 1];
                    if (previousPoint > 0 && previousPoint != _buFVGPoint)
                    {
                        Chart.RemoveObject($"bull_distal_{previousPoint}");
                        Chart.RemoveObject($"bull_proximal_{previousPoint}");
                        Chart.RemoveObject($"bull_fill_{previousPoint}");
                        Chart.RemoveObject($"bull_lbl_{previousPoint}");
                        RemoveBullDiscountPremium(previousPoint);
                    }
                }
            }

            if (_validityBeFVG && _beFVGPoint > 0 && index <= _beFVGPoint + ValLenFVG)
            {
                var bearDistalId = $"bear_distal_{_beFVGPoint}";
                var bearProximalId = $"bear_proximal_{_beFVGPoint}";
                var bearFillId = $"bear_fill_{_beFVGPoint}";
                var bearLabelId = $"bear_lbl_{_beFVGPoint}";

                Chart.DrawTrendLine(bearDistalId, _beFVGPoint, _distalLvLBe[index], index, _distalLvLBe[index], Color.FromArgb(186, 8, 8, 8), 1, LineStyle.LinesDots);
                Chart.DrawTrendLine(bearProximalId, _beFVGPoint, _proximalLvLBe[index], index, _proximalLvLBe[index], Color.FromArgb(186, 8, 8, 8), 1, LineStyle.LinesDots);
                var bearFill = Chart.DrawRectangle(bearFillId, _beFVGPoint, _distalLvLBe[index], index, _proximalLvLBe[index], Color.FromArgb(77, 255, 49, 49));
                bearFill.IsFilled = true;
                bearFill.IsInteractive = false;
                Chart.DrawText(bearLabelId, "FVG", _beFVGPoint + 1, _distalLvLBe[index], Color.Black);
            }

            if (_bearFVG)
            {
                DrawBearDiscountPremium(_beFVGPoint);
                if (!SASS)
                {
                    var previousPoint = (int)_bePointSeries[index - 1];
                    if (previousPoint > 0 && previousPoint != _beFVGPoint)
                    {
                        Chart.RemoveObject($"bear_distal_{previousPoint}");
                        Chart.RemoveObject($"bear_proximal_{previousPoint}");
                        Chart.RemoveObject($"bear_fill_{previousPoint}");
                        Chart.RemoveObject($"bear_lbl_{previousPoint}");
                        RemoveBearDiscountPremium(previousPoint);
                    }
                }
            }
        }

        private void DrawBullDiscountPremium(int point)
        {
            if (!DisANDPre)
                return;

            var discountLineId = $"bull_discount_line_{point}";
            var premiumLineId = $"bull_premium_line_{point}";
            var equilibriumLineId = $"bull_equilibrium_line_{point}";
            var disRectId = $"bull_discount_fill_{point}";
            var premRectId = $"bull_premium_fill_{point}";
            var disLabelId = $"bull_discount_lbl_{point}";
            var premLabelId = $"bull_premium_lbl_{point}";
            var equiLabelId = $"bull_equilibrium_lbl_{point}";

            Chart.DrawTrendLine(discountLineId, point + 12, _buDiscount, point + 20, _buDiscount, Color.Transparent);
            Chart.DrawTrendLine(premiumLineId, point + 12, _buPremium, point + 20, _buPremium, Color.Transparent);
            Chart.DrawTrendLine(equilibriumLineId, point + 12, _buEquilibrium, point + 20, _buEquilibrium, Color.Transparent);

            var disRect = Chart.DrawRectangle(disRectId, point + 12, _buDiscount, point + 20, _buEquilibrium, Color.FromArgb(110, 211, 238, 255));
            disRect.IsFilled = true;
            disRect.IsInteractive = false;

            var premRect = Chart.DrawRectangle(premRectId, point + 12, _buPremium, point + 20, _buEquilibrium, Color.FromArgb(75, 255, 165, 100));
            premRect.IsFilled = true;
            premRect.IsInteractive = false;

            Chart.DrawText(disLabelId, "Discount", point + 16, _buDiscount, Color.Black);
            Chart.DrawText(premLabelId, "Premium", point + 16, _buPremium, Color.Black);
            Chart.DrawText(equiLabelId, "EQU", point + 16, _buEquilibrium, Color.Black);
        }

        private void DrawBearDiscountPremium(int point)
        {
            if (!DisANDPre)
                return;

            var premiumLineId = $"bear_premium_line_{point}";
            var discountLineId = $"bear_discount_line_{point}";
            var equilibriumLineId = $"bear_equilibrium_line_{point}";
            var disRectId = $"bear_discount_fill_{point}";
            var premRectId = $"bear_premium_fill_{point}";
            var disLabelId = $"bear_discount_lbl_{point}";
            var premLabelId = $"bear_premium_lbl_{point}";
            var equiLabelId = $"bear_equilibrium_lbl_{point}";

            Chart.DrawTrendLine(premiumLineId, point + 12, _bePremium, point + 20, _bePremium, Color.Transparent);
            Chart.DrawTrendLine(discountLineId, point + 12, _beDiscount, point + 20, _beDiscount, Color.Transparent);
            Chart.DrawTrendLine(equilibriumLineId, point + 12, _beEquilibrium, point + 20, _beEquilibrium, Color.Transparent);

            var disRect = Chart.DrawRectangle(disRectId, point + 12, _beDiscount, point + 20, _beEquilibrium, Color.FromArgb(110, 211, 238, 255));
            disRect.IsFilled = true;
            disRect.IsInteractive = false;

            var premRect = Chart.DrawRectangle(premRectId, point + 12, _bePremium, point + 20, _beEquilibrium, Color.FromArgb(75, 255, 165, 100));
            premRect.IsFilled = true;
            premRect.IsInteractive = false;

            Chart.DrawText(disLabelId, "Discount", point + 16, _beDiscount, Color.Black);
            Chart.DrawText(premLabelId, "Premium", point + 16, _bePremium, Color.Black);
            Chart.DrawText(equiLabelId, "EQU", point + 16, _beEquilibrium, Color.Black);
        }

        private void RemoveBullDiscountPremium(int point)
        {
            if (point <= 0)
                return;

            Chart.RemoveObject($"bull_discount_line_{point}");
            Chart.RemoveObject($"bull_premium_line_{point}");
            Chart.RemoveObject($"bull_equilibrium_line_{point}");
            Chart.RemoveObject($"bull_discount_fill_{point}");
            Chart.RemoveObject($"bull_premium_fill_{point}");
            Chart.RemoveObject($"bull_discount_lbl_{point}");
            Chart.RemoveObject($"bull_premium_lbl_{point}");
            Chart.RemoveObject($"bull_equilibrium_lbl_{point}");
        }

        private void RemoveBearDiscountPremium(int point)
        {
            if (point <= 0)
                return;

            Chart.RemoveObject($"bear_discount_line_{point}");
            Chart.RemoveObject($"bear_premium_line_{point}");
            Chart.RemoveObject($"bear_equilibrium_line_{point}");
            Chart.RemoveObject($"bear_discount_fill_{point}");
            Chart.RemoveObject($"bear_premium_fill_{point}");
            Chart.RemoveObject($"bear_discount_lbl_{point}");
            Chart.RemoveObject($"bear_premium_lbl_{point}");
            Chart.RemoveObject($"bear_equilibrium_lbl_{point}");
        }

        private void EmitAlerts(int index)
        {
            if (AlertSetting != "On")
                return;

            if (_longSignal && _lastLongAlertBar != index)
            {
                Print("{0} | LONG | Bar={1} | {2}", AlertName, index, MessageBull);
                _lastLongAlertBar = index;
            }

            if (_shortSignal && _lastShortAlertBar != index)
            {
                Print("{0} | SHORT | Bar={1} | {2}", AlertName, index, MessageBear);
                _lastShortAlertBar = index;
            }
        }
    }
}
