using System;
using cAlgo.API;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class IctSetup01FvgLiquidityHuntIndicator : Indicator
    {
        [Parameter("FVG Detector Multiplier Factor", DefaultValue = 1.0, MinValue = 1.0, Group = "FVGs Setting")]
        public double FvgDetectorMultiplier { get; set; }

        [Parameter("FVG Validity Period", DefaultValue = 15, MinValue = 2, Group = "FVGs Setting")]
        public int FvgValidityPeriod { get; set; }

        [Parameter("Level in Low Risk Zone", DefaultValue = false, Group = "Discount & Premium")]
        public bool UseDiscountAndPremium { get; set; }

        [Parameter("Issuing Signals Method", DefaultValue = "Hunt", Group = "Signal")]
        public string SignalMethod { get; set; }

        [Parameter("The number of signals allowed from a Zone", DefaultValue = 3, MinValue = 1, Group = "Signal")]
        public int SignalsAllowedPerZone { get; set; }

        [Parameter("Signal after Hunts/Sweeps", DefaultValue = false, Group = "Signal")]
        public bool SignalAfterHunts { get; set; }

        [Parameter("How Many Hunts/Sweeps?", DefaultValue = 2, MinValue = 1, Group = "Signal")]
        public int RequiredHunts { get; set; }

        [Parameter("Show All Long Setup", DefaultValue = false, Group = "Show or Hide")]
        public bool ShowAllLongSetups { get; set; }

        [Parameter("Show All Short Setup", DefaultValue = false, Group = "Show or Hide")]
        public bool ShowAllShortSetups { get; set; }

        [Parameter("Alert", DefaultValue = "On", Group = "Alert")]
        public string AlertSetting { get; set; }

        [Parameter("Alert Name", DefaultValue = "ICT Setup 01 Alerts [TradingFinder]", Group = "Alert")]
        public string AlertName { get; set; }

        [Parameter("Message Frequency", DefaultValue = "Once Per Bar", Group = "Alert")]
        public string Frequency { get; set; }

        [Parameter("Show Alert time by Time Zone", DefaultValue = "UTC", Group = "Alert")]
        public string AlertTimeZone { get; set; }

        [Parameter("Long Position Message", DefaultValue = "Long Signal Position Based on ICT Setup 01 [FVG Hunts]", Group = "Alert")]
        public string LongPositionMessage { get; set; }

        [Parameter("Short Position Message", DefaultValue = "Short Signal Position Based on ICT Setup 01 [FVG Hunts]", Group = "Alert")]
        public string ShortPositionMessage { get; set; }

        private const int AtrLength = 55;
        private const string LongSignalColorHex = "#008304";
        private const string ShortSignalColorHex = "#d30000";
        private const int DiscountPremiumDrawOffsetStart = 12;
        private const int DiscountPremiumDrawOffsetLabel = 16;
        private const int DiscountPremiumDrawOffsetEnd = 20;

        private IndicatorDataSeries _trueRange;
        private IndicatorDataSeries _averageTrueRange;

        private IndicatorDataSeries _bullishDistalLevel;
        private IndicatorDataSeries _bullishProximalLevel;
        private IndicatorDataSeries _bearishDistalLevel;
        private IndicatorDataSeries _bearishProximalLevel;
        private IndicatorDataSeries _bullishFvgPointSeries;
        private IndicatorDataSeries _bearishFvgPointSeries;

        private double _bullishFvgDistal;
        private double _bullishFvgProximal;
        private int _bullishFvgPoint;
        private double _bullishPremium;
        private double _bullishDiscount;
        private double _bullishEquilibrium;

        private double _bearishFvgDistal;
        private double _bearishFvgProximal;
        private int _bearishFvgPoint;
        private double _bearishPremium;
        private double _bearishDiscount;
        private double _bearishEquilibrium;

        private bool _isBullishFvgValid = true;
        private bool _isBearishFvgValid = true;

        private double _lowTracker;
        private double _highTracker;

        private int _longSignalCount;
        private int _shortSignalCount;

        private bool _isLongSignal;
        private bool _isShortSignal;

        private bool _isBullishFvg;
        private bool _isBearishFvg;

        private int _lastLongAlertBar = -1;
        private int _lastShortAlertBar = -1;
        private DateTime _lastLiveBarOpenTime = DateTime.MinValue;
        private bool _lastLiveLongSignal;
        private bool _lastLiveShortSignal;


        protected override void Initialize()
        {
            _trueRange = CreateDataSeries();
            _averageTrueRange = CreateDataSeries();

            _bullishDistalLevel = CreateDataSeries();
            _bullishProximalLevel = CreateDataSeries();
            _bearishDistalLevel = CreateDataSeries();
            _bearishProximalLevel = CreateDataSeries();
            _bullishFvgPointSeries = CreateDataSeries();
            _bearishFvgPointSeries = CreateDataSeries();
        }

        public override void Calculate(int index)
        {
            CalculateAtr(index);

            if (index == 0)
            {
                _bullishDistalLevel[index] = 0.0;
                _bullishProximalLevel[index] = 0.0;
                _bearishDistalLevel[index] = 0.0;
                _bearishProximalLevel[index] = 0.0;
                _bullishFvgPointSeries[index] = 0.0;
                _bearishFvgPointSeries[index] = 0.0;
                return;
            }

            _bullishDistalLevel[index] = _bullishDistalLevel[index - 1];
            _bullishProximalLevel[index] = _bullishProximalLevel[index - 1];
            _bearishDistalLevel[index] = _bearishDistalLevel[index - 1];
            _bearishProximalLevel[index] = _bearishProximalLevel[index - 1];

            var high = Bars.HighPrices[index];
            var low = Bars.LowPrices[index];
            var close = Bars.ClosePrices[index];

            _isBullishFvg = false;
            _isBearishFvg = false;

            if (index >= 2)
            {
                var high2 = Bars.HighPrices[index - 2];
                var low2 = Bars.LowPrices[index - 2];
                var high1 = Bars.HighPrices[index - 1];
                var low1 = Bars.LowPrices[index - 1];
                var atrValue = _averageTrueRange[index];
                if (double.IsNaN(atrValue))
                    atrValue = 0.0;

                if ((high - low2) > (FvgDetectorMultiplier * atrValue))
                {
                    if (low > high2 && low2 < low1 && high1 < high && (high + low2) / 2.0 >= high2)
                    {
                        _bullishFvgDistal = high2;
                        _bullishFvgProximal = low;
                        _bullishFvgPoint = index;
                        _bullishDiscount = low2;
                        _bullishPremium = high;
                        _bullishEquilibrium = (high + low2) / 2.0;
                        _isBullishFvg = true;
                    }
                }

                if ((high2 - low) > (FvgDetectorMultiplier * atrValue))
                {
                    if (low2 > high && high2 > high1 && low1 > low && (low + high2) / 2.0 <= low2)
                    {
                        _bearishFvgDistal = low2;
                        _bearishFvgProximal = high;
                        _bearishFvgPoint = index;
                        _bearishDiscount = low;
                        _bearishPremium = high2;
                        _bearishEquilibrium = (low + high2) / 2.0;
                        _isBearishFvg = true;
                    }
                }
            }

            if (UseDiscountAndPremium)
            {
                if (_isBullishFvg)
                {
                    _bullishDistalLevel[index] = _bullishFvgDistal;
                    _bullishProximalLevel[index] = _bullishEquilibrium >= _bullishFvgProximal ? _bullishFvgProximal : _bullishEquilibrium;
                }

                if (_isBearishFvg)
                {
                    _bearishDistalLevel[index] = _bearishFvgDistal;
                    _bearishProximalLevel[index] = _bearishEquilibrium <= _bearishFvgProximal ? _bearishFvgProximal : _bearishEquilibrium;
                }
            }
            else
            {
                if (_isBullishFvg)
                {
                    _bullishDistalLevel[index] = _bullishFvgDistal;
                    _bullishProximalLevel[index] = _bullishFvgProximal;
                }

                if (_isBearishFvg)
                {
                    _bearishDistalLevel[index] = _bearishFvgDistal;
                    _bearishProximalLevel[index] = _bearishFvgProximal;
                }
            }

            var body1 = Bars.ClosePrices[index - 1] - Bars.OpenPrices[index - 1];
            var prevDistalBu = _bullishDistalLevel[index - 1];
            var prevProximalBu = _bullishProximalLevel[index - 1];
            var prevDistalBe = _bearishDistalLevel[index - 1];
            var prevProximalBe = _bearishProximalLevel[index - 1];

            if (_isBullishFvgValid)
            {
                var bullProximal = _bullishProximalLevel[index];
                _isBullishFvgValid = UpdateZoneValidity(index, body1, true, _bullishFvgPoint, prevDistalBu, prevProximalBu, _longSignalCount, ref bullProximal);
                _bullishProximalLevel[index] = bullProximal;
            }

            if (_isBearishFvgValid)
            {
                var bearProximal = _bearishProximalLevel[index];
                _isBearishFvgValid = UpdateZoneValidity(index, body1, false, _bearishFvgPoint, prevDistalBe, prevProximalBe, _shortSignalCount, ref bearProximal);
                _bearishProximalLevel[index] = bearProximal;
            }

            if (_bullishFvgPointSeries[index - 1] != _bullishFvgPoint)
            {
                _isBullishFvgValid = true;
                _lowTracker = 0.0;
                _longSignalCount = 0;
                _isLongSignal = false;
            }

            if (_bearishFvgPointSeries[index - 1] != _bearishFvgPoint)
            {
                _isBearishFvgValid = true;
                _highTracker = 0.0;
                _shortSignalCount = 0;
                _isShortSignal = false;
            }

            if (_isBullishFvgValid)
            {
                if (_lowTracker == 0.0 && low < _bullishProximalLevel[index])
                    _lowTracker = low;

                if (low < _lowTracker && _lowTracker > 0.0)
                {
                    _lowTracker = low;
                    if (close >= _bullishProximalLevel[index])
                    {
                        _longSignalCount += 1;
                        if (SignalAfterHunts)
                            _isLongSignal = _longSignalCount == RequiredHunts;
                        else
                            _isLongSignal = true;
                    }
                    else
                    {
                        _isLongSignal = false;
                    }
                }
                else
                {
                    _isLongSignal = false;
                }
            }
            else
            {
                _lowTracker = 0.0;
                _longSignalCount = 0;
                _isLongSignal = false;
            }

            if (_isBearishFvgValid)
            {
                if (_highTracker == 0.0 && high > _bearishProximalLevel[index])
                    _highTracker = high;

                if (high > _highTracker && _highTracker > 0.0)
                {
                    _highTracker = high;
                    if (close <= _bearishProximalLevel[index])
                    {
                        _shortSignalCount += 1;
                        if (SignalAfterHunts)
                            _isShortSignal = _shortSignalCount == RequiredHunts;
                        else
                            _isShortSignal = true;
                    }
                    else
                    {
                        _isShortSignal = false;
                    }
                }
                else
                {
                    _isShortSignal = false;
                }
            }
            else
            {
                _highTracker = 0.0;
                _shortSignalCount = 0;
                _isShortSignal = false;
            }

            _bullishFvgPointSeries[index] = _bullishFvgPoint;
            _bearishFvgPointSeries[index] = _bearishFvgPoint;

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
                _trueRange[index] = high - low;
                _averageTrueRange[index] = double.NaN;
                return;
            }

            var prevClose = Bars.ClosePrices[index - 1];
            var tr1 = high - low;
            var tr2 = Math.Abs(high - prevClose);
            var tr3 = Math.Abs(low - prevClose);
            _trueRange[index] = Math.Max(tr1, Math.Max(tr2, tr3));

            if (index < AtrLength - 1)
            {
                _averageTrueRange[index] = double.NaN;
                return;
            }

            if (index == AtrLength - 1)
            {
                var sum = 0.0;
                for (var i = 0; i < AtrLength; i++)
                    sum += _trueRange[i];

                _averageTrueRange[index] = sum / AtrLength;
                return;
            }

            _averageTrueRange[index] = ((_averageTrueRange[index - 1] * (AtrLength - 1)) + _trueRange[index]) / AtrLength;
        }

        private void DrawSignals(int index)
        {
            var longSignalId = $"long_signal_{index}";
            var shortSignalId = $"short_signal_{index}";

            if (_isLongSignal)
                Chart.DrawIcon(longSignalId, ChartIconType.UpTriangle, index, Bars.LowPrices[index], Color.FromHex(LongSignalColorHex));

            if (_isShortSignal)
                Chart.DrawIcon(shortSignalId, ChartIconType.DownTriangle, index, Bars.HighPrices[index], Color.FromHex(ShortSignalColorHex));
        }

        private bool UpdateZoneValidity(int index, double body1, bool isBull, int zonePoint, double prevDistal, double prevProximal, int signalCount, ref double updatedProximal)
        {
            var useOpenForBodyDirection = isBull ? body1 > 0 : body1 <= 0;
            var selectedPrice = useOpenForBodyDirection ? Bars.OpenPrices[index - 1] : Bars.ClosePrices[index - 1];
            var sweepPrice = isBull
                ? (SignalMethod == "Sweeps" ? selectedPrice : Bars.LowPrices[index - 1])
                : (SignalMethod == "Sweeps" ? selectedPrice : Bars.HighPrices[index - 1]);

            var sweepCheck = isBull ? sweepPrice < prevDistal : sweepPrice > prevDistal;
            var expired = index > zonePoint + FvgValidityPeriod;
            var signalLimitExceeded = !SignalAfterHunts && signalCount > SignalsAllowedPerZone - 1;

            if (sweepCheck || expired || signalLimitExceeded)
                return false;

            var movedInsideZone = isBull
                ? selectedPrice < prevProximal && selectedPrice > prevDistal
                : selectedPrice > prevProximal && selectedPrice < prevDistal;

            if (movedInsideZone)
                updatedProximal = selectedPrice;

            return true;
        }

        private void DrawCurrentZones(int index)
        {
            if (_isBullishFvgValid && _bullishFvgPoint > 0 && index <= _bullishFvgPoint + FvgValidityPeriod)
            {
                var bullDistalId = $"bull_distal_{_bullishFvgPoint}";
                var bullProximalId = $"bull_proximal_{_bullishFvgPoint}";
                var bullFillId = $"bull_fill_{_bullishFvgPoint}";
                var bullLabelId = $"bull_lbl_{_bullishFvgPoint}";

                Chart.DrawTrendLine(bullDistalId, _bullishFvgPoint, _bullishDistalLevel[index], index, _bullishDistalLevel[index], Color.FromArgb(186, 8, 8, 8), 1, LineStyle.LinesDots);
                Chart.DrawTrendLine(bullProximalId, _bullishFvgPoint, _bullishProximalLevel[index], index, _bullishProximalLevel[index], Color.FromArgb(186, 8, 8, 8), 1, LineStyle.LinesDots);
                var bullFill = Chart.DrawRectangle(bullFillId, _bullishFvgPoint, _bullishDistalLevel[index], index, _bullishProximalLevel[index], Color.FromArgb(77, 76, 175, 79));
                bullFill.IsFilled = true;
                bullFill.IsInteractive = false;
                Chart.DrawText(bullLabelId, "FVG", _bullishFvgPoint + 1, _bullishDistalLevel[index], Color.Black);
            }

            if (_isBullishFvg)
            {
                DrawBullDiscountPremium(_bullishFvgPoint);
                if (!ShowAllLongSetups)
                {
                    var previousPoint = (int)_bullishFvgPointSeries[index - 1];
                    if (previousPoint > 0 && previousPoint != _bullishFvgPoint)
                    {
                        Chart.RemoveObject($"bull_distal_{previousPoint}");
                        Chart.RemoveObject($"bull_proximal_{previousPoint}");
                        Chart.RemoveObject($"bull_fill_{previousPoint}");
                        Chart.RemoveObject($"bull_lbl_{previousPoint}");
                        RemoveBullDiscountPremium(previousPoint);
                    }
                }
            }

            if (_isBearishFvgValid && _bearishFvgPoint > 0 && index <= _bearishFvgPoint + FvgValidityPeriod)
            {
                var bearDistalId = $"bear_distal_{_bearishFvgPoint}";
                var bearProximalId = $"bear_proximal_{_bearishFvgPoint}";
                var bearFillId = $"bear_fill_{_bearishFvgPoint}";
                var bearLabelId = $"bear_lbl_{_bearishFvgPoint}";

                Chart.DrawTrendLine(bearDistalId, _bearishFvgPoint, _bearishDistalLevel[index], index, _bearishDistalLevel[index], Color.FromArgb(186, 8, 8, 8), 1, LineStyle.LinesDots);
                Chart.DrawTrendLine(bearProximalId, _bearishFvgPoint, _bearishProximalLevel[index], index, _bearishProximalLevel[index], Color.FromArgb(186, 8, 8, 8), 1, LineStyle.LinesDots);
                var bearFill = Chart.DrawRectangle(bearFillId, _bearishFvgPoint, _bearishDistalLevel[index], index, _bearishProximalLevel[index], Color.FromArgb(77, 255, 49, 49));
                bearFill.IsFilled = true;
                bearFill.IsInteractive = false;
                Chart.DrawText(bearLabelId, "FVG", _bearishFvgPoint + 1, _bearishDistalLevel[index], Color.Black);
            }

            if (_isBearishFvg)
            {
                DrawBearDiscountPremium(_bearishFvgPoint);
                if (!ShowAllShortSetups)
                {
                    var previousPoint = (int)_bearishFvgPointSeries[index - 1];
                    if (previousPoint > 0 && previousPoint != _bearishFvgPoint)
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
            if (!UseDiscountAndPremium)
                return;

            var discountLineId = $"bull_discount_line_{point}";
            var premiumLineId = $"bull_premium_line_{point}";
            var equilibriumLineId = $"bull_equilibrium_line_{point}";
            var disRectId = $"bull_discount_fill_{point}";
            var premRectId = $"bull_premium_fill_{point}";
            var disLabelId = $"bull_discount_lbl_{point}";
            var premLabelId = $"bull_premium_lbl_{point}";
            var equiLabelId = $"bull_equilibrium_lbl_{point}";

            Chart.DrawTrendLine(discountLineId, point + DiscountPremiumDrawOffsetStart, _bullishDiscount, point + DiscountPremiumDrawOffsetEnd, _bullishDiscount, Color.Transparent);
            Chart.DrawTrendLine(premiumLineId, point + DiscountPremiumDrawOffsetStart, _bullishPremium, point + DiscountPremiumDrawOffsetEnd, _bullishPremium, Color.Transparent);
            Chart.DrawTrendLine(equilibriumLineId, point + DiscountPremiumDrawOffsetStart, _bullishEquilibrium, point + DiscountPremiumDrawOffsetEnd, _bullishEquilibrium, Color.Transparent);

            var disRect = Chart.DrawRectangle(disRectId, point + DiscountPremiumDrawOffsetStart, _bullishDiscount, point + DiscountPremiumDrawOffsetEnd, _bullishEquilibrium, Color.FromArgb(110, 211, 238, 255));
            disRect.IsFilled = true;
            disRect.IsInteractive = false;

            var premRect = Chart.DrawRectangle(premRectId, point + DiscountPremiumDrawOffsetStart, _bullishPremium, point + DiscountPremiumDrawOffsetEnd, _bullishEquilibrium, Color.FromArgb(75, 255, 165, 100));
            premRect.IsFilled = true;
            premRect.IsInteractive = false;

            Chart.DrawText(disLabelId, "Discount", point + DiscountPremiumDrawOffsetLabel, _bullishDiscount, Color.Black);
            Chart.DrawText(premLabelId, "Premium", point + DiscountPremiumDrawOffsetLabel, _bullishPremium, Color.Black);
            Chart.DrawText(equiLabelId, "EQU", point + DiscountPremiumDrawOffsetLabel, _bullishEquilibrium, Color.Black);
        }

        private void DrawBearDiscountPremium(int point)
        {
            if (!UseDiscountAndPremium)
                return;

            var premiumLineId = $"bear_premium_line_{point}";
            var discountLineId = $"bear_discount_line_{point}";
            var equilibriumLineId = $"bear_equilibrium_line_{point}";
            var disRectId = $"bear_discount_fill_{point}";
            var premRectId = $"bear_premium_fill_{point}";
            var disLabelId = $"bear_discount_lbl_{point}";
            var premLabelId = $"bear_premium_lbl_{point}";
            var equiLabelId = $"bear_equilibrium_lbl_{point}";

            Chart.DrawTrendLine(premiumLineId, point + DiscountPremiumDrawOffsetStart, _bearishPremium, point + DiscountPremiumDrawOffsetEnd, _bearishPremium, Color.Transparent);
            Chart.DrawTrendLine(discountLineId, point + DiscountPremiumDrawOffsetStart, _bearishDiscount, point + DiscountPremiumDrawOffsetEnd, _bearishDiscount, Color.Transparent);
            Chart.DrawTrendLine(equilibriumLineId, point + DiscountPremiumDrawOffsetStart, _bearishEquilibrium, point + DiscountPremiumDrawOffsetEnd, _bearishEquilibrium, Color.Transparent);

            var disRect = Chart.DrawRectangle(disRectId, point + DiscountPremiumDrawOffsetStart, _bearishDiscount, point + DiscountPremiumDrawOffsetEnd, _bearishEquilibrium, Color.FromArgb(110, 211, 238, 255));
            disRect.IsFilled = true;
            disRect.IsInteractive = false;

            var premRect = Chart.DrawRectangle(premRectId, point + DiscountPremiumDrawOffsetStart, _bearishPremium, point + DiscountPremiumDrawOffsetEnd, _bearishEquilibrium, Color.FromArgb(75, 255, 165, 100));
            premRect.IsFilled = true;
            premRect.IsInteractive = false;

            Chart.DrawText(disLabelId, "Discount", point + DiscountPremiumDrawOffsetLabel, _bearishDiscount, Color.Black);
            Chart.DrawText(premLabelId, "Premium", point + DiscountPremiumDrawOffsetLabel, _bearishPremium, Color.Black);
            Chart.DrawText(equiLabelId, "EQU", point + DiscountPremiumDrawOffsetLabel, _bearishEquilibrium, Color.Black);
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
            if (!string.Equals(AlertSetting, "On", StringComparison.OrdinalIgnoreCase))
                return;

            if (index != Bars.Count - 1)
                return;

            var frequencyMode = Frequency ?? "Once Per Bar";
            var isNewLiveBar = _lastLiveBarOpenTime != DateTime.MinValue && Bars.OpenTimes[index] != _lastLiveBarOpenTime;

            if (string.Equals(frequencyMode, "Per Bar Close", StringComparison.OrdinalIgnoreCase))
            {
                if (isNewLiveBar)
                {
                    var closedBarIndex = index - 1;
                    if (_lastLiveLongSignal && _lastLongAlertBar != closedBarIndex)
                    {
                        Print("{0} | LONG | Bar={1} | TZ={2} | {3}", AlertName, closedBarIndex, AlertTimeZone, LongPositionMessage);
                        _lastLongAlertBar = closedBarIndex;
                    }

                    if (_lastLiveShortSignal && _lastShortAlertBar != closedBarIndex)
                    {
                        Print("{0} | SHORT | Bar={1} | TZ={2} | {3}", AlertName, closedBarIndex, AlertTimeZone, ShortPositionMessage);
                        _lastShortAlertBar = closedBarIndex;
                    }
                }

                _lastLiveLongSignal = _isLongSignal;
                _lastLiveShortSignal = _isShortSignal;
                _lastLiveBarOpenTime = Bars.OpenTimes[index];
                return;
            }

            if (_isLongSignal)
            {
                if (string.Equals(frequencyMode, "All", StringComparison.OrdinalIgnoreCase) || _lastLongAlertBar != index)
                {
                    Print("{0} | LONG | Bar={1} | TZ={2} | {3}", AlertName, index, AlertTimeZone, LongPositionMessage);
                    if (!string.Equals(frequencyMode, "All", StringComparison.OrdinalIgnoreCase))
                        _lastLongAlertBar = index;
                }
            }

            if (_isShortSignal)
            {
                if (string.Equals(frequencyMode, "All", StringComparison.OrdinalIgnoreCase) || _lastShortAlertBar != index)
                {
                    Print("{0} | SHORT | Bar={1} | TZ={2} | {3}", AlertName, index, AlertTimeZone, ShortPositionMessage);
                    if (!string.Equals(frequencyMode, "All", StringComparison.OrdinalIgnoreCase))
                        _lastShortAlertBar = index;
                }
            }

            _lastLiveLongSignal = _isLongSignal;
            _lastLiveShortSignal = _isShortSignal;
            _lastLiveBarOpenTime = Bars.OpenTimes[index];
        }
    }
}
