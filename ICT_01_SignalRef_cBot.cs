using System;
using cAlgo.API;
using cAlgo.API.Indicators;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class ICT_01_SignalRef_cBot : Robot
    {
        [Parameter("FVG Detector Multiplier", DefaultValue = 1.0, MinValue = 1.0, Group = "ICT Setup 01")]
        public double FvgDetectorMultiplier { get; set; }

        [Parameter("FVG Validity Period (bars)", DefaultValue = 15, MinValue = 2, Group = "ICT Setup 01")]
        public int FvgValidityPeriod { get; set; }

        [Parameter("Use Discount & Premium Zone", DefaultValue = false, Group = "ICT Setup 01")]
        public bool UseDiscountAndPremium { get; set; }

        [Parameter("Signal Method (Hunt / Sweeps)", DefaultValue = "Hunt", Group = "ICT Setup 01")]
        public string SignalMethod { get; set; }

        [Parameter("Signals Allowed Per Zone", DefaultValue = 3, MinValue = 1, Group = "ICT Setup 01")]
        public int SignalsAllowedPerZone { get; set; }

        [Parameter("Signal After Hunts", DefaultValue = false, Group = "ICT Setup 01")]
        public bool SignalAfterHunts { get; set; }

        [Parameter("Required Hunts Count", DefaultValue = 2, MinValue = 1, Group = "ICT Setup 01")]
        public int RequiredHunts { get; set; }

        [Parameter("Risk Per Trade (%)", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 10.0, Group = "Risk Management")]
        public double RiskPercent { get; set; }

        [Parameter("ATR Period (SL sizing)", DefaultValue = 14, MinValue = 5, Group = "Risk Management")]
        public int AtrPeriod { get; set; }

        [Parameter("ATR Multiplier for SL", DefaultValue = 2.0, MinValue = 0.5, MaxValue = 10.0, Group = "Risk Management")]
        public double AtrMultiplier { get; set; }

        [Parameter("Min SL Distance (pips)", DefaultValue = 5.0, MinValue = 1.0, Group = "Risk Management")]
        public double MinSlPips { get; set; }

        [Parameter("Max SL Distance (pips)", DefaultValue = 500.0, MinValue = 10.0, Group = "Risk Management")]
        public double MaxSlPips { get; set; }

        private const string BotLabel = "ICT01_Ref";

        private ICT_01 _ictIndicator;
        private AverageTrueRange _atr;

        private int _lastLongBar = -1;
        private int _lastShortBar = -1;

        protected override void OnStart()
        {
            _ictIndicator = Indicators.GetIndicator<ICT_01>(
                FvgDetectorMultiplier,
                FvgValidityPeriod,
                UseDiscountAndPremium,
                SignalMethod,
                SignalsAllowedPerZone,
                SignalAfterHunts,
                RequiredHunts,
                false,
                false,
                "Off",
                "ICT Setup 01 Alerts [TradingFinder]",
                "Once Per Bar",
                "UTC",
                "Long Signal Position Based on ICT Setup 01 [FVG Hunts]",
                "Short Signal Position Based on ICT Setup 01 [FVG Hunts]");

            // cTrader enum member is WilderSmoothing (not Wilder).
            _atr = Indicators.AverageTrueRange(AtrPeriod, MovingAverageType.WilderSmoothing);
        }

        protected override void OnBar()
        {
            int signalBar = Bars.Count - 2;
            if (signalBar < 2)
                return;

            bool isLong = IsSignal(_ictIndicator.LongSignal[signalBar]);
            bool isShort = IsSignal(_ictIndicator.ShortSignal[signalBar]);

            if (isLong && _lastLongBar != signalBar)
            {
                _lastLongBar = signalBar;
                ClosePositionsByType(TradeType.Sell);
                OpenLong(signalBar);
            }
            else if (isShort && _lastShortBar != signalBar)
            {
                _lastShortBar = signalBar;
                ClosePositionsByType(TradeType.Buy);
                OpenShort(signalBar);
            }
        }

        private bool IsSignal(double value)
        {
            return !double.IsNaN(value) && value != 0.0;
        }

        private void ClosePositionsByType(TradeType tradeType)
        {
            foreach (var position in Positions.FindAll(BotLabel, SymbolName, tradeType))
                ClosePosition(position);
        }

        private void OpenLong(int signalBar)
        {
            double slPips = GetSlPips(signalBar);
            if (slPips <= 0)
                return;

            double volume = CalculateVolume(slPips);
            if (volume <= 0)
                return;

            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, BotLabel, slPips, null);
        }

        private void OpenShort(int signalBar)
        {
            double slPips = GetSlPips(signalBar);
            if (slPips <= 0)
                return;

            double volume = CalculateVolume(slPips);
            if (volume <= 0)
                return;

            ExecuteMarketOrder(TradeType.Sell, SymbolName, volume, BotLabel, slPips, null);
        }

        private double GetSlPips(int signalBar)
        {
            double atr = _atr.Result[signalBar];
            if (double.IsNaN(atr) || atr <= 0)
                return 0;

            double slPips = (atr * AtrMultiplier) / Symbol.PipSize;
            if (slPips < MinSlPips)
                slPips = MinSlPips;
            if (slPips > MaxSlPips)
                return 0;

            return slPips;
        }

        private double CalculateVolume(double slPips)
        {
            if (slPips <= 0)
                return 0;

            double riskAmount = Account.Equity * (RiskPercent / 100.0);
            double rawVolume = Symbol.VolumeForFixedRisk(riskAmount, slPips);
            double normalized = Symbol.NormalizeVolumeInUnits(rawVolume, RoundingMode.Down);

            if (normalized < Symbol.VolumeInUnitsMin)
                return 0;

            if (normalized > Symbol.VolumeInUnitsMax)
                normalized = Symbol.VolumeInUnitsMax;

            return normalized;
        }
    }
}
