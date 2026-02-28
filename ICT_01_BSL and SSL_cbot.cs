using System;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class ICT_01_BSL_and_SSL_cbot : Robot
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

        [Parameter("Pivot Left", DefaultValue = 5, MinValue = 1, Group = "BSL/SSL")]
        public int PivotLeft { get; set; }

        [Parameter("Pivot Right", DefaultValue = 5, MinValue = 1, Group = "BSL/SSL")]
        public int PivotRight { get; set; }

        [Parameter("Risk Per Trade (%)", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 10.0, Group = "Risk Management")]
        public double RiskPercent { get; set; }

        [Parameter("Min SL Distance (pips)", DefaultValue = 3.0, MinValue = 0.5, Group = "Risk Management")]
        public double MinSlPips { get; set; }

        [Parameter("Max SL Distance (pips)", DefaultValue = 500.0, MinValue = 10.0, Group = "Risk Management")]
        public double MaxSlPips { get; set; }

        [Parameter("Liquidity Lookback Bars", DefaultValue = 300, MinValue = 1, Group = "BSL/SSL")]
        public int LiquidityLookbackBars { get; set; }

        private const string BotLabel = "ICT01_BSL_SSL";

        private ICT_01 _ict;
        private BSL_SSL _bslSsl;

        private int _lastLongSignalBar = -1;
        private int _lastShortSignalBar = -1;

        protected override void OnStart()
        {
            _ict = Indicators.GetIndicator<ICT_01>(
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

            _bslSsl = Indicators.GetIndicator<BSL_SSL>(
                PivotLeft,
                PivotRight,
                1,
                BSL_SSL.LiquidityLineStyle.Dots,
                "Teal",
                "Red",
                2.0);
        }

        protected override void OnBar()
        {
            int signalBar = Bars.Count - 2;
            if (signalBar < 2)
                return;

            bool isLongSignal = IsSignal(_ict.LongSignal[signalBar]);
            bool isShortSignal = IsSignal(_ict.ShortSignal[signalBar]);

            if (isLongSignal && _lastLongSignalBar != signalBar)
            {
                _lastLongSignalBar = signalBar;
                ClosePositionsByType(TradeType.Sell);
                TryOpenLong(signalBar);
            }
            else if (isShortSignal && _lastShortSignalBar != signalBar)
            {
                _lastShortSignalBar = signalBar;
                ClosePositionsByType(TradeType.Buy);
                TryOpenShort(signalBar);
            }
        }

        private bool IsSignal(double value)
        {
            return !double.IsNaN(value) && value != 0.0;
        }

        private void ClosePositionsByType(TradeType type)
        {
            foreach (var position in Positions.FindAll(BotLabel, SymbolName, type))
                ClosePosition(position);
        }

        private void TryOpenLong(int signalBar)
        {
            double entry = Symbol.Ask;
            double sslLevel = GetLatestLiquidity(_bslSsl.CurrentSSL, signalBar);

            if (double.IsNaN(sslLevel) || sslLevel >= entry)
                return;

            double slPips = (entry - sslLevel) / Symbol.PipSize;
            if (!IsValidSl(slPips))
                return;

            double volume = CalculateVolume(slPips);
            if (volume <= 0)
                return;

            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, BotLabel, slPips, null);
        }

        private void TryOpenShort(int signalBar)
        {
            double entry = Symbol.Bid;
            double bslLevel = GetLatestLiquidity(_bslSsl.CurrentBSL, signalBar);

            if (double.IsNaN(bslLevel) || bslLevel <= entry)
                return;

            double slPips = (bslLevel - entry) / Symbol.PipSize;
            if (!IsValidSl(slPips))
                return;

            double volume = CalculateVolume(slPips);
            if (volume <= 0)
                return;

            ExecuteMarketOrder(TradeType.Sell, SymbolName, volume, BotLabel, slPips, null);
        }

        private bool IsValidSl(double slPips)
        {
            return slPips >= MinSlPips && slPips <= MaxSlPips;
        }

        private double GetLatestLiquidity(IndicatorDataSeries series, int signalBar)
        {
            int start = Math.Max(0, signalBar - LiquidityLookbackBars);
            for (int i = signalBar; i >= start; i--)
            {
                double level = series[i];
                if (!double.IsNaN(level) && level > 0)
                    return level;
            }

            return double.NaN;
        }

        private double CalculateVolume(double slPips)
        {
            if (slPips <= 0)
                return 0;

            double riskAmount = Account.Equity * (RiskPercent / 100.0);
            double rawVolume = Symbol.VolumeForFixedRisk(riskAmount, slPips);
            double volume = Symbol.NormalizeVolumeInUnits(rawVolume, RoundingMode.Down);

            if (volume < Symbol.VolumeInUnitsMin)
                return 0;

            if (volume > Symbol.VolumeInUnitsMax)
                volume = Symbol.VolumeInUnitsMax;

            return volume;
        }
    }
}
