using System;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class ICT_01_cBot : Robot
    {
        [Parameter("FVG Detector Multiplier", DefaultValue = 1.0, MinValue = 1.0, Group = "ICT_01 Signal Settings")]
        public double FvgDetectorMultiplier { get; set; }

        [Parameter("FVG Validity Period (bars)", DefaultValue = 15, MinValue = 2, Group = "ICT_01 Signal Settings")]
        public int FvgValidityPeriod { get; set; }

        [Parameter("Use Discount/Premium Zone", DefaultValue = false, Group = "ICT_01 Signal Settings")]
        public bool UseDiscountAndPremium { get; set; }

        [Parameter("Signal Method (Hunt / Sweeps)", DefaultValue = "Hunt", Group = "ICT_01 Signal Settings")]
        public string SignalMethod { get; set; }

        [Parameter("Max Signals Per Zone", DefaultValue = 3, MinValue = 1, Group = "ICT_01 Signal Settings")]
        public int SignalsAllowedPerZone { get; set; }

        [Parameter("Signal After Hunts/Sweeps", DefaultValue = false, Group = "ICT_01 Signal Settings")]
        public bool SignalAfterHunts { get; set; }

        [Parameter("Required Hunts/Sweeps Count", DefaultValue = 2, MinValue = 1, Group = "ICT_01 Signal Settings")]
        public int RequiredHunts { get; set; }

        [Parameter("Pivot Left", DefaultValue = 5, MinValue = 1, Group = "BSL/SSL Liquidity Settings")]
        public int PivotLeft { get; set; }

        [Parameter("Pivot Right", DefaultValue = 5, MinValue = 1, Group = "BSL/SSL Liquidity Settings")]
        public int PivotRight { get; set; }

        [Parameter("Risk Per Trade (%)", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 10.0, Group = "Risk Management")]
        public double RiskPercent { get; set; }

        [Parameter("Reward : Risk Ratio", DefaultValue = 2.0, MinValue = 1.0, Group = "Risk Management")]
        public double RewardRiskRatio { get; set; }

        [Parameter("Max Open Positions", DefaultValue = 3, MinValue = 1, MaxValue = 10, Group = "Risk Management")]
        public int MaxOpenPositions { get; set; }

        [Parameter("Min SL Distance (pips)", DefaultValue = 3.0, MinValue = 0.5, Group = "Risk Management")]
        public double MinSlPips { get; set; }

        [Parameter("Max SL Distance (pips)", DefaultValue = 500.0, MinValue = 10.0, Group = "Risk Management")]
        public double MaxSlPips { get; set; }

        [Parameter("Liquidity Lookback Bars", DefaultValue = 300, MinValue = 1, Group = "BSL/SSL Liquidity Settings")]
        public int LiquidityLookbackBars { get; set; }

        private const string BotLabel = "ICT_01_cBot";

        private ICT_01 _ict01;
        private BSL_SSL _bslSsl;

        private int _lastLongEntryBar = -1;
        private int _lastShortEntryBar = -1;

        protected override void OnStart()
        {
            _ict01 = Indicators.GetIndicator<ICT_01>(
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

            Print("{0} started. Risk={1}% | RR={2} | MaxPos={3}",
                BotLabel, RiskPercent, RewardRiskRatio, MaxOpenPositions);
        }

        protected override void OnBar()
        {
            int signalBar = Bars.Count - 2;
            if (signalBar < 2)
                return;

            bool longSignal = IsSignal(_ict01.LongSignal[signalBar]);
            bool shortSignal = IsSignal(_ict01.ShortSignal[signalBar]);

            if (longSignal && _lastLongEntryBar != signalBar)
            {
                _lastLongEntryBar = signalBar;
                TryEnterLong(signalBar);
            }

            if (shortSignal && _lastShortEntryBar != signalBar)
            {
                _lastShortEntryBar = signalBar;
                TryEnterShort(signalBar);
            }
        }

        private bool IsSignal(double value)
        {
            return !double.IsNaN(value) && value != 0.0;
        }

        private void TryEnterLong(int signalBar)
        {
            if (Positions.FindAll(BotLabel, SymbolName).Length >= MaxOpenPositions)
                return;

            double entry = Symbol.Ask;
            double ssl = GetLatestLiquidity(_bslSsl.CurrentSSL, signalBar);

            if (double.IsNaN(ssl) || ssl >= entry)
                return;

            double slPips = (entry - ssl) / Symbol.PipSize;
            if (!SlPipsInRange(slPips))
                return;

            double tpPips = slPips * RewardRiskRatio;
            double volume = CalculateVolume(slPips);
            if (volume <= 0)
                return;

            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, BotLabel, slPips, tpPips);
        }

        private void TryEnterShort(int signalBar)
        {
            if (Positions.FindAll(BotLabel, SymbolName).Length >= MaxOpenPositions)
                return;

            double entry = Symbol.Bid;
            double bsl = GetLatestLiquidity(_bslSsl.CurrentBSL, signalBar);

            if (double.IsNaN(bsl) || bsl <= entry)
                return;

            double slPips = (bsl - entry) / Symbol.PipSize;
            if (!SlPipsInRange(slPips))
                return;

            double tpPips = slPips * RewardRiskRatio;
            double volume = CalculateVolume(slPips);
            if (volume <= 0)
                return;

            ExecuteMarketOrder(TradeType.Sell, SymbolName, volume, BotLabel, slPips, tpPips);
        }

        private double GetLatestLiquidity(IndicatorDataSeries series, int signalBar)
        {
            int start = Math.Max(0, signalBar - LiquidityLookbackBars);
            for (int i = signalBar; i >= start; i--)
            {
                var level = series[i];
                if (!double.IsNaN(level) && level > 0)
                    return level;
            }

            return double.NaN;
        }

        private bool SlPipsInRange(double slPips)
        {
            return slPips >= MinSlPips && slPips <= MaxSlPips;
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
