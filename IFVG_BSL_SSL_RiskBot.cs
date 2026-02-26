using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Indicators;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class IFVGBslSslRiskBot : Robot
    {
        [Parameter("Risk % per trade", Group = "Risk", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 10.0)]
        public double RiskPercent { get; set; }

        [Parameter("Reward Multiple", Group = "Risk", DefaultValue = 2.0, MinValue = 1.0)]
        public double RewardMultiple { get; set; }

        [Parameter("Allow Multiple Positions", Group = "Execution", DefaultValue = true)]
        public bool AllowMultiplePositions { get; set; }

        [Parameter("Label", Group = "Execution", DefaultValue = "IFVG_BSL_SSL")]
        public string BotLabel { get; set; }

        [Parameter("FVG Search Lookback (Bars)", Group = "IFVG", DefaultValue = 15, MinValue = 1)]
        public int FvgGapBars { get; set; }

        [Parameter("Min FVG Size (Pips/Points)", Group = "IFVG", DefaultValue = 0.0)]
        public double MinFvgPips { get; set; }

        [Parameter("FVG Epsilon (Price Units)", Group = "IFVG", DefaultValue = 0.0)]
        public double FvgEpsilonPoints { get; set; }

        [Parameter("MA Period", Group = "IFVG", DefaultValue = 21, MinValue = 1)]
        public int MaPeriod { get; set; }

        [Parameter("MA Type", Group = "IFVG", DefaultValue = "EMA")]
        public string MaType { get; set; }

        [Parameter("Pivot Left", Group = "BSL/SSL", DefaultValue = 5, MinValue = 1)]
        public int PivotLeft { get; set; }

        [Parameter("Pivot Right", Group = "BSL/SSL", DefaultValue = 5, MinValue = 1)]
        public int PivotRight { get; set; }

        private sealed class Pivot
        {
            public double Price;
            public int BarIndex;
            public int Type; // 1 = high(BSL), -1 = low(SSL)
        }

        private sealed class ExternalLiquidity
        {
            public double Price;
            public int PivotIndex;
        }

        private readonly LinkedList<Pivot> _pivots = new LinkedList<Pivot>();
        private readonly LinkedList<ExternalLiquidity> _buysidePools = new LinkedList<ExternalLiquidity>();
        private readonly LinkedList<ExternalLiquidity> _sellsidePools = new LinkedList<ExternalLiquidity>();

        private const int MaxPivotsToKeep = 10;

        private IndicatorDataSeries _maSeries;
        private SimpleMovingAverage _sma;
        private ExponentialMovingAverage _ema;

        protected override void OnStart()
        {
            _maSeries = CreateDataSeries();
            _sma = Indicators.SimpleMovingAverage(Bars.ClosePrices, MaPeriod);
            _ema = Indicators.ExponentialMovingAverage(Bars.ClosePrices, MaPeriod);
        }

        protected override void OnBar()
        {
            // OnBar is fired when a new bar opens. To act right after a signal bar closes,
            // we must evaluate the just-closed bar at Bars.Count - 2.
            var signalIndex = Bars.Count - 2;
            if (signalIndex < Math.Max(PivotLeft + PivotRight + 1, 3))
                return;

            UpdateLiquidity(signalIndex);

            var maValue = CalculateMa(signalIndex);
            var signalDir = DetectIfvgSignal(signalIndex, maValue);

            if (signalDir == 1)
                TryEnterLong(signalIndex);
            else if (signalDir == -1)
                TryEnterShort(signalIndex);
        }

        private void UpdateLiquidity(int index)
        {
            DetectAndStoreConfirmedPivots(index);
            AddExternalLiquidityFromNewPivot(index);
            ClearMitigated(index);
        }

        private void DetectAndStoreConfirmedPivots(int currentIndex)
        {
            var pivotIndex = currentIndex - PivotRight;
            var leftStart = pivotIndex - PivotLeft;
            var rightEnd = pivotIndex + PivotRight;

            if (pivotIndex <= 0 || leftStart < 0 || rightEnd >= Bars.Count)
                return;

            var candidateHigh = Bars.HighPrices[pivotIndex];
            var candidateLow = Bars.LowPrices[pivotIndex];

            if (IsPivotHigh(candidateHigh, leftStart, rightEnd))
                UnshiftPivot(new Pivot { Price = candidateHigh, BarIndex = pivotIndex, Type = 1 });

            if (IsPivotLow(candidateLow, leftStart, rightEnd))
                UnshiftPivot(new Pivot { Price = candidateLow, BarIndex = pivotIndex, Type = -1 });
        }

        private bool IsPivotHigh(double candidate, int start, int end)
        {
            var max = double.MinValue;
            for (var i = start; i <= end; i++)
                if (Bars.HighPrices[i] > max)
                    max = Bars.HighPrices[i];

            return candidate == max;
        }

        private bool IsPivotLow(double candidate, int start, int end)
        {
            var min = double.MaxValue;
            for (var i = start; i <= end; i++)
                if (Bars.LowPrices[i] < min)
                    min = Bars.LowPrices[i];

            return candidate == min;
        }

        private void UnshiftPivot(Pivot pivot)
        {
            if (_pivots.First != null &&
                _pivots.First.Value.BarIndex == pivot.BarIndex &&
                _pivots.First.Value.Type == pivot.Type &&
                Math.Abs(_pivots.First.Value.Price - pivot.Price) < Symbol.PipSize * 0.1)
                return;

            _pivots.AddFirst(pivot);

            while (_pivots.Count > MaxPivotsToKeep)
                _pivots.RemoveLast();
        }

        private void AddExternalLiquidityFromNewPivot(int currentIndex)
        {
            var confirmedPivotIndex = currentIndex - PivotRight;

            foreach (var pivot in _pivots)
            {
                if (pivot.BarIndex != confirmedPivotIndex)
                    continue;

                if (pivot.Type == 1)
                    AddExternalLiquidity(_buysidePools, pivot);
                else if (pivot.Type == -1)
                    AddExternalLiquidity(_sellsidePools, pivot);
            }
        }

        private static void AddExternalLiquidity(LinkedList<ExternalLiquidity> poolList, Pivot pivot)
        {
            poolList.AddFirst(new ExternalLiquidity { Price = pivot.Price, PivotIndex = pivot.BarIndex });
        }

        private void ClearMitigated(int index)
        {
            var node = _sellsidePools.First;
            while (node != null)
            {
                var next = node.Next;
                if (Bars.LowPrices[index] <= node.Value.Price)
                    _sellsidePools.Remove(node);
                node = next;
            }

            node = _buysidePools.First;
            while (node != null)
            {
                var next = node.Next;
                if (Bars.HighPrices[index] >= node.Value.Price)
                    _buysidePools.Remove(node);
                node = next;
            }
        }

        private double CurrentBslPrice => _buysidePools.First != null ? _buysidePools.First.Value.Price : double.NaN;
        private double CurrentSslPrice => _sellsidePools.First != null ? _sellsidePools.First.Value.Price : double.NaN;

        private double CalculateMa(int index)
        {
            if (string.Equals(MaType, "SMA", StringComparison.OrdinalIgnoreCase))
                _maSeries[index] = _sma.Result[index];
            else
                _maSeries[index] = _ema.Result[index];

            return _maSeries[index];
        }

        private int DetectIfvgSignal(int index, double maValue)
        {
            var minSizeValue = MinFvgPips * Symbol.PipSize;

            for (var i = 1; i <= FvgGapBars; i++)
            {
                var fvgType = DetectFvg(index, i, FvgEpsilonPoints);
                if (fvgType == 0)
                    continue;

                int signalDir;
                if (TryProcessFvgCandidate(index, i, fvgType, minSizeValue, maValue, out signalDir))
                    return signalDir;
            }

            return 0;
        }

        private int DetectFvg(int currentIndex, int idx, double epsVal)
        {
            if (idx + 2 > currentIndex)
                return 0;

            var h2 = Bars.HighPrices[currentIndex - (idx + 2)];
            var l2 = Bars.LowPrices[currentIndex - (idx + 2)];
            var lt = Bars.LowPrices[currentIndex - idx];
            var ht = Bars.HighPrices[currentIndex - idx];

            if (lt > h2 - epsVal)
                return 1;

            if (ht < l2 + epsVal)
                return -1;

            return 0;
        }

        private bool TryProcessFvgCandidate(int index, int i, int fvgType, double minSizeValue, double maValue, out int signalDir)
        {
            signalDir = 0;

            var isBearishGap = fvgType == 1;
            var gapLow = isBearishGap ? Bars.HighPrices[index - (i + 2)] : Bars.HighPrices[index - i];
            var gapHigh = isBearishGap ? Bars.LowPrices[index - i] : Bars.LowPrices[index - (i + 2)];

            if ((gapHigh - gapLow) < minSizeValue)
                return false;

            var alreadyBroken = false;
            if (i > 1)
            {
                for (var k = i - 1; k >= 1; k--)
                {
                    var close = Bars.ClosePrices[index - k];
                    if ((isBearishGap && close < gapLow) || (!isBearishGap && close > gapHigh))
                    {
                        alreadyBroken = true;
                        break;
                    }
                }
            }

            if (alreadyBroken)
                return false;

            var breakout = isBearishGap ? Bars.ClosePrices[index] < gapLow : Bars.ClosePrices[index] > gapHigh;
            if (!breakout)
                return false;

            var maReady = !double.IsNaN(maValue) && !double.IsNaN(_maSeries[index - 1]);
            var maCondition = isBearishGap
                ? maReady && maValue < _maSeries[index - 1] && Bars.ClosePrices[index] < maValue
                : maReady && maValue > _maSeries[index - 1] && Bars.ClosePrices[index] > maValue;

            if (!maCondition)
                return false;

            signalDir = isBearishGap ? -1 : 1;
            return true;
        }

        private void TryEnterLong(int signalIndex)
        {
            if (!AllowMultiplePositions && HasOpenPosition(TradeType.Buy))
            {
                Print("[LONG SKIP] Existing long position blocks new entry at bar {0}", signalIndex);
                return;
            }

            var sslPrice = CurrentSslPrice;
            if (double.IsNaN(sslPrice) || sslPrice <= 0)
            {
                Print("[LONG SKIP] SSL unavailable at bar {0}", signalIndex);
                return;
            }

            var entry = Symbol.Ask;
            var slDistancePips = (entry - sslPrice) / Symbol.PipSize;
            if (slDistancePips <= 0)
            {
                Print("[LONG SKIP] Invalid SL distance at bar {0}. Entry={1}, SSL={2}", signalIndex, entry, sslPrice);
                return;
            }

            var volume = GetRiskVolume(slDistancePips);
            if (volume <= 0)
            {
                Print("[LONG SKIP] Volume <= 0 at bar {0}. SL pips={1}", signalIndex, slDistancePips);
                return;
            }

            var takeProfitPips = slDistancePips * RewardMultiple;
            Print("[LONG FIRE] bar={0}, slPips={1}, tpPips={2}, vol={3}", signalIndex, slDistancePips, takeProfitPips, volume);
            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, BotLabel, slDistancePips, takeProfitPips);
        }

        private void TryEnterShort(int signalIndex)
        {
            if (!AllowMultiplePositions && HasOpenPosition(TradeType.Sell))
            {
                Print("[SHORT SKIP] Existing short position blocks new entry at bar {0}", signalIndex);
                return;
            }

            var bslPrice = CurrentBslPrice;
            if (double.IsNaN(bslPrice) || bslPrice <= 0)
            {
                Print("[SHORT SKIP] BSL unavailable at bar {0}", signalIndex);
                return;
            }

            var entry = Symbol.Bid;
            var slDistancePips = (bslPrice - entry) / Symbol.PipSize;
            if (slDistancePips <= 0)
            {
                Print("[SHORT SKIP] Invalid SL distance at bar {0}. Entry={1}, BSL={2}", signalIndex, entry, bslPrice);
                return;
            }

            var volume = GetRiskVolume(slDistancePips);
            if (volume <= 0)
            {
                Print("[SHORT SKIP] Volume <= 0 at bar {0}. SL pips={1}", signalIndex, slDistancePips);
                return;
            }

            var takeProfitPips = slDistancePips * RewardMultiple;
            Print("[SHORT FIRE] bar={0}, slPips={1}, tpPips={2}, vol={3}", signalIndex, slDistancePips, takeProfitPips, volume);
            ExecuteMarketOrder(TradeType.Sell, SymbolName, volume, BotLabel, slDistancePips, takeProfitPips);
        }

        private double GetRiskVolume(double stopLossPips)
        {
            var riskAmount = Account.Equity * (RiskPercent / 100.0);
            var volume = Symbol.VolumeForFixedRisk(riskAmount, stopLossPips);
            volume = Symbol.NormalizeVolumeInUnits(volume, RoundingMode.Down);

            if (volume < Symbol.VolumeInUnitsMin)
                return 0;

            if (volume > Symbol.VolumeInUnitsMax)
                volume = Symbol.VolumeInUnitsMax;

            return volume;
        }

        private bool HasOpenPosition(TradeType tradeType)
        {
            foreach (var position in Positions)
            {
                if (position.SymbolName == SymbolName && position.Label == BotLabel && position.TradeType == tradeType)
                    return true;
            }

            return false;
        }
    }
}
