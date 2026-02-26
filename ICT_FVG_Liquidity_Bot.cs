// =============================================================================
// ICT FVG + Liquidity Sweep cBot
// =============================================================================
// Combines:
//   1) ICT Setup 01 [TradingFinder] FVG + Liquidity Sweeps/Hunt signal logic
//      → Long signal: price sweeps below bullish FVG, closes back above proximal
//      → Short signal: price sweeps above bearish FVG, closes back below proximal
//   2) BSL & SSL (Buyside/Sellside Liquidity) pivot-based levels
//      → Long SL at nearest SSL; Short SL at nearest BSL
//   3) Risk management: 1% equity risk per trade, R:R = 1:2
// =============================================================================

using System;
using System.Collections.Generic;
using System.Linq;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class ICT_FVG_Liquidity_Bot : Robot
    {
        // =====================================================================
        // Parameters – ICT FVG Signal Settings
        // =====================================================================

        [Parameter("FVG Detector Multiplier", DefaultValue = 1.0, MinValue = 1.0, Group = "FVG Settings")]
        public double FvgDetectorMultiplier { get; set; }

        [Parameter("FVG Validity Period", DefaultValue = 15, MinValue = 2, Group = "FVG Settings")]
        public int FvgValidityPeriod { get; set; }

        [Parameter("Level in Low Risk Zone", DefaultValue = false, Group = "FVG Settings")]
        public bool UseDiscountAndPremium { get; set; }

        [Parameter("Signal Method (Hunt/Sweeps)", DefaultValue = "Hunt", Group = "FVG Settings")]
        public string SignalMethod { get; set; }

        [Parameter("Signals Allowed Per Zone", DefaultValue = 3, MinValue = 1, Group = "FVG Settings")]
        public int SignalsAllowedPerZone { get; set; }

        [Parameter("Signal After Hunts", DefaultValue = false, Group = "FVG Settings")]
        public bool SignalAfterHunts { get; set; }

        [Parameter("Required Hunts Count", DefaultValue = 2, MinValue = 1, Group = "FVG Settings")]
        public int RequiredHunts { get; set; }

        // =====================================================================
        // Parameters – BSL / SSL Liquidity Settings
        // =====================================================================

        [Parameter("Pivot Left", DefaultValue = 5, MinValue = 1, Group = "Liquidity")]
        public int PivotLeft { get; set; }

        [Parameter("Pivot Right", DefaultValue = 5, MinValue = 1, Group = "Liquidity")]
        public int PivotRight { get; set; }

        // =====================================================================
        // Parameters – Risk Management
        // =====================================================================

        [Parameter("Risk Per Trade (%)", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 10.0, Group = "Risk Management")]
        public double RiskPercent { get; set; }

        [Parameter("Reward:Risk Ratio", DefaultValue = 2.0, MinValue = 1.0, Group = "Risk Management")]
        public double RewardRiskRatio { get; set; }

        [Parameter("Max Open Positions", DefaultValue = 1, MinValue = 1, MaxValue = 5, Group = "Risk Management")]
        public int MaxOpenPositions { get; set; }

        [Parameter("Mirror ICT Signals Exactly", DefaultValue = true, Group = "Signal Execution")]
        public bool MirrorIctSignalsExactly { get; set; }

        [Parameter("Close Opposite Positions", DefaultValue = true, Group = "Signal Execution")]
        public bool CloseOppositePositions { get; set; }

        [Parameter("Min SL Distance (pips)", DefaultValue = 5.0, MinValue = 1.0, Group = "Risk Management")]
        public double MinSlPips { get; set; }

        [Parameter("Max SL Distance (pips)", DefaultValue = 200.0, MinValue = 10.0, Group = "Risk Management")]
        public double MaxSlPips { get; set; }

        [Parameter("SL Buffer (pips)", DefaultValue = 2.0, MinValue = 0.0, Group = "Risk Management")]
        public double SlBufferPips { get; set; }

        // =====================================================================
        // Constants
        // =====================================================================

        private const int AtrLength = 55;
        private const string BotLabel = "ICT_FVG_LIQ";

        // =====================================================================
        // ICT FVG Signal State
        // =====================================================================

        private double[] _trueRange;
        private double[] _averageTrueRange;
        private double[] _bullishDistalLevel;
        private double[] _bullishProximalLevel;
        private double[] _bearishDistalLevel;
        private double[] _bearishProximalLevel;
        private double[] _bullishFvgPointSeries;
        private double[] _bearishFvgPointSeries;

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

        private int _fvgArraySize;

        // =====================================================================
        // BSL / SSL Liquidity State
        // =====================================================================

        private sealed class LiquidityLevel
        {
            public double Price;
            public int PivotBarIndex;
            public bool Mitigated;
        }

        // Sorted newest-first
        private readonly List<LiquidityLevel> _bslLevels = new List<LiquidityLevel>();  // Buyside (pivot highs)
        private readonly List<LiquidityLevel> _sslLevels = new List<LiquidityLevel>();  // Sellside (pivot lows)

        private int _lastProcessedBarIndex = -1;
        private int _lastLongTradeSignalBar = -1;
        private int _lastShortTradeSignalBar = -1;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        protected override void OnStart()
        {
            // Pre-allocate FVG arrays to cover loaded bar history
            _fvgArraySize = Bars.Count + 5000; // some extra room
            InitFvgArrays(_fvgArraySize);

            // Run the full history to initialize FVG state and liquidity levels
            for (int i = 0; i < Bars.Count - 1; i++)
            {
                ProcessBar(i, isLive: false);
            }

            Print("ICT FVG + Liquidity Bot started. History processed: {0} bars", Bars.Count);
            Print("Current BSL levels: {0}, SSL levels: {1}", _bslLevels.Count, _sslLevels.Count);
        }

        protected override void OnBar()
        {
            // OnBar fires when a new bar opens, so the last completed bar is index = Bars.Count - 2
            int completedBarIndex = Bars.Count - 2;

            // Expand arrays if needed
            EnsureFvgArrayCapacity(completedBarIndex + 1);

            ProcessBar(completedBarIndex, isLive: true);
        }

        protected override void OnStop()
        {
            Print("ICT FVG + Liquidity Bot stopped.");
        }

        // =====================================================================
        // Main Processing (per completed bar)
        // =====================================================================

        private void ProcessBar(int index, bool isLive)
        {
            // --- 1. ATR ---
            CalculateAtr(index);

            // --- 2. FVG Signal Detection ---
            CalculateFvgSignals(index);

            // --- 3. BSL/SSL Liquidity Level Detection ---
            DetectLiquidityLevels(index);

            // --- 4. Mitigate swept levels ---
            MitigateLevels(index);

            // --- 5. Trade execution (only on live bars) ---
            if (isLive)
            {
                ExecuteTradeLogic(index);
            }
        }

        // =====================================================================
        // ICT FVG Signal Logic (ported from indicator)
        // =====================================================================

        private void InitFvgArrays(int size)
        {
            _trueRange = new double[size];
            _averageTrueRange = new double[size];
            _bullishDistalLevel = new double[size];
            _bullishProximalLevel = new double[size];
            _bearishDistalLevel = new double[size];
            _bearishProximalLevel = new double[size];
            _bullishFvgPointSeries = new double[size];
            _bearishFvgPointSeries = new double[size];

            for (int i = 0; i < size; i++)
            {
                _averageTrueRange[i] = double.NaN;
            }
        }

        private void EnsureFvgArrayCapacity(int requiredIndex)
        {
            if (requiredIndex < _fvgArraySize)
                return;

            int newSize = Math.Max(_fvgArraySize * 2, requiredIndex + 1000);
            ResizeArray(ref _trueRange, newSize);
            ResizeArray(ref _averageTrueRange, newSize, double.NaN);
            ResizeArray(ref _bullishDistalLevel, newSize);
            ResizeArray(ref _bullishProximalLevel, newSize);
            ResizeArray(ref _bearishDistalLevel, newSize);
            ResizeArray(ref _bearishProximalLevel, newSize);
            ResizeArray(ref _bullishFvgPointSeries, newSize);
            ResizeArray(ref _bearishFvgPointSeries, newSize);
            _fvgArraySize = newSize;
        }

        private void ResizeArray(ref double[] arr, int newSize, double fillValue = 0.0)
        {
            var newArr = new double[newSize];
            Array.Copy(arr, newArr, arr.Length);
            if (fillValue != 0.0)
            {
                for (int i = arr.Length; i < newSize; i++)
                    newArr[i] = fillValue;
            }
            arr = newArr;
        }

        private void CalculateAtr(int index)
        {
            double high = Bars.HighPrices[index];
            double low = Bars.LowPrices[index];

            if (index == 0)
            {
                _trueRange[index] = high - low;
                _averageTrueRange[index] = double.NaN;
                return;
            }

            double prevClose = Bars.ClosePrices[index - 1];
            double tr1 = high - low;
            double tr2 = Math.Abs(high - prevClose);
            double tr3 = Math.Abs(low - prevClose);
            _trueRange[index] = Math.Max(tr1, Math.Max(tr2, tr3));

            if (index < AtrLength - 1)
            {
                _averageTrueRange[index] = double.NaN;
                return;
            }

            if (index == AtrLength - 1)
            {
                double sum = 0.0;
                for (int i = 0; i < AtrLength; i++)
                    sum += _trueRange[i];
                _averageTrueRange[index] = sum / AtrLength;
                return;
            }

            _averageTrueRange[index] =
                ((_averageTrueRange[index - 1] * (AtrLength - 1)) + _trueRange[index]) / AtrLength;
        }

        private void CalculateFvgSignals(int index)
        {
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

            // Carry forward
            _bullishDistalLevel[index] = _bullishDistalLevel[index - 1];
            _bullishProximalLevel[index] = _bullishProximalLevel[index - 1];
            _bearishDistalLevel[index] = _bearishDistalLevel[index - 1];
            _bearishProximalLevel[index] = _bearishProximalLevel[index - 1];

            double high = Bars.HighPrices[index];
            double low = Bars.LowPrices[index];
            double close = Bars.ClosePrices[index];

            _isBullishFvg = false;
            _isBearishFvg = false;

            if (index >= 2)
            {
                double high2 = Bars.HighPrices[index - 2];
                double low2 = Bars.LowPrices[index - 2];
                double high1 = Bars.HighPrices[index - 1];
                double low1 = Bars.LowPrices[index - 1];
                double atrValue = _averageTrueRange[index];
                if (double.IsNaN(atrValue))
                    atrValue = 0.0;

                // Bullish FVG detection
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

                // Bearish FVG detection
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

            // Update zone levels
            if (UseDiscountAndPremium)
            {
                if (_isBullishFvg)
                {
                    _bullishDistalLevel[index] = _bullishFvgDistal;
                    _bullishProximalLevel[index] = _bullishEquilibrium >= _bullishFvgProximal
                        ? _bullishFvgProximal : _bullishEquilibrium;
                }
                if (_isBearishFvg)
                {
                    _bearishDistalLevel[index] = _bearishFvgDistal;
                    _bearishProximalLevel[index] = _bearishEquilibrium <= _bearishFvgProximal
                        ? _bearishFvgProximal : _bearishEquilibrium;
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

            // Zone validity checks
            double body1 = Bars.ClosePrices[index - 1] - Bars.OpenPrices[index - 1];
            double prevDistalBu = _bullishDistalLevel[index - 1];
            double prevProximalBu = _bullishProximalLevel[index - 1];
            double prevDistalBe = _bearishDistalLevel[index - 1];
            double prevProximalBe = _bearishProximalLevel[index - 1];

            if (_isBullishFvgValid)
            {
                double bullProximal = _bullishProximalLevel[index];
                _isBullishFvgValid = UpdateZoneValidity(index, body1, true, _bullishFvgPoint,
                    prevDistalBu, prevProximalBu, _longSignalCount, ref bullProximal);
                _bullishProximalLevel[index] = bullProximal;
            }

            if (_isBearishFvgValid)
            {
                double bearProximal = _bearishProximalLevel[index];
                _isBearishFvgValid = UpdateZoneValidity(index, body1, false, _bearishFvgPoint,
                    prevDistalBe, prevProximalBe, _shortSignalCount, ref bearProximal);
                _bearishProximalLevel[index] = bearProximal;
            }

            // Reset on new FVG
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

            // Long signal logic
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
                        _isLongSignal = SignalAfterHunts
                            ? _longSignalCount == RequiredHunts
                            : true;
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

            // Short signal logic
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
                        _isShortSignal = SignalAfterHunts
                            ? _shortSignalCount == RequiredHunts
                            : true;
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
        }

        private bool UpdateZoneValidity(int index, double body1, bool isBull, int zonePoint,
            double prevDistal, double prevProximal, int signalCount, ref double updatedProximal)
        {
            bool useOpenForBodyDirection = isBull ? body1 > 0 : body1 <= 0;
            double selectedPrice = useOpenForBodyDirection
                ? Bars.OpenPrices[index - 1] : Bars.ClosePrices[index - 1];
            double sweepPrice = isBull
                ? (SignalMethod == "Sweeps" ? selectedPrice : Bars.LowPrices[index - 1])
                : (SignalMethod == "Sweeps" ? selectedPrice : Bars.HighPrices[index - 1]);

            bool sweepCheck = isBull ? sweepPrice < prevDistal : sweepPrice > prevDistal;
            bool expired = index > zonePoint + FvgValidityPeriod;
            bool signalLimitExceeded = !SignalAfterHunts && signalCount > SignalsAllowedPerZone - 1;

            if (sweepCheck || expired || signalLimitExceeded)
                return false;

            bool movedInsideZone = isBull
                ? selectedPrice < prevProximal && selectedPrice > prevDistal
                : selectedPrice > prevProximal && selectedPrice < prevDistal;

            if (movedInsideZone)
                updatedProximal = selectedPrice;

            return true;
        }

        // =====================================================================
        // BSL / SSL Liquidity Detection (ported from indicator)
        // =====================================================================

        private void DetectLiquidityLevels(int index)
        {
            int pivotIndex = index - PivotRight;
            if (pivotIndex - PivotLeft < 0)
                return;

            int leftStart = pivotIndex - PivotLeft;
            int rightEnd = pivotIndex + PivotRight;

            if (rightEnd >= Bars.Count)
                return;

            double candidateHigh = Bars.HighPrices[pivotIndex];
            double candidateLow = Bars.LowPrices[pivotIndex];

            // Non-strict pivot high (BSL)
            if (IsPivotHigh(candidateHigh, leftStart, rightEnd))
            {
                // Avoid duplicates
                if (!_bslLevels.Any(l => l.PivotBarIndex == pivotIndex && !l.Mitigated
                    && Math.Abs(l.Price - candidateHigh) < Symbol.PipSize * 0.1))
                {
                    _bslLevels.Insert(0, new LiquidityLevel
                    {
                        Price = candidateHigh,
                        PivotBarIndex = pivotIndex,
                        Mitigated = false
                    });
                }
            }

            // Non-strict pivot low (SSL)
            if (IsPivotLow(candidateLow, leftStart, rightEnd))
            {
                if (!_sslLevels.Any(l => l.PivotBarIndex == pivotIndex && !l.Mitigated
                    && Math.Abs(l.Price - candidateLow) < Symbol.PipSize * 0.1))
                {
                    _sslLevels.Insert(0, new LiquidityLevel
                    {
                        Price = candidateLow,
                        PivotBarIndex = pivotIndex,
                        Mitigated = false
                    });
                }
            }

            // Keep lists from growing unbounded
            while (_bslLevels.Count > 50)
                _bslLevels.RemoveAt(_bslLevels.Count - 1);
            while (_sslLevels.Count > 50)
                _sslLevels.RemoveAt(_sslLevels.Count - 1);
        }

        private bool IsPivotHigh(double candidate, int start, int end)
        {
            double max = double.MinValue;
            for (int i = start; i <= end; i++)
                if (Bars.HighPrices[i] > max) max = Bars.HighPrices[i];
            return candidate == max;
        }

        private bool IsPivotLow(double candidate, int start, int end)
        {
            double min = double.MaxValue;
            for (int i = start; i <= end; i++)
                if (Bars.LowPrices[i] < min) min = Bars.LowPrices[i];
            return candidate == min;
        }

        private void MitigateLevels(int index)
        {
            double barHigh = Bars.HighPrices[index];
            double barLow = Bars.LowPrices[index];

            // BSL mitigated when price trades at or above it
            foreach (var bsl in _bslLevels)
            {
                if (!bsl.Mitigated && barHigh >= bsl.Price)
                    bsl.Mitigated = true;
            }

            // SSL mitigated when price trades at or below it
            foreach (var ssl in _sslLevels)
            {
                if (!ssl.Mitigated && barLow <= ssl.Price)
                    ssl.Mitigated = true;
            }
        }

        // =====================================================================
        // Trade Execution
        // =====================================================================

        private void ExecuteTradeLogic(int index)
        {
            if (_lastProcessedBarIndex == index)
                return;

            _lastProcessedBarIndex = index;

            // Count open positions from this bot
            int openCount = Positions.FindAll(BotLabel, SymbolName).Length;

            if (!MirrorIctSignalsExactly && openCount >= MaxOpenPositions)
                return;

            // Signal is computed on the completed bar (index), entry must mirror execution at next bar open.
            int entryBarIndex = Math.Min(index + 1, Bars.Count - 1);
            double entryPrice = Bars.OpenPrices[entryBarIndex];

            if (_isLongSignal && (!MirrorIctSignalsExactly || _lastLongTradeSignalBar != index))
            {
                if (CloseOppositePositions)
                    CloseBotPositions(TradeType.Sell);

                if (!MirrorIctSignalsExactly)
                {
                    openCount = Positions.FindAll(BotLabel, SymbolName).Length;
                    if (openCount >= MaxOpenPositions)
                        return;
                }

                TryEnterLong(entryPrice, index);
                _lastLongTradeSignalBar = index;
            }

            if (_isShortSignal && (!MirrorIctSignalsExactly || _lastShortTradeSignalBar != index))
            {
                if (CloseOppositePositions)
                    CloseBotPositions(TradeType.Buy);

                if (!MirrorIctSignalsExactly)
                {
                    openCount = Positions.FindAll(BotLabel, SymbolName).Length;
                    if (openCount >= MaxOpenPositions)
                        return;
                }

                TryEnterShort(entryPrice, index);
                _lastShortTradeSignalBar = index;
            }
        }

        private void CloseBotPositions(TradeType tradeType)
        {
            foreach (var position in Positions.FindAll(BotLabel, SymbolName, tradeType))
            {
                ClosePosition(position);
            }
        }

        private void TryEnterLong(double entryPrice, int barIndex)
        {
            // Find the nearest non-mitigated SSL below current price for stop loss
            double? sslLevel = GetNearestSslBelow(entryPrice);

            if (!sslLevel.HasValue && MirrorIctSignalsExactly)
            {
                double fallback = _bullishDistalLevel[barIndex];
                if (fallback > 0 && fallback < entryPrice)
                    sslLevel = fallback;
            }

            if (!sslLevel.HasValue)
            {
                Print("Bar {0}: LONG signal, but no active SSL level found below price {1}. Skipping.",
                    barIndex, entryPrice);
                return;
            }

            double slPrice = sslLevel.Value - (SlBufferPips * Symbol.PipSize); // buffer below SSL
            double slDistancePrice = entryPrice - slPrice;

            if (slDistancePrice <= 0)
            {
                Print("Bar {0}: LONG signal, but SSL+buffer ({1:F5}) is not below entry ({2:F5}). Skipping.",
                    barIndex, slPrice, entryPrice);
                return;
            }

            double slPips = slDistancePrice / Symbol.PipSize;

            if (slPips < MinSlPips)
            {
                Print("Bar {0}: LONG signal, SL distance {1:F1} pips < minimum {2:F1}. Skipping.",
                    barIndex, slPips, MinSlPips);
                return;
            }

            if (slPips > MaxSlPips)
            {
                Print("Bar {0}: LONG signal, SL distance {1:F1} pips > maximum {2:F1}. Skipping.",
                    barIndex, slPips, MaxSlPips);
                return;
            }

            double tpPips = slPips * RewardRiskRatio;

            // Calculate volume for 1% risk
            double riskAmount = Account.Equity * (RiskPercent / 100.0);
            double volume = CalculateVolume(riskAmount, slPips);

            if (volume <= 0)
            {
                Print("Bar {0}: LONG signal, calculated volume is 0. Skipping.", barIndex);
                return;
            }

            Print("Bar {0}: LONG entry | SL={1:F5} ({2:F1} pips) | TP={3:F1} pips | Vol={4}",
                barIndex, slPrice, slPips, tpPips, volume);

            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, BotLabel, slPips, tpPips);
        }

        private void TryEnterShort(double entryPrice, int barIndex)
        {
            // Find the nearest non-mitigated BSL above current price for stop loss
            double? bslLevel = GetNearestBslAbove(entryPrice);

            if (!bslLevel.HasValue && MirrorIctSignalsExactly)
            {
                double fallback = _bearishDistalLevel[barIndex];
                if (fallback > 0 && fallback > entryPrice)
                    bslLevel = fallback;
            }

            if (!bslLevel.HasValue)
            {
                Print("Bar {0}: SHORT signal, but no active BSL level found above price {1}. Skipping.",
                    barIndex, entryPrice);
                return;
            }

            double slPrice = bslLevel.Value + (SlBufferPips * Symbol.PipSize); // buffer above BSL
            double slDistancePrice = slPrice - entryPrice;

            if (slDistancePrice <= 0)
            {
                Print("Bar {0}: SHORT signal, but BSL+buffer ({1:F5}) is not above entry ({2:F5}). Skipping.",
                    barIndex, slPrice, entryPrice);
                return;
            }

            double slPips = slDistancePrice / Symbol.PipSize;

            if (slPips < MinSlPips)
            {
                Print("Bar {0}: SHORT signal, SL distance {1:F1} pips < minimum {2:F1}. Skipping.",
                    barIndex, slPips, MinSlPips);
                return;
            }

            if (slPips > MaxSlPips)
            {
                Print("Bar {0}: SHORT signal, SL distance {1:F1} pips > maximum {2:F1}. Skipping.",
                    barIndex, slPips, MaxSlPips);
                return;
            }

            double tpPips = slPips * RewardRiskRatio;

            // Calculate volume for 1% risk
            double riskAmount = Account.Equity * (RiskPercent / 100.0);
            double volume = CalculateVolume(riskAmount, slPips);

            if (volume <= 0)
            {
                Print("Bar {0}: SHORT signal, calculated volume is 0. Skipping.", barIndex);
                return;
            }

            Print("Bar {0}: SHORT entry | SL={1:F5} ({2:F1} pips) | TP={3:F1} pips | Vol={4}",
                barIndex, slPrice, slPips, tpPips, volume);

            ExecuteMarketOrder(TradeType.Sell, SymbolName, volume, BotLabel, slPips, tpPips);
        }

        // =====================================================================
        // Liquidity Level Lookups
        // =====================================================================

        /// <summary>
        /// Find the nearest non-mitigated SSL (pivot low) that is below the given price.
        /// Returns the one closest to price (highest SSL below price).
        /// </summary>
        private double? GetNearestSslBelow(double price)
        {
            double? nearest = null;

            foreach (var ssl in _sslLevels)
            {
                if (ssl.Mitigated) continue;
                if (ssl.Price >= price) continue;

                if (!nearest.HasValue || ssl.Price > nearest.Value)
                    nearest = ssl.Price;
            }

            return nearest;
        }

        /// <summary>
        /// Find the nearest non-mitigated BSL (pivot high) that is above the given price.
        /// Returns the one closest to price (lowest BSL above price).
        /// </summary>
        private double? GetNearestBslAbove(double price)
        {
            double? nearest = null;

            foreach (var bsl in _bslLevels)
            {
                if (bsl.Mitigated) continue;
                if (bsl.Price <= price) continue;

                if (!nearest.HasValue || bsl.Price < nearest.Value)
                    nearest = bsl.Price;
            }

            return nearest;
        }

        // =====================================================================
        // Volume Calculation
        // =====================================================================

        /// <summary>
        /// Calculates the trade volume in units for a given risk amount and SL distance in pips.
        /// Uses Symbol.PipValue (value of 1 pip for 1 unit of volume).
        /// </summary>
        private double CalculateVolume(double riskAmount, double slPips)
        {
            if (slPips <= 0 || Symbol.PipValue <= 0)
                return 0;

            // PipValue = value of 1 pip for 1 unit of the symbol
            // Volume = RiskAmount / (slPips * PipValue)
            double rawVolume = riskAmount / (slPips * Symbol.PipValue);

            // Normalize to the nearest valid volume (broker lot constraints)
            double normalizedVolume = Symbol.NormalizeVolumeInUnits(rawVolume, RoundingMode.Down);

            // Ensure minimum volume
            if (normalizedVolume < Symbol.VolumeInUnitsMin)
                normalizedVolume = Symbol.VolumeInUnitsMin;

            return normalizedVolume;
        }
    }
}
