using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    // ════════════════════════════════════════════════════════════════════════════
    //  Swing_OB_Detector_cBot  —  SELF-CONTAINED (no external indicator)
    //
    //  All swing OB detection logic is embedded directly in this file.
    //  There is NO dependency on the Swing_OB_Detector.cs indicator.
    //  This guarantees the cBot runs exactly the code written here,
    //  with no installation ambiguity, no DLL caching issues, and no
    //  version mismatch between indicator and cBot.
    //
    //  ── DETECTION ENGINE (identical to Swing_OB_Detector.cs) ────────────────
    //
    //  1. Every bar: ComputeLeg() tracks the swing pivot direction using
    //     SwingsLengthInput bars on each side.
    //
    //  2. When the swing leg changes, _lastSwingHigh / _lastSwingLow are updated.
    //
    //  3. ProcessSwingStructureCrosses():
    //     • Bullish BOS/CHoCH (close > _lastSwingHigh):
    //         → StoreObFromPivot(bias=+1) stores the bar with the lowest
    //           parsed-low in [_lastSwingHighIndex .. breakIndex] as a BULL OB.
    //     • Bearish BOS/CHoCH (close < _lastSwingLow):
    //         → StoreObFromPivot(bias=−1) stores the bar with the highest
    //           parsed-high in [_lastSwingLowIndex .. breakIndex] as a BEAR OB.
    //
    //  4. ManageObList() runs every bar:
    //     • Bull OB: if Low ≤ ob.Top → touch. If filters pass, _swingObSignal
    //       is set with Point=ob.Top, IsBull=true, ObSlLevel=ob.Bottom.
    //     • Bear OB: if High ≥ ob.Bottom → touch. _swingObSignal set with
    //       Point=ob.Bottom, IsBull=false, ObSlLevel=ob.Top.
    //     • Mitigation (HighLow mode): High > ob.Top (bear) or Low < ob.Bottom
    //       (bull) → OB removed from pool.
    //
    //  5. EvaluateSignal():
    //     • Long:  close > ob.Top  AND bullish candle (close>open) AND Entry=false
    //              → condSwing = +1   (LONG signal)
    //     • Short: close < ob.Bottom AND bearish candle (close<open) AND Entry=false
    //              → condSwing = −1   (SHORT signal)
    //
    //  6. Signal output:
    //     • condSwing=+1 → _lastLongObBottom  = ob.Bottom  (non-NaN this bar only)
    //     • condSwing=−1 → _lastShortObTop    = ob.Top     (non-NaN this bar only)
    //     Both reset to NaN at the start of every bar.
    //
    //  ── TRADE RULES ─────────────────────────────────────────────────────────
    //  • Long  signal on bar N → BUY  market order at bar N+1 open.
    //  • Short signal on bar N → SELL market order at bar N+1 open.
    //  • No stop loss.  No take profit.  No cap on concurrent positions.
    //  • Every position is closed at bar (entryBar + 2) open.
    //  • Position size = 0.5% of Account.Balance / Symbol.PipValue (per entry).
    // ════════════════════════════════════════════════════════════════════════════

    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class Swing_OB_Detector_cBot : Robot
    {
        // ════════════════════════════════════════════════════════════════════════
        //  Parameters
        // ════════════════════════════════════════════════════════════════════════

        // ── Swing structure ───────────────────────────────────────────────────
        [Parameter("Swings Length", DefaultValue = 50, MinValue = 10, Group = "Swing Detection")]
        public int SwingsLengthInput { get; set; }

        [Parameter("BOS/CHoCH Source (Close=0 / HighLow=1)", DefaultValue = 0, MinValue = 0, MaxValue = 1, Group = "Swing Detection")]
        public int StructureSourceInput { get; set; }   // 0=Close  1=HighLow

        // ── Order Block ───────────────────────────────────────────────────────
        [Parameter("OB Filter ATR Period", DefaultValue = 200, MinValue = 1, MaxValue = 500, Group = "Swing Detection")]
        public int ObFilterAtrPeriod { get; set; }

        [Parameter("Order Block Mitigation (Close=0 / HighLow=1)", DefaultValue = 1, MinValue = 0, MaxValue = 1, Group = "Swing Detection")]
        public int MitigationModeInput { get; set; }    // 0=Close  1=HighLow

        [Parameter("Min Dist OB (bars)", DefaultValue = 1, MinValue = 1, Group = "Swing Detection")]
        public int MinDist { get; set; }

        [Parameter("Min Bars After Structure Break", DefaultValue = 0, MinValue = 0, MaxValue = 200, Group = "Swing Detection")]
        public int MinBarsAfterStructureBreak { get; set; }

        [Parameter("Show All Historical OBs", DefaultValue = true, Group = "Swing Detection")]
        public bool ShowAllHistoricalObs { get; set; }

        [Parameter("Swing OB Size", DefaultValue = 5, MinValue = 1, MaxValue = 20, Group = "Swing Detection")]
        public int SwingOBSize { get; set; }

        // ── Trade ─────────────────────────────────────────────────────────────
        [Parameter("Trade Long Signals",  DefaultValue = true, Group = "Trade")]
        public bool TradeLong  { get; set; }

        [Parameter("Trade Short Signals", DefaultValue = true, Group = "Trade")]
        public bool TradeShort { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Inner types
        // ════════════════════════════════════════════════════════════════════════

        private sealed class OrderBlock
        {
            public int    Index;               // pivot bar index
            public double Top;                 // parsed high of pivot bar
            public double Bottom;              // parsed low  of pivot bar
            public bool   Bullish;
            public bool   SignalFired;         // consumed on first touch
            public bool   Mitigated;
            public int    StructureBreakIndex; // bar that created this OB
        }

        private sealed class SignalState
        {
            public double Point     = double.NaN; // OB top (bull) or bottom (bear)
            public double ObSlLevel = double.NaN; // OB bottom (bull) or top (bear)
            public bool   IsBull;
            public bool   Entry;                  // true = already consumed
        }

        private sealed class TradeRecord
        {
            public string Label;
            public int    EntryBarIndex;
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Private fields — detection engine
        // ════════════════════════════════════════════════════════════════════════
        private readonly List<OrderBlock> _bullObs      = new List<OrderBlock>();
        private readonly List<OrderBlock> _bearObs      = new List<OrderBlock>();
        private readonly List<double>     _parsedHighs  = new List<double>();
        private readonly List<double>     _parsedLows   = new List<double>();
        private SignalState               _swingObSignal;

        private int    _swingLeg;
        private double _lastSwingHigh      = double.NaN;
        private double _lastSwingLow       = double.NaN;
        private int    _lastSwingHighIndex = -1;
        private int    _lastSwingLowIndex  = -1;
        private bool   _swingHighCrossed;
        private bool   _swingLowCrossed;

        private int    _obIdCounter;
        private const int MaxOBs = 500;

        // Per-bar signal values (reset each bar)
        private double _longObLevel;   // non-NaN when long signal fires this bar
        private double _shortObLevel;  // non-NaN when short signal fires this bar

        private int _minBarsWarmup;

        // ── Trade tracking ────────────────────────────────────────────────────
        private readonly List<TradeRecord> _openTrades = new List<TradeRecord>();
        private int _tradeCount;

        // ════════════════════════════════════════════════════════════════════════
        //  OnStart
        // ════════════════════════════════════════════════════════════════════════
        protected override void OnStart()
        {
            _minBarsWarmup  = SwingsLengthInput + 5;
            _swingObSignal  = NewEmptySignal();

            Print($"[SwingOB cBot] Started — SwingsLen={SwingsLengthInput} " +
                  $"MinDist={MinDist} MinBarsAfterBreak={MinBarsAfterStructureBreak} " +
                  $"ATRPeriod={ObFilterAtrPeriod} Mitigation={(MitigationModeInput==1?"HighLow":"Close")} " +
                  $"StructSrc={(StructureSourceInput==1?"HighLow":"Close")}");
        }

        // ════════════════════════════════════════════════════════════════════════
        //  OnBar — fires when a new bar opens (bar N+1), last closed bar = N
        // ════════════════════════════════════════════════════════════════════════
        protected override void OnBar()
        {
            int currentBarIndex = Bars.Count - 1;   // bar that just opened
            int signalBarIndex  = Bars.Count - 2;   // bar that just closed

            // ── Close positions open for 2 full bars ─────────────────────────
            CloseExpiredPositions(currentBarIndex);

            // ── Advance the detection engine through signalBarIndex ───────────
            // ProcessBar() runs the full OB engine for every bar from where
            // we left off up to and including signalBarIndex.
            // _longObLevel / _shortObLevel are set by ProcessBar(signalBarIndex).
            ProcessBar(signalBarIndex);

            // ── Warmup guard ──────────────────────────────────────────────────
            if (signalBarIndex < _minBarsWarmup)
                return;

            // ── Open trades based on signal from the just-closed bar ──────────
            if (TradeLong  && !double.IsNaN(_longObLevel))
                OpenTrade(TradeType.Buy,  currentBarIndex, _longObLevel);

            if (TradeShort && !double.IsNaN(_shortObLevel))
                OpenTrade(TradeType.Sell, currentBarIndex, _shortObLevel);
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Detection engine — ProcessBar
        //
        //  Runs the full swing OB pipeline for bar `index`.
        //  Sets _longObLevel / _shortObLevel for that bar.
        //  Called once per OnBar() for the just-closed bar.
        // ════════════════════════════════════════════════════════════════════════

        // Track the last bar index the engine has processed (to avoid re-processing)
        private int _lastProcessedIndex = -1;

        private void ProcessBar(int index)
        {
            if (index <= _lastProcessedIndex) return;

            // Fill parsed arrays from _lastProcessedIndex+1 up to index
            for (int i = _lastProcessedIndex + 1; i <= index; i++)
            {
                UpdateParsedArrays(i);
            }

            // Run the full engine for the new bar only
            _lastProcessedIndex = index;

            // Reset per-bar signal outputs
            _longObLevel  = double.NaN;
            _shortObLevel = double.NaN;

            if (index < _minBarsWarmup) return;

            // ── Swing leg and structure crosses ───────────────────────────────
            int sLen = Math.Max(5, SwingsLengthInput);
            int swingLegNow = ComputeLeg(index, sLen, _swingLeg);
            int dc = swingLegNow - _swingLeg;
            if (dc != 0)
            {
                if (dc == 1)
                {
                    _lastSwingLow      = Bars.LowPrices[index - sLen];
                    _lastSwingLowIndex = index - sLen;
                    _swingLowCrossed   = false;
                }
                else
                {
                    _lastSwingHigh      = Bars.HighPrices[index - sLen];
                    _lastSwingHighIndex = index - sLen;
                    _swingHighCrossed   = false;
                }
            }
            _swingLeg = swingLegNow;

            ProcessSwingStructureCrosses(index);

            // ── OB lifecycle + touch detection ────────────────────────────────
            ManageObList(_bullObs, index, bullish: true);
            ManageObList(_bearObs, index, bullish: false);

            // ── Signal confirmation ────────────────────────────────────────────
            double close     = Bars.ClosePrices[index];
            double open      = Bars.OpenPrices[index];
            int    candleDir = close > open ? 1 : -1;

            int condSwing = EvaluateSignal(_swingObSignal, close, candleDir);

            if (condSwing ==  1) _longObLevel  = _swingObSignal.ObSlLevel;
            if (condSwing == -1) _shortObLevel = _swingObSignal.ObSlLevel;
        }

        // ── Parsed arrays (volatility-adjusted OB anchoring) ──────────────────
        // On a high-volatility bar (range ≥ 2 × ATR), the parsed High and Low
        // are swapped so OBs anchor to the candle body rather than the wick.
        private void UpdateParsedArrays(int index)
        {
            double atr      = AverageTrueRange(index, ObFilterAtrPeriod);
            bool   highVol  = (Bars.HighPrices[index] - Bars.LowPrices[index]) >= 2.0 * atr;
            _parsedHighs.Add(highVol ? Bars.LowPrices[index]  : Bars.HighPrices[index]);
            _parsedLows.Add( highVol ? Bars.HighPrices[index] : Bars.LowPrices[index]);
        }

        private double AverageTrueRange(int index, int period)
        {
            int    start = Math.Max(1, index - period + 1);
            double sum   = 0.0;
            int    n     = 0;
            for (int i = start; i <= index; i++)
            {
                double pc = Bars.ClosePrices[i - 1];
                sum += Math.Max(Bars.HighPrices[i] - Bars.LowPrices[i],
                       Math.Max(Math.Abs(Bars.HighPrices[i] - pc),
                                Math.Abs(Bars.LowPrices[i]  - pc)));
                n++;
            }
            return n > 0 ? sum / n : Symbol.TickSize;
        }

        // ── Swing leg direction ────────────────────────────────────────────────
        // Returns 0 (down leg) or 1 (up leg). Unchanged if neither extreme is clear.
        private int ComputeLeg(int index, int size, int previousLeg)
        {
            if (index - size < 1) return previousLeg;
            double highest = double.MinValue, lowest = double.MaxValue;
            for (int i = Math.Max(0, index - size + 1); i <= index; i++)
            {
                if (Bars.HighPrices[i] > highest) highest = Bars.HighPrices[i];
                if (Bars.LowPrices[i]  < lowest)  lowest  = Bars.LowPrices[i];
            }
            if (Bars.HighPrices[index - size] > highest) return 0;
            if (Bars.LowPrices[index  - size] < lowest)  return 1;
            return previousLeg;
        }

        // ── BOS / CHoCH detection and OB creation ─────────────────────────────
        private void ProcessSwingStructureCrosses(int index)
        {
            // Bullish cross: price closes above the last swing high
            if (!double.IsNaN(_lastSwingHigh) && !_swingHighCrossed
                && CrossedUp(index, _lastSwingHigh))
            {
                _swingHighCrossed = true;
                StoreObFromPivot(_lastSwingHighIndex, bias: 1, breakIndex: index);
            }

            // Bearish cross: price closes below the last swing low
            if (!double.IsNaN(_lastSwingLow) && !_swingLowCrossed
                && CrossedDown(index, _lastSwingLow))
            {
                _swingLowCrossed = true;
                StoreObFromPivot(_lastSwingLowIndex, bias: -1, breakIndex: index);
            }
        }

        private bool CrossedUp(int index, double level)
        {
            return StructureSourceInput == 1
                ? Bars.HighPrices[index] > level
                : Bars.ClosePrices[index] > level;
        }

        private bool CrossedDown(int index, double level)
        {
            return StructureSourceInput == 1
                ? Bars.LowPrices[index] < level
                : Bars.ClosePrices[index] < level;
        }

        /// <summary>
        /// Creates a swing OB from the pivot bar that preceded a structure break.
        ///   bias = +1 (bullish break): search [pivotIndex..breakIndex] for the bar
        ///          with the LOWEST parsed-low → that bar's body is the demand zone.
        ///   bias = −1 (bearish break): search [pivotIndex..breakIndex] for the bar
        ///          with the HIGHEST parsed-high → that bar's body is the supply zone.
        /// </summary>
        private void StoreObFromPivot(int pivotIndex, int bias, int breakIndex)
        {
            if (pivotIndex < 0 || pivotIndex >= breakIndex
                || breakIndex >= _parsedHighs.Count) return;

            int parsedIndex = pivotIndex;
            if (bias == -1)
            {
                double maxV = double.MinValue;
                for (int i = pivotIndex; i <= breakIndex; i++)
                {
                    if (_parsedHighs[i] > maxV) { maxV = _parsedHighs[i]; parsedIndex = i; }
                }
            }
            else
            {
                double minV = double.MaxValue;
                for (int i = pivotIndex; i <= breakIndex; i++)
                {
                    if (_parsedLows[i] < minV) { minV = _parsedLows[i]; parsedIndex = i; }
                }
            }

            bool bullish = (bias == 1);
            var  ob = new OrderBlock
            {
                Index               = parsedIndex,
                Top                 = _parsedHighs[parsedIndex],
                Bottom              = _parsedLows[parsedIndex],
                Bullish             = bullish,
                SignalFired         = false,
                Mitigated           = false,
                StructureBreakIndex = breakIndex
            };
            _obIdCounter++;

            var list = bullish ? _bullObs : _bearObs;
            if (list.Count >= MaxOBs)
                list.RemoveAt(list.Count - 1);
            list.Insert(0, ob);
        }

        // ── OB lifecycle + touch → signal ─────────────────────────────────────
        /// <summary>
        /// Processes touch detection, mitigation, and optional pruning for one OB list.
        ///
        /// Touch (first-touch rule):
        ///   Bull OB: Low ≤ ob.Top   → ob.SignalFired=true; if filters pass, _swingObSignal updated.
        ///   Bear OB: High ≥ ob.Bottom → same.
        ///   Once SignalFired=true the OB never triggers again.
        ///
        /// Filters:
        ///   Filter 1 – same-bar block: the touch cannot be on the same bar as the
        ///              structure break that created the OB.
        ///   Filter 2 – MinDist: ob.Index + MinDist must be < index.
        ///   Filter 3 – MinBarsAfterStructureBreak cooldown.
        ///
        /// Mitigation (HighLow mode default):
        ///   Bear OB: High > ob.Top   → mitigated, removed.
        ///   Bull OB: Low  < ob.Bottom → mitigated, removed.
        /// </summary>
        private void ManageObList(List<OrderBlock> list, int index, bool bullish)
        {
            for (int i = list.Count - 1; i >= 0; i--)
            {
                OrderBlock ob = list[i];

                // ── Touch detection ───────────────────────────────────────────
                if (!ob.SignalFired && ob.Index < index)
                {
                    if (bullish && Bars.LowPrices[index] <= ob.Top)
                    {
                        ob.SignalFired = true;

                        bool sameBar   = (index == ob.StructureBreakIndex)
                                       && (Bars.ClosePrices[index] > Bars.OpenPrices[index]);
                        bool cooldownOk = index > ob.StructureBreakIndex + MinBarsAfterStructureBreak;

                        if (ob.Index + MinDist < index && !sameBar && cooldownOk)
                        {
                            // Point = ob.Top : confirmation requires close > ob.Top
                            // ObSlLevel = ob.Bottom : stop-loss reference
                            _swingObSignal = new SignalState
                            {
                                Point     = ob.Top,
                                ObSlLevel = ob.Bottom,
                                IsBull    = true,
                                Entry     = false
                            };
                        }
                    }
                    else if (!bullish && Bars.HighPrices[index] >= ob.Bottom)
                    {
                        ob.SignalFired = true;

                        bool sameBar   = (index == ob.StructureBreakIndex)
                                       && (Bars.ClosePrices[index] < Bars.OpenPrices[index]);
                        bool cooldownOk = index > ob.StructureBreakIndex + MinBarsAfterStructureBreak;

                        if (ob.Index + MinDist < index && !sameBar && cooldownOk)
                        {
                            // Point = ob.Bottom : confirmation requires close < ob.Bottom
                            // ObSlLevel = ob.Top : stop-loss reference
                            _swingObSignal = new SignalState
                            {
                                Point     = ob.Bottom,
                                ObSlLevel = ob.Top,
                                IsBull    = false,
                                Entry     = false
                            };
                        }
                    }
                }

                // ── Mitigation ────────────────────────────────────────────────
                double bearSrc = MitigationModeInput == 0
                    ? Bars.ClosePrices[index] : Bars.HighPrices[index];
                double bullSrc = MitigationModeInput == 0
                    ? Bars.ClosePrices[index] : Bars.LowPrices[index];

                bool mitigated = (!bullish && bearSrc > ob.Top)
                              || ( bullish && bullSrc < ob.Bottom);

                if (mitigated)
                {
                    ob.Mitigated = true;
                    list.RemoveAt(i);
                    continue;
                }

                // ── Prune outside size window (when ShowAllHistoricalObs=false) ─
                if (!ShowAllHistoricalObs && i >= SwingOBSize)
                {
                    list.RemoveAt(i);
                }
            }
        }

        // ── Signal confirmation ────────────────────────────────────────────────
        /// <summary>
        /// Checks the pending _swingObSignal against the current bar.
        ///   Long  (condSwing = +1): close > ob.Top   AND bullish bar AND Entry=false
        ///   Short (condSwing = −1): close < ob.Bottom AND bearish bar AND Entry=false
        /// Entry is set true once fired — signal cannot fire again until a new OB touch.
        /// </summary>
        private int EvaluateSignal(SignalState signal, double close, int candleDir)
        {
            if (double.IsNaN(signal.Point)) return 0;

            if (close > signal.Point && signal.IsBull && candleDir == 1 && !signal.Entry)
            {
                signal.Entry = true;
                return 1;
            }
            if (close < signal.Point && !signal.IsBull && candleDir == -1 && !signal.Entry)
            {
                signal.Entry = true;
                return -1;
            }
            return 0;
        }

        private static SignalState NewEmptySignal()
            => new SignalState { Point = double.NaN, ObSlLevel = double.NaN };

        // ════════════════════════════════════════════════════════════════════════
        //  Trade execution
        // ════════════════════════════════════════════════════════════════════════

        private void CloseExpiredPositions(int currentBarIndex)
        {
            for (int i = _openTrades.Count - 1; i >= 0; i--)
            {
                TradeRecord record = _openTrades[i];
                if (currentBarIndex < record.EntryBarIndex + 2) continue;

                bool found = false;
                foreach (var pos in Positions)
                {
                    if (pos.Label == record.Label && pos.SymbolName == SymbolName)
                    {
                        var result = ClosePosition(pos);
                        if (result.IsSuccessful)
                            Print($"[SwingOB] CLOSED '{record.Label}' " +
                                  $"@ {Bars.OpenTimes[currentBarIndex]:yyyy-MM-dd HH:mm} " +
                                  $"P&L={pos.NetProfit:F2}");
                        else
                            Print($"[SwingOB] CLOSE FAILED '{record.Label}' err={result.Error}");
                        found = true;
                        break;
                    }
                }
                if (!found)
                    Print($"[SwingOB] '{record.Label}' not found at close (closed externally?)");
                _openTrades.RemoveAt(i);
            }
        }

        private void OpenTrade(TradeType type, int entryBarIndex, double obLevel)
        {
            double riskAmount    = Account.Balance * 0.005;
            double rawUnits      = riskAmount / Symbol.PipValue;
            double volumeInUnits = Symbol.NormalizeVolumeInUnits(
                                       Math.Max(rawUnits, Symbol.VolumeInUnitsMin));

            string dir   = type == TradeType.Buy ? "Long" : "Short";
            string label = $"SwingOB_{dir}_{_tradeCount++}";

            var result = ExecuteMarketOrder(
                tradeType      : type,
                symbolName     : SymbolName,
                volume         : volumeInUnits,
                label          : label,
                stopLossPips   : null,
                takeProfitPips : null);

            if (result.IsSuccessful)
            {
                _openTrades.Add(new TradeRecord
                {
                    Label         = label,
                    EntryBarIndex = entryBarIndex
                });
                Print($"[SwingOB] OPENED '{label}' " +
                      $"@ {Bars.OpenTimes[entryBarIndex]:yyyy-MM-dd HH:mm} " +
                      $"price={result.Position.EntryPrice:F5} " +
                      $"vol={volumeInUnits} " +
                      $"risk={riskAmount:F2} " +
                      $"OB={obLevel:F5} " +
                      $"closeAt={entryBarIndex+2}");
            }
            else
            {
                Print($"[SwingOB] ORDER FAILED '{label}' err={result.Error}");
            }
        }

        protected override void OnStop()
        {
            Print($"[SwingOB] Stopped — trades={_tradeCount} " +
                  $"openRecords={_openTrades.Count} " +
                  $"balance={Account.Balance:F2}");
        }
    }
}
