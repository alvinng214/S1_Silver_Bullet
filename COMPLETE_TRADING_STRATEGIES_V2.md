================================================================================
COMPLETE TRADING STRATEGIES — ENHANCED v2.0
================================================================================
Source References:
- Original Claude complete_trading_strategies.md
- GPT Critical Review and Enhancements
- ALL IN ONE SR.txt (Primary Source for Hard Rules)
- Episodes 141-143 (New Content)

================================================================================
TAXONOMY CLARIFICATION
================================================================================

This document distinguishes between:

STRATEGY = Entry/exit model with codable trigger, stop, target (tradeable)
FRAMEWORK/FILTER = Gates trades or selects direction but does NOT define entry
COMPONENT = Building block used within strategies (OB, FVG, etc.)

Items previously over-claimed as "fully specified strategies" have been 
reclassified where appropriate.

================================================================================
SECTION A: FRAMEWORKS AND FILTERS (NOT STANDALONE STRATEGIES)
================================================================================

------------------------------------------------------------------------------
FRAMEWORK F1: TOP-DOWN BIAS + POI SELECTION
------------------------------------------------------------------------------
Classification: FILTER (Required prerequisite for all strategies)

PURPOSE: Determine directional bias and identify allowed trading zones

HARD RULES:
F1.1 Determine bias on Daily/4H first: bullish | bearish | no-trade
F1.2 Only trade at POI: pre-defined unmitigated HTF PD arrays (OB/FVG/supply-demand/breaker)
F1.3 IF no HTF bias OR price mid-range with no POI nearby → NO TRADE (agent must refuse)
F1.4 Entry timeframe must be at least 2 timeframes lower than analysis timeframe

OUTPUTS:
- bias: bullish | bearish | no_trade
- allowed_POIs: list of zones with timeframe + boundaries
- nearest_liquidity_targets: list (BSL/SSL)

CODABLE: YES - Clear binary outputs, zone identification rules defined

------------------------------------------------------------------------------
FRAMEWORK F2: DAILY BIAS MODEL
------------------------------------------------------------------------------
Classification: FILTER (Directional gate)

PURPOSE: Establish daily directional bias before session trading

HARD RULES:
F2.1 Define daily bias from HTF structure and PD array control
F2.2 Only take setups aligned with bias
F2.3 Exception: Clear HTF reversal confirmed by MAJOR CoC + POI mitigation allows counter-bias trades

CODABLE: YES - Structure-based bias determination

------------------------------------------------------------------------------
FRAMEWORK F3: KILLZONE SESSION FILTERS
------------------------------------------------------------------------------
Classification: FILTER (Time gate)

PURPOSE: Restrict trading to high-probability time windows

LONDON KILLZONE: 2:00 AM - 5:00 AM EST
- Scenarios: (a) Sweep Asian range then continue HTF, (b) Liquidity already taken pre-KZ then continue, (c) Sweep then full reversal (requires MAJOR CoC)

NEW YORK KILLZONE: 7:00 AM - 10:00 AM EST
- Same scenario logic as London; typically sees manipulation and distribution phases

HARD RULES:
F3.1 Only execute trades during defined killzone windows
F3.2 Use one of the entry strategies (Silver Bullet/CoC/CRT) inside the window
F3.3 Session open = elevated inducement risk; require sweep confirmation before breakout entries

CODABLE: YES - Time-based gates with scenario logic

------------------------------------------------------------------------------
FRAMEWORK F4: SMT DIVERGENCE (CONFLUENCE FILTER)
------------------------------------------------------------------------------
Classification: FILTER/CONFIRMATION (Not standalone entry)

PURPOSE: Add confluence to other entry models by detecting institutional divergence

CORRELATED PAIRS:
- Forex: EUR/USD vs GBP/USD (positive correlation)
- Forex: DXY inverse to EUR/USD and GBP/USD
- Indices: S&P 500 vs NASDAQ vs Dow Jones
- Crypto: Bitcoin vs Ethereum

HARD RULES (FROM ALL IN ONE SR - PREVIOUSLY MISSING):

SMT.1 CORRELATION BASELINE REQUIREMENT:
- Establish synchronized swing points FIRST
- Both pairs must show similar price patterns (matching highs/lows at same times)
- This creates the "normal" baseline against which divergence becomes visible
- Without baseline = cannot detect meaningful divergence

SMT.2 INEFFICIENCY PRECURSOR CONCEPT:
- Large inefficiency on one pair can PRECEDE reversal movement on the other
- Use as early warning signal
- Example: GU pushes lower with significant inefficiency RIGHT BEFORE EU reverses

SMT.3 RELIABILITY FILTER:
- Divergence must occur AT/INTO HTF PD array
- Must agree with market profile direction
- Bullish profile requires bullish divergence; bearish profile requires bearish divergence
- Without this alignment, SMT signal is NOT reliable

SMT.4 CONFLUENCE ONLY:
- SMT is NOT a standalone entry signal
- Use SMT to confirm other entry methods (Silver Bullet, CoC, FVG/OB)
- Entry via primary model; SMT adds confidence

BULLISH DIVERGENCE:
- Pair 1 makes lower low (takes SSL)
- Pair 2 makes higher low (refuses to take SSL)
- Indicates bulls gaining control in Pair 2 (institutional accumulation)

BEARISH DIVERGENCE:
- Pair 1 makes higher high (takes BSL)
- Pair 2 makes lower high (refuses to take BSL)
- Indicates bears gaining control in Pair 2 (institutional distribution)

CODABLE: YES with correlation baseline + alignment requirements


================================================================================
SECTION B: COMPLETE TRADING STRATEGIES (CODABLE ENTRY/EXIT MODELS)
================================================================================

------------------------------------------------------------------------------
STRATEGY S1: SILVER BULLET (LIQUIDITY SWEEP → MSS → FVG)
------------------------------------------------------------------------------
Classification: STRATEGY (Session-based reversal)

PREREQUISITES:
S1.1 Active killzone window (London or NY)
S1.2 Clear external liquidity target (day high/low, session high/low, equal highs/lows)
S1.3 Sweep must occur FIRST (wick beyond liquidity then close back inside)

SETUP LOGIC:
1. Mark current day high/low and session high/low
2. Wait for sweep of liquidity in OPPOSITE direction of intended trade
3. Confirm MSS/CHOCH: candle CLOSE beyond the last internal swing (not wick-only)
4. Identify FVG formed DURING the MSS displacement leg
5. Entry at FVG (or CE midpoint)

ENTRY:
- Limit at FVG boundary (zone must be created during MSS leg)

STOP:
- Beyond sweep extreme OR beyond MSS swing that caused shift

TARGETS:
- Primary: Internal structure / next liquidity pool
- Secondary: Opposite side external liquidity

INVALIDATION:
- Price closes fully through FVG against position
- Price breaks sweep extreme

CODABLE: YES

------------------------------------------------------------------------------
STRATEGY S2: POWER OF THREE / AMD (SESSION PHASE MODEL)
------------------------------------------------------------------------------
Classification: STRATEGY (Intraday phase model)

PHASES:
- Accumulation: Define range (typically Asian session)
- Manipulation: Sweep one side of range (liquidity raid)
- Distribution: Move toward opposite liquidity objective

SETUP LOGIC:
1. Identify accumulation range
2. Wait for manipulation raid (sweep beyond range)
3. Confirm MSS/CHOCH after raid
4. Enter via FVG/OB aligned with distribution direction

ENTRY:
- FVG or OB created during manipulation-to-distribution transition

STOP:
- Beyond raid extreme

TARGETS:
- Opposite side of accumulation range
- Next HTF liquidity

CODABLE: YES

------------------------------------------------------------------------------
STRATEGY S3: CANDLE RANGE THEORY (CRT) (3-CANDLE PATTERN)
------------------------------------------------------------------------------
Classification: STRATEGY (Time-based raid-and-return)

CORE PATTERN:
- Candle 1: Defines range (high and low)
- Candle 2: Sweeps beyond Candle 1 extreme, closes BACK INSIDE range
- Candle 3: Confirms direction (entry trigger or LTF confirmation)

SETUP LOGIC:
1. Identify key institutional time (1am, 5am, 9am, 1pm, 3pm, 6pm, 9pm)
2. Mark Candle 1 range
3. Candle 2 must sweep beyond range AND close back inside
4. If sweep doesn't close back inside = genuine breakout, not CRT

ENTRY:
- Conservative: Wait for LTF FVG/OB in Candle 3 direction
- Aggressive: Enter at Candle 2 close (inside range)

STOP:
- Beyond sweep extreme (Candle 2 wick)

TARGETS:
- First: Range midpoint (conservative)
- Second: Opposite range boundary

CODABLE: YES

------------------------------------------------------------------------------
STRATEGY S4: MAJOR CHANGE OF CHARACTER (CoC) ENTRY
------------------------------------------------------------------------------
Classification: STRATEGY (Reversal entry model)

CRITICAL DISTINCTION - MAJOR vs MINOR CoC (FROM ALL IN ONE SR):

MINOR CoC:
- Breaks structure WITHOUT creating BOS
- LOW probability setup
- SKIP - Do not trade

MAJOR CoC:
- Breaks structure AND creates BOS (break of structure)
- HIGH probability setup
- VALID for trading

RULE: Only MAJOR CoC signals trigger entry models. Minor CoC = noise.

PREREQUISITES:
S4.1 Price reaches HTF supply/demand POI
S4.2 MAJOR CoC occurs (break + close beyond most recent structural swing WITH BOS creation)
S4.3 HTF zone mitigation must precede CoC for validity

AGGRESSIVE ENTRY (Single-TF):
1. On analysis TF, identify zone created during MAJOR CoC wave
2. Place limit at zone extreme (lowest for sells, highest for buys)
3. Stop: Few pips beyond zone extreme or CoC swing extreme
4. TP: Nearest unmitigated opposing HTF zone or major liquidity pool

CONSERVATIVE ENTRY (Multi-TF):
Timeframe Rule: Entry TF must be at least 2 steps lower than analysis TF
- 15m analysis → 1m entry
- 1H analysis → 5m entry  
- 4H analysis → 15m entry

1. After MAJOR CoC on HTF, wait for price to retrace to HTF zone
2. On LTF, require another MAJOR CoC in intended direction
3. Identify OB created by LTF CoC wave
4. Enter at LTF OB extreme
5. Stop: Beyond LTF OB extreme
6. TP: HTF opposing zone or local structure liquidity

CODABLE: YES - BOS qualification requirement makes CoC objectively identifiable

------------------------------------------------------------------------------
STRATEGY S5: FLIP ZONE ENTRY (COMPLETE RULES FROM ALL IN ONE SR)
------------------------------------------------------------------------------
Classification: STRATEGY (Role-reversal entry)

CORE CONCEPT:
- Flip = specific supply/demand rejection → test → break sequence
- EVERY flip IS a Change of Character
- NOT every Change of Character IS a flip
- Flip represents distinct institutional setup with specific requirements

FOUR NON-NEGOTIABLE VALIDITY RULES (FROM ALL IN ONE SR):

RULE 1 - HTF PREREQUISITE:
- Price MUST first mitigate AND get rejected from HTF supply/demand zone
- Rejection signifies shift in market sentiment, sets stage for flip
- IF price does NOT mitigate HTF zone but simply creates flip pattern = NOT VALID
- This is institutional prerequisite confirmation

RULE 2 - INEFFICIENCY/DISPLACEMENT REQUIREMENT:
- High-quality flip MUST leave significant inefficiency behind
- Inefficiency = clear noticeable break of structure with swift, powerful movement
- Inefficiency = imbalance (gaps within candles) showing disequilibrium
- Three-candle sequence WITHOUT gaps = efficiency (weak flip)
- Three-candle sequence WITH gaps = inefficiency (strong flip)
- The break must be SHARP and FAST

RULE 3 - PULLBACK/REACTION REQUIREMENT:
- Price MUST react to supply/demand zone and create pullback BEFORE breaking
- The reaction shows the zone was tested
- IF price does NOT create pullback / does NOT show reaction = NOT A VALID FLIP
- "No reaction = no flip" is absolute rule

RULE 4 - UNMITIGATED (ONE-TIME USE):
- Flip zones are ONE-TIME USE only
- Once flip zone is mitigated (touched), it is NO LONGER valid
- Price moves beyond boundaries = inefficiency resolved = loses significance
- Subsequent retests of flip zone may NOT yield same opportunities
- Focus on FIRST entry when price enters flip zone

FLIP FORMATION SEQUENCE (BEARISH - INVERT FOR BULLISH):
1. Price rejects from HTF supply zone (Rule 1 satisfied)
2. Price reaches last demand zone, produces visible reaction/pullback (Rule 3 satisfied)
3. Price breaks through demand zone with speed and displacement leaving inefficiency (Rule 2 satisfied)
4. The zone created by the flip wave = FLIP ZONE (Rule 4: must be unmitigated)

AGGRESSIVE FLIP ENTRY:
1. Identify valid flip zone (all 4 rules met)
2. Place sell limit at LOWEST point of flip supply zone (buy limit at HIGHEST point for bullish)
3. Stop: Few pips beyond flip zone extreme
4. TP: Next unmitigated demand zone (or supply for bullish)

CONSERVATIVE FLIP ENTRY (MULTI-TF):
1. Identify flip on HTF
2. Zoom to LTF
3. Wait for price to return to flip zone AND demonstrate MAJOR CoC on LTF
4. Identify OB created by LTF CoC wave
5. Enter at LTF OB extreme
6. Stop: Beyond LTF OB extreme
7. TP: HTF opposing zone or swing low/high

CONTINUATION FLIP PATTERNS:
- Supply-to-Demand Flip: Selling pressure subsides, buyers take control → bullish continuation
- Demand-to-Supply Flip: Buying pressure subsides, sellers take control → bearish continuation
- Zone break indicates lack of opposing momentum = shift in control

CODABLE: YES - All 4 rules have objective criteria

------------------------------------------------------------------------------
STRATEGY S6: OPTIMAL TRADE ENTRY (OTE) (61.8-78.6 RETRACE)
------------------------------------------------------------------------------
Classification: STRATEGY (Trend continuation pullback)

PREREQUISITES:
S6.1 Clear impulse leg defines dealing range
S6.2 Bias aligned with trend direction
S6.3 Fibonacci 50% validation: Impulse must be REAL (not structural inducement)
     - If previous move didn't retrace to 50% before creating impulse, may be fake

SETUP LOGIC:
1. Identify trending market (HH/HL for longs, LH/LL for shorts)
2. Mark most recent impulse (swing low to swing high for bullish)
3. Apply Fibonacci from impulse start to end
4. Wait for retrace into 61.8% - 78.6% zone (OTE zone)
5. Require trigger inside OTE zone: FVG, OB, or MSS on LTF

ENTRY:
- 70.5% level (midpoint of OTE zone)
- OR at FVG/OB that coincides with OTE zone

STOP:
- Beyond impulse origin

TARGETS:
- First: Prior swing high/low
- Second: -27% Fibonacci extension
- Third: -61.8% Fibonacci extension

PREMIUM/DISCOUNT RULE:
- Only buy in DISCOUNT (below 50%)
- Only sell in PREMIUM (above 50%)

CODABLE: YES

------------------------------------------------------------------------------
STRATEGY S7: CONTINUATION ENTRY (TWO METHODS FROM ALL IN ONE SR)
------------------------------------------------------------------------------
Classification: STRATEGY (Momentum-following after missed reversal)

CONTEXT: Trader missed CoC/flip entry OR price action didn't provide entry opportunity

FOUR ESSENTIAL CRITERIA (ALL MUST BE MET - NO EXCEPTIONS):
1. Price must form CoC or flip pattern after reaching HTF supply/demand zone
2. Must patiently await BOS and zone breakout in price movement
3. Price should leave inefficiency behind when forming BOS
4. Presence of static liquidity zones near identified POI (double/triple bottoms, trendline liquidity)

METHOD 1: BOS-CREATED POI CONTINUATION
1. Confirm trend direction
2. After BOS, identify supply/demand zone created by the BOS leg
3. DO NOT place limit order at newly generated zone immediately (inducement risk)
4. Wait for pullback + liquidity zone formation (double bottom, triple bottom, trendline liquidity)
5. Wait for liquidity sweep confirmation BEFORE zone taps
6. Enter at zone (first touch) in trend direction

CRITICAL CONSTRAINT (FROM ALL IN ONE SR):
- DO NOT open new positions once price reaches next unmitigated opposing HTF zone
- This is risk boundary enforcement
- Example: Long positions only allowed UNTIL price hasn't reached unmitigated supply zone

STOP:
- Below zone (longs) / Above zone (shorts)

TARGET:
- Next unmitigated opposing zone

METHOD 2: ZONE-BREAK CONTINUATION
1. Price reaches HTF supply/demand zone
2. Price creates tiny pullback (small reaction)
3. Price breaks through the zone (no reversal, just continuation)
4. Identify new zone created by the wave that broke through (role shift)
5. Enter in original direction at that newly formed zone

STOP:
- Beyond new zone

TARGET:
- Original direction liquidity

MULTIPLE POSITION MANAGEMENT RULES:
- Maximum 3 open positions simultaneously
- Position sizing: First entry = full size, second entry = half, third entry = half
- Example: 2 lots first → 1 lot second → 1 lot third
- Worst case (all stopped): controlled drawdown

CODABLE: YES - Four criteria + constraint on new positions make this rule-based

------------------------------------------------------------------------------
STRATEGY S8: TURTLE SOUP (FALSE BREAKOUT + FVG)
------------------------------------------------------------------------------
Classification: STRATEGY (Liquidity raid continuation)

PREREQUISITES:
S8.1 Clear trend direction on HTF
S8.2 Equal highs/lows present (double/triple formations)
S8.3 FVG positioned between current price and liquidity pool

SETUP LOGIC:
1. Identify trend (bullish for long turtle soup)
2. Locate equal lows (bullish) or equal highs (bearish) as liquidity target
3. Identify FVG resting above liquidity (bullish) or below (bearish)
4. Wait for price to sweep liquidity beyond equal levels
5. Watch for rejection at FVG zone
6. Enter when price returns inside previous range

ENTRY:
- Bullish: Price sweeps below equal lows, rejects at FVG, returns inside range
- Bearish: Price sweeps above equal highs, rejects at FVG, returns inside range

STOP:
- Beyond FVG zone

TARGET:
- Next significant liquidity in trend direction

CODABLE: YES

------------------------------------------------------------------------------
STRATEGY S9: LIQUIDITY SWEEP + OB/FVG COMBO (GENERIC REVERSAL)
------------------------------------------------------------------------------
Classification: STRATEGY (Non-session-specific reversal)

PREREQUISITES:
S9.1 External liquidity pool exists
S9.2 Sweep occurs (wick beyond, close inside)
S9.3 MSS/CHOCH after sweep
S9.4 OB or FVG forms during displacement

SETUP LOGIC:
1. Identify external liquidity (swing highs/lows, equal levels)
2. Wait for sweep (wick beyond, immediate close back)
3. Confirm MAJOR CoC (break + close with BOS creation)
4. Mark OB or FVG created during displacement leg
5. Enter at OB/FVG

ENTRY:
- Limit at OB or FVG created by displacement

STOP:
- Beyond OB/FVG invalidation point
- Conservative: Also beyond sweep extreme

TARGET:
- Nearest opposing liquidity pool
- Next HTF PD array

CODABLE: YES

------------------------------------------------------------------------------
STRATEGY S10: EXTERNAL LIQUIDITY SWEEP ENTRY
------------------------------------------------------------------------------
Classification: STRATEGY (Highest-probability sweep model)

CONTEXT: Focus on EXTERNAL liquidity sweeps (trading range boundaries) for highest probability

PREREQUISITES:
S10.1 Trading range clearly defined from HTF (1-2 TF above entry)
S10.2 External liquidity identified (range high/low, session extremes)
S10.3 High-volume session window active (before/after major session open)

SETUP LOGIC:
1. Identify trading range bounded by last valid BOS impulse
2. Mark external liquidity: Range HIGH = buyside, Range LOW = sellside
3. Wait for price to sweep external liquidity
4. Sweep confirmation: Price takes level, immediately closes back inside
5. Once confirmed, price drives toward OPPOSITE external liquidity
6. Enter at MSS/FVG/OB formed during sweep reversal

INTERNAL vs EXTERNAL SWEEP RULE:
- EXTERNAL sweeps = HIGHEST probability (focus here)
- INTERNAL sweeps = LOWER probability (often inducements/traps)
- EXCEPTION: Internal sweep + unmitigated FVG on opposite side = INCREASED probability

ENTRY:
- After buyside sweep: Sell at bearish FVG/OB from reversal
- After sellside sweep: Buy at bullish FVG/OB from reversal

STOP:
- Beyond sweep wick/extreme

TARGET:
- Opposite external liquidity pool

CODABLE: YES

------------------------------------------------------------------------------
STRATEGY S11: CISD (CHANGE IN STATE OF DELIVERY)
------------------------------------------------------------------------------
Classification: STRATEGY (Momentum reversal)

PATTERN:
- Strong directional delivery abruptly shifts
- Creates opposite FVG and role flip
- Support becomes resistance (or vice versa)

PREREQUISITES:
S11.1 HTF POI tap or liquidity sweep first
S11.2 Sudden momentum shift visible
S11.3 MSS/CHOCH confirmation after shift

SETUP LOGIC:
1. Identify extended trend approaching HTF level
2. Wait for sudden momentum reversal
3. Confirm support/resistance flip (price closes through)
4. Mark IFVG or FVG created during shift
5. Mark OB at shift origin
6. Enter on pullback to flipped zone

ENTRY:
- At IFVG, displaced FVG, or OB created in shift leg

STOP:
- Beyond shift swing extreme

TARGET:
- Opposite liquidity pool (swept liquidity becomes history; target fresh external)

CODABLE: YES

------------------------------------------------------------------------------
STRATEGY S12: FVG IN FVG (TWO-TIMEFRAME FVG ENTRY)
------------------------------------------------------------------------------
Classification: STRATEGY (Dual-timeframe precision)

FRAMEWORK:
- HTF (4H/1H): Analysis - structure, direction, FVG identification
- LTF (15M/5M): Confirmation and precise entry
- Entry TF must be at least 2 timeframes lower than analysis TF

SETUP LOGIC:
1. Analyze market structure on HTF
2. Mark HTF FVG in DISCOUNT zone (longs) or PREMIUM zone (shorts)
3. Apply retracement tool to identify 50% level
4. Wait for price to enter HTF FVG zone
5. Zoom to LTF
6. Look for: Liquidity sweep, CHOCH signal, NEW LTF FVG formation
7. Enter at beginning of LTF FVG zone

ENTRY:
- Buy limit at LTF bullish FVG after LTF CHOCH within HTF FVG zone
- Sell limit at LTF bearish FVG after LTF CHOCH within HTF FVG zone

STOP:
- Small FVG: Beyond HTF FVG extreme (larger stop)
- Large FVG: At LTF FVG extreme or HTF FVG midpoint (tighter stop)

TARGET:
- First important structure level on HTF

CODABLE: YES

------------------------------------------------------------------------------
STRATEGY S13: ORDER BLOCK ENTRY (STANDARD)
------------------------------------------------------------------------------
Classification: STRATEGY (Institutional zone entry)

FOUR RULES FOR VALID OB:
1. IMBALANCE: Must have one-sided institutional move away
2. BOS: Must break structural level after leaving OB
3. INEFFICIENCY: Zone must contain FVG/gap showing unfilled orders
4. UNMITIGATED: First touch only; previously tested OBs invalid

SETUP LOGIC:
1. Identify last opposite-color candle before displacement
2. Verify all 4 rules are met
3. Mark OB boundaries (use wick if shadow > body)
4. Wait for first retest

ENTRY:
- Limit at OB extreme

STOP:
- Beyond OB

TARGET:
- Next liquidity pool or opposing PD array

CODABLE: YES

------------------------------------------------------------------------------
STRATEGY S14: BREAKER BLOCK ENTRY
------------------------------------------------------------------------------
Classification: STRATEGY (Failed OB role-flip)

PATTERN:
- OB fails (price breaks through)
- OB flips role (support becomes resistance or vice versa)
- Entry on retest from opposite side

FORMATION (BULLISH BREAKER):
- Swing low → Swing high → Lower low with expansion opposite direction
- Former resistance becomes support

SETUP LOGIC:
1. Identify failed OB (price broke through with close beyond)
2. Mark breaker zone (area between original swing and failure point)
3. Wait for retest from opposite side
4. Enter at breaker zone

ENTRY:
- Buy limit at lowest point of bullish breaker
- Sell limit at highest point of bearish breaker

STOP:
- Beyond breaker zone

TARGET:
- Next liquidity in new direction

CODABLE: YES

------------------------------------------------------------------------------
STRATEGY S15: INVERSE FVG (IFVG) ROLE-FLIP
------------------------------------------------------------------------------
Classification: STRATEGY (Failed FVG entry)

PATTERN:
- FVG is violated (price closes completely through)
- Role flips (bullish FVG becomes resistance, bearish FVG becomes support)
- Entry on retrace to IFVG

SETUP LOGIC:
1. Identify FVG on analysis timeframe
2. Price breaks straight through FVG with no reaction
3. Mark as IFVG (role inverted)
4. Wait for price to retrace back to IFVG
5. Enter at IFVG with expectation of continuation in new direction

ENTRY:
- Buy limit at bullish IFVG (former bearish FVG)
- Sell limit at bearish IFVG (former bullish FVG)

STOP:
- Beyond IFVG zone

TARGET:
- Next liquidity in new direction

FVG BEHAVIOR RULE:
- FVG RESPECT (rejection at midpoint): Continuation expected in rejection direction
- FVG DISRESPECT (break through): IFVG formation, role flips

CODABLE: YES

------------------------------------------------------------------------------
STRATEGY S16: CAMERON'S MODEL (DRAW → STOP-RAID → FVG)
------------------------------------------------------------------------------
Classification: STRATEGY (Three-component setup)

THREE COMPONENTS:
1. DRAW ON LIQUIDITY: Target price is moving toward (key level)
2. STOP RATE: Opposite swing that gets taken out (confirms institutional activity)
3. ENTRY: FVG formed after stop rate sweep

SETUP LOGIC:
1. Identify draw on liquidity (equal highs/lows, swing level, session extreme)
2. Identify stop rate (opposite direction swing)
3. Wait for stop rate to be taken (sweep confirmation)
4. Mark FVG formed after sweep
5. Enter at FVG

ENTRY:
- Buy limit at bullish FVG after sellside (stop rate) taken
- Sell limit at bearish FVG after buyside (stop rate) taken

STOP:
- Beyond stop rate swing

TARGET:
- Draw on liquidity level

CODABLE: YES

------------------------------------------------------------------------------
STRATEGY S17: 50 EMA PULLBACK (TREND CONTINUATION)
------------------------------------------------------------------------------
Classification: STRATEGY (Moving average confluence)

HARDENED RULES (FOR CODABILITY):
- Trend definition: HH/HL for longs, LH/LL for shorts on chosen TF
- EMA condition: Price closes above 50 EMA for longs (below for shorts)
- Entry requires PD array alignment (OB/FVG at EMA)

SETUP LOGIC:
1. Identify clear trend (structure-based)
2. Confirm price above/below 50 EMA
3. Wait for first pullback to 50 EMA
4. Require PD array (OB or FVG) at pullback zone
5. Enter at pullback PD array

ENTRY:
- At OB/FVG that aligns with 50 EMA pullback

STOP:
- Beyond pullback swing

TARGET:
- Next liquidity pool / swing

CODABLE: YES (with structure-based trend definition + PD array requirement)


================================================================================
SECTION C: COMPONENT REFERENCE (BUILDING BLOCKS)
================================================================================

FAIR VALUE GAP (FVG):
- Definition: Three-candle formation; wicks of candles 1 and 3 don't overlap
- Quality filter: Must be in discount (longs) or premium (shorts)
- Refinement: Zoom lower TF for precise FVG inside larger zone

ORDER BLOCK (OB):
- Definition: Last opposite-color candle before displacement
- Four rules: Imbalance, BOS, Inefficiency, Unmitigated
- Wick rule: If shadow > body, use shadow as zone

LIQUIDITY:
- External: Outside trading range (highest probability targets)
- Internal: Inside trading range (fuel for external moves)

MARKET STRUCTURE:
- BOS: Break beyond swing in trend direction (wick acceptable)
- CHOCH/MSS: Break beyond swing OPPOSITE to trend (requires close)
- Major CoC: BOS-qualified (high probability)
- Minor CoC: Non-BOS-qualified (low probability, skip)


================================================================================
SECTION D: DEDUPED STRATEGY MENU FOR AI AGENT
================================================================================

When suggesting strategies, use this compact menu:

REVERSAL STRATEGIES:
- Silver Bullet (session-based, sweep→MSS→FVG)
- Major CoC Entry (HTF POI reversal)
- Flip Zone Entry (4-rule validated)
- Turtle Soup (equal high/low sweep)
- CISD (momentum shift)
- IFVG (role-flip entry)
- External Liquidity Sweep (range boundary)

CONTINUATION STRATEGIES:
- OTE Fibonacci (61.8-78.6 pullback)
- CRT (3-candle raid-and-return)
- Continuation Method 1 (BOS-OB with position constraint)
- Continuation Method 2 (Zone-break)
- 50 EMA Pullback (trend + MA confluence)

FRAMEWORKS/FILTERS (Apply before entry):
- Top-Down Bias (direction + POI selection)
- Daily Bias (daily direction gate)
- Killzones (time gate)
- SMT Divergence (correlation confluence with baseline + inefficiency precursor)


================================================================================
SECTION E: VALIDATION REQUIREMENT
================================================================================

HARD RULE:
- Backtest at least 100 occurrences per strategy, per market, per timeframe
- If validation evidence cannot be produced, label outputs as "HYPOTHESES" only
- Track performance by strategy type, not just overall results


================================================================================
END OF DOCUMENT
================================================================================

Version: 2.0 Enhanced
Changes from v1.0:
1. Reclassified frameworks vs strategies (addressed over-claiming)
2. Added Flip Zone 4 non-negotiable rules from ALL IN ONE SR
3. Added Continuation Method 1/Method 2 with explicit constraints
4. Added SMT correlation baseline + inefficiency precursor requirements
5. Added Major vs Minor CoC distinction (BOS-qualified)
6. Added External vs Internal Liquidity Sweep distinction
7. Added Four Essential Criteria for continuation trades
8. Added position management rules (3 max, halving rule)
9. Unified Supply/Demand Flip into Flip Zone Strategy
10. Added validation requirement
