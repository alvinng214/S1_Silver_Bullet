import sys
import os
sys.path.insert(0, "/home/user/S1_Silver_Bullet")

from Silver_bullet_backtrader import (
    load_data,
    _compute_htf_zones,
    _zones_to_in_zone_series,
)

def find_contiguous_periods(series):
    """Find contiguous True periods in a boolean series."""
    periods = []
    in_period = False
    start = None
    for i, (ts, val) in enumerate(series.items()):
        if val and not in_period:
            start = ts
            in_period = True
            bar_count = 1
        elif val and in_period:
            bar_count += 1
        elif not val and in_period:
            periods.append((start, prev_ts, bar_count))
            in_period = False
        prev_ts = ts
    if in_period:
        periods.append((start, prev_ts, bar_count))
    return periods

def main():
    csv_file = "/home/user/S1_Silver_Bullet/PEPPERSTONE_XAUUSD, 5.csv"
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return

    print("Loading data...")
    df = load_data(csv_file)
    print(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")

    print("\nComputing 1H order block zones...")
    zones_1h = _compute_htf_zones(df, resolution="60")
    print(f"  1H timestamps with zones: {sum(1 for v in zones_1h.values() if v)}")

    print("Computing 4H order block zones...")
    zones_4h = _compute_htf_zones(df, resolution="240")
    print(f"  4H timestamps with zones: {sum(1 for v in zones_4h.values() if v)}")

    bull_1h = _zones_to_in_zone_series(df, zones_1h, side="bull")
    bear_1h = _zones_to_in_zone_series(df, zones_1h, side="bear")
    bull_4h = _zones_to_in_zone_series(df, zones_4h, side="bull")
    bear_4h = _zones_to_in_zone_series(df, zones_4h, side="bear")

    print(f"\n1H Bull bars in OB: {bull_1h.sum()}")
    print(f"1H Bear bars in OB: {bear_1h.sum()}")
    print(f"4H Bull bars in OB: {bull_4h.sum()}")
    print(f"4H Bear bars in OB: {bear_4h.sum()}")

    # Print details for each
    for label, series in [
        ("1H BULLISH Order Block", bull_1h),
        ("1H BEARISH Order Block", bear_1h),
        ("4H BULLISH Order Block", bull_4h),
        ("4H BEARISH Order Block", bear_4h),
    ]:
        periods = find_contiguous_periods(series)
        print(f"\n{'='*80}")
        print(f"  {label} Periods ({len(periods)} total)")
        print(f"{'='*80}")
        if not periods:
            print("  (none)")
        else:
            print(f"  {'#':<4} {'Start':<22} {'End':<22} {'Bars':<6}")
            print(f"  {'-'*54}")
            for idx, (start, end, bars) in enumerate(periods, 1):
                print(f"  {idx:<4} {str(start):<22} {str(end):<22} {bars:<6}")

    # Also print the actual zone levels
    print(f"\n{'='*80}")
    print("  ZONE LEVELS DETAIL")
    print(f"{'='*80}")
    
    for label, zones_dict in [("1H", zones_1h), ("4H", zones_4h)]:
        # Collect all unique zones seen
        all_zones = {}
        for ts, zones in sorted(zones_dict.items()):
            for z in zones:
                key = (z.direction, z.source_time, z.high, z.low)
                if key not in all_zones:
                    all_zones[key] = {"zone": z, "first_seen": ts, "last_seen": ts}
                else:
                    all_zones[key]["last_seen"] = ts
        
        print(f"\n  --- {label} Order Block Zones ---")
        print(f"  {'Dir':<6} {'Source Time':<22} {'High':<12} {'Low':<12} {'Avg':<12} {'Active From':<22} {'Active Until':<22}")
        print(f"  {'-'*108}")
        for key, info in sorted(all_zones.items(), key=lambda x: x[1]["first_seen"]):
            z = info["zone"]
            print(f"  {z.direction:<6} {str(z.source_time):<22} {z.high:<12.2f} {z.low:<12.2f} {z.avg:<12.2f} {str(info['first_seen']):<22} {str(info['last_seen']):<22}")

if __name__ == "__main__":
    main()
