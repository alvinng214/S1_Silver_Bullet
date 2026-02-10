from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RUN_OUTPUT = Path("backtrader_run_output.txt")
CSV_REPORT = Path("backtest_hk_trade_report.csv")
MD_REPORT = Path("backtest_hk_detailed_report.md")


def parse_trade_lines(text: str) -> pd.DataFrame:
    pattern = re.compile(
        r"\s*(\d+)\s+(LONG|SHORT)\s+"
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
        r"\$([\-0-9.]+)\s+(WIN|LOSS)\s+(.+?)\s*$"
    )

    rows: list[dict] = []
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        trade_num, direction, entry_time, entry, stop, target, pnl, result, signal = match.groups()
        rows.append(
            {
                "trade_num": int(trade_num),
                "direction": direction,
                "entry_time_utc": entry_time,
                "entry": float(entry),
                "stop_loss": float(stop),
                "take_profit": float(target),
                "pnl": float(pnl),
                "result": result,
                "signal": signal.strip(),
            }
        )

    if not rows:
        raise ValueError("No trades parsed from backtrader output.")

    df = pd.DataFrame(rows).sort_values("trade_num")
    df["entry_time_hk"] = (
        pd.to_datetime(df["entry_time_utc"], utc=True)
        .dt.tz_convert("Asia/Hong_Kong")
        .dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    )
    return df


def write_markdown_report(df: pd.DataFrame) -> None:
    wins = int((df["result"] == "WIN").sum())
    losses = int((df["result"] == "LOSS").sum())
    total = int(len(df))

    lines: list[str] = [
        "# Silver Bullet Backtrader Report (PEPPERSTONE_XAUUSD, 5m)",
        "",
        f"- Total trades: **{total}**",
        f"- Wins: **{wins}**",
        f"- Losses: **{losses}**",
        f"- Win rate: **{(wins / total * 100):.2f}%**",
        "",
        "## Performance by signal",
    ]

    for signal, grouped in df.groupby("signal"):
        signal_wins = int((grouped["result"] == "WIN").sum())
        signal_losses = int((grouped["result"] == "LOSS").sum())
        signal_total = len(grouped)
        signal_win_rate = signal_wins / signal_total * 100 if signal_total else 0
        lines.append(
            f"- {signal}: {signal_wins} wins / {signal_losses} losses "
            f"({signal_win_rate:.2f}% win rate)"
        )

    lines.extend(
        [
            "",
            "## Trade details (entry time in HK time)",
            "",
            "| # | Side | Entry time (HK) | Entry signal | Entry | Stop loss | Take profit | Result | PnL |",
            "|---:|:-----|:----------------|:-------------|------:|----------:|------------:|:------|----:|",
        ]
    )

    for _, row in df.iterrows():
        lines.append(
            "| {trade_num} | {direction} | {entry_time_hk} | {signal} | "
            "{entry:.2f} | {stop_loss:.2f} | {take_profit:.2f} | {result} | {pnl:.2f} |".format(
                **row.to_dict()
            )
        )

    MD_REPORT.write_text("\n".join(lines))


def main() -> None:
    text = RUN_OUTPUT.read_text()
    trades = parse_trade_lines(text)
    trades.to_csv(CSV_REPORT, index=False)
    write_markdown_report(trades)


if __name__ == "__main__":
    main()
