"""Turn the Hackathon Data CSVs into readable per-match / per-player summaries.

Writes .md files into ../data_processed/ (relative to the CoachEYE repo).
Focuses on Northwestern's 2024 season and their Big Ten opponents.
"""
import os
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT.parent / "Hackathon Data"
OUT_DIR = REPO_ROOT / "data_processed"

TEAM_CSV = DATA_ROOT / "team-data" / "big-ten-combined-data.csv"
PLAYER_CSV = DATA_ROOT / "player-data" / "player-combined-data.csv"
GENERAL_PLAYER_CSV = DATA_ROOT / "player-data" / "general-player-data.csv"
TEAM_GLOSSARY = DATA_ROOT / "team-data" / "variable_name_glossary.md"
PLAYER_GLOSSARY = DATA_ROOT / "player-data" / "variable_name.md"


def slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s or "untitled"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def build_team_match_reports() -> int:
    df = pd.read_csv(TEAM_CSV)
    df["team"] = df["team"].str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    count = 0
    for (match, date), grp in df.groupby(["match", "date"], sort=False):
        if grp.shape[0] != 2:
            continue
        row_a, row_b = grp.iloc[0], grp.iloc[1]
        title = f"{match} — {date.strftime('%Y-%m-%d')}"
        lines = [f"# {title}", ""]
        for row in (row_a, row_b):
            lines.append(f"## {row['team']}")
            lines.append(f"- Formation: {row['scheme']}")
            lines.append(
                f"- Score line: {int(row['goals'])} goals scored, "
                f"{int(row['goals_against'])} conceded"
            )
            lines.append(
                f"- xG: {row['xg']:.2f} for, allowed shots against: {int(row['shots_against'])} "
                f"(on target: {int(row['sot_against'])})"
            )
            lines.append(
                f"- Shots: {int(row['shots'])} total, {int(row['sot'])} on target "
                f"({row['sotr']:.1f}% on-target rate), avg distance {row['mean_shot_dist']:.1f} m"
            )
            lines.append(
                f"- Possession: {row['possession']:.1f}%, "
                f"passes {int(row['passes'])} ({row['pass_success_rate']:.1f}% accurate), "
                f"match tempo {row['match_tempo']:.2f}"
            )
            lines.append(
                f"- Attacks: {int(row['posit_attacks'])} positional "
                f"({row['pawsr']:.1f}% ended in shot), "
                f"{int(row['counters'])} counters "
                f"({row['countwsr']:.1f}% ended in shot)"
            )
            lines.append(
                f"- Set pieces: {int(row['corners'])} corners "
                f"({row['cornwsr']:.1f}% led to shot), "
                f"{int(row['free_kicks'])} free kicks "
                f"({row['fkwsr']:.1f}% led to shot)"
            )
            lines.append(
                f"- Crosses: {int(row['crosses'])} ({row['acc_cross_rate']:.1f}% accurate); "
                f"box entries: {int(row['box_entries'])} "
                f"({int(row['box_entries_run'])} by run, {int(row['box_entries_cross'])} by cross); "
                f"touches in box: {int(row['touches_in_box'])}"
            )
            lines.append(
                f"- Defensive duels: {int(row['def_duels'])} attempted, "
                f"{row['defdwr']:.1f}% won; "
                f"aerial duels: {int(row['air_duels'])} ({row['airdwr']:.1f}% won); "
                f"interceptions: {int(row['interceptions'])}, "
                f"clearances: {int(row['clearences'])}"
            )
            lines.append(
                f"- Pressing: PPDA {row['ppda']:.2f}; "
                f"recoveries by third — low {int(row['recovery_low'])}, "
                f"mid {int(row['recovery_med'])}, high {int(row['recovery_high'])}"
            )
            lines.append(
                f"- Losses by third — low {int(row['loss_low'])}, "
                f"mid {int(row['loss_med'])}, high {int(row['loss_high'])}"
            )
            lines.append(
                f"- Progressive passes: {int(row['prog_passes'])} "
                f"({row['succ_prog_passes_rate']:.1f}% completed); "
                f"final-third passes: {int(row['fin_third_passes'])} "
                f"({row['succ_fin_third_passes_rate']:.1f}% completed)"
            )
            lines.append("")
        fname = f"match__{slugify(match)}__{date.strftime('%Y-%m-%d')}.md"
        write(OUT_DIR / "team-matches" / fname, "\n".join(lines))
        count += 1
    return count


def build_northwestern_player_reports() -> int:
    df = pd.read_csv(PLAYER_CSV)
    df = df[df["team"].str.strip().str.lower() == "northwestern wildcats"].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    count = 0
    for player, grp in df.groupby("player_name"):
        grp = grp.sort_values("Date", ascending=False)
        totals = grp[
            [
                "Minutes played",
                "goals",
                "assists",
                "shots",
                "shots_on_target",
                "xg",
                "xa",
                "total_passes",
                "total_passes_completed",
                "dribbles",
                "successful_dribbles",
                "total_duels",
                "total_duels_won",
                "interceptions",
                "recoveries",
                "losses",
                "fouls",
            ]
        ].sum(numeric_only=True)
        lines = [
            f"# Northwestern 2024 — {player}",
            "",
            f"Matches logged: {len(grp)}",
            f"Total minutes: {int(totals['Minutes played'])}",
            f"Goals: {int(totals['goals'])}, xG: {totals['xg']:.2f}",
            f"Assists: {int(totals['assists'])}, xA: {totals['xa']:.2f}",
            f"Shots: {int(totals['shots'])} ({int(totals['shots_on_target'])} on target)",
            f"Passes: {int(totals['total_passes'])} attempted, "
            f"{int(totals['total_passes_completed'])} completed",
            f"Dribbles: {int(totals['dribbles'])} attempted, "
            f"{int(totals['successful_dribbles'])} successful",
            f"Duels: {int(totals['total_duels'])} attempted, "
            f"{int(totals['total_duels_won'])} won",
            f"Defensive actions: {int(totals['interceptions'])} interceptions, "
            f"{int(totals['recoveries'])} recoveries",
            f"Losses: {int(totals['losses'])}; fouls committed: {int(totals['fouls'])}",
            "",
            "## Match log (most recent first)",
        ]
        for _, row in grp.iterrows():
            date = row["Date"].strftime("%Y-%m-%d") if pd.notna(row["Date"]) else "?"
            lines.append(
                f"- {date} — {row['Match']} — position {row['Position']}, "
                f"{int(row['Minutes played'])} min, "
                f"{int(row['goals'])}G/{int(row['assists'])}A, "
                f"xG {row['xg']:.2f}, xA {row['xa']:.2f}, "
                f"{int(row['shots'])} shots ({int(row['shots_on_target'])} on target)"
            )
        write(OUT_DIR / "players-northwestern" / f"{slugify(player)}.md", "\n".join(lines))
        count += 1
    return count


def build_roster() -> None:
    df = pd.read_csv(GENERAL_PLAYER_CSV)
    df = df[df["Team"].str.strip().str.lower() == "northwestern"].copy()
    lines = ["# Northwestern Men's Soccer — 2024 Roster", ""]
    for _, r in df.sort_values("Number").iterrows():
        weight = f", {int(r['Weight'])} lb" if pd.notna(r["Weight"]) else ""
        lines.append(
            f"- #{int(r['Number'])} {r['Name']} — {r['Position']}, "
            f"{r['Height']}, {r['Class']}{weight}"
        )
    write(OUT_DIR / "roster" / "northwestern-2024.md", "\n".join(lines))


def copy_glossaries() -> None:
    for src in (TEAM_GLOSSARY, PLAYER_GLOSSARY):
        if src.exists():
            write(OUT_DIR / "glossary" / src.name, src.read_text(encoding="utf-8"))


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Writing to", OUT_DIR)
    n_matches = build_team_match_reports()
    print(f"  team-match summaries: {n_matches}")
    n_players = build_northwestern_player_reports()
    print(f"  Northwestern player summaries: {n_players}")
    build_roster()
    print("  roster: 1")
    copy_glossaries()
    print("  glossaries: 2")


if __name__ == "__main__":
    main()
