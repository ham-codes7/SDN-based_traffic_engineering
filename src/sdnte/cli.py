"""Command-line interface: `uv run sdnte --help`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from sdnte.experiments import run_episode
from sdnte.routing import BASELINES, make_te_app
from sdnte.topology import REPO_ROOT, Topology
from sdnte.traffic import PATTERNS, generate_sequence

app = typer.Typer(help="SDN traffic engineering with deep reinforcement learning.", no_args_is_help=True)
console = Console()

TABLE_COLUMNS = ["step", "mlu", "mean_util", "loss_pct", "throughput_ratio", "avg_delay_ms", "decision_ms"]


@app.command()
def topologies() -> None:
    """List the available topologies."""
    table = Table("name", "switches", "links (undirected)", "capacity range (Mbps)")
    for name in Topology.available():
        t = Topology.load(name)
        table.add_row(name, str(t.num_nodes), str(t.num_links // 2), f"{t.capacities.min():g} - {t.capacities.max():g}")
    console.print(table)


@app.command()
def simulate(
    topo: str = typer.Option("toy6", help=f"Topology: {', '.join(Topology.available())}"),
    method: str = typer.Option("ospf_hop", help=f"TE method: {', '.join(BASELINES)}"),
    pattern: str = typer.Option("gravity", help=f"Traffic pattern: {', '.join(PATTERNS)}"),
    load: float = typer.Option(0.6, help="Load relative to hop-count OSPF saturation (1.0 = saturated)"),
    steps: int = typer.Option(20, help="Control intervals in the episode"),
    seed: int = typer.Option(0),
    out: Path | None = typer.Option(None, help="CSV output path (default: results/simulate_*.csv)"),
) -> None:
    """Run one episode of a TE method and print per-step metrics."""
    topology = Topology.load(topo)
    tm_seq = generate_sequence(topology, pattern, steps, load, seed)
    try:
        df = run_episode(topology, make_te_app(method), tm_seq)
    except NotImplementedError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    table = Table(*TABLE_COLUMNS, title=f"{method} on {topo} | {pattern} traffic | load {load}")
    for _, row in df.iterrows():
        table.add_row(str(int(row["step"])), *(f"{row[c]:.3f}" for c in TABLE_COLUMNS[1:]))
    console.print(table)

    mean = df[TABLE_COLUMNS[1:]].mean()
    console.print(
        f"[bold]Mean:[/bold] MLU {mean['mlu']:.3f} | loss {mean['loss_pct']:.2f}% | "
        f"throughput {mean['throughput_ratio']:.3f} | delay {mean['avg_delay_ms']:.2f} ms | "
        f"flow rules {int(df['flow_rules'].iloc[-1])}"
    )

    out = out or REPO_ROOT / "results" / f"simulate_{topo}_{method}_{pattern}_load{load}_seed{seed}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    console.print(f"Saved {out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}")


def _not_yet(what: str, week: str, owner: str) -> None:
    console.print(f"[yellow]{what} is not implemented yet ({week}, owner {owner}).[/yellow]")
    raise typer.Exit(1)


@app.command()
def train(topo: str = "abilene", algo: str = "ppo") -> None:
    """Train a DRL TE agent (PPO or A2C)."""
    _not_yet("Training", "week 3", "P2")


@app.command("run-all")
def run_all() -> None:
    """Run the full experiment grid from configs/default.yaml."""
    _not_yet("The experiment grid", "week 4", "P3")


@app.command()
def plots() -> None:
    """Generate report figures from results/ into figures/."""
    _not_yet("Plotting", "week 4", "P3")


if __name__ == "__main__":
    app()
