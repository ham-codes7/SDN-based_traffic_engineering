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
    method: str = typer.Option("ospf_hop", help=f"TE method: {', '.join(BASELINES)}, or a trained agent (ppo, a2c, ppo_gravity)"),
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
        df = run_episode(topology, make_te_app(method, topo), tm_seq)
    except FileNotFoundError as e:
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


@app.command()
def train(
    topo: list[str] = typer.Option(None, help="Topology (repeatable). Default: experiments.topologies in the config"),
    agent: list[str] = typer.Option(None, help="Agent from training.agents (repeatable). Default: all"),
    timesteps: int | None = typer.Option(None, help="Override training.timesteps"),
    seed: int = typer.Option(0),
) -> None:
    """Train DRL TE agents (PPO, A2C, ...) and save them to models/."""
    from sdnte.agents import train as train_agent
    from sdnte.config import load_config

    config = load_config()
    for t in topo or config["experiments"]["topologies"]:
        for name in agent or list(config["training"]["agents"]):
            console.print(f"Training [bold]{name}[/bold] on [bold]{t}[/bold] ...")
            path = train_agent(t, name, config, timesteps=timesteps, seed=seed)
            console.print(f"  saved {path.relative_to(REPO_ROOT)}")


def _not_yet(what: str, week: str, owner: str) -> None:
    console.print(f"[yellow]{what} is not implemented yet ({week}, owner {owner}).[/yellow]")
    raise typer.Exit(1)


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
