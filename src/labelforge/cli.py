"""
Command Line Interface for LabelForge.
"""

import click
import logging
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .datasets import load_example_data, load_from_csv, load_from_jsonl
from .lf import apply_lfs, get_lf_summary, LF_REGISTRY
from .label_model import LabelModel


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    """LabelForge: Programmatic data labeling with weak supervision."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")


@main.command()
@click.option(
    "--dataset",
    "-d",
    default="medical_texts",
    help="Dataset to use (medical_texts, sentiment, spam)",
)
@click.option("--csv-file", help="Load from CSV file instead")
@click.option("--jsonl-file", help="Load from JSONL file instead")
@click.option("--output", "-o", help="Output directory for results")
def run(
    dataset: str,
    csv_file: Optional[str],
    jsonl_file: Optional[str],
    output: Optional[str],
) -> None:
    """Run the full LabelForge pipeline."""
    click.echo(f"🔨 LabelForge v{__version__} - Running pipeline...")

    # Load data
    if csv_file:
        examples = load_from_csv(csv_file)
    elif jsonl_file:
        examples = load_from_jsonl(jsonl_file)
    else:
        examples = load_example_data(dataset)

    click.echo(f"📊 Loaded {len(examples)} examples")

    # Check for registered LFs
    if not LF_REGISTRY:
        click.echo("⚠️  No labeling functions found! Please register some LFs first.")
        click.echo("Example:")
        click.echo("  from labelforge import lf")
        click.echo("  @lf()")
        click.echo("  def my_lf(example):")
        click.echo("      return 1 if 'keyword' in example.text else 0")
        sys.exit(1)

    # Apply LFs
    click.echo(f"🏷️  Applying {len(LF_REGISTRY)} labeling functions...")
    lf_output = apply_lfs(examples)

    # Show LF statistics
    click.echo("\n📈 LF Statistics:")
    coverage = lf_output.coverage()
    for i, lf_name in enumerate(lf_output.lf_names):
        click.echo(f"  {lf_name}: {coverage[i]:.2%} coverage")

    # Train label model
    click.echo("\n🧠 Training label model...")
    label_model = LabelModel(verbose=True)
    label_model.fit(lf_output)

    # Get predictions
    probs = label_model.predict_proba(lf_output)
    preds = label_model.predict(lf_output)

    click.echo(f"✅ Generated predictions for {len(examples)} examples")

    # Save results if output directory specified
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        import numpy as np

        np.save(output_dir / "predictions.npy", preds)
        np.save(output_dir / "probabilities.npy", probs)

        # Save LF stats
        import json

        with open(output_dir / "lf_stats.json", "w") as f:
            json.dump(get_lf_summary(), f, indent=2)

        click.echo(f"💾 Results saved to {output_dir}")

    click.echo("🎉 Pipeline completed successfully!")


@main.command()
def lf_list() -> None:
    """List all registered labeling functions."""
    if not LF_REGISTRY:
        click.echo("No labeling functions registered.")
        return

    click.echo(f"📋 Registered Labeling Functions ({len(LF_REGISTRY)}):")
    for name, lf in LF_REGISTRY.items():
        tags_str = ", ".join(f"{k}={v}" for k, v in lf.tags.items())
        click.echo(f"  • {name}")
        if tags_str:
            click.echo(f"    Tags: {tags_str}")
        click.echo(f"    Description: {lf.description}")
        if lf.n_calls > 0:
            click.echo(
                f"    Coverage: {lf.coverage:.2%}, Error rate: {lf.error_rate:.2%}"
            )


@main.command()
@click.option("--dataset", "-d", default="medical_texts", help="Dataset to test on")
@click.option(
    "--sample-size", "-n", default=10, type=int, help="Number of examples to test"
)
def lf_test(dataset: str, sample_size: int) -> None:
    """Test labeling functions on a sample dataset."""
    if not LF_REGISTRY:
        click.echo("No labeling functions registered.")
        return

    # Load sample data
    examples = load_example_data(dataset)[:sample_size]

    click.echo(f"🧪 Testing {len(LF_REGISTRY)} LFs on {len(examples)} examples...")

    # Apply LFs
    lf_output = apply_lfs(examples)

    # Show results
    click.echo("\n📊 Test Results:")
    click.echo(f"{'Example':<15} {'Text':<50} {'LF Votes':<20}")
    click.echo("-" * 85)

    for i in range(min(10, len(examples))):
        example = examples[i]
        text = example.text[:47] + "..." if len(example.text) > 50 else example.text
        votes = lf_output.votes[i]
        votes_str = str(votes)
        click.echo(f"{example.id:<15} {text:<50} {votes_str:<20}")

    # Show coverage and conflicts
    click.echo("\n📈 Coverage:")
    coverage = lf_output.coverage()
    for i, lf_name in enumerate(lf_output.lf_names):
        click.echo(f"  {lf_name}: {coverage[i]:.2%}")

    click.echo("\n⚔️  Conflicts:")
    conflicts = lf_output.conflict()
    for i, lf1 in enumerate(lf_output.lf_names):
        for j, lf2 in enumerate(lf_output.lf_names):
            if i < j and conflicts[i, j] > 0:
                click.echo(f"  {lf1} vs {lf2}: {conflicts[i, j]:.2%}")


@main.command()
@click.argument("name")
@click.option("--tags", help="Comma-separated tags (e.g., type=regex,domain=medical)")
@click.option("--description", help="Description of the LF")
def lf_create(name: str, tags: Optional[str], description: Optional[str]) -> None:
    """Create a new labeling function template."""
    # Parse tags
    tag_dict = {}
    if tags:
        for tag in tags.split(","):
            if "=" in tag:
                key, value = tag.split("=", 1)
                tag_dict[key.strip()] = value.strip()

    # Generate template
    template = f'''"""
Labeling function: {name}
"""
from labelforge import lf

@lf(name="{name}"'''

    if tag_dict:
        template += f", tags={tag_dict}"

    if description:
        template += f', description="{description}"'

    template += f''')
def {name}(example):
    """
    {description or f"Labeling function: {name}"}

    Args:
        example: Example object with .text, .metadata, .id attributes

    Returns:
        Label (int): 1 for positive, 0 for negative, -1 for abstain
    """
    # TODO: Implement your labeling logic here
    # Example:
    # if "keyword" in example.text.lower():
    #     return 1
    # else:
    #     return 0

    return -1  # Abstain by default
'''

    # Save to file
    filename = f"{name}.py"
    with open(filename, "w") as f:
        f.write(template)

    click.echo(f"✅ Created labeling function template: {filename}")
    click.echo(
        "Edit the file to implement your labeling logic, "
        "then import it to register the LF."
    )


@main.command()
@click.option(
    "--format",
    "output_format",
    default="table",
    type=click.Choice(["table", "json"]),
    help="Output format",
)
def lf_stats(output_format: str) -> None:
    """Show statistics for all registered labeling functions."""
    if not LF_REGISTRY:
        click.echo("No labeling functions registered.")
        return

    stats = get_lf_summary()

    if output_format == "json":
        import json

        click.echo(json.dumps(stats, indent=2))
    else:
        click.echo(f"📊 LF Statistics ({stats['total_lfs']} functions):")
        click.echo(
            f"{'Name':<20} {'Coverage':<10} {'Error Rate':<12} {'Calls':<8} {'Tags'}"
        )
        click.echo("-" * 70)

        for name, lf_stats in stats["lfs"].items():
            coverage = f"{lf_stats['coverage']:.2%}"
            error_rate = f"{lf_stats['error_rate']:.2%}"
            calls = str(lf_stats["n_calls"])
            tags = ", ".join(f"{k}={v}" for k, v in lf_stats["tags"].items())

            click.echo(f"{name:<20} {coverage:<10} {error_rate:<12} {calls:<8} {tags}")


@main.command()
@click.option("--port", "-p", default=8501, help="Port to run the web interface on")
@click.option("--host", "-h", default="localhost", help="Host to run the web interface on")
def web(port: int, host: str) -> None:
    """Launch the interactive web interface."""
    try:
        import streamlit.web.cli as stcli
        import sys
        from pathlib import Path
        
        # Get the path to the web app
        web_app_path = Path(__file__).parent / "web" / "app.py"
        
        if not web_app_path.exists():
            click.echo("❌ Web interface not found. Make sure LabelForge is properly installed.")
            sys.exit(1)
        
        click.echo(f"🚀 Starting LabelForge web interface on {host}:{port}")
        click.echo(f"📂 App location: {web_app_path}")
        click.echo("💡 Press Ctrl+C to stop the server")
        
        # Set up streamlit arguments
        sys.argv = [
            "streamlit",
            "run",
            str(web_app_path),
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        # Run streamlit
        stcli.main()
        
    except ImportError:
        click.echo("❌ Streamlit not installed. Install web dependencies:")
        click.echo("   pip install labelforge[web]")
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error starting web interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
