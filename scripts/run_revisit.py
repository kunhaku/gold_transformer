from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from pprint import pprint
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import DataConfig, ModelConfig


def _disable_visual_tool() -> None:
    """Monkey patch the visual tool to avoid launching Dash during CLI runs."""

    try:
        visual_tool = importlib.import_module("visual_tool")
    except ModuleNotFoundError:
        return

    def _noop(*args, **kwargs) -> None:
        print("visual_tool disabled for CLI run")

    visual_tool.run_visual_tool = _noop  # type: ignore[attr-defined]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the revisit workflow from the CLI.")
    parser.add_argument("--epochs", type=int, help="Override the number of training epochs.")
    args = parser.parse_args()

    data_config = DataConfig()
    model_config = ModelConfig()

    if args.epochs is not None:
        model_config.epochs = args.epochs

    _disable_visual_tool()

    from pipelines.revisit_pipeline import run_revisit_workflow

    result = run_revisit_workflow(data_config=data_config, model_config=model_config)

    print("workflow keys:", list(result.keys()))
    decision = result.get("decision") or {}
    print("decision keys:", list(decision.keys()))
    thesis_summaries = decision.get("thesis_summaries") or {}
    print("num thesis:", len(thesis_summaries))
    print("inference metrics:")
    pprint(result.get("inference_metrics"))


if __name__ == "__main__":
    main()
