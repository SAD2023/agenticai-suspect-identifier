"""

  python main.py train   --data <path>
      Train the model on a labeled dataset. Must be run at least once before
      the 'predict' command will work. Requires FEATURE_COLUMNS and
      TARGET_COLUMN to be defined in config.py.

  python main.py predict --input <path> [--output <path>] [--format excel|pdf]
      Apply the trained model to a list of individuals. Produces a
      color-coded, prioritized output report for analyst review.

"""

import argparse
import logging
import os
import sys
import pandas as pd
from datetime import datetime

from config import FEATURE_CONFIG, FILE_CONFIG

os.makedirs("logs", exist_ok=True)
log_filename = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)



def cmd_train(args: argparse.Namespace) -> None:
    """
    Handler for:  python main.py train --data <path>

    Args:
        args.data (str): Path to the labeled training dataset (CSV or Excel).
    """
    feature_columns     = FEATURE_CONFIG["FEATURE_COLUMNS"]
    target_column       = FEATURE_CONFIG["TARGET_COLUMN"]
    categorical_columns = FEATURE_CONFIG["CATEGORICAL_COLUMNS"]

    # Guard: fail early with a clear message if config.py has not been filled in.
    if not feature_columns:
        logger.error(
            "FEATURE_COLUMNS is empty in config.py.\n"
            
        )
        sys.exit(1)

    if target_column is None:
        logger.error(
            "TARGET_COLUMN is None in config.py.\n"
            
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE STARTING")
    logger.info(f"  Dataset:              {args.data}")
    logger.info(f"  Feature columns ({len(feature_columns)}): {feature_columns}")
    logger.info(f"  Target column:        {target_column}")
    logger.info(f"  Categorical columns:  {categorical_columns or '(none)'}")
    logger.info("=" * 60)

    from model_trainer import run_training_pipeline
    run_training_pipeline(
        training_data_path  = args.data,
        feature_columns     = feature_columns,
        target_column       = target_column,
        categorical_columns = categorical_columns,
    )

    logger.info("Training complete. Run 'python main.py predict --input <file>' to score individuals.")


def cmd_predict(args: argparse.Namespace) -> None:
    """
    Handler for:  python main.py predict --input <path> [--output <path>] [--format excel|pdf]

    Args:
        args.input  (str): Path to the analyst input file (.xlsx, .csv, or .pdf).
        args.output (str): Optional output file path. Auto-generated if not provided.
        args.format (str): Output format — "excel" (default) or "pdf".
    """
    input_path  = args.input
    fmt         = args.format.lower()
    ext         = ".xlsx" if fmt == "excel" else ".pdf"

    if args.output:
        output_path = args.output
    else:
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            FILE_CONFIG["DEFAULT_OUTPUT_DIR"],
            f"prioritized_report_{timestamp}{ext}",
        )

    logger.info("=" * 60)
    logger.info("PREDICTION PIPELINE STARTING")
    logger.info(f"  Input file:   {input_path}")
    logger.info(f"  Output file:  {output_path}")
    logger.info(f"  Output format: {fmt.upper()}")
    logger.info("=" * 60)

    from predictor import run_prediction_pipeline
    df_result = run_prediction_pipeline(
        input_filepath  = input_path,
        output_filepath = output_path,
        output_format   = fmt,
    )

    _print_console_summary(df_result)
    print(f"\n  Full report saved to: {output_path}\n")


def cmd_template(args: argparse.Namespace) -> None:
    """
    Handler for:  python main.py template [--output <path>]

    Args:
        args.output (str): Output path for the template file. Defaults to
                           "outputs/input_template.xlsx".
    """
    from create_sample_template import generate_template

    output_path = args.output or os.path.join(
        FILE_CONFIG["DEFAULT_OUTPUT_DIR"], "input_template.xlsx"
    )
    generate_template(output_path)
    print(f"\n  Input template saved to: {output_path}")
    print(
        "  Open this file, fill in rows for each individual to score, "
        "then run:\n"
        f"  python main.py predict --input {output_path}\n"
    )


def _print_console_summary(df: pd.DataFrame) -> None:
    """
    Prints a brief analyst-facing summary to stdout after a prediction run.

    Args:
        df (pd.DataFrame): Final prioritized + explained DataFrame.
    """
    id_col   = FEATURE_CONFIG.get("ID_COLUMN",   "person_id")
    name_col = FEATURE_CONFIG.get("NAME_COLUMN", "name")

    print()
    print("=" * 60)
    print("  PRIORITIZATION COMPLETE")
    print("=" * 60)
    print(f"  Total individuals analyzed:  {len(df):,}")
    print()

    for tier in ["HIGH", "MEDIUM", "LOW"]:
        count = (df["priority_tier"] == tier).sum()
        bar   = "█" * min(count, 30)
        print(f"  {tier:<8s}  {count:>4d}  {bar}")

    print()
    df_high = df[df["priority_tier"] == "HIGH"].head(5)

    if df_high.empty:
        print("  No individuals exceeded the HIGH priority threshold.")
    else:
        print("  Top HIGH priority individuals (up to 5 shown):")
        print(f"  {'Score':<8}  {'ID':<15}  Name")
        print(f"  {'-'*8}  {'-'*15}  {'-'*20}")
        for _, row in df_high.iterrows():
            pid  = str(row.get(id_col,   "N/A"))[:15]
            name = str(row.get(name_col, "N/A"))[:30] if name_col else "N/A"
            print(f"  {row['risk_score']:.4f}    {pid:<15}  {name}")

    print("=" * 60)



def _build_parser() -> argparse.ArgumentParser:
    """
    Constructs the full CLI argument parser with three subcommands:
    'train', 'predict', and 'template'.

    Returns:
        argparse.ArgumentParser: Fully configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Suspect Prioritization System\n"
            "Helps analysts triage large lists of individuals using a trained ML model.\n\n"
            "Typical workflow:\n"
            "  1. Edit config.py to define FEATURE_COLUMNS and TARGET_COLUMN.\n"
            "  2. python main.py train   --data <approved_dataset.csv>\n"
            "  3. python main.py predict --input <todays_list.xlsx>\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # -- train subcommand --
    p_train = sub.add_parser("train", help="Train the model on a labeled dataset.")
    p_train.add_argument(
        "--data",
        required=True,
        metavar="FILE",
        help="Path to the labeled training dataset (.csv, .xlsx, or .xls).",
    )

    # -- predict subcommand --
    p_predict = sub.add_parser(
        "predict",
        help="Score and prioritize a list of individuals.",
    )
    p_predict.add_argument(
        "--input",
        required=True,
        metavar="FILE",
        help="Path to the analyst input file (.xlsx, .xls, .csv, or .pdf).",
    )
    p_predict.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help=(
            "Path for the output report. "
            "Default: outputs/prioritized_report_<timestamp>.<ext>"
        ),
    )
    p_predict.add_argument(
        "--format",
        choices=["excel", "pdf"],
        default="excel",
        metavar="FORMAT",
        help="Output format: 'excel' (default) or 'pdf'.",
    )

    # -- template subcommand --
    p_template = sub.add_parser(
        "template",
        help="Generate a blank Excel input template for analysts.",
    )
    p_template.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Output path for the template. Default: outputs/input_template.xlsx",
    )

    return parser


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    logger.info(f"Command: {args.command}  |  Log: {log_filename}")

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "template":
        cmd_template(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
