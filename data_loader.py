"""
Handles all file reading for both:

  1. Training data  — A labeled dataset used once during model training.
                      Each row is a historical case with known outcomes.

  2. Analyst input  — A list of individuals submitted by an analyst for
                      daily triage. Accepts .xlsx, .xls, .csv, and .pdf.

PUBLIC FUNCTIONS:
  load_training_data(filepath)          Load a labeled training dataset from disk.
  load_input_file(filepath)             Load an analyst-provided list for scoring.
  validate_dataframe(df, cols, context) Assert that required columns are present.

"""

import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_training_data(filepath: str) -> pd.DataFrame:
    """
    Loads the labeled training dataset from disk.

    The file must be a CSV or Excel spreadsheet where:
      - Each row represents one individual from a historical case.
      - Feature columns contain behavioral indicators as defined in config.py.
      - The target column contains a binary label: 0 (not a suspect) or 1 (suspect).

    Args:
        filepath (str): Path to the training dataset (.csv, .xlsx, or .xls).

    Returns:
        pd.DataFrame: The raw training dataset.

    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Training data file not found: '{filepath}'\n"
            "Check the path and try again."
        )

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        df = _read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        df = _read_excel(filepath)
    else:
        raise ValueError(
            f"Unsupported training data format: '{ext}'. "
            f"Accepted formats: .csv, .xlsx, .xls"
        )

    df = _normalize_columns(df)
    logger.info(
        f"Loaded training data: {df.shape[0]:,} rows × {df.shape[1]} columns "
        f"from '{filepath}'."
    )
    return df


def load_input_file(filepath: str) -> pd.DataFrame:
    """
    Loads an analyst-submitted list of individuals to be scored and prioritized.

    Accepted formats: .xlsx, .xls, .csv, .pdf
    

    The input file must contain:
      - The same feature columns the model was trained on (defined in config.py).
      - An ID column to identify each person in the output report.
      - Optionally a name column for display purposes.

    Args:
        filepath (str): Path to the analyst input file.

    Returns:
        pd.DataFrame: Each row is one individual to be scored.

    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Input file not found: '{filepath}'\n"
            "Ensure the file exists and the path is correct."
        )

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        df = _read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        df = _read_excel(filepath)
    elif ext == ".pdf":
        df = _read_pdf_table(filepath)
    else:
        raise ValueError(
            f"Unsupported input format: '{ext}'. "
            f"Accepted: .xlsx, .xls, .csv, .pdf"
        )

    df = _normalize_columns(df)
    logger.info(
        f"Loaded analyst input: {df.shape[0]:,} individuals "
        f"from '{os.path.basename(filepath)}'."
    )
    return df


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list,
    context: str = "",
) -> None:
    """
    Asserts that a DataFrame contains all required columns.

    Args:
        df               (pd.DataFrame): DataFrame to validate.
        required_columns (list):         Column names that must be present.
        context          (str):          Optional label for the error message,
                                         e.g. "training data" or "input file".
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        ctx_label = f"[{context}] " if context else ""
        raise ValueError(
            f"{ctx_label}The following required columns are missing: {missing}\n"
            f"Columns present in file: {list(df.columns)}\n"
            "Check that the file matches the expected template "
            "(run create_sample_template.py to generate one)."
        )


def _read_excel(filepath: str) -> pd.DataFrame:
    """
    Reads the first sheet of an Excel file (.xlsx or .xls) into a DataFrame.

    Args:
        filepath (str): Path to the Excel file.

    Returns:
        pd.DataFrame
    """
    try:
        return pd.read_excel(filepath, sheet_name=0)
    except Exception as exc:
        raise ValueError(f"Could not read Excel file '{filepath}': {exc}") from exc


def _read_csv(filepath: str) -> pd.DataFrame:
    """
    Reads a CSV file with automatic encoding detection.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame
    """
    try:
        return pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        logger.debug(f"UTF-8 decode failed for '{filepath}'; retrying with Latin-1.")
        return pd.read_csv(filepath, encoding="latin-1")


def _read_pdf_table(filepath: str) -> pd.DataFrame:
    """
    Extracts the first structured table found across the pages of a PDF.


    Args:
        filepath (str): Path to the PDF file.

    Returns:
        pd.DataFrame: The extracted table, with numeric columns coerced.

    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required to read PDF input files.\n"
            "Install it with:  pip install pdfplumber"
        )

    found_table = None
    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages):
            raw = page.extract_table()
            if raw:
                logger.debug(f"Table found on PDF page {page_num + 1}.")
                found_table = raw
                break

    if found_table is None:
        raise ValueError(
            f"No table data found in '{filepath}'.\n"
            "Ensure the PDF contains a structured table with a header row, "
            "and that it is not a scanned/image-only PDF."
        )

    # First row is treated as the header
    headers = [
        str(h).strip() if h else f"col_{i}"
        for i, h in enumerate(found_table[0])
    ]
    rows = found_table[1:]
    df = pd.DataFrame(rows, columns=headers)

    # Coerce columns that look numeric to float/int
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    logger.info(f"Extracted {len(df):,} rows from PDF table in '{filepath}'.")
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strips leading/trailing whitespace from column names.

    Args:
        df (pd.DataFrame): DataFrame with potentially dirty column names.

    Returns:
        pd.DataFrame: Same DataFrame with cleaned column names.
    """
    df.columns = [str(col).strip() for col in df.columns]
    return df
