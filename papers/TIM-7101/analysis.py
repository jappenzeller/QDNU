"""
TIM-7101 Paper Analysis
=======================
Starter template for pulling data from QDNU core.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now you can import from qdnu
from qdnu import quantum_agate

# Example: access existing data
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Paper-specific output
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    """Main analysis entry point."""
    print(f"QDNU Project Root: {PROJECT_ROOT}")
    print(f"Data available at: {DATA_DIR}")
    print(f"Output will be saved to: {OUTPUT_DIR}")

    # Add your analysis here
    pass


if __name__ == "__main__":
    main()
