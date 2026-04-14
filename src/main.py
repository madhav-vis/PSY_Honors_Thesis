"""
Main pipeline orchestrator.

Dependencies:
  01 eeg_preprocess  — independent
  02 et_preprocess   — independent
  03 fuse_eeg_et     — requires 01 + 02
  04 extract_features — requires 03
  05 dl_prep         — requires 04
  06 sanity_checks   — standalone audit (requires 04)
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_step(name, module_run):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    module_run()


def run_all():
    import eeg_preprocess
    import et_preprocess
    import fuse_eeg_et
    import extract_features
    import dl_prep
    import sanity_checks

    run_step("01 — EEG Preprocessing", eeg_preprocess.run)
    run_step("02 — ET Preprocessing", et_preprocess.run)
    run_step("03 — EEG + ET Fusion", fuse_eeg_et.run)
    run_step("04 — Feature Extraction", extract_features.run)
    run_step("05 — DL Preparation", dl_prep.run)
    run_step("06 — Sanity Checks", sanity_checks.run)

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")


def main():
    steps = {
        "eeg": ("01 — EEG Preprocessing", "eeg_preprocess"),
        "et": ("02 — ET Preprocessing", "et_preprocess"),
        "fuse": ("03 — EEG + ET Fusion", "fuse_eeg_et"),
        "features": ("04 — Feature Extraction", "extract_features"),
        "dl": ("05 — DL Preparation", "dl_prep"),
        "checks": ("06 — Sanity Checks", "sanity_checks"),
    }

    parser = argparse.ArgumentParser(description="Run the EEG+ET processing pipeline.")
    parser.add_argument(
        "steps",
        nargs="*",
        default=["all"],
        help=f"Steps to run: {', '.join(steps.keys())}, or 'all' (default: all)",
    )
    args = parser.parse_args()

    if "all" in args.steps:
        run_all()
        return

    for step_key in args.steps:
        if step_key not in steps:
            print(f"Unknown step: {step_key}. Choose from: {', '.join(steps.keys())}")
            sys.exit(1)

        name, module_name = steps[step_key]
        module = __import__(module_name)
        run_step(name, module.run)

    print(f"\n{'='*60}")
    print("  SELECTED STEPS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
