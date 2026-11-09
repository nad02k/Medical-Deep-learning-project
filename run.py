"""Point d'entrée (remplace le Makefile, compatible Windows).
  python run.py mimic     # MIMIC-IV demo -> tables ehr_*
  python run.py nih       # NIH ChestX-ray14 (sample)
  python run.py pneumonia # Kaggle Chest X-Ray Pneumonia
  python run.py all
  python run.py check     # contrôles sur la base
"""
import json
import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "check":
        from src import db, config
        res = {"post_checks": db.post_checks(), "post_checks_ehr": db.post_checks_ehr()}
        print(json.dumps(res, indent=2))
        config.REPORTS.mkdir(exist_ok=True)
        (config.REPORTS / "checks_latest.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        return
    from flows import pipeline as p
    runs = {"mimic": p.ingest_mimic_demo, "nih": p.ingest_nih, "pneumonia": p.ingest_pneumonia}
    if cmd == "all":
        for name in ("mimic", "nih", "pneumonia"):
            print(f"\n=== {name} ===")
            runs[name]()
    elif cmd in runs:
        runs[cmd]()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
