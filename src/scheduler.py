"""Planificateur de collecte automatisée périodique (Bonus DataOps Tâche 3).

Permet d'exécuter la collecte à intervalles réguliers (mode daemon ou cron)
pour surveiller l'arrivée de nouvelles données sur Kaggle ou PhysioNet.
"""
from flows import pipeline
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_scheduled_job(source: str = "all"):
    now = datetime.now(timezone.utc).isoformat()
    print(f"[{now}] ⏰ Déclenchement automatique du job de collecte planifiée (source: {source})...")
    if source == "mimic":
        pipeline.ingest_mimic_demo()
    elif source == "nih":
        pipeline.ingest_nih()
    elif source == "pneumonia":
        pipeline.ingest_pneumonia()
    elif source == "all":
        pipeline.ingest_mimic_demo()
        pipeline.ingest_nih()
        pipeline.ingest_pneumonia()
    print(f"[{datetime.now(timezone.utc).isoformat()}] ✅ Job de collecte planifiée terminé avec succès.")


def start_scheduler(interval_hours: float = 24.0,
                    source: str = "all", run_immediately: bool = False):
    print(
        f"🚀 Planificateur DataOps actif. Intervalle d'exécution : {interval_hours} heures.")
    if run_immediately:
        run_scheduled_job(source)

    interval_seconds = int(interval_hours * 3600)
    try:
        while True:
            time.sleep(interval_seconds)
            run_scheduled_job(source)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt du planificateur DataOps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Planificateur de collecte de données DataOps")
    parser.add_argument(
        "--interval",
        type=float,
        default=24.0,
        help="Intervalle en heures entre chaque collecte")
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=[
            "nih",
            "pneumonia",
            "mimic",
            "all"])
    parser.add_argument(
        "--now",
        action="store_true",
        help="Exécuter immédiatement au démarrage")
    args = parser.parse_args()

    start_scheduler(
        interval_hours=args.interval,
        source=args.source,
        run_immediately=args.now)
