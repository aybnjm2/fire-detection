"""
src/agent.py

Autonomous Wildfire Monitoring Agent (Discord Webhook Edition - No Emojis)
-------------------------------------------------------------------------
"""

import os
import time
import logging
import json
import requests
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Charge les variables d'environnement (ex: DISCORD_WEBHOOK_URL)
load_dotenv()

from src.inference import predict_new_anomaly
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FireAgent")

# Configuration
ALERT_LOG_PATH = os.path.join(DATA_PROCESSED_DIR, "agent_alert_log.json")

# Severity thresholds
FRP_MEDIUM_THRESHOLD = 30.0   # MW
FRP_HIGH_THRESHOLD   = 60.0   # MW
FRP_CRITICAL_THRESHOLD = 100.0 # MW

FIRE_TYPE_LABELS = {
    0: "Volcanic / Geothermal",
    1: "Industrial Burn",
    2: "Forest / Vegetation Fire",
    3: "Other / Noise",
}

SEVERITY_LEVELS = {
    "LOW":      {"escalate": False, "hex": 5763719},
    "MEDIUM":   {"escalate": False, "hex": 16776960},
    "HIGH":     {"escalate": True,  "hex": 16753920},
    "CRITICAL": {"escalate": True,  "hex": 16711680},
}

@dataclass
class FireAlert:
    timestamp: str
    latitude: float
    longitude: float
    predicted_frp_mw: float
    fire_type: str
    type_confidence: float
    cluster_id: int
    severity: str
    escalated: bool
    reasoning: str
    raw_input: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def summary(self) -> str:
        return (
            f"[{self.severity}] {self.fire_type} detected at "
            f"({self.latitude:.3f}, {self.longitude:.3f}) — "
            f"FRP: {self.predicted_frp_mw:.1f} MW | "
            f"Confidence: {self.type_confidence:.1f}%"
        )


class AgentTools:
    @staticmethod
    def fetch_new_observations() -> list[dict]:
        logger.info("[Tool] fetch_observations")
        raw_path = os.path.join(DATA_RAW_DIR, "MODIS_fires.csv")

        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path)
            sample = df.sample(min(5, len(df)), random_state=int(time.time()) % 1000)
            records = []
            for _, row in sample.iterrows():
                records.append({
                    "latitude":   float(row.get("latitude",  36.0)),
                    "longitude":  float(row.get("longitude", 38.0)),
                    "brightness": float(row.get("brightness", np.random.uniform(300, 370))),
                    "bright_t31": float(row.get("bright_t31", np.random.uniform(280, 310))),
                    "scan":       float(row.get("scan",  1.2)),
                    "track":      float(row.get("track", 1.2)),
                    "confidence": int(row.get("confidence", 75)),
                    "daynight":   str(row.get("daynight", "D")),
                    "satellite":  str(row.get("satellite", "Terra")),
                })
            return records
        else:
            return [{
                "latitude": 36.74, "longitude": 36.21,
                "brightness": 330.0, "bright_t31": 300.0,
                "scan": 1.2, "track": 1.2, "confidence": 80,
                "daynight": "D", "satellite": "Terra"
            }]

    @staticmethod
    def run_inference(observation: dict) -> dict:
        return predict_new_anomaly(observation)

    @staticmethod
    def assess_severity(frp_mw: float, fire_type: int, confidence: float) -> tuple[str, str]:
        reasons = []

        if frp_mw >= FRP_CRITICAL_THRESHOLD:
            severity = "CRITICAL"
            reasons.append(f"FRP of {frp_mw:.1f} MW exceeds critical threshold.")
        elif frp_mw >= FRP_HIGH_THRESHOLD:
            severity = "HIGH"
            reasons.append(f"FRP of {frp_mw:.1f} MW exceeds high threshold.")
        elif frp_mw >= FRP_MEDIUM_THRESHOLD:
            severity = "MEDIUM"
            reasons.append(f"FRP of {frp_mw:.1f} MW is moderate.")
        else:
            severity = "LOW"
            reasons.append(f"FRP of {frp_mw:.1f} MW is low.")

        if fire_type == 2:
            if severity == "MEDIUM":
                severity = "HIGH"
                reasons.append("Upgraded to HIGH: Vegetation fires spread rapidly.")
            elif severity == "LOW":
                severity = "MEDIUM"
                reasons.append("Upgraded to MEDIUM: Vegetation fire.")

        return severity, " ".join(reasons)

    @staticmethod
    def log_alert(alert: FireAlert, log_path: str = ALERT_LOG_PATH):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        existing = []
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    existing = json.load(f)
            except Exception: pass
        existing.append(alert.to_dict())
        with open(log_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    @staticmethod
    def send_discord_alert(alert: FireAlert, webhook_url: str):
        hex_color = SEVERITY_LEVELS[alert.severity]["hex"]

        embed = {
            "title": f"Wildfire Alert: {alert.severity}",
            "description": f"**{alert.fire_type}** detected.",
            "color": hex_color,
            "fields": [
                {"name": "Location (Lat, Lon)", "value": f"`{alert.latitude:.4f}, {alert.longitude:.4f}`", "inline": True},
                {"name": "Predicted Power", "value": f"`{alert.predicted_frp_mw:.1f} MW`", "inline": True},
                {"name": "Confidence", "value": f"`{alert.type_confidence:.1f}%`", "inline": True},
                {"name": "System Reasoning", "value": alert.reasoning, "inline": False}
            ],
            "footer": {"text": f"FireMonitoringAgent - {alert.timestamp}"}
        }

        try:
            response = requests.post(webhook_url, json={"embeds": [embed]})
            response.raise_for_status()
            logger.info("Discord Webhook sent successfully.")
        except Exception as exc:
            logger.error("Failed to send Discord Webhook: %s", exc)


class FireMonitoringAgent:
    def __init__(self,
                 poll_interval_seconds: int = 60,
                 max_cycles: Optional[int] = None,
                 enable_discord: bool = False,
                 generate_test_critical: bool = False):
        self.tools = AgentTools()
        self.poll_interval = poll_interval_seconds
        self.max_cycles = max_cycles
        self.enable_discord = enable_discord
        self.generate_test_critical = generate_test_critical
        self.cycle_count = 0

    def run_cycle(self) -> list[FireAlert]:
        self.cycle_count += 1
        alerts_this_cycle = []

        # 1. OBSERVE : On récupère toujours un nombre aléatoire d'observations normales
        observations = self.tools.fetch_new_observations()

        # Si le mode test est activé, on INJECTE 1 ou 2 cas critiques dans le lot
        if self.generate_test_critical:
            import random
            num_critical = random.choice([1, 2])
            logger.warning(f"TEST MODE ACTIVE: Injecting {num_critical} critical artificial fire(s) into the batch...")
            
            for i in range(num_critical):
                observations.append({
                    "latitude": 36.8 + (i * 0.3), 
                    "longitude": 36.2 - (i * 0.7), 
                    "brightness": 450.0 + (i * 20), 
                    "bright_t31": 310.0, 
                    "scan": 1.2, 
                    "track": 1.2, 
                    "confidence": 100, 
                    "daynight": "D", 
                    "satellite": "Terra",
                    "is_injected_test": True # Marqueur pour forcer les valeurs critiques
                })

        # 2. THINK
        for i, obs in enumerate(observations):
            try:
                # Vérifie si c'est une de nos données injectées pour le test
                if obs.pop("is_injected_test", False):
                    # Force les valeurs critiques pour déclencher Discord à coup sûr
                    frp = 160.5 + (i * 25.3) 
                    fire_type = 2 # 2 = Vegetation
                    confidence = 99.9
                    cluster_id = 999
                else:
                    # Analyse normale par le modèle ML pour les données aléatoires
                    result = self.tools.run_inference(obs)
                    frp = result["Predicted_FRP_MW"]
                    fire_type = result["Predicted_Fire_Type"]
                    confidence = result["Type_Confidence"]
                    cluster_id = result["Assigned_Cluster_ID"]

                severity, reasoning = self.tools.assess_severity(frp, fire_type, confidence)
                escalate = SEVERITY_LEVELS[severity]["escalate"]

                alert = FireAlert(
                    timestamp = datetime.now().isoformat(timespec="seconds"),
                    latitude = obs["latitude"], longitude = obs["longitude"],
                    predicted_frp_mw = frp, fire_type = FIRE_TYPE_LABELS.get(fire_type, "Unknown"),
                    type_confidence = confidence, cluster_id = cluster_id,
                    severity = severity, escalated = escalate, reasoning = reasoning, raw_input=obs
                )

                self.tools.log_alert(alert)
                alerts_this_cycle.append(alert)
                
            except Exception as exc:
                logger.error("Inference failed: %s", exc)

        # 3. ACT (Discord Webhooks)
        escalated_alerts = [a for a in alerts_this_cycle if a.escalated]
        if self.enable_discord and escalated_alerts:
            webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
            if not webhook_url:
                logger.warning("Discord is enabled but DISCORD_WEBHOOK_URL is missing in .env")
            else:
                for alert in escalated_alerts:
                    self.tools.send_discord_alert(alert, webhook_url)

        return alerts_this_cycle
    
    def run(self):
        try:
            while True:
                self.run_cycle()
                if self.max_cycles and self.cycle_count >= self.max_cycles: break
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logger.info("Agent stopped.")

    @staticmethod
    def load_alert_log(log_path: str = ALERT_LOG_PATH) -> pd.DataFrame:
        if not os.path.exists(log_path): return pd.DataFrame()
        try:
            with open(log_path, "r") as f: return pd.DataFrame(json.load(f))
        except Exception: return pd.DataFrame()

if __name__ == "__main__":
    # Test immédiat du webhook avec la génération de feux critiques
    agent = FireMonitoringAgent(
        max_cycles=1,                  # S'arrête après 1 cycle de test
        enable_discord=True,           # Active l'envoi sur Discord
        generate_test_critical=True    # FORCE LA CRÉATION DE 2 FEUX CRITIQUES > 150MW
    )
    agent.run()