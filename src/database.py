import sys
sys.path.insert(0, '/c/Projects/mlmodel')

import sqlite3
import json
from datetime import datetime
from src.config import DB_PATH


class PredictionDatabase:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(DB_PATH)
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database with predictions table."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cattle_id TEXT NOT NULL,
                disease TEXT NOT NULL,
                confidence REAL NOT NULL,
                risk_level TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                input_features TEXT,
                recommendation TEXT
            )
        """)

        conn.commit()
        conn.close()

    def log_prediction(self, cattle_id: str, disease: str, confidence: float, risk_level: str, input_features: dict, recommendation: str):
        """Log a prediction to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO predictions
            (cattle_id, disease, confidence, risk_level, input_features, recommendation)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            cattle_id,
            disease,
            confidence,
            risk_level,
            json.dumps(input_features),
            recommendation
        ))

        conn.commit()
        conn.close()

    def get_recent_predictions(self, limit: int = 10) -> list:
        """Get recent predictions from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, cattle_id, disease, confidence, risk_level, timestamp
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row[0],
                "cattle_id": row[1],
                "disease": row[2],
                "confidence": row[3],
                "risk_level": row[4],
                "timestamp": row[5]
            }
            for row in rows
        ]

    def get_statistics(self) -> dict:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]

        cursor.execute("""
            SELECT disease, COUNT(*) as count
            FROM predictions
            GROUP BY disease
        """)
        disease_counts = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()

        return {
            "total_predictions": total_predictions,
            "disease_distribution": disease_counts
        }


# Global database instance
_db = None


def get_database(db_path: str = None) -> PredictionDatabase:
    """Get or create database singleton."""
    global _db
    if _db is None:
        _db = PredictionDatabase(db_path)
    return _db
