from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class PatientRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    age INTEGER,
                    sex TEXT,
                    main_complaint TEXT,
                    allergies TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_exams (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT,
                    exam_name TEXT,
                    status TEXT,
                    FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
                )
                """
            )
            conn.commit()

    def seed_demo_data(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO patients (patient_id, age, sex, main_complaint, allergies)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("P001", 67, "M", "febre e hipotensão", "penicilina"),
            )
            conn.execute("DELETE FROM pending_exams WHERE patient_id = ?", ("P001",))
            conn.executemany(
                """
                INSERT INTO pending_exams (patient_id, exam_name, status)
                VALUES (?, ?, ?)
                """,
                [
                    ("P001", "Lactato", "pending"),
                    ("P001", "Hemocultura", "pending"),
                ],
            )
            conn.commit()

    def get_patient_context(self, patient_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            patient = conn.execute(
                "SELECT patient_id, age, sex, main_complaint, allergies FROM patients WHERE patient_id = ?",
                (patient_id,),
            ).fetchone()

            exams = conn.execute(
                "SELECT exam_name, status FROM pending_exams WHERE patient_id = ?",
                (patient_id,),
            ).fetchall()

        if not patient:
            return {
                "patient_id": patient_id,
                "found": False,
                "message": "Paciente não encontrado na base.",
                "pending_exams": [],
            }

        return {
            "patient_id": patient[0],
            "age": patient[1],
            "sex": patient[2],
            "main_complaint": patient[3],
            "allergies": patient[4],
            "found": True,
            "pending_exams": [{"exam_name": row[0], "status": row[1]} for row in exams],
        }
