import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

class UserMemory:
    def __init__(self, db_path: str = "user_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Table for chat turns (each row = 1 user turn, with all context)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_message TEXT,
                    bot_message TEXT,
                    flow_name TEXT,
                    flow_step INTEGER,
                    meta JSON
                )
            ''')
            conn.commit()

    def append_user_turn(
        self,
        user_id: str,
        user_message: str,
        bot_message: str,
        flow_name: Optional[str] = None,
        flow_step: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ):
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()
        if meta is None:
            meta = {}
        meta_json = json.dumps(meta)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_chat_history (
                    user_id, timestamp, user_message, bot_message, flow_name, flow_step, meta
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, timestamp, user_message, bot_message, flow_name, flow_step, meta_json
            ))
            conn.commit()

    def get_user_history(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            sql = '''
                SELECT timestamp, user_message, bot_message, flow_name, flow_step, meta
                FROM user_chat_history
                WHERE user_id = ?
                ORDER BY id ASC
            '''
            if limit:
                sql += ' LIMIT ?'
                cursor.execute(sql, (user_id, limit))
            else:
                cursor.execute(sql, (user_id,))
            rows = cursor.fetchall()
            history = []
            for row in rows:
                meta = json.loads(row[5]) if row[5] else {}
                history.append({
                    "timestamp": row[0],
                    "user_message": row[1],
                    "bot_message": row[2],
                    "flow_name": row[3],
                    "flow_step": row[4],
                    "meta": meta
                })
            return history

    def clear_user_history(self, user_id: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM user_chat_history WHERE user_id = ?', (user_id,))
            conn.commit()

    def get_last_turn(self, user_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, user_message, bot_message, flow_name, flow_step, meta
                FROM user_chat_history
                WHERE user_id = ?
                ORDER BY id DESC LIMIT 1
            ''', (user_id,))
            row = cursor.fetchone()
            if row:
                meta = json.loads(row[5]) if row[5] else {}
                return {
                    "timestamp": row[0],
                    "user_message": row[1],
                    "bot_message": row[2],
                    "flow_name": row[3],
                    "flow_step": row[4],
                    "meta": meta
                }
            return None

