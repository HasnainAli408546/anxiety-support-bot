from typing import Dict, List, Optional
from src.memory.user_memory import UserMemory
from collections import Counter

class UserAnalytics:
    def __init__(self, memory_db_path: str = "user_memory.db"):
        self.user_memory = UserMemory(memory_db_path)

    def get_user_session_stats(self, user_id: str) -> Dict:
        """
        Basic analytics for a single user: turns, session length, most frequent flows, flagged crisis/safety issues.
        """
        history = self.user_memory.get_user_history(user_id)
        if not history:
            return {"message_count": 0, "sessions": 0, "frequent_flows": [], "crisis_count": 0}

        flows = [h["flow_name"] for h in history if h["flow_name"]]
        crisis_turns = [h for h in history if h["meta"].get("crisis_detected", False)]

        return {
            "message_count": len(history),
            "distinct_sessions": len(set(f for f in flows if f)),
            "most_frequent_flows": Counter(flows).most_common(2),
            "crisis_flags": len(crisis_turns)
        }

    def get_global_stats(self, user_ids: List[str]) -> Dict:
        """
        Aggregate stats over all user_ids: active users, total messages, crisis events, etc.
        """
        total_msgs = 0
        total_crisis = 0
        all_flows = []
        for user_id in user_ids:
            stats = self.get_user_session_stats(user_id)
            total_msgs += stats["message_count"]
            total_crisis += stats["crisis_flags"]
            all_flows += [f for f, _ in stats["most_frequent_flows"]]

        return {
            "active_users": len(user_ids),
            "total_messages": total_msgs,
            "total_crisis_flags": total_crisis,
            "top_flows": Counter(all_flows).most_common(3)
        }

    def get_last_session_summary(self, user_id: str) -> Dict:
        """
        Quick summary (last 10 messages) for user: for display or session review.
        """
        hist = self.user_memory.get_user_history(user_id, limit=10)
        return {"recent_history": hist}

# EXAMPLE USAGE:
if __name__ == "__main__":
    analytics = UserAnalytics()
    print(analytics.get_user_session_stats("testuser1"))
    print(analytics.get_last_session_summary("testuser1"))
