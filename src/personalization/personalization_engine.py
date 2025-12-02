from typing import Dict, List, Optional
from src.memory.user_memory import UserMemory

class PersonalizationEngine:
    def __init__(self, memory_db_path: str = "user_memory.db"):
        self.user_memory = UserMemory(memory_db_path)

    def get_personalized_recommendations(self, user_id: str, context: Optional[dict] = None) -> Dict:
        """
        Analyze user memory and recent history to personalize advice/interventions.
        Returns a dict with recommendations for the pipeline/flows.
        """
        history = self.user_memory.get_user_history(user_id, limit=20)
        # Simple examples: count steps, find last successful intervention, check recurring struggles, etc.

        recs = {}
        if not history:
            recs['priority_technique'] = None
            recs['needs_encouragement'] = True
            return recs

        # Find most-used technique or lack of progress pattern
        recent_flows = [h["flow_name"] for h in history if h.get("flow_name")]
        if recent_flows:
            recs['most_frequent_flow'] = max(set(recent_flows), key=recent_flows.count)
        else:
            recs['most_frequent_flow'] = None

        # If the last few bot messages include "did not help" or "still anxious", suggest escalation
        trouble_phrases = ["still anxious", "did not help", "not working"]
        if any(tp in (h["bot_message"] or "") for h in history[-5:] for tp in trouble_phrases):
            recs['escalate_support'] = True
        else:
            recs['escalate_support'] = False

        # Example: If user successfully completed a flow recently, reinforce that technique
        for h in reversed(history):
            if h["flow_step"] == 0 and "completed" in (h["bot_message"] or ""):
                recs["highlight_success"] = True
                break

        return recs

    def personalize_step(self, user_id: str, step: Dict, context: dict = {}) -> Dict:
        """
        Optionally mutate a step dict based on personalization logic (e.g., reorder, skip, suggest a technique type).
        """
        recommendations = self.get_personalized_recommendations(user_id, context)
        # Example: If escalation needed, insert extra reassurance step
        if recommendations.get("escalate_support", False):
            step['suggested_intervention'] = "extra_reassurance"
        return step

# EXAMPLE USAGE:
if __name__ == "__main__":
    personalizer = PersonalizationEngine()
    recs = personalizer.get_personalized_recommendations("testuser1")
    print(recs)
