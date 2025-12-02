from typing import Dict, Optional, Tuple
import re
from datetime import datetime

class ScenarioRouter:
    def __init__(self):
        # Enhanced crisis detection patterns with severity levels
        self.crisis_patterns = {
            'high_risk': [
                r'\b(want to die|wanna die|wish i was dead)\b',
                r'\b(kill myself|killing myself|end my life)\b',
                r'\b(suicide|suicidal thoughts|suicidal)\b',
                r'\b(better off dead|world better without me)\b',
                r'\b(no point living|no reason to live)\b',
                r'\b(end it all|ending it all)\b'
            ],
            'medium_risk': [
                r'\b(can\'t go on|cannot go on|give up)\b',
                r'\b(hopeless|no hope|nothing left)\b',
                r'\b(can\'t take it anymore|cannot take it)\b',
                r'\b(harm myself|hurt myself)\b'
            ]
        }
        self.scenario_weights = {
            'panic': 3.0,
            'sleep': 2.0,
            'isolation': 2.5,
            'pre_event': 2.0,
            'uncertainty': 1.5,
            'decision_making': 1.0,
            'physical_triggers': 1.8
        }
        self.scenario_to_flow = {
            'panic': 'acute_anxiety_flow',
            'sleep': 'nighttime_flow',
            'isolation': 'isolation_flow',
            'pre_event': 'pre_event_flow',
            'uncertainty': 'uncertainty_flow',
            'decision_making': 'decision_making_flow',
            'physical_triggers': 'physical_triggers_flow'
        }

    def detect_crisis(self, text: str) -> Dict:
        """Enhanced crisis detection with severity levels"""
        text_lower = text.lower()
        result = {
            'is_crisis': False,
            'severity': 'none',
            'patterns_found': []
        }
        # Check high-risk patterns first
        for pattern in self.crisis_patterns['high_risk']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                result['is_crisis'] = True
                result['severity'] = 'high'
                result['patterns_found'].append(pattern)
        # If no high-risk, check medium-risk
        if not result['is_crisis']:
            for pattern in self.crisis_patterns['medium_risk']:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    result['is_crisis'] = True
                    result['severity'] = 'medium'
                    result['patterns_found'].append(pattern)
        return result

    def route_scenario(
        self, 
        intent_scores: Dict[str, float], 
        emotion_scores: Dict[str, float],
        text: str = "",
        context: Optional[Dict] = None
    ) -> Tuple[str, Dict]:
        """Main routing function for scenario flows. Always returns (selected_flow, metadata)"""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'crisis_detected': False,
            'crisis_severity': 'none',
            'scenario': None,
            'intensity': None,
            'confidence': 0.0,
            'fallback_used': False
        }
        # Priority 1: Crisis Detection
        crisis_info = self.detect_crisis(text)
        if crisis_info['is_crisis']:
            metadata['crisis_detected'] = True
            metadata['crisis_severity'] = crisis_info['severity']
            if crisis_info['severity'] == 'high':
                return 'crisis_emergency_flow', metadata
            else:
                return 'crisis_support_flow', metadata

        # Scenario from top intent
        if intent_scores:
            top_scenario = max(intent_scores, key=intent_scores.get)
            top_score = intent_scores[top_scenario]
            # Get mapped clinical flow name
            mapped_flow = self.scenario_to_flow.get(top_scenario, None)
            metadata['scenario'] = top_scenario
            metadata['confidence'] = top_score
            metadata['fallback_used'] = False
            if mapped_flow:
                # Estimate intensity from emotion scores
                intensity_score = max(emotion_scores.values()) if emotion_scores else 0.5
                if intensity_score >= 0.8:
                    intensity = 'high'
                elif intensity_score >= 0.5:
                    intensity = 'medium'
                else:
                    intensity = 'low'
                metadata['intensity'] = intensity
                return mapped_flow, metadata

        # If no scenario confidently matched, fallback
        metadata['fallback_used'] = True
        return 'general_anxiety_flow', metadata

    def get_top_scenarios(self, text: str, top_k: int = 3) -> list:
        """Dummy utility: Find top scenarios in the text using weights/keywords (customize as needed)"""
        # This could apply keyword/embedding matching. For now, returns highest weights as dummy.
        candidates = sorted(self.scenario_weights.items(), key=lambda x: -x[1])
        return candidates[:top_k]

    def has_multiple_scenarios(self, text: str) -> bool:
        """Dummy utility: Check if text matches multiple scenario patterns (customize as needed)"""
        matches = 0
        text_lower = text.lower()
        for scenario in self.scenario_weights:
            if scenario in text_lower:
                matches += 1
        return matches > 1
