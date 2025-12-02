import re
from typing import Dict, List, Tuple

class CrisisDetector:
    """
    Detects high-risk crisis, suicidal intent, or self-harm language in text using regex patterns.
    Returns risk level, patterns found, and recommended action.
    """

    def __init__(self):
        # High-risk patterns categorized by severity
        self.high_risk_patterns = {
            'suicidal_ideation': [
                r'\b(want to die|wanna die|wish i was dead)\b',
                r'\b(kill myself|killing myself|end my life)\b',
                r'\b(suicide|suicidal thoughts|suicidal)\b',
                r'\b(better off dead|world better without me)\b',
                r'\b(no point living|no reason to live)\b'
            ],
            'self_harm': [
                r'\b(cut myself|cutting myself|hurt myself)\b',
                r'\b(harm myself|harming myself)\b',
                r'\b(self harm|self-harm)\b'
            ],
            'hopelessness': [
                r'\b(can\'t go on|cannot go on|give up)\b',
                r'\b(hopeless|no hope|nothing left)\b',
                r'\b(end it all|ending it all)\b'
            ]
        }

        # Medium-risk patterns for monitoring (distress, breakdown)
        self.medium_risk_patterns = {
            'severe_distress': [
                r'\b(can\'t take it anymore|cannot take it)\b',
                r'\b(overwhelmed|breaking down|falling apart)\b',
                r'\b(desperate|desperation)\b'
            ]
        }

    def detect_crisis_level(self, text: str) -> Dict:
        """
        Detect crisis level and return detailed information

        Returns:
            Dict with crisis level, patterns found, and recommended action
        """
        text_lower = text.lower()
        result = {
            'crisis_level': 'none',
            'patterns_found': [],
            'categories': [],
            'recommended_action': 'continue',
            'confidence': 0.0
        }

        high_risk_found = []
        medium_risk_found = []

        # Check high-risk patterns
        for category, patterns in self.high_risk_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    high_risk_found.extend([(category, match) for match in matches])

        # Check medium-risk patterns
        for category, patterns in self.medium_risk_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    medium_risk_found.extend([(category, match) for match in matches])

        # Determine crisis level
        if high_risk_found:
            result['crisis_level'] = 'high'
            result['recommended_action'] = 'immediate_intervention'
            result['confidence'] = 0.9
        elif medium_risk_found:
            result['crisis_level'] = 'medium'
            result['recommended_action'] = 'elevated_support'
            result['confidence'] = 0.6

        # Populate details
        result['patterns_found'] = high_risk_found + medium_risk_found
        result['categories'] = list(set([cat for cat, _ in result['patterns_found']]))

        return result

# --- Module-level API for pipeline integration ---
def detect_crisis_keywords(text: str) -> bool:
    """
    Simple API to check if provided text indicates crisis (high or medium risk).
    Returns True if crisis detected, else False.
    """
    cd = CrisisDetector()
    result = cd.detect_crisis_level(text)
    # Return True if risk is "high" or "medium"
    return result['crisis_level'] in ['high', 'medium']
