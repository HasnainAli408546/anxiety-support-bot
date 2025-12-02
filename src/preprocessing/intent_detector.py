import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict

class IntentDetector:
    def __init__(self):
        # Define scenario keywords with weights
        self.scenario_keywords = {
            "panic": {
                "primary": ["racing heart", "heart racing", "can't breathe", "cannot breathe", "dizzy", "trembling", "shaking", "chest tight", "palpitations"],
                "secondary": ["panic", "scared", "terrified", "overwhelmed", "nauseous", "sweating", "hyperventilating"]
            },
            "sleep": {
                "primary": ["can't sleep", "cannot sleep", "thoughts won't stop", "racing mind", "racing thoughts", "lying awake"],
                "secondary": ["insomnia", "restless", "tossing turning", "mind racing", "overthinking", "ruminating", "bedtime"]
            },
            "pre_event": {
                "primary": ["interview", "exam", "test", "presentation", "meeting", "tomorrow", "next week"],
                "secondary": ["nervous", "worried about", "preparing for", "upcoming", "performance", "evaluation", "speech"]
            },
            "isolation": {
                "primary": ["alone", "lonely", "no one to talk to", "nobody understands", "isolated", "by myself"],
                "secondary": ["abandoned", "disconnected", "empty", "friendless", "solitary", "withdrawn"]
            },
            "uncertainty": {
                "primary": ["waiting for", "don't know", "what if", "uncertain", "unknown", "unclear"],
                "secondary": ["confused", "unsure", "doubtful", "ambiguous", "unpredictable", "worrying about"]
            },
            "decision_making": {
                "primary": ["don't know what to", "can't decide", "cannot decide", "choices", "options", "confused about"],
                "secondary": ["indecisive", "torn between", "struggling with", "difficulty choosing", "overwhelmed by options"]
            },
            "physical_triggers": {
                "primary": ["caffeine", "tired", "exhausted", "crowded", "noisy", "loud", "bright lights"],
                "secondary": ["stimulants", "coffee", "energy drink", "fatigue", "overstimulated", "sensory overload"]
            }
        }
        
        # Compile regex patterns for efficient matching
        self.compiled_patterns = {}
        for scenario, keywords in self.scenario_keywords.items():
            patterns = []
            for category in ['primary', 'secondary']:
                for keyword in keywords[category]:
                    # Create word boundary pattern for exact matches
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    patterns.append((pattern, category))
            self.compiled_patterns[scenario] = patterns
        
        # Weights for scoring
        self.weights = {
            'primary': 2.0,
            'secondary': 1.0
        }
    
    def detect_intent(self, text: str) -> Dict[str, float]:
        """
        Detect intent from text and return confidence scores for each scenario
        
        Args:
            text: Normalized input text
            
        Returns:
            Dictionary with scenario names as keys and confidence scores as values
        """
        if not text:
            return {}
        
        text = text.lower()
        scenario_scores = defaultdict(float)
        
        # Score each scenario based on keyword matches
        for scenario, patterns in self.compiled_patterns.items():
            for pattern, category in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    weight = self.weights[category]
                    # Add score based on number of matches and weight
                    scenario_scores[scenario] += len(matches) * weight
        
        # Normalize scores (optional)
        if scenario_scores:
            max_score = max(scenario_scores.values())
            normalized_scores = {k: v/max_score for k, v in scenario_scores.items()}
            return dict(normalized_scores)
        
        return {}
    
    def get_top_scenarios(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top K scenarios with highest confidence scores
        
        Args:
            text: Input text
            top_k: Number of top scenarios to return
            
        Returns:
            List of tuples (scenario_name, confidence_score)
        """
        scores = self.detect_intent(text)
        if not scores:
            return []
        
        # Sort by score in descending order
        sorted_scenarios = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scenarios[:top_k]
    
    def has_multiple_scenarios(self, text: str, threshold: float = 0.5) -> bool:
        """
        Check if text contains multiple scenarios above threshold
        
        Args:
            text: Input text
            threshold: Minimum confidence threshold
            
        Returns:
            Boolean indicating multiple scenarios detected
        """
        scores = self.detect_intent(text)
        high_confidence_scenarios = [s for s, score in scores.items() if score >= threshold]
        return len(high_confidence_scenarios) > 1
    
    def get_keywords_found(self, text: str, scenario: str) -> List[str]:
        """
        Get the actual keywords found for a specific scenario
        
        Args:
            text: Input text
            scenario: Scenario name
            
        Returns:
            List of keywords found in the text
        """
        if scenario not in self.compiled_patterns:
            return []
        
        text = text.lower()
        found_keywords = []
        
        for pattern, category in self.compiled_patterns[scenario]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Extract the original keyword from pattern
                keyword = pattern.replace(r'\b', '').replace('\\', '')
                found_keywords.extend([keyword] * len(matches))
        
        return found_keywords
