from typing import Dict, List, Tuple, Optional
from .text_normalizer import TextNormalizer
from .intent_detector import IntentDetector

class PreprocessingPipeline:
    def __init__(self):
        self.text_normalizer = TextNormalizer()
        self.intent_detector = IntentDetector()
    
    def process(self, user_input: str) -> Dict:
        """
        Complete preprocessing pipeline
        
        Args:
            user_input: Raw user input text
            
        Returns:
            Dictionary containing processed results
        """
        if not user_input or not isinstance(user_input, str):
            return self._empty_result()
        
        # Step 1: Normalize text
        normalized_text = self.text_normalizer.normalize_text(user_input)
        
        if not normalized_text:
            return self._empty_result()
        
        # Step 2: Detect intent/scenarios
        intent_scores = self.intent_detector.detect_intent(normalized_text)
        top_scenarios = self.intent_detector.get_top_scenarios(normalized_text)
        
        # Step 3: Extract additional metadata
        has_multiple_scenarios = self.intent_detector.has_multiple_scenarios(normalized_text)
        
        # Step 4: Get keywords found for top scenario
        keywords_found = {}
        if top_scenarios:
            for scenario, score in top_scenarios:
                keywords_found[scenario] = self.intent_detector.get_keywords_found(
                    normalized_text, scenario
                )
        
        return {
            'original_text': user_input,
            'normalized_text': normalized_text,
            'intent_scores': intent_scores,
            'top_scenarios': top_scenarios,
            'primary_scenario': top_scenarios[0][0] if top_scenarios else None,
            'confidence': top_scenarios[0][1] if top_scenarios else 0.0,
            'has_multiple_scenarios': has_multiple_scenarios,
            'keywords_found': keywords_found,
            'text_length': len(normalized_text),
            'word_count': len(normalized_text.split())
        }
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'original_text': '',
            'normalized_text': '',
            'intent_scores': {},
            'top_scenarios': [],
            'primary_scenario': None,
            'confidence': 0.0,
            'has_multiple_scenarios': False,
            'keywords_found': {},
            'text_length': 0,
            'word_count': 0
        }
    
    def batch_process(self, inputs: List[str]) -> List[Dict]:
        """
        Process multiple inputs at once
        
        Args:
            inputs: List of input texts
            
        Returns:
            List of processing results
        """
        return [self.process(text) for text in inputs]
    
    def get_scenario_summary(self, results: List[Dict]) -> Dict:
        """
        Get summary statistics for multiple processing results
        
        Args:
            results: List of processing results
            
        Returns:
            Summary statistics
        """
        if not results:
            return {}
        
        scenario_counts = {}
        total_processed = len(results)
        
        for result in results:
            primary = result.get('primary_scenario')
            if primary:
                scenario_counts[primary] = scenario_counts.get(primary, 0) + 1
        
        return {
            'total_processed': total_processed,
            'scenario_distribution': scenario_counts,
            'most_common_scenario': max(scenario_counts.items(), key=lambda x: x[1])[0] if scenario_counts else None
        }
