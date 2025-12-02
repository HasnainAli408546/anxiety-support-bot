# src/flows/general/general_anxiety_flow.py

from ..base_flow import ClinicalTherapeuticFlow
from ...rag.content_retriever import ContentRetriever
from typing import Dict, Optional, List
from datetime import datetime
import re

class GeneralAnxietyFlow(ClinicalTherapeuticFlow):
    """
    Fallback clinical flow for generalized anxiety or unclassified emotional inputs.
    Uses dynamic RAG-backed content, education, and simple supportive engagement.
    """

    def __init__(self):
        super().__init__(
            flow_name="General Anxiety Support",
            scenario="general_anxiety",
            intensity="variable"
        )
        self.retriever = ContentRetriever()
        self.clinical_steps = [
            {
                'intervention': 'emotion_check_in',
                'step_type': 'assessment',
                'requires_input': True
            },
            {
                'intervention': 'psychoeducation_general',
                'step_type': 'education',
                'requires_input': False
            },
            {
                'intervention': 'coping_options',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': 'reflection',
                'step_type': 'reassurance',
                'requires_input': False
            },
            {
                'intervention': 'effectiveness_review',
                'step_type': 'effectiveness_check',
                'requires_input': True
            }
        ]

    def start_flow(self, clinical_context: Dict = None) -> Dict:
        # Start the flow: Greet and begin emotional check-in
        greeting = self.retriever.get_reassurance_content("general", 0.7)
        prompt = "I'm here for you. How are you feeling right now? You can tell me as much or as little as you want."
        return {
            'message': f"{greeting}\n\n{prompt}",
            'step_info': {
                'current_step': 1,
                'total_steps': len(self.clinical_steps),
                'intervention_type': self.clinical_steps[0]['intervention']
            },
            'requires_input': True
        }

    def process_user_input(self, user_input: str) -> Dict:
        """
        Advance through generic steps using dynamic RAG-backed support.
        Adapts responses based on emotion, input, and step type.
        """
        user_input_lower = user_input.lower().strip()
        step_idx = self.state.current_step
        current_step = self.clinical_steps[step_idx]

        # Step 0: Emotion check-in
        if step_idx == 0:
            reassurance = self.retriever.get_reassurance_content("general", 0.5)
            return {
                'message': (
                    f"{reassurance}\n\nThanks for sharing. Let’s talk a bit about general anxiety and how it can affect you."
                ),
                'advance_step': True,
                'step_info': {
            'current_step': self.state.current_step + 1,
            'total_steps': len(self.clinical_steps),
            'intervention_type': current_step['intervention']
                }
            }

        # Step 1: Psychoeducation (general)
        elif step_idx == 1:
            education = self.retriever.get_educational_content("general")
            return {
                'message': education,
                'advance_step': True,
                'step_info': {
                    'current_step': self.state.current_step + 1,
                    'total_steps': len(self.clinical_steps),
                    'intervention_type': current_step['intervention']
                }
            }

        # Step 2: Coping techniques/options
        elif step_idx == 2:
            technique = self.retriever.get_technique_for_scenario(
                scenario="general",
                intensity="variable",
                user_context={"user_input": user_input}
            )
            return {
                'message': (
                    f"Here’s a simple grounding technique you can try:\n\n{technique}\n\n"
                    "Would you like to give it a try now or learn about other options?"
                ),
                'advance_step': True,
                'step_info': {
                    'current_step': self.state.current_step + 1,
                    'total_steps': len(self.clinical_steps),
                    'intervention_type': current_step['intervention']
                }
            }

        # Step 3: Reflection/reassurance
        elif step_idx == 3:
            reflection = self.retriever.get_reassurance_content("general", 0.3)
            return {
                'message': (
                    f"{reflection}\n\nRemember, you're not alone in this. Every improvement counts, no matter how small."
                ),
                'advance_step': True,
                'step_info': {
                    'current_step': self.state.current_step + 1,
                    'total_steps': len(self.clinical_steps),
                    'intervention_type': current_step['intervention']
                }
            }

        # Step 4: Effectiveness review
        elif step_idx == 4:
            # Try to extract a simple rating (1–10 or words)
            rating = self.extract_effectiveness_rating(user_input)
            if rating is not None:
                self.state.effectiveness_ratings.append({
                    'rating': rating,
                    'timestamp': datetime.now().isoformat()
                })
                summary = (
                    f"Thank you for sharing. On a scale of 1 to 10, you've reported {rating}. "
                    "Would you like to continue or end for now?"
                )
            else:
                summary = (
                    "Let me know how much these steps helped you, using a number (1–10) or a few words."
                )
            return {
                'message': summary,
                'flow_completed': True if rating is not None else False,
                'advance_step': True,
                'step_info': {
                    'current_step': self.state.current_step + 1,
                    'total_steps': len(self.clinical_steps),
                    'intervention_type': current_step['intervention']
                }
            }

        # Beyond last step: Complete flow
        else:
            return self.complete_clinical_flow()

    def extract_effectiveness_rating(self, user_input: str) -> Optional[int]:
        numbers = re.findall(r'\b([1-9]|10)\b', user_input)
        if numbers:
            return int(numbers[0])
        word_ratings = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        for word, rating in word_ratings.items():
            if word in user_input.lower():
                return rating
        return None

    def get_clinical_resources(self) -> List[Dict]:
        return [
            {
                'type': 'general_education',
                'title': 'Understanding General Anxiety',
                'evidence_base': 'CBT, mindfulness, clinical guidelines',
                'description': 'General anxiety is common and manageable, even without a specific trigger.'
            },
            {
                'type': 'basic_technique',
                'title': 'Diaphragmatic Breathing',
                'evidence_base': 'Evidence for anxiety reduction',
                'description': 'Breathe slowly and deeply, focusing on your diaphragm. Repeat for 2 minutes.'
            },
            {
                'type': 'resources',
                'title': 'Mental Health Helpline',
                'evidence_base': 'Support networks',
                'description': 'You can reach out for confidential support anytime.'
            }
        ]
