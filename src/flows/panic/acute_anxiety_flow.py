from ..base_flow import ClinicalTherapeuticFlow
from ...rag.content_retriever import ContentRetriever  # Relative import may need adjusting
import re
from datetime import datetime
from typing import Dict, Optional, List

class AcuteAnxietyFlow(ClinicalTherapeuticFlow):
    """
    Clinical panic flow using dynamic, RAG-powered intervention content.
    """

    def __init__(self):
        super().__init__(
            flow_name="Acute Anxiety Support",
            scenario="panic",
            intensity="high"
        )
        self.retriever = ContentRetriever()
        self.clinical_steps = [
            {
                'intervention': 'safety_assessment',
                'step_type': 'reassurance',
                'requires_input': True
            },
            {
                'intervention': 'psychoeducation',
                'step_type': 'education',
                'requires_input': False
            },
            {
                'intervention': '4_7_8_breathing',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': '5_4_3_2_1_grounding',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': 'progressive_muscle_relaxation',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': 'clinical_reassurance',
                'step_type': 'reassurance',
                'requires_input': False
            },
            {
                'intervention': 'effectiveness_assessment',
                'step_type': 'effectiveness_check',
                'requires_input': True
            }
        ]

    def process_clinical_step(self, user_input: str) -> Dict:
        if self.state.current_step >= len(self.clinical_steps):
            return self.complete_clinical_flow()
        step_idx = self.state.current_step
        current_step = self.clinical_steps[step_idx]
        user_input_lower = user_input.lower().strip()

        def step_info():
            return {
                'current_step': step_idx + 1,
                'total_steps': len(self.clinical_steps),
                'intervention_type': current_step.get('intervention', '')
            }

        # 1. Safety step: Provide contextual reassurance
        if step_idx == 0:
            reassurance = self.retriever.get_reassurance_content("panic", 0.7)
            question = "Are you in a safe location?"
            if any(word in user_input_lower for word in ['no', 'not safe', 'danger']):
                return {
                    'message': (
                        f"{reassurance}\n\nYour safety is the priority. Please consider calling emergency services (911) or going to your nearest emergency room. "
                        "We can continue with grounding techniques while you decide."
                    ),
                    'safety_concern': True,
                    'escalation_recommended': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
            return {
                'message': f"{reassurance}\n\n{question}", 
                'advance_step': True,
                'step_info': step_info()
            }

        # 2. Psychoeducation step
        elif step_idx == 1:
            education = self.retriever.get_educational_content("panic")
            return {
                'message': education,
                'advance_step': True,
                'step_info': step_info()
            }

        # 3. Evidence-based breathing technique
        elif step_idx == 2:
            technique = self.retriever.get_technique_for_scenario(
                scenario="panic",
                intensity="high",
                user_context={"preferred_technique_type": "breathing"}
            )
            if any(phrase in user_input_lower for phrase in ["can't", "difficult", "not working"]):
                return {
                    'message': (
                        "That's okay—breathing techniques can feel difficult during panic. "
                        "Let's try natural breathing: just breathe at your own pace and count each breath from 1 to 10. "
                        "Reply 'done' when ready."
                    ),
                    'advance_step': False,
                    'step_info': step_info()
                }
            prompt = (
                f"Let's use a powerful technique for panic:\n\n{technique}\n\n"
                "Type 'done' when you've completed it."
            )
            return {'message': prompt, 'advance_step': True, 'step_info': step_info()}

        # 4. 5-4-3-2-1 Grounding
        elif step_idx == 3:
            technique = self.retriever.get_technique_for_scenario(
                scenario="panic", 
                intensity="medium",
                user_context={"preferred_technique_type": "grounding"}
            )
            sensory_words = ['see', 'hear', 'touch', 'smell', 'taste']
            if any(word in user_input_lower for word in sensory_words):
                return {
                    'message': (
                        "Excellent work engaging your senses. "
                        "This helps interrupt the panic cycle by grounding you in reality."
                    ),
                    'advance_step': True,
                    'step_info': step_info()
                }
            return {
                'message': f"Let's ground you in the moment:\n\n{technique}\n\nStart with the 5 things you can see and work through the list.",
                'advance_step': True,
                'step_info': step_info()
            }

        # 5. Progressive muscle relaxation
        elif step_idx == 4:
            technique = self.retriever.get_technique_for_scenario(
                scenario="panic", 
                intensity="low",
                user_context={"preferred_technique_type": "relaxation"}
            )
            relaxation_note = (
                "Notice the contrast between tension and relaxation. Type 'done' when finished."
            )
            return {'message': f"{technique}\n\n{relaxation_note}", 'advance_step': True, 'step_info': step_info()}

        # 6. Clinical reassurance (dynamic)
        elif step_idx == 5:
            reassurance = self.retriever.get_reassurance_content("panic", 0.4)
            return {
                'message': (
                    f"{reassurance}\n\nYou've just used several evidence-based techniques. Panic attacks, while frightening, are temporary. "
                    "Your symptoms will continue to decrease as your nervous system calms."
                ),
                'advance_step': True,
                'step_info': step_info()
            }

        # 7. Effectiveness assessment step
        elif step_idx == 6:
            rating = self.extract_anxiety_rating(user_input)
            if rating:
                self.state.effectiveness_ratings.append({
                    'rating': rating,
                    'timestamp': datetime.now().isoformat(),
                    'intervention_phase': 'post_technique'
                })
                if rating <= 3:
                    response_msg = f"Excellent! Your anxiety has decreased significantly to {rating}/10. The techniques worked well for you."
                elif rating <= 6:
                    response_msg = f"Good progress—you're down to {rating}/10. These skills keep getting more effective with practice."
                else:
                    response_msg = (
                        f"You're still experiencing high anxiety at {rating}/10. "
                        "That's normal—recovery isn't always linear. Would you like to try another technique?"
                    )
                return {
                    'message': response_msg,
                    'clinical_outcome': rating,
                    'advance_step': True,
                    'step_info': step_info()
                }

        response = self.get_current_step_response()
        response['step_info'] = step_info()
        return response

    def extract_anxiety_rating(self, user_input: str) -> Optional[int]:
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
                'type': 'clinical_technique',
                'title': '4-7-8 Breathing',
                'evidence_base': 'Parasympathetic nervous system activation',
                'description': 'Practice daily when calm for 2 minutes to build familiarity'
            },
            {
                'type': 'clinical_technique', 
                'title': '5-4-3-2-1 Grounding',
                'evidence_base': 'Sensory grounding interrupts panic spiral',
                'description': 'Use when you feel anxiety beginning to escalate'
            },
            {
                'type': 'clinical_education',
                'title': 'Understanding Panic',
                'evidence_base': 'Psychoeducation reduces panic frequency',
                'description': 'Panic attacks peak in 10 minutes and are not medically dangerous'
            },
            {
                'type': 'clinical_referral',
                'title': 'When to Seek Professional Help',
                'evidence_base': 'Clinical guidelines for care escalation',
                'description': 'If panic attacks occur frequently or interfere with daily functioning, consider consulting a mental health professional'
            }
        ]
