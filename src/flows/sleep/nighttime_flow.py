from ..base_flow import ClinicalTherapeuticFlow
from ...rag.content_retriever import ContentRetriever
from datetime import datetime
from typing import Dict, Optional, List
import re

class NighttimeFlow(ClinicalTherapeuticFlow):
    """
    Clinical sleep flow with dynamic RAG-powered interventions for:
    - Sleep hygiene reminders
    - Cognitive restructuring for rumination
    - Body scan relaxation techniques
    - Bedtime routine suggestions
    """

    def __init__(self):
        super().__init__(
            flow_name="Nighttime Sleep Support",
            scenario="sleep", 
            intensity="medium"
        )
        self.retriever = ContentRetriever()
        self.clinical_steps = [
            {
                'intervention': 'sleep_assessment',
                'step_type': 'assessment',
                'requires_input': True
            },
            {
                'intervention': 'sleep_psychoeducation',
                'step_type': 'education',
                'requires_input': False
            },
            {
                'intervention': 'cognitive_restructuring',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': 'body_scan_relaxation',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': 'sleep_hygiene',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': '20_minute_rule',
                'step_type': 'education',
                'requires_input': False
            },
            {
                'intervention': 'effectiveness_check',
                'step_type': 'assessment',
                'requires_input': True
            }
        ]

    def process_clinical_step(self, user_input: str) -> Dict:

        if self.state.current_step >= len(self.clinical_steps):
            return self.complete_clinical_flow()

        user_input_lower = user_input.lower().strip()
        step_idx = self.state.current_step
        current_step = self.clinical_steps[step_idx]

        def step_info():
            return {
                'current_step': step_idx + 1,
                'total_steps': len(self.clinical_steps),
                'intervention_type': current_step.get('intervention', '')
            }

        # Step 0: Sleep assessment
        if step_idx == 0:
            reassurance = self.retriever.get_reassurance_content("sleep", 0.6)
            if any(time_word in user_input_lower for time_word in ['hours', 'minutes', 'long time']):
                return {
                    'message': (
                        f"{reassurance}\n\nIt sounds like you've been struggling for a while. "
                        "Let's work on some techniques to help calm your mind and body for sleep."
                    ),
                    'sleep_duration_noted': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': (
                        f"{reassurance}\n\nWhat time is it for you right now, and how long have you been trying to sleep?"
                    ),
                    'advance_step': True,
                    'step_info': step_info()
                }

        # Step 1: Sleep psychoeducation
        elif step_idx == 1:
            education = self.retriever.get_educational_content(
                scenario="sleep",
                topic="sleep anxiety cycle"
            )
            return {
                'message': education,
                'advance_step': True,
                'step_info': step_info()
            }

        # Step 2: Cognitive restructuring for racing thoughts
        elif step_idx == 2:
            technique = self.retriever.get_technique_for_scenario(
                scenario="sleep",
                intensity="medium",
                user_context={"preferred_technique_type": "thought_stopping"}
            )
            if 'difficult' in user_input_lower or 'not working' in user_input_lower:
                return {
                    'message': (
                        "Thought stopping can be challenging at first. Try this gentler approach: "
                        "Instead of fighting the thoughts, acknowledge them: 'I notice I'm having worried thoughts, "
                        "and that's okay. Right now, I choose to focus on rest.'"
                    ),
                    'technique_adaptation': True,
                    'advance_step': False,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': (
                        f"Racing thoughts are common at bedtime. Let's use a technique to quiet your mind:\n\n{technique}\n\n"
                        "Try this now with any current worrying thoughts. Reply 'done' when ready."
                    ),
                    'advance_step': True,
                    'step_info': step_info()
                }

        # Step 3: Body scan relaxation
        elif step_idx == 3:
            technique = self.retriever.get_technique_for_scenario(
                scenario="sleep",
                intensity="low",
                user_context={"preferred_technique_type": "body_scan"}
            )
            if 'done' in user_input_lower or 'finished' in user_input_lower:
                return {
                    'message': (
                        "Excellent. Body scanning helps signal to your nervous system that it's time to shift into rest mode. "
                        "Notice if your body feels heavier or more relaxed."
                    ),
                    'relaxation_noted': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': (
                        f"Now let's prepare your body for sleep:\n\n{technique}\n\n"
                        "Spend 10-15 seconds on each area. Take your time."
                    ),
                    'advance_step': True,
                    'step_info': step_info()
                }

        # Step 4: Sleep hygiene check
        elif step_idx == 4:
            technique = self.retriever.get_technique_for_scenario(
                scenario="sleep",
                intensity="low",
                user_context={"preferred_technique_type": "sleep_hygiene"}
            )
            return {
                'message': (
                    f"Quick sleep environment optimization:\n\n{technique}\n\n"
                    "Make any quick adjustments you can right now."
                ),
                'advance_step': True,
                'step_info': step_info()
            }

        # Step 5: 20-minute rule education
        elif step_idx == 5:
            education = self.retriever.get_educational_content(
                scenario="sleep",
                topic="20 minute rule"
            )
            return {
                'message': education,
                'advance_step': True,
                'step_info': step_info()
            }

        # Step 6: Effectiveness check and sleepiness rating
        elif step_idx == 6:
            rating = self.extract_sleepiness_rating(user_input)
            if rating:
                self.state.effectiveness_ratings.append({
                    'sleepiness_rating': rating,
                    'timestamp': datetime.now().isoformat()
                })
                if rating >= 6:
                    response_msg = (
                        f"Good—a sleepiness level of {rating}/10 suggests the techniques are helping. "
                        "Try to sleep now while you're feeling drowsy."
                    )
                elif rating >= 3:
                    response_msg = (
                        f"You're at {rating}/10 for sleepiness. Consider doing some gentle reading or "
                        "listening to calm music for 15-20 minutes before trying to sleep again."
                    )
                else:
                    response_msg = (
                        f"At {rating}/10, you're still quite alert. The 20-minute rule applies here—"
                        "engage in a quiet activity until you feel more drowsy."
                    )
                return {
                    'message': response_msg,
                    'sleep_guidance': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': (
                        "How are you feeling now? More relaxed and ready for sleep, about the same, or more anxious? "
                        "Rate your current sleepiness level 1-10 (1=wide awake, 10=very drowsy)."
                    ),
                    'advance_step': False,
                    'step_info': step_info()
                }

        response = self.get_current_step_response()
        response['step_info'] = step_info()
        return response

    def get_current_step_response(self) -> Dict:
        if self.state.current_step >= len(self.clinical_steps):
            res = self.complete_clinical_flow()
            res['step_info'] = {
                'current_step': len(self.clinical_steps),
                'total_steps': len(self.clinical_steps),
                'intervention_type': 'completed'
            }
            return res

        step_idx = self.state.current_step
        current_step = self.clinical_steps[step_idx]
        step_type = current_step.get('step_type', 'general')

        if step_type == 'reassurance':
            content = self.retriever.get_reassurance_content("sleep", 0.5)
        elif step_type == 'education':
            content = self.retriever.get_educational_content("sleep")
        elif step_type == 'technique':
            content = self.retriever.get_technique_for_scenario("sleep", "medium")
        elif step_type == 'assessment':
            content = "Let's assess your current sleep situation and how I can help."
        else:
            content = "Let's continue working on preparing you for restful sleep."

        return {
            'message': content,
            'intervention_type': current_step['intervention'],
            'step_number': step_idx + 1,
            'total_steps': len(self.clinical_steps),
            'flow_name': self.flow_name,
            'scenario': self.scenario,
            'requires_input': current_step.get('requires_input', True),
            'advance_step': True,
            'step_info': {
                'current_step': step_idx + 1,
                'total_steps': len(self.clinical_steps),
                'intervention_type': current_step['intervention']
            }
        }

    def extract_sleepiness_rating(self, user_input: str) -> Optional[int]:
        numbers = re.findall(r'\b([1-9]|10)\b', user_input)
        return int(numbers[0]) if numbers else None

    def get_clinical_resources(self) -> List[Dict]:
        return [
            {
                'type': 'sleep_hygiene',
                'title': 'Daily Sleep Routine',
                'evidence_base': 'Sleep hygiene protocols',
                'description': 'Consistent bedtime, cool room, no screens 1 hour before sleep'
            },
            {
                'type': 'cognitive_technique',
                'title': 'Thought Stopping for Sleep',
                'evidence_base': 'CBT for insomnia',
                'description': 'Practice the STOP technique during the day to strengthen it for nighttime use'
            },
            {
                'type': 'relaxation_technique',
                'title': 'Body Scan Meditation',
                'evidence_base': 'Mindfulness-based stress reduction',
                'description': '15-20 minute body scan recordings can be helpful for sleep preparation'
            },
            {
                'type': 'sleep_education',
                'title': 'Understanding Sleep and Anxiety',
                'evidence_base': 'Sleep medicine and CBT-I',
                'description': 'Learning about the sleep-anxiety cycle helps break the pattern'
            }
        ]
