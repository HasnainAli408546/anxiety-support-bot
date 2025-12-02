from ..base_flow import ClinicalTherapeuticFlow
from ...rag.content_retriever import ContentRetriever
from typing import Dict, Optional, List

class DecisionMakingFlow(ClinicalTherapeuticFlow):
    """
    Clinical decision-making flow with dynamic RAG intervention content.
    """
    def __init__(self):
        super().__init__(
            flow_name="Decision-Making Support",
            scenario="decision_making",
            intensity="medium"
        )
        self.retriever = ContentRetriever()
        self.clinical_steps = [
            {
                'intervention': 'decision_assessment',
                'step_type': 'reassurance',
                'requires_input': True
            },
            {
                'intervention': 'perfectionism_check',
                'step_type': 'reassurance',
                'requires_input': True
            },
            {
                'intervention': 'decision_framework',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': 'values_clarification',
                'step_type': 'education',
                'requires_input': True
            },
            {
                'intervention': 'good_enough_principle',
                'step_type': 'technique',
                'requires_input': False
            },
            {
                'intervention': 'time_limit_technique',
                'step_type': 'education',
                'requires_input': True
            },
            {
                'intervention': 'decision_confidence',
                'step_type': 'reassurance',
                'requires_input': True
            }
        ]

    def process_clinical_step(self, user_input: str) -> Dict:
        if self.state.current_step >= len(self.clinical_steps):
            return self.complete_clinical_flow()
            
        user_input_lower = user_input.lower().strip()
        step_idx = self.state.current_step
        current_step = self.clinical_steps[step_idx]
        step_type = current_step.get('step_type', '')

        def step_info():
            return {
                'current_step': step_idx + 1,
                'total_steps': len(self.clinical_steps),
                'intervention_type': current_step.get('intervention', '')
            }

        # Step 0: Reassure and open the conversation
        if step_idx == 0:
            reassurance = self.retriever.get_reassurance_content("decision_making", 0.5)
            return {
                'message': f"{reassurance}\n\nWhat decision are you struggling with right now?",
                'advance_step': True,
                'step_info': step_info()
            }
        # Step 1: Perfectionism check
        elif step_idx == 1:
            indicator_matches = ['perfect', 'right choice', 'best', 'wrong', 'mess up', 'regret']
            if any(indicator in user_input_lower for indicator in indicator_matches):
                reassurance = self.retriever.get_reassurance_content("decision_making", 0.8)
                return {
                    'message': (
                        f"{reassurance}\n\nThe pressure to choose perfectly is often what makes decisions feel impossible. "
                        "Let's work on making a 'good enough' choice."
                    ),
                    'perfectionism_identified': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': "Thanks for sharing how you're feeling about this decision. Let's move forward to look at your options.",
                    'advance_step': True,
                    'step_info': step_info()
                }

        # Step 2: Decision framework (identify options)
        elif step_idx == 2:
            technique = self.retriever.get_technique_for_scenario(
                scenario="decision_making",
                intensity="medium",
                user_context={"preferred_technique_type": "framework"}
            )
            option_indicators = ['or', 'either', 'option', 'choice', 'could']
            if any(indicator in user_input_lower for indicator in option_indicators):
                return {
                    'message': (
                        f"Good—you have multiple possibilities (which is a strength, even if it feels overwhelming)\n\n"
                        f"Here's a decision framework to guide you:\n\n{technique}\n\n"
                        "Now, what matters most to you in making this choice?"
                    ),
                    'advance_step': True,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': f"Let's clarify your options first. {technique}\n\nList 2-3 main possibilities you're considering.",
                    'advance_step': False,
                    'step_info': step_info()
                }

        # Step 3: Values clarification
        elif step_idx == 3:
            education = self.retriever.get_educational_content(
                scenario="decision_making",
                topic="values clarification"
            )
            return {
                'message': (
                    f"Values matter deeply in our decisions. {education}\n\n"
                    "What feels most important to you in this decision?"
                ),
                'advance_step': True,
                'step_info': step_info()
            }

        # Step 4: "Good Enough" principle (dynamic technique)
        elif step_idx == 4:
            technique = self.retriever.get_technique_for_scenario(
                scenario="decision_making",
                intensity="low",
                user_context={"preferred_technique_type": "good_enough"}
            )
            return {
                'message': (
                    f"Consider this principle:\n\n{technique}\n\n"
                    "Instead of seeking perfection, look for what is 'good enough' for your values and needs."
                ),
                'advance_step': True,
                'step_info': step_info()
            }

        # Step 5: Time limit for decision process, using educational content
        elif step_idx == 5:
            education = self.retriever.get_educational_content(
                scenario="decision_making",
                topic="decision time limits"
            )
            time_indicators = ['day', 'week', 'month', 'tomorrow', 'soon', 'deadline']
            if any(indicator in user_input_lower for indicator in time_indicators):
                return {
                    'message': (
                        f"{education}\n\nThat sounds like a reasonable timeframe. "
                        "Having that kind of deadline can actually reduce anxiety because it creates closure—you can make a good decision within that limit."
                    ),
                    'deadline_set': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': (
                        f"{education}\n\nTo help with decision paralysis, try picking a specific date or event by which you'll decide. "
                        "When would feel right?"
                    ),
                    'advance_step': False,
                    'step_info': step_info()
                }

        # Step 6: Decision confidence—validation/reassurance
        elif step_idx == 6:
            reassurance = self.retriever.get_reassurance_content("decision_making", 0.4)
            return {
                'message': (
                    f"{reassurance}\n\nRemember: you've made many decisions in your life, including difficult ones. "
                    "What qualities or strategies have helped you make good decisions in the past?"
                ),
                'advance_step': True,
                'step_info': step_info()
            }

        response = self.get_current_step_response()
        response['step_info'] = step_info()
        return response

    def get_clinical_resources(self) -> List[Dict]:
        return [
            {
                'type': 'decision_framework',
                'title': 'Structured Decision Process',
                'evidence_base': 'Decision science and cognitive therapy',
                'description': 'Use the options-values-outcomes framework for future decisions'
            },
            {
                'type': 'perfectionism_management',
                'title': 'Good Enough Decision Making',
                'evidence_base': 'Satisficing theory (Herbert Simon)',
                'description': 'Practice making "good enough" decisions to reduce analysis paralysis'
            },
            {
                'type': 'values_work',
                'title': 'Values Clarification',
                'evidence_base': 'Acceptance and Commitment Therapy',
                'description': 'Regular values reflection improves decision confidence'
            }
        ]
