from ..base_flow import ClinicalTherapeuticFlow
from ...rag.content_retriever import ContentRetriever
from typing import Dict, Optional, List

class UncertaintyFlow(ClinicalTherapeuticFlow):
    """
    Clinical uncertainty flow with dynamic RAG-powered interventions for:
    - Uncertainty tolerance building
    - Worry time technique
    - Cognitive restructuring for catastrophic thinking
    - Present moment grounding
    """
    
    def __init__(self):
        super().__init__(
            flow_name="Uncertainty and Worry Support",
            scenario="uncertainty", 
            intensity="medium"
        )
        
        self.retriever = ContentRetriever()
        self.clinical_steps = [
            {
                'intervention': 'uncertainty_assessment',
                'step_type': 'assessment',
                'requires_input': True
            },
            {
                'intervention': 'uncertainty_normalization',
                'step_type': 'reassurance',
                'requires_input': False
            },
            {
                'intervention': 'worry_vs_problem_solving',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': 'worry_time_technique',
                'step_type': 'technique',
                'requires_input': False
            },
            {
                'intervention': 'uncertainty_tolerance',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': 'present_moment_grounding',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': 'coping_confidence',
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

        def step_info():
            return {
                'current_step': step_idx + 1,
                'total_steps': len(self.clinical_steps),
                'intervention_type': current_step.get('intervention', '')
            }

        # Step 0: Uncertainty assessment
        if step_idx == 0:
            reassurance = self.retriever.get_reassurance_content("uncertainty", 0.7)
            return {
                'message': (
                    f"{reassurance}\n\nWhat specific situation or outcome are you most worried about not knowing?"
                ),
                'advance_step': True,
                'step_info': step_info()
            }

        # Step 1: Uncertainty normalization
        elif step_idx == 1:
            reassurance = self.retriever.get_reassurance_content("uncertainty", 0.5)
            education = self.retriever.get_educational_content(
                scenario="uncertainty",
                topic="uncertainty normalization"
            )
            return {
                'message': f"{education}\n\n{reassurance}",
                'advance_step': True,
                'step_info': step_info()
            }

        # Step 2: Problem-solving vs worry distinction
        elif step_idx == 2:
            technique = self.retriever.get_technique_for_scenario(
                scenario="uncertainty",
                intensity="medium",
                user_context={"preferred_technique_type": "problem_solving"}
            )
            action_indicators = ['can', 'could', 'action', 'step', 'do something']
            if any(indicator in user_input_lower for indicator in action_indicators):
                return {
                    'message': (
                        "Great—it sounds like there are some actions you can take. Focus your mental energy "
                        "on those actionable steps rather than the parts you can't control. What feels like "
                        "the most important action to take first?"
                    ),
                    'problem_solving_mode': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': (
                        "This sounds like it falls into the 'worry' category—something important to you that "
                        "you can't directly control right now. That's when worry management techniques become most helpful."
                    ),
                    'worry_category': True,
                    'advance_step': True,
                    'step_info': step_info()
                }

        # Step 3: Worry time technique
        elif step_idx == 3:
            technique = self.retriever.get_technique_for_scenario(
                scenario="uncertainty",
                intensity="medium",
                user_context={"preferred_technique_type": "worry_time"}
            )
            return {
                'message': f"For worries we can't act on right now:\n\n{technique}",
                'advance_step': True,
                'step_info': step_info()
            }

        # Step 4: Uncertainty tolerance practice
        elif step_idx == 4:
            technique = self.retriever.get_technique_for_scenario(
                scenario="uncertainty",
                intensity="high",
                user_context={"preferred_technique_type": "uncertainty_tolerance"}
            )
            if any(word in user_input_lower for word in ['hard', 'difficult', 'uncomfortable', 'scary']):
                return {
                    'message': (
                        "Yes, it is difficult—that's exactly the point. You're practicing tolerating discomfort "
                        "rather than avoiding it. The discomfort won't hurt you, even though it feels unpleasant."
                    ),
                    'validate_difficulty': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
            elif any(word in user_input_lower for word in ['okay', 'better', 'calming', 'helpful']):
                return {
                    'message': (
                        "That's wonderful! You're building uncertainty tolerance, which is like strengthening a muscle. "
                        "Each time you practice, it gets a little easier."
                    ),
                    'technique_success': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': (
                        f"Let's practice uncertainty tolerance:\n\n{technique}\n\n"
                        "Try saying this phrase—how does it feel? The goal isn't to like uncertainty, "
                        "just to tolerate it without letting it control your day."
                    ),
                    'advance_step': True,
                    'step_info': step_info()
                }

        # Step 5: Present moment grounding
        elif step_idx == 5:
            technique = self.retriever.get_technique_for_scenario(
                scenario="uncertainty",
                intensity="low",
                user_context={"preferred_technique_type": "present_moment"}
            )
            grounding_indicators = ['breathing', 'sitting', 'see', 'hear', 'feel', 'notice']
            if any(indicator in user_input_lower for indicator in grounding_indicators):
                return {
                    'message': (
                        "Excellent! You're anchoring yourself in the present moment. This is where your power lies—"
                        "not in the uncertain future, but in the reality of right now."
                    ),
                    'grounding_success': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': (
                        f"Uncertainty anxiety pulls us into the future. Let's anchor in the present:\n\n{technique}\n\n"
                        "Share what you notice in this moment."
                    ),
                    'advance_step': True,
                    'step_info': step_info()
                }

        # Step 6: Coping confidence building
        elif step_idx == 6:
            reassurance = self.retriever.get_reassurance_content("uncertainty", 0.3)
            return {
                'message': (
                    f"{reassurance}\n\nThink about other times you've faced uncertainty in your life. "
                    "You've handled unknown situations before, even when they felt overwhelming. "
                    "What strengths or coping skills did you use then that you still have now?"
                ),
                'advance_step': True,
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
            content = self.retriever.get_reassurance_content("uncertainty", 0.5)
        elif step_type == 'education':
            content = self.retriever.get_educational_content("uncertainty")
        elif step_type == 'technique':
            content = self.retriever.get_technique_for_scenario("uncertainty", "medium")
        elif step_type == 'assessment':
            content = "Let's explore what uncertain situation is causing you anxiety."
        else:
            content = "Let's continue working on managing uncertainty and worry."

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

    def get_clinical_resources(self) -> List[Dict]:
        return [
            {
                'type': 'worry_management',
                'title': 'Scheduled Worry Time',
                'evidence_base': 'GAD treatment protocols',
                'description': 'Practice 15-minute daily worry sessions to contain anxiety'
            },
            {
                'type': 'uncertainty_tolerance',
                'title': 'Daily Uncertainty Practice',
                'evidence_base': 'Intolerance of uncertainty therapy',
                'description': 'Practice the uncertainty tolerance phrase during low-anxiety moments'
            },
            {
                'type': 'mindfulness',
                'title': 'Present Moment Awareness',
                'evidence_base': 'Mindfulness-based anxiety treatment',
                'description': 'Regular grounding in present reality counters future-focused worry'
            },
            {
                'type': 'cognitive_techniques',
                'title': 'Problem-Solving vs Worry Distinction',
                'evidence_base': 'Cognitive Behavioral Therapy',
                'description': 'Learn to identify when concerns are actionable vs when they require acceptance'
            }
        ]
