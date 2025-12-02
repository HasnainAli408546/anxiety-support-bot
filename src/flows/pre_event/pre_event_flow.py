from ..base_flow import ClinicalTherapeuticFlow
from ...rag.content_retriever import ContentRetriever
from typing import Dict, Optional, List

class PreEventFlow(ClinicalTherapeuticFlow):
    """
    Clinical pre-event flow with dynamic RAG-powered interventions for:
    - Preparation strategies and checklists
    - Confidence-building affirmations
    - Visualization techniques
    - Post-event processing scripts
    """
    
    def __init__(self):
        super().__init__(
            flow_name="Pre-Event Anxiety Support",
            scenario="pre_event",
            intensity="medium"
        )
        
        self.retriever = ContentRetriever()
        self.clinical_steps = [
            {
                'intervention': 'event_assessment',
                'step_type': 'assessment',
                'requires_input': True
            },
            {
                'intervention': 'normalization',
                'step_type': 'reassurance',
                'requires_input': False
            },
            {
                'intervention': 'preparation_checklist',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': 'visualization',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': 'confidence_affirmations',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': 'post_event_planning',
                'step_type': 'technique',
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
        
        # Step 0: Event assessment
        if step_idx == 0:
            reassurance = self.retriever.get_reassurance_content("pre_event", 0.6)
            return {
                'message': (
                    f"{reassurance}\n\nTell me briefly: what event is causing you anxiety, and when is it happening?"
                ),
                'advance_step': True,
                'step_info': step_info()
            }
        
        # Step 1: Normalization and reframing
        elif step_idx == 1:
            education = self.retriever.get_educational_content(
                scenario="pre_event",
                topic="anxiety normalization"
            )
            return {
                'message': education,
                'advance_step': True,
                'step_info': step_info()
            }
        
        # Step 2: Preparation checklist
        elif step_idx == 2:
            technique = self.retriever.get_technique_for_scenario(
                scenario="pre_event",
                intensity="medium",
                user_context={"preferred_technique_type": "preparation"}
            )
            preparation_indicators = ['practice', 'prepare', 'study', 'rehearse', 'plan', 'ready']
            if any(indicator in user_input_lower for indicator in preparation_indicators):
                return {
                    'message': (
                        "Excellent! Those are concrete steps that will build your confidence. "
                        "Preparation is one of the most effective ways to reduce pre-event anxiety."
                    ),
                    'preparation_identified': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': (
                        f"Let's build your confidence through preparation:\n\n{technique}\n\n"
                        "Share what comes to mind—even small steps count."
                    ),
                    'advance_step': True,
                    'step_info': step_info()
                }
        
        # Step 3: Visualization technique
        elif step_idx == 3:
            technique = self.retriever.get_technique_for_scenario(
                scenario="pre_event",
                intensity="medium",
                user_context={"preferred_technique_type": "visualization"}
            )
            if 'done' in user_input_lower or 'finished' in user_input_lower:
                return {
                    'message': (
                        "Great job with the visualization! Mental rehearsal is a powerful tool used by athletes and performers. "
                        "The more vivid and positive your mental practice, the more confident you'll feel."
                    ),
                    'visualization_completed': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': (
                        f"Let's use positive visualization:\n\n{technique}\n\n"
                        "Spend 2-3 minutes on this mental rehearsal. Reply 'done' when finished."
                    ),
                    'advance_step': True,
                    'step_info': step_info()
                }
        
        # Step 4: Confidence affirmations
        elif step_idx == 4:
            technique = self.retriever.get_technique_for_scenario(
                scenario="pre_event",
                intensity="low",
                user_context={"preferred_technique_type": "affirmations"}
            )
            confidence_indicators = ['capable', 'prepared', 'can do', 'strong', 'ready', 'confident']
            if any(indicator in user_input_lower for indicator in confidence_indicators):
                return {
                    'message': (
                        "I can hear the strength in your words! These positive self-statements will serve you well. "
                        "Keep repeating them, especially as the event approaches."
                    ),
                    'affirmations_resonated': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': (
                        f"Let's build some positive self-statements:\n\n{technique}\n\n"
                        "Which of these resonates with you, or do you have your own empowering statement?"
                    ),
                    'advance_step': True,
                    'step_info': step_info()
                }
        
        # Step 5: Post-event self-compassion planning
        elif step_idx == 5:
            technique = self.retriever.get_technique_for_scenario(
                scenario="pre_event",
                intensity="low",
                user_context={"preferred_technique_type": "self_compassion"}
            )
            return {
                'message': (
                    f"Finally, let's plan for after the event:\n\n{technique}\n\n"
                    "What's one nice thing you can do for yourself after the event?"
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
            content = self.retriever.get_reassurance_content("pre_event", 0.5)
        elif step_type == 'education':
            content = self.retriever.get_educational_content("pre_event")
        elif step_type == 'technique':
            content = self.retriever.get_technique_for_scenario("pre_event", "medium")
        elif step_type == 'assessment':
            content = "Let's explore what upcoming event is causing you anxiety."
        else:
            content = "Let's continue preparing you for your upcoming event."

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
                'type': 'preparation_strategy',
                'title': 'Event Preparation Checklist',
                'evidence_base': 'Self-efficacy theory',
                'description': 'Thorough preparation reduces uncertainty and builds confidence'
            },
            {
                'type': 'cognitive_technique',
                'title': 'Positive Visualization',
                'evidence_base': 'Mental rehearsal for performance',
                'description': 'Practice visualization daily leading up to the event'
            },
            {
                'type': 'self_compassion',
                'title': 'Post-Event Self-Care',
                'evidence_base': 'Self-compassion reduces performance anxiety',
                'description': 'Plan recovery activities regardless of event outcome'
            },
            {
                'type': 'performance_techniques',
                'title': 'Event Day Strategies',
                'evidence_base': 'Performance psychology',
                'description': 'Breathing techniques and grounding exercises for the day of the event'
            }
        ]
