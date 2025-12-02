from ..base_flow import ClinicalTherapeuticFlow
from ...rag.content_retriever import ContentRetriever
from datetime import datetime
from typing import Dict, Optional, List

class IsolationFlow(ClinicalTherapeuticFlow):
    """
    Clinical isolation flow with dynamic RAG-powered interventions for:
    - Validation and normalization of loneliness
    - Self-compassion exercises
    - Connection activation strategies
    - Social anxiety management
    """
    
    def __init__(self):
        super().__init__(
            flow_name="Isolation and Loneliness Support",
            scenario="isolation",
            intensity="medium"
        )
        
        self.retriever = ContentRetriever()
        self.clinical_steps = [
            {
                'intervention': 'loneliness_validation',
                'step_type': 'reassurance',
                'requires_input': True
            },
            {
                'intervention': 'safety_assessment',
                'step_type': 'safety_check',
                'requires_input': True
            },
            {
                'intervention': 'loneliness_psychoeducation',
                'step_type': 'education',
                'requires_input': False
            },
            {
                'intervention': 'self_compassion_exercise',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': 'connection_inventory',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': 'small_connection_step',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': 'self_care_planning',
                'step_type': 'technique',
                'requires_input': True
            }
        ]

    def process_clinical_step(self, user_input: str) -> Dict:
        """Isolation-specific clinical interventions with dynamic RAG content"""
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
        
        # Step 0: Loneliness validation with dynamic reassurance
        if step_idx == 0:
            reassurance = self.retriever.get_reassurance_content("isolation", 0.7)
            return {
                'message': f"{reassurance}\n\nCan you tell me a bit about what's making you feel most alone right now?",
                'advance_step': True,
                'step_info': step_info()
            }
        
        # Step 1: Safety assessment for self-harm risk
        elif step_idx == 1:
            risk_indicators = ['yes', 'sometimes', 'hurt myself', 'end it', 'die']
            if any(indicator in user_input_lower for indicator in risk_indicators):
                self.state.safety_flags.append('self_harm_risk_identified')
                return {
                    'message': (
                        "I'm very concerned about your safety. Please reach out for immediate help:\n\n"
                        "• National Suicide Prevention Lifeline: 988\n"
                        "• Crisis Text Line: Text HOME to 741741\n"
                        "• Emergency Services: 911\n\n"
                        "Your life has value, and there are people who want to help. Can you reach out to one of these resources right now?"
                    ),
                    'crisis_intervention': True,
                    'safety_concern': True,
                    'advance_step': False,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': "Thank you for letting me know. I'm glad you're reaching out for support instead of dealing with this alone.",
                    'advance_step': True,
                    'step_info': step_info()
                }
        
        # Step 2: Loneliness psychoeducation
        elif step_idx == 2:
            education = self.retriever.get_educational_content(
                scenario="isolation",
                topic="loneliness understanding"
            )
            return {
                'message': education,
                'advance_step': True,
                'step_info': step_info()
            }
        
        # Step 3: Self-compassion exercise
        elif step_idx == 3:
            technique = self.retriever.get_technique_for_scenario(
                scenario="isolation",
                intensity="medium",
                user_context={"preferred_technique_type": "self_compassion"}
            )

            # Process user feedback on self-compassion
            if any(word in user_input_lower for word in ['good', 'better', 'helpful', 'calming']):
                return {
                    'message': (
                        "I'm glad that felt helpful. Self-compassion is like building an internal supportive friend. "
                        "The more you practice, the stronger that voice becomes."
                    ),
                    'technique_success': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
            elif any(word in user_input_lower for word in ['weird', 'hard', 'difficult', 'fake']):
                return {
                    'message': (
                        "It can feel strange at first—many people aren't used to being kind to themselves. "
                        "That's okay. Even trying is an act of self-compassion."
                    ),
                    'normalize_difficulty': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': f"Let's practice self-compassion:\n\n{technique}\n\nHow does that feel?",
                    'advance_step': True,
                    'step_info': step_info()
                }
        
        # Step 4: Connection inventory
        elif step_idx == 4:
            technique = self.retriever.get_technique_for_scenario(
                scenario="isolation",
                intensity="low",
                user_context={"preferred_technique_type": "connection_mapping"}
            )

            if any(word in user_input_lower for word in ['no', 'nobody', 'no one', 'can\'t']):
                return {
                    'message': (
                        "It feels like there's no one right now, and that's incredibly hard. Sometimes connection starts with very indirect ways—"
                        "like reading comments on a forum, watching a livestream, or even calling a helpline just to hear a kind voice. These count too."
                    ),
                    'alternative_connections': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': (
                        "That's wonderful that you could think of a possibility. Even having that awareness is the beginning of reconnection."
                    ),
                    'connection_identified': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
        
        # Step 5: Small connection step
        elif step_idx == 5:
            technique = self.retriever.get_technique_for_scenario(
                scenario="isolation",
                intensity="low",
                user_context={"preferred_technique_type": "small_steps"}
            )
            return {
                'message': f"{technique}\n\nWhat feels like the smallest, most manageable step you could take today?",
                'advance_step': True,
                'step_info': step_info()
            }
        
        # Step 6: Self-care planning
        elif step_idx == 6:
            technique = self.retriever.get_technique_for_scenario(
                scenario="isolation",
                intensity="low",
                user_context={"preferred_technique_type": "self_care"}
            )
            return {
                'message': (
                    f"While you're working on building connections, it's important to care for yourself:\n\n{technique}\n\n"
                    "What's one nurturing thing you can do for yourself today? Something that feels like giving yourself a gentle hug?"
                ),
                'advance_step': True,
                'step_info': step_info()
            }

        response = self.get_current_step_response()
        response['step_info'] = step_info()
        return response
    
    def get_current_step_response(self) -> Dict:
        """Get current step response with RAG content if no specific processing needed"""
        if self.state.current_step >= len(self.clinical_steps):
            res = self.complete_clinical_flow()
            res['step_info'] = {
                'current_step': len(self.clinical_steps),
                'total_steps': len(self.clinical_steps),
                'intervention_type': 'completed'
            }
            return res
            
        current_step = self.clinical_steps[self.state.current_step]
        step_type = current_step.get('step_type', 'general')
        
        # Use RAG to generate appropriate content based on step type
        if step_type == 'reassurance':
            content = self.retriever.get_reassurance_content("isolation", 0.6)
        elif step_type == 'education':
            content = self.retriever.get_educational_content("isolation")
        elif step_type == 'technique':
            content = self.retriever.get_technique_for_scenario("isolation", "medium")
        else:
            content = "Let's continue working through this together."
        
        return {
            'message': content,
            'intervention_type': current_step['intervention'],
            'step_number': self.state.current_step + 1,
            'total_steps': len(self.clinical_steps),
            'flow_name': self.flow_name,
            'scenario': self.scenario,
            'requires_input': current_step.get('requires_input', True),
            'advance_step': True,
            'step_info': {
                'current_step': self.state.current_step + 1,
                'total_steps': len(self.clinical_steps),
                'intervention_type': current_step['intervention']
            }
        }
    
    def get_clinical_resources(self) -> List[Dict]:
        """Clinical follow-up resources for isolation and loneliness"""
        return [
            {
                'type': 'self_compassion',
                'title': 'Daily Self-Compassion Practice',
                'evidence_base': 'Kristin Neff\'s self-compassion research',
                'description': 'Practice the self-compassion phrases daily, especially during difficult moments'
            },
            {
                'type': 'connection_strategy',
                'title': 'Gradual Social Reconnection',
                'evidence_base': 'Behavioral activation for depression',
                'description': 'Start with small, low-pressure social interactions and gradually build up'
            },
            {
                'type': 'crisis_resource',
                'title': 'Loneliness Support Resources',
                'evidence_base': 'Crisis intervention protocols',
                'description': 'Crisis Text Line (HOME to 741741) available 24/7 for isolation and loneliness support'
            },
            {
                'type': 'online_communities',
                'title': 'Safe Online Connection Spaces',
                'evidence_base': 'Digital mental health interventions',
                'description': 'Moderated support groups and forums for people experiencing loneliness'
            }
        ]
