from ..base_flow import ClinicalTherapeuticFlow
from ...rag.content_retriever import ContentRetriever
from typing import Dict, Optional, List

class PhysicalTriggersFlow(ClinicalTherapeuticFlow):
    """
    Clinical physical triggers flow with dynamic RAG-powered interventions for:
    - Trigger identification and mapping
    - Somatic awareness training
    - Environmental modification strategies
    - Body-based anxiety management
    """

    def __init__(self):
        super().__init__(
            flow_name="Physical Triggers Management",
            scenario="physical_triggers",
            intensity="medium"
        )
        self.retriever = ContentRetriever()
        self.clinical_steps = [
            {
                'intervention': 'trigger_assessment',
                'step_type': 'assessment',
                'requires_input': True
            },
            {
                'intervention': 'trigger_validation',
                'step_type': 'reassurance',
                'requires_input': False
            },
            {
                'intervention': 'body_awareness',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': 'trigger_differentiation',
                'step_type': 'education',
                'requires_input': True
            },
            {
                'intervention': 'immediate_modifications',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': 'grounding_through_body',
                'step_type': 'technique',
                'requires_input': True
            },
            {
                'intervention': 'trigger_prevention',
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

        # Step 0: Trigger assessment
        if step_idx == 0:
            reassurance = self.retriever.get_reassurance_content("physical_triggers", 0.6)
            common_triggers = ['caffeine', 'tired', 'crowd', 'noise', 'light', 'heat', 'cold', 'hungry']
            mentioned_triggers = [trigger for trigger in common_triggers if trigger in user_input_lower]
            if mentioned_triggers:
                return {
                    'message': (
                        f"{reassurance}\n\nI can see that {', '.join(mentioned_triggers)} are triggers for you. "
                        "These are very common anxiety triggers—you're definitely not alone in this experience."
                    ),
                    'triggers_identified': mentioned_triggers,
                    'advance_step': True,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': (
                        f"{reassurance}\n\nWhat specific physical experiences or environments tend to make "
                        "your anxiety worse? (Examples: caffeine, crowds, bright lights, fatigue)"
                    ),
                    'advance_step': True,
                    'step_info': step_info()
                }

        # Step 1: Trigger validation
        elif step_idx == 1:
            reassurance = self.retriever.get_reassurance_content("physical_triggers", 0.5)
            return {
                'message': reassurance,
                'advance_step': True,
                'step_info': step_info()
            }

        # Step 2: Body awareness training
        elif step_idx == 2:
            technique = self.retriever.get_technique_for_scenario(
                scenario="physical_triggers",
                intensity="medium",
                user_context={"preferred_technique_type": "body_awareness"}
            )
            body_awareness_indicators = ['tight', 'tense', 'shallow', 'fast', 'racing', 'stomach', 'shoulders', 'jaw']
            if any(indicator in user_input_lower for indicator in body_awareness_indicators):
                return {
                    'message': (
                        "Good awareness! Noticing these physical sensations is the first step in managing them. "
                        "Your body is giving you information about your stress level."
                    ),
                    'body_awareness_success': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': f"Let's increase your body awareness:\n\n{technique}\n\nJust observe without trying to change anything.",
                    'advance_step': True,
                    'step_info': step_info()
                }

        # Step 3: Trigger differentiation education
        elif step_idx == 3:
            education = self.retriever.get_educational_content(
                scenario="physical_triggers",
                topic="sensation differentiation"
            )
            return {
                'message': (
                    f"{education}\n\nWhich category do your current physical sensations fall into?"
                ),
                'advance_step': True,
                'step_info': step_info()
            }

        # Step 4: Immediate environmental modifications
        elif step_idx == 4:
            technique = self.retriever.get_technique_for_scenario(
                scenario="physical_triggers",
                intensity="low",
                user_context={"preferred_technique_type": "environmental_modification"}
            )
            if any(word in user_input_lower for word in ['yes', 'can', 'will', 'need']):
                return {
                    'message': (
                        "Excellent! Making these immediate adjustments can often provide quick relief. "
                        "Even small environmental changes can have a big impact on how you feel."
                    ),
                    'modifications_accepted': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': f"Let's identify immediate changes you can make:\n\n{technique}\n\nWhat feels most needed right now?",
                    'advance_step': True,
                    'step_info': step_info()
                }

        # Step 5: Somatic grounding techniques
        elif step_idx == 5:
            technique = self.retriever.get_technique_for_scenario(
                scenario="physical_triggers",
                intensity="medium",
                user_context={"preferred_technique_type": "somatic_grounding"}
            )
            if any(word in user_input_lower for word in ['better', 'calmer', 'helped', 'good']):
                return {
                    'message': (
                        "That's wonderful! Physical grounding techniques work because they give your nervous system "
                        "concrete, safe sensations to focus on instead of anxiety symptoms."
                    ),
                    'grounding_success': True,
                    'advance_step': True,
                    'step_info': step_info()
                }
            else:
                return {
                    'message': (
                        f"Let's use your body for grounding:\n\n{technique}\n\n"
                        "Try one of these and notice how it affects your anxiety."
                    ),
                    'advance_step': True,
                    'step_info': step_info()
                }

        # Step 6: Future trigger prevention planning
        elif step_idx == 6:
            technique = self.retriever.get_technique_for_scenario(
                scenario="physical_triggers",
                intensity="low",
                user_context={"preferred_technique_type": "trigger_prevention"}
            )
            return {
                'message': (
                    f"For future trigger management:\n\n{technique}\n\n"
                    "What feels like the most helpful prevention strategy for you?"
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

        step_idx = self.state.current_step
        current_step = self.clinical_steps[step_idx]
        step_type = current_step.get('step_type', 'general')

        if step_type == 'reassurance':
            content = self.retriever.get_reassurance_content("physical_triggers", 0.6)
        elif step_type == 'education':
            content = self.retriever.get_educational_content("physical_triggers")
        elif step_type == 'technique':
            content = self.retriever.get_technique_for_scenario("physical_triggers", "medium")
        elif step_type == 'assessment':
            content = "Let's explore what physical experiences trigger your anxiety."
        else:
            content = "Let's continue working on managing your physical triggers."

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
        """Clinical follow-up resources for physical trigger management"""
        return [
            {
                'type': 'trigger_management',
                'title': 'Personal Trigger Map',
                'evidence_base': 'Somatic experiencing and trauma therapy',
                'description': 'Create a personalized map of your physical triggers and effective responses'
            },
            {
                'type': 'body_awareness',
                'title': 'Daily Body Scan Practice',
                'evidence_base': 'Mindfulness-based stress reduction',
                'description': 'Regular body awareness practice increases sensitivity to early anxiety warning signs'
            },
            {
                'type': 'environmental_modification',
                'title': 'Trigger Prevention Toolkit',
                'evidence_base': 'Sensory processing and anxiety management',
                'description': 'Assemble a personal toolkit for managing triggering environments'
            },
            {
                'type': 'somatic_techniques',
                'title': 'Body-Based Grounding Methods',
                'evidence_base': 'Somatic experiencing therapy',
                'description': 'Physical grounding techniques for immediate anxiety relief'
            }
        ]
