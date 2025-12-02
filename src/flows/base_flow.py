"""
Base therapeutic flow following clinical guidelines from your implementation guide
Implements evidence-based CBT, breathing, and grounding techniques with RAG integration
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from ..safety.crisis_detector import detect_crisis_keywords  # From your Phase 2

class FlowState:
    """Manages therapeutic flow session state with RAG tracking"""
    def __init__(self):
        self.current_step = 0
        self.user_responses = {}
        self.effectiveness_ratings = []
        self.start_time = datetime.now()
        self.completion_status = 'in_progress'
        self.safety_flags = []
        self.technique_preferences = {}
        # RAG-specific state
        self.content_retrievals = 0
        self.dynamic_content_used = []
        self.personalization_applied = False

class ClinicalTherapeuticFlow:
    """
    Base class implementing clinical best practices with RAG-powered content delivery
    Follows CBT, mindfulness, and grounding exercise protocols
    """
    
    def __init__(self, flow_name: str, scenario: str, intensity: str):
        self.flow_name = flow_name
        self.scenario = scenario  # panic, sleep, pre_event, isolation, uncertainty, decision_making, physical_triggers
        self.intensity = intensity  # crisis, high, medium, low
        self.state = FlowState()
        self.clinical_steps = []
        
        # RAG integration will be initialized by child classes
        # (Each flow initializes its own ContentRetriever in __init__)
        
    def start_flow(self, user_context: Dict = None) -> Dict:
        """Initialize flow with safety and clinical protocols"""
        self.state = FlowState()
        
        # Store user context for personalization
        if user_context:
            self.state.user_context = user_context
            emotion_scores = user_context.get('emotion_scores', {})
            if any(score > 0.7 for score in emotion_scores.values()):
                self.state.personalization_applied = True
        
        # Immediate crisis detection per your Phase 2 requirements
        if user_context and user_context.get('text'):
            crisis_detected = detect_crisis_keywords(user_context['text'])
            if crisis_detected:
                return self.activate_crisis_protocol()
        
        return self.get_current_step_response()
    
    def activate_crisis_protocol(self) -> Dict:
        """Crisis override per your implementation guide"""
        self.state.safety_flags.append('crisis_protocol_activated')
        
        return {
            'flow_type': 'crisis_override',
            'message': 'I notice you might be in crisis. Your safety is the priority.',
            'crisis_resources': [
                {
                    'name': 'National Suicide Prevention Lifeline',
                    'contact': '988',
                    'available': '24/7'
                },
                {
                    'name': 'Crisis Text Line',
                    'contact': 'Text HOME to 741741',
                    'available': '24/7'
                },
                {
                    'name': 'Emergency Services',
                    'contact': '911',
                    'available': 'Immediate emergency response'
                }
            ],
            'disable_free_form': True,  # Per your Phase 4 safety requirements
            'immediate_escalation': True,
            'crisis_override': True
        }
    
    def process_user_input(self, user_input: str) -> Dict:
        """Process input with safety monitoring and RAG context tracking"""
        # Store response for clinical tracking
        self.state.user_responses[self.state.current_step] = {
            'input': user_input,
            'timestamp': datetime.now().isoformat(),
            'safety_screened': True,
            'step_intervention': self.clinical_steps[self.state.current_step]['intervention'] if self.state.current_step < len(self.clinical_steps) else 'completed'
        }
        
        # Continuous crisis monitoring per your guide
        if detect_crisis_keywords(user_input):
            self.state.safety_flags.append('crisis_detected_during_flow')
            return self.activate_crisis_protocol()
        
        # Process clinical step logic (may use RAG in child classes)
        response = self.process_clinical_step(user_input)
        
        # Track if RAG content was likely used (child classes will have retriever)
        if hasattr(self, 'retriever') and response.get('message'):
            self.state.content_retrievals += 1
            if self.state.current_step < len(self.clinical_steps):
                intervention_type = self.clinical_steps[self.state.current_step].get('intervention', 'unknown')
                self.state.dynamic_content_used.append(intervention_type)
        
        # Advance step if indicated
        if response.get('advance_step', True):
            self.state.current_step += 1
            
        return response
    
    def process_clinical_step(self, user_input: str) -> Dict:
        """Clinical step processing - to be overridden by specific flows"""
        return self.get_current_step_response()
    
    def get_current_step_response(self) -> Dict:
        """Get current clinical intervention step"""
        if self.state.current_step >= len(self.clinical_steps):
            return self.complete_clinical_flow()
            
        current_step = self.clinical_steps[self.state.current_step]
        
        # Base response structure
        response = {
            'intervention_type': current_step.get('intervention', 'general'),
            'step_number': self.state.current_step + 1,
            'total_steps': len(self.clinical_steps),
            'flow_name': self.flow_name,
            'scenario': self.scenario,
            'requires_input': current_step.get('requires_input', True),
            'clinical_guidance': current_step.get('clinical_notes', ''),
            'advance_step': True
        }
        
        # Add message - child classes with RAG will override this method
        if 'clinical_message' in current_step:
            response['message'] = current_step['clinical_message']
        else:
            response['message'] = f"Continuing with {self.flow_name} - step {self.state.current_step + 1}"
        
        return response
    
    def complete_clinical_flow(self) -> Dict:
        """Complete with clinical outcome tracking including RAG analytics"""
        self.state.completion_status = 'completed'
        
        # Calculate completion metrics
        duration = (datetime.now() - self.state.start_time).total_seconds()
        completion_rate = min(self.state.current_step, len(self.clinical_steps)) / len(self.clinical_steps) if self.clinical_steps else 1.0

        return {
            'message': f"You've completed the {self.flow_name}. These evidence-based techniques become more effective with practice.",
            'flow_completed': True,
            'clinical_outcomes': {
                'duration_seconds': duration,
                'completion_rate': completion_rate,
                'effectiveness_ratings': self.state.effectiveness_ratings,
                'safety_flags': self.state.safety_flags,
                'techniques_used': [step.get('intervention', 'unknown') for step in self.clinical_steps[:self.state.current_step]],
                
                # RAG-specific outcomes
                'content_retrievals': self.state.content_retrievals,
                'dynamic_content_used': self.state.dynamic_content_used,
                'personalization_applied': self.state.personalization_applied,
                'rag_enhanced': hasattr(self, 'retriever')
            },
            'follow_up_clinical_resources': self.get_clinical_resources(),
            'session_summary': {
                'scenario': self.scenario,
                'intensity': self.intensity,
                'steps_completed': self.state.current_step,
                'clinical_success': completion_rate >= 0.8 and len(self.state.safety_flags) == 0
            },
            'step_info': {
                'current_step': len(self.clinical_steps),
                'total_steps': len(self.clinical_steps),
                'intervention_type': 'completed'
            }
        }
    
    def get_clinical_resources(self) -> List[Dict]:
        """Evidence-based follow-up resources per your guide"""
        return [
            {
                'type': 'clinical_technique',
                'title': f'{self.scenario.title()} Management Techniques',
                'evidence_base': 'CBT, mindfulness, clinical guidelines',
                'description': 'Practice these techniques daily when calm for better effectiveness during anxiety',
                'scenario': self.scenario
            },
            {
                'type': 'professional_support',
                'title': 'When to Seek Additional Help',
                'evidence_base': 'Clinical care guidelines',
                'description': 'Consider professional support if symptoms persist or worsen',
                'scenario': self.scenario
            }
        ]
    
    def get_personalization_context(self) -> Dict:
        """Get user context for RAG personalization (used by child classes)"""
        context = {
            'scenario': self.scenario,
            'intensity': self.intensity,
            'current_step': self.state.current_step,
            'safety_flags': self.state.safety_flags,
            'previous_effectiveness': self.state.effectiveness_ratings
        }
        
        # Add user context if available
        if hasattr(self.state, 'user_context'):
            context.update({
                'emotion_scores': self.state.user_context.get('emotion_scores', {}),
                'user_preferences': self.state.technique_preferences
            })
        
        return context
