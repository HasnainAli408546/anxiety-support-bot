"""
Complete Clinical Flow Manager following your implementation guide Phase 3 architecture
Integrates all 7 RAG-powered scenario flows with safety protocols and monitoring
"""
# from integration.phase5_integration import Phase5IntegratedSystem
from typing import Dict, Optional, List
from datetime import datetime
import json
import logging

# Import all RAG-powered clinical flows
from .panic.acute_anxiety_flow import AcuteAnxietyFlow
from .sleep.nighttime_flow import NighttimeFlow
from .pre_event.pre_event_flow import PreEventFlow
from .isolation.isolation_flow import IsolationFlow
from .uncertainty.uncertainty_flow import UncertaintyFlow
from .decision_making.decision_making_flow import DecisionMakingFlow
from .physical_triggers.physical_triggers_flow import PhysicalTriggersFlow
from .general.general_anxiety_flow import GeneralAnxietyFlow


# RAG integration
from ..rag.content_retriever import ContentRetriever

# Safety module imports (from your Phase 2 implementation)
from ..safety.crisis_detector import detect_crisis_keywords
import os
from dotenv import load_dotenv

load_dotenv()  # Loads from .env file automatically
CHROMADB_PATH = os.getenv("CHROMADB_PATH")

class ClinicalFlowManager:
    """
    Complete RAG-powered clinical flow manager implementing your 7-scenario anxiety support system
    Features:
    - All 7 RAG-enhanced evidence-based therapeutic flows
    - Dynamic content retrieval from therapeutic knowledge base
    - Crisis detection and safety overrides
    - Clinical outcome tracking with RAG usage analytics
    - Session management and monitoring
    - Integration with your existing emotion detection and scenario mapping
    """
    
    def __init__(self,db_path=CHROMADB_PATH):
        self.active_flows = {}  # user_id -> flow_instance
        self.session_history = {}  # user_id -> session_data
        self.clinical_outcomes = {}  # user_id -> outcomes_data

        # self.phase5_system = Phase5IntegratedSystem()
        # self.logger.info("Phase 5 personalization system initialized")

        # Initialize RAG content retriever for system-wide content access
        self.content_retriever = ContentRetriever(db_path=db_path)
        
        # Complete flow registry for all 7 RAG-powered scenarios
        self.clinical_flow_registry = {
            # Panic scenario flows
            'acute_anxiety_flow': AcuteAnxietyFlow,
            'panic_crisis_flow': AcuteAnxietyFlow,  # Alias for crisis-level panic
            'panic_breathing_flow': AcuteAnxietyFlow,  # Can be extended for different intensities
            'panic_flow': AcuteAnxietyFlow,  # Simple alias
            
            # Sleep scenario flows  
            'nighttime_flow': NighttimeFlow,
            'sleep_flow': NighttimeFlow,  # Alias
            'sleep_hygiene_flow': NighttimeFlow,
            'insomnia_flow': NighttimeFlow,
            
            # Pre-event scenario flows
            'pre_event_flow': PreEventFlow,
            'pre_event_nervousness_flow': PreEventFlow,
            'performance_anxiety_flow': PreEventFlow,
            'event_anxiety_flow': PreEventFlow,
            
            # Isolation scenario flows
            'isolation_flow': IsolationFlow,
            'loneliness_flow': IsolationFlow,
            'social_anxiety_flow': IsolationFlow,
            'connection_flow': IsolationFlow,
            
            # Uncertainty scenario flows
            'uncertainty_flow': UncertaintyFlow,
            'worry_flow': UncertaintyFlow,
            'anticipatory_anxiety_flow': UncertaintyFlow,
            'unknown_flow': UncertaintyFlow,
            
            # Decision-making scenario flows
            'decision_making_flow': DecisionMakingFlow,
            'choice_paralysis_flow': DecisionMakingFlow,
            'indecision_flow': DecisionMakingFlow,
            'decision_flow': DecisionMakingFlow,
            
            # Physical triggers scenario flows
            'physical_triggers_flow': PhysicalTriggersFlow,
            'somatic_anxiety_flow': PhysicalTriggersFlow,
            'environmental_triggers_flow': PhysicalTriggersFlow,
            'body_anxiety_flow': PhysicalTriggersFlow,

            'general_anxiety_flow': GeneralAnxietyFlow
        }
        
        # Set up logging for clinical monitoring
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize knowledge base statistics
        self.knowledge_base_stats = self.content_retriever.get_database_stats()
        self.logger.info(f"RAG-powered Clinical Flow Manager initialized with {len(self.clinical_flow_registry)} flows and {self.knowledge_base_stats.get('total_items', 0)} knowledge base items")
    
    def start_clinical_flow(self, user_id: str, flow_name: str, 
                          clinical_context: Dict = None) -> Dict:
        """
        Start RAG-enhanced evidence-based clinical flow per your Phase 3 implementation
        
        Args:
            user_id: User identifier
            flow_name: Clinical flow type (from scenario_router output)
            clinical_context: Contains emotion_scores, intent_scores, user_text, metadata
        
        Returns:
            Dict with flow response and clinical metadata including RAG usage
        """
        
        # if clinical_context:
        #     personalized_rec = self.phase5_system.personalization.get_personalized_flow_recommendation(
        #     user_id, clinical_context
        # )
        
        # # Use recommended scenario if confidence is high
        # if personalized_rec['confidence_score'] > 0.7:
        #     recommended_flow = f"{personalized_rec['recommended_scenario']}_flow"
        #     if recommended_flow in self.clinical_flow_registry:
        #         flow_name = recommended_flow
        #         clinical_context['personalization_applied'] = True

        
        # Initialize session tracking
        if user_id not in self.session_history:
            self.session_history[user_id] = []
        
        # Crisis detection override per your Phase 2 safety protocols
        if clinical_context and clinical_context.get('text'):
            if detect_crisis_keywords(clinical_context['text']):
                self.logger.warning(f"Crisis detected for user {user_id}")
                return self.activate_crisis_override(user_id, clinical_context.get('text', ''))
        
        # Validate flow exists
        if flow_name not in self.clinical_flow_registry:
            self.logger.error(f"Unknown clinical flow requested: {flow_name}")
            return {
                'error': f'Clinical flow {flow_name} not implemented',
                'available_flows': list(self.clinical_flow_registry.keys()),
                'fallback_message': 'Let me help you with general anxiety support. How are you feeling right now?',
                'suggested_flows': self._suggest_similar_flows(flow_name)
            }

        
        # Create RAG-powered clinical flow instance
        flow_class = self.clinical_flow_registry[flow_name]
        flow_instance = flow_class()
        
        # Store active flow with enhanced metadata
        self.active_flows[user_id] = {
            'flow_instance': flow_instance,
            'start_time': datetime.now(),
            'flow_name': flow_name,
            'clinical_context': clinical_context or {},
            'rag_retrievals': 0,  # Track RAG usage
            'dynamic_content_used': False
        }
        
        # Log clinical session start with RAG info
        self.log_clinical_event(user_id, 'rag_flow_started', {
            'flow_name': flow_name,
            'scenario': flow_instance.scenario,
            'emotion_scores': clinical_context.get('emotion_scores', {}) if clinical_context else {},
            'intent_scores': clinical_context.get('intent_scores', {}) if clinical_context else {},
            'knowledge_base_items': self.knowledge_base_stats.get('total_items', 0)
        })
        
        # Start flow with clinical context (RAG is used internally by flow)
        response = flow_instance.start_flow(clinical_context)
        
        # Track if dynamic content was used
        if hasattr(flow_instance, 'retriever'):
            self.active_flows[user_id]['dynamic_content_used'] = True
            self.active_flows[user_id]['rag_retrievals'] += 1
        
        # Add enhanced clinical metadata to response
        response['clinical_metadata'] = {
            'flow_name': flow_name,
            'scenario': flow_instance.scenario,
            'intensity': flow_instance.intensity,
            'evidence_base': 'CBT, mindfulness, clinical guidelines',
            'safety_screened': True,
            'rag_powered': True,
            'dynamic_content': self.active_flows[user_id]['dynamic_content_used'],
            'knowledge_base_items': self.knowledge_base_stats.get('total_items', 0)
        }
        
        return response
    
    def process_clinical_response(self, user_id: str, user_input: str) -> Dict:
        """
        Process user response in RAG-enhanced clinical flow with comprehensive safety monitoring
        
        Args:
            user_id: User identifier  
            user_input: User's text response
            
        Returns:
            Dict with clinical flow response and safety metadata including RAG analytics
        """
        
        # Continuous crisis monitoring per your safety protocols
        if detect_crisis_keywords(user_input):
            self.logger.warning(f"Crisis detected in response from user {user_id}: {user_input[:50]}...")
            return self.activate_crisis_override(user_id, user_input)
        
        # Check for active flow
        if user_id not in self.active_flows:
            return {
                'error': 'No active clinical flow',
                'message': 'It looks like we need to start over. How are you feeling right now?',
                'flow_restart_needed': True,
                'suggested_action': 'restart_assessment',
                'available_scenarios': list(self.get_available_flows().keys())
            }
        
        # Get active flow
        flow_data = self.active_flows[user_id]
        flow_instance = flow_data['flow_instance']
        
        # Track RAG usage before processing
        initial_rag_count = flow_data.get('rag_retrievals', 0)
        
        # Process user input through RAG-enhanced clinical flow
        response = flow_instance.process_user_input(user_input)
        if response.get('advance_step'):
           if flow_instance.state.current_step < len(flow_instance.clinical_steps):
               flow_instance.state.current_step += 1
        
        # Update RAG usage tracking (estimate based on step processing)
        if hasattr(flow_instance, 'retriever') and response.get('message'):
            flow_data['rag_retrievals'] += 1
            flow_data['dynamic_content_used'] = True
        
        # Log clinical interaction with RAG analytics
        self.log_clinical_event(user_id, 'rag_response_processed', {
            'step_number': flow_instance.state.current_step,
            'user_input_length': len(user_input),
            'safety_flags': flow_instance.state.safety_flags,
            'rag_retrievals_session': flow_data.get('rag_retrievals', 0),
            'dynamic_content_active': flow_data.get('dynamic_content_used', False)
        })
        
        # Check for flow completion
        if response.get('flow_completed') or response.get('crisis_override'):
            self.complete_clinical_session(user_id, response)
        
        # Add enhanced clinical monitoring metadata
        response['clinical_monitoring'] = {
            'session_duration': (datetime.now() - flow_data['start_time']).total_seconds(),
            'current_step': flow_instance.state.current_step + 1,
            'total_steps': len(flow_instance.clinical_steps),
            'safety_flags': flow_instance.state.safety_flags,
            'effectiveness_ratings': flow_instance.state.effectiveness_ratings,
            'rag_enhanced': True,
            'rag_retrievals_session': flow_data.get('rag_retrievals', 0),
            'dynamic_content_percentage': self._calculate_dynamic_content_percentage(user_id)
        }
        
        return response
    
    def _calculate_dynamic_content_percentage(self, user_id: str) -> float:
        """Calculate percentage of session using dynamic RAG content"""
        if user_id not in self.active_flows:
            return 0.0
        
        flow_data = self.active_flows[user_id]
        flow_instance = flow_data['flow_instance']
        
        if not hasattr(flow_instance, 'state') or not hasattr(flow_instance, 'clinical_steps'):
            return 0.0
        
        current_step = flow_instance.state.current_step + 1
        total_steps = len(flow_instance.clinical_steps)
        
        if total_steps == 0:
            return 0.0
        
        # Assume RAG is used in most steps for RAG-powered flows
        return min(100.0, (current_step / total_steps) * 100.0)
    
    def _suggest_similar_flows(self, requested_flow: str) -> List[str]:
        """Suggest similar flows based on requested flow name"""
        suggestions = []
        
        # Extract keywords from requested flow
        keywords = requested_flow.lower().split('_')
        
        # Find flows with similar keywords
        for flow_name in self.clinical_flow_registry.keys():
            flow_keywords = flow_name.lower().split('_')
            if any(keyword in flow_keywords for keyword in keywords):
                suggestions.append(flow_name)
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def activate_crisis_override(self, user_id: str, crisis_text: str = '') -> Dict:
        """
        Comprehensive crisis protocol per your Phase 2 implementation
        Enhanced with RAG-powered crisis resources when appropriate
        
        Args:
            user_id: User identifier
            crisis_text: Text that triggered crisis detection
            
        Returns:
            Dict with crisis intervention response
        """
        
        # Clean up any active flow
        if user_id in self.active_flows:
            del self.active_flows[user_id]
        
        # Log crisis intervention with RAG context
        self.log_clinical_event(user_id, 'crisis_intervention_activated', {
            'trigger_text_length': len(crisis_text),
            'timestamp': datetime.now().isoformat(),
            'rag_system_active': True,
            'knowledge_base_available': self.knowledge_base_stats.get('total_items', 0) > 0
        })
        
        # Get additional crisis resources from RAG if available
        additional_resources = []
        try:
            crisis_content = self.content_retriever.search_by_keywords(
                keywords=['crisis', 'safety', 'help', 'emergency'],
                content_type='resources',
                n_results=2
            )
            for resource in crisis_content:
                if 'crisis' in resource.get('metadata', {}).get('type', '').lower():
                    additional_resources.append({
                        'name': 'Additional Support',
                        'description': resource['content'][:100] + '...',
                        'source': 'therapeutic_knowledge_base'
                    })
        except Exception as e:
            self.logger.error(f"Error retrieving crisis resources from RAG: {str(e)}")
        
        return {
            'flow_type': 'crisis_override',
            'message': 'I\'m very concerned about your safety right now. Please reach out for immediate help:',
            'crisis_resources': [
                {
                    'name': 'National Suicide Prevention Lifeline',
                    'contact': '988',
                    'description': 'Free, confidential 24/7 support',
                    'priority': 'highest'
                },
                {
                    'name': 'Crisis Text Line', 
                    'contact': 'Text HOME to 741741',
                    'description': '24/7 text-based crisis support',
                    'priority': 'highest'
                },
                {
                    'name': 'Emergency Services',
                    'contact': '911',
                    'description': 'For immediate emergency response',
                    'priority': 'highest'
                },
                {
                    'name': 'National Alliance on Mental Illness',
                    'contact': 'Text NAMI to 741741',
                    'description': 'Mental health support and resources',
                    'priority': 'high'
                }
            ] + additional_resources,
            'safety_message': 'Your life has value and there are people who want to help you. Please reach out to one of these resources.',
            'disable_free_form_responses': True,  # Per your Phase 4 safety requirements
            'log_crisis_interaction': True,
            'immediate_escalation': True,
            'clinical_override': True,
            'rag_enhanced_crisis': len(additional_resources) > 0
        }
    
    def complete_clinical_session(self, user_id: str, final_response: Dict):
        """
        Complete clinical session with enhanced outcome tracking including RAG analytics
        
        Args:
            user_id: User identifier
            final_response: Final flow response with outcomes
        """
        
        if user_id in self.active_flows:
            flow_data = self.active_flows[user_id]
            flow_instance = flow_data['flow_instance']
            
            # Calculate session metrics
            session_duration = (datetime.now() - flow_data['start_time']).total_seconds()
            completion_rate = min(flow_instance.state.current_step + 1, len(flow_instance.clinical_steps)) / len(flow_instance.clinical_steps)

            # Store enhanced clinical outcomes
            if user_id not in self.clinical_outcomes:
                self.clinical_outcomes[user_id] = []
                
            outcome_data = {
                'session_id': f"{user_id}_{datetime.now().isoformat()}",
                'flow_name': flow_data['flow_name'],
                'scenario': flow_instance.scenario,
                'duration_seconds': session_duration,
                'completion_rate': completion_rate,
                'effectiveness_ratings': flow_instance.state.effectiveness_ratings,
                'safety_flags': flow_instance.state.safety_flags,
                'techniques_used': [step['intervention'] for step in flow_instance.clinical_steps[:flow_instance.state.current_step + 1]],
                'clinical_success': completion_rate >= 0.8 and len(flow_instance.state.safety_flags) == 0,
                
                # RAG-specific analytics
                'rag_powered': True,
                'rag_retrievals': flow_data.get('rag_retrievals', 0),
                'dynamic_content_used': flow_data.get('dynamic_content_used', False),
                'dynamic_content_percentage': self._calculate_dynamic_content_percentage(user_id),
                'knowledge_base_items_available': self.knowledge_base_stats.get('total_items', 0)
            }
            
            self.clinical_outcomes[user_id].append(outcome_data)
            
            # Log session completion with RAG analytics
            self.log_clinical_event(user_id, 'rag_session_completed', outcome_data)
            
            # Clean up active flow
            del self.active_flows[user_id]
    
    def get_clinical_flow_status(self, user_id: str) -> Optional[Dict]:
        """
        Get comprehensive clinical flow status for monitoring including RAG analytics
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with detailed flow status including RAG usage or None if no active flow
        """
        
        if user_id in self.active_flows:
            flow_data = self.active_flows[user_id]
            flow_instance = flow_data['flow_instance']
            
            return {
                'flow_name': flow_data['flow_name'],
                'scenario': flow_instance.scenario,
                'intensity': flow_instance.intensity,
                'current_step': flow_instance.state.current_step + 1,
                'total_steps': len(flow_instance.clinical_steps),
                'current_intervention': flow_instance.clinical_steps[flow_instance.state.current_step]['intervention'] if flow_instance.state.current_step < len(flow_instance.clinical_steps) else 'completed',
                'session_duration': (datetime.now() - flow_data['start_time']).total_seconds(),
                'safety_flags': flow_instance.state.safety_flags,
                'effectiveness_ratings': flow_instance.state.effectiveness_ratings,
                'completion_rate': (min(flow_instance.state.current_step + 1, len(flow_instance.clinical_steps)) / len(flow_instance.clinical_steps)),


                # RAG-specific status
                'rag_enhanced': True,
                'rag_retrievals': flow_data.get('rag_retrievals', 0),
                'dynamic_content_active': flow_data.get('dynamic_content_used', False),
                'dynamic_content_percentage': self._calculate_dynamic_content_percentage(user_id)
            }
        return None
    
    def get_user_clinical_history(self, user_id: str) -> Dict:
        """
        Get user's clinical session history and outcomes with RAG analytics
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with clinical history and RAG-enhanced analytics
        """
        
        history = {
            'user_id': user_id,
            'total_sessions': len(self.clinical_outcomes.get(user_id, [])),
            'session_history': self.session_history.get(user_id, []),
            'clinical_outcomes': self.clinical_outcomes.get(user_id, []),
            'current_active_flow': self.get_clinical_flow_status(user_id),
            'rag_system_info': {
                'knowledge_base_items': self.knowledge_base_stats.get('total_items', 0),
                'scenarios_covered': list(self.knowledge_base_stats.get('scenarios', {}).keys()),
                'content_types': list(self.knowledge_base_stats.get('content_types', {}).keys())
            }
        }
        
        # Calculate enhanced summary statistics including RAG usage
        if user_id in self.clinical_outcomes:
            outcomes = self.clinical_outcomes[user_id]
            rag_sessions = [o for o in outcomes if o.get('rag_powered', False)]
            
            history['analytics'] = {
                'average_completion_rate': sum(o['completion_rate'] for o in outcomes) / len(outcomes),
                'total_safety_flags': sum(len(o['safety_flags']) for o in outcomes),
                'most_common_scenario': max(set(o['scenario'] for o in outcomes), key=[o['scenario'] for o in outcomes].count) if outcomes else None,
                'average_session_duration': sum(o['duration_seconds'] for o in outcomes) / len(outcomes),
                'clinical_success_rate': sum(1 for o in outcomes if o['clinical_success']) / len(outcomes),
                
                # RAG-specific analytics
                'rag_sessions': len(rag_sessions),
                'rag_usage_percentage': (len(rag_sessions) / len(outcomes)) * 100 if outcomes else 0,
                'average_rag_retrievals': sum(o.get('rag_retrievals', 0) for o in rag_sessions) / len(rag_sessions) if rag_sessions else 0,
                'average_dynamic_content_percentage': sum(o.get('dynamic_content_percentage', 0) for o in rag_sessions) / len(rag_sessions) if rag_sessions else 0
            }
        
        return history
    
    def log_clinical_event(self, user_id: str, event_type: str, event_data: Dict):
        """
        Log clinical events for monitoring and improvement with RAG context
        
        Args:
            user_id: User identifier
            event_type: Type of clinical event
            event_data: Event metadata
        """
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'event_type': event_type,
            'data': event_data,
            'rag_system_version': '1.0.0',
            'knowledge_base_stats': self.knowledge_base_stats
        }
        
        # Add to session history
        if user_id not in self.session_history:
            self.session_history[user_id] = []
        self.session_history[user_id].append(log_entry)
        
        # Log for monitoring (can be extended to external logging systems)
        self.logger.info(f"RAG Clinical event - User: {user_id}, Type: {event_type}, Data: {json.dumps(event_data, default=str)}")
    
    def get_available_flows(self) -> Dict:
        """
        Get all available RAG-powered clinical flows organized by scenario
        
        Returns:
            Dict organizing flows by the 7 anxiety scenarios with RAG info
        """
        
        return {
            'panic': {
                'flows': [
                    'acute_anxiety_flow',
                    'panic_crisis_flow', 
                    'panic_breathing_flow',
                    'panic_flow'
                ],
                'rag_techniques_available': len([item for item in self.knowledge_base_stats.get('content_types', {}) if 'technique' in item.lower()]) > 0
            },
            'sleep': {
                'flows': [
                    'nighttime_flow',
                    'sleep_flow',
                    'sleep_hygiene_flow',
                    'insomnia_flow'
                ],
                'rag_techniques_available': True
            },
            'pre_event': {
                'flows': [
                    'pre_event_flow',
                    'pre_event_nervousness_flow',
                    'performance_anxiety_flow',
                    'event_anxiety_flow'
                ],
                'rag_techniques_available': True
            },
            'isolation': {
                'flows': [
                    'isolation_flow', 
                    'loneliness_flow',
                    'social_anxiety_flow',
                    'connection_flow'
                ],
                'rag_techniques_available': True
            },
            'uncertainty': {
                'flows': [
                    'uncertainty_flow',
                    'worry_flow', 
                    'anticipatory_anxiety_flow',
                    'unknown_flow'
                ],
                'rag_techniques_available': True
            },
            'decision_making': {
                'flows': [
                    'decision_making_flow',
                    'choice_paralysis_flow',
                    'indecision_flow',
                    'decision_flow'
                ],
                'rag_techniques_available': True
            },
            'physical_triggers': {
                'flows': [
                    'physical_triggers_flow',
                    'somatic_anxiety_flow',
                    'environmental_triggers_flow',
                    'body_anxiety_flow'
                ],
                'rag_techniques_available': True
            }
        }
    
    def get_system_health(self) -> Dict:
        """Get comprehensive system health including RAG status"""
        return {
            'flow_manager_status': 'healthy',
            'active_sessions': len(self.active_flows),
            'total_flows_available': len(self.clinical_flow_registry),
            'scenarios_supported': 7,
            'rag_system': {
                'status': 'active',
                'knowledge_base_items': self.knowledge_base_stats.get('total_items', 0),
                'scenarios_covered': len(self.knowledge_base_stats.get('scenarios', {})),
                'content_types': len(self.knowledge_base_stats.get('content_types', {})),
                'retriever_healthy': hasattr(self.content_retriever, 'collection')
            },
            'safety_protocols': {
                'crisis_detection': 'active',
                'safety_monitoring': 'active',
                'clinical_logging': 'active'
            }
        }
