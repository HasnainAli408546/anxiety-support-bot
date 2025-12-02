"""
Complete Anxiety Bot Pipeline - Updated with Clinical Flows
Integrates preprocessing, emotion detection, scenario mapping, and clinical therapeutic flows
Following your 7-scenario implementation guide
"""

from src.personalization.personalization_engine import PersonalizationEngine
from src.analytics.user_analytics import UserAnalytics
from src.memory.user_memory import UserMemory
from typing import Dict, Optional, List
from datetime import datetime
import logging

# Phase 1: Preprocessing imports (your existing modules)
from src.preprocessing.text_normalizer import normalize_text
from src.preprocessing.intent_detector import IntentDetector

# Phase 2: Emotion Detection imports (your trained model)  
from src.emotion_detection.emotion_predictor import EmotionPredictor

# Phase 3: Scenario Mapping imports (your existing modules)
from src.scenario_mapping.scenario_router import ScenarioRouter

# Phase 3: Clinical Flows imports (new clinical implementation)
from src.flows.clinical_flow_manager import ClinicalFlowManager

# Safety imports
from src.safety.crisis_detector import detect_crisis_keywords

from dotenv import load_dotenv
import os

load_dotenv()  # Loads from .env file automatically
CHROMADB_PATH = os.getenv("CHROMADB_PATH")


class AnxietyBotPipeline:
    """
    Enhanced Anxiety Bot Pipeline with Clinical Therapeutic Flows
    
    Features:
    - Text preprocessing and intent detection
    - Emotion detection using trained DistilBERT model
    - Scenario routing for 7 anxiety scenarios
    - Clinical therapeutic flows with evidence-based interventions
    - Crisis detection and safety protocols
    - Session management and outcome tracking
    """
    
    def __init__(self, model_path=None):

        # Initialize core pipeline components
        if model_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            model_path = os.path.join(project_root, "models", "distilbert_emotion_model", "checkpoint-16563")

        self.intent_detector = IntentDetector()
        self.emotion_predictor = EmotionPredictor(model_path)
        self.scenario_router = ScenarioRouter()
        self.clinical_flow_manager = ClinicalFlowManager(db_path=CHROMADB_PATH)
        self.user_memory = UserMemory()
        self.personalizer = PersonalizationEngine()   # uses default user_memory.db
        self.analytics = UserAnalytics()              # uses default user_memory.db

        # Session management
        self.active_sessions = {}  # user_id -> session_data
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("✅ Anxiety Bot Pipeline initialized with clinical therapeutic flows")
    
    def process_message(self, user_text: str, user_id: str = "default_user", 
                       context: dict = None):
        """
        Enhanced main processing pipeline with clinical flow integration
        
        Args:
            user_text: Raw user input text
            user_id: User identifier for session management
            context: Optional additional context
            
        Returns:
            Dict with comprehensive response including clinical flow guidance
        """
        
        processing_start = datetime.now()
        
        try:
            print(f"\n=== Processing Message for User: {user_id} ===")
            print(f"Input: {user_text[:100]}{'...' if len(user_text) > 100 else ''}")
            
            # STEP 1: Text Preprocessing
            normalized_text = normalize_text(user_text)
            print(f"Normalized: {normalized_text}")
            
            # STEP 2: Intent Detection
            intent_scores = self.intent_detector.detect_intent(normalized_text)
            top_scenarios = self.intent_detector.get_top_scenarios(normalized_text, top_k=3)
            print(f"Intent Scores: {intent_scores}")
            print(f"Top Scenarios: {top_scenarios}")
            
            # STEP 3: Emotion Detection
            emotion_scores = self.emotion_predictor.predict_emotions(normalized_text)
            print(f"Emotion Scores: {emotion_scores}")
            
            # STEP 4: Crisis Detection
            crisis_detected = detect_crisis_keywords(user_text)
            print(f"Crisis Detected: {crisis_detected}")

            personalization_recs = self.personalizer.get_personalized_recommendations(user_id)
            
            # STEP 5: Scenario Routing
            clinical_context = {
                'text': user_text,
                'normalized_text': normalized_text,
                'emotion_scores': emotion_scores,
                'intent_scores': intent_scores,
                'crisis_detected': crisis_detected,
                'user_context': context or {},
                'timestamp': datetime.now().isoformat(),
                'personalization_recs': personalization_recs
            }
            
            selected_flow, routing_metadata = self.scenario_router.route_scenario(
                intent_scores=intent_scores,
                emotion_scores=emotion_scores,
                text=normalized_text,
                context=clinical_context
            )
            
            print(f"Selected Flow: {selected_flow}")
            print(f"Routing Metadata: {routing_metadata}")
            
            # STEP 6: Clinical Flow Execution
            if crisis_detected or routing_metadata.get('crisis_detected'):
                # Crisis Override
                clinical_response = self.clinical_flow_manager.activate_crisis_override(
                    user_id, user_text
                )
                print("🚨 Crisis Override Activated")
            else:
                # Check if user has active flow
                if user_id in self.clinical_flow_manager.active_flows:
                    # Continue existing clinical flow
                    clinical_response = self.clinical_flow_manager.process_clinical_response(
                        user_id, user_text
                    )
                    print("📋 Continuing existing clinical flow")
                else:
                    # Start new clinical flow
                    clinical_response = self.clinical_flow_manager.start_clinical_flow(
                        user_id, selected_flow, clinical_context
                    )
                    print("🆕 Starting new clinical flow")
            
            # STEP 7: Response Assembly
            processing_duration = (datetime.now() - processing_start).total_seconds()
            
            complete_response = {
                # Core response data (your original structure)
                'original_text': user_text,
                'normalized_text': normalized_text,
                'intent_scores': intent_scores,
                'emotion_scores': emotion_scores,
                'selected_flow': selected_flow,
                'metadata': routing_metadata,
                
                # Clinical flow response (new)
                'clinical_response': {
                    'message': clinical_response.get('message', ''),
                    'flow_type': clinical_response.get('flow_type', selected_flow),
                    'requires_input': clinical_response.get('requires_input', True),
                    'suggested_responses': clinical_response.get('suggested_responses', []),
                    'step_info': {
                        'current_step': clinical_response.get('step_number', 0),
                        'total_steps': clinical_response.get('total_steps', 0),
                        'intervention_type': clinical_response.get('intervention_type', '')
                    }
                },
                
                # Session management (new)
                'session': {
                    'user_id': user_id,
                    'flow_active': user_id in self.clinical_flow_manager.active_flows,
                    'flow_status': self.clinical_flow_manager.get_clinical_flow_status(user_id),
                    'processing_time_ms': processing_duration * 1000
                },
                
                # Safety information (new)
                'safety': {
                    'crisis_detected': crisis_detected,
                    'safety_resources': clinical_response.get('crisis_resources', []),
                    'emergency_protocols': clinical_response.get('immediate_escalation', False)
                },
                
                # Analytics (enhanced)
                'analytics': {
                    'top_intent': max(intent_scores, key=intent_scores.get) if intent_scores else 'None',
                    'top_emotion': max(emotion_scores, key=emotion_scores.get) if emotion_scores else 'None',
                    'confidence': routing_metadata.get('confidence', 0.0),
                    'multiple_scenarios': self.intent_detector.has_multiple_scenarios(normalized_text),
                    'processing_successful': True
                }
            }
            
            # Add flow completion info if applicable
            if clinical_response.get('flow_completed'):
                complete_response['flow_completion'] = {
                    'completed': True,
                    'clinical_outcomes': clinical_response.get('clinical_outcomes', {}),
                    'follow_up_resources': clinical_response.get('follow_up_resources', []),
                    'session_summary': clinical_response.get('clinical_monitoring', {})
                }
                print("✅ Clinical flow completed")
            
            print(f"✅ Processing completed in {processing_duration:.3f}s")
            # Log turn to user memory
            self.user_memory.append_user_turn(
                user_id=user_id,
                user_message=user_text,
                bot_message=complete_response['clinical_response']['message'],
                flow_name=complete_response['selected_flow'],
                flow_step=complete_response['clinical_response']['step_info'].get('current_step', 0),
                meta={
                    "intent_scores": complete_response["intent_scores"],
                    "emotion_scores": complete_response["emotion_scores"],
                    "crisis_detected": complete_response["safety"]["crisis_detected"],
                    "timestamp": datetime.now().isoformat()
                }
            )

            return complete_response
            
        except Exception as e:
            self.logger.error(f"Error processing message for user {user_id}: {str(e)}")
            return self.handle_pipeline_error(user_id, user_text, str(e))
    
    def handle_pipeline_error(self, user_id: str, user_text: str, error_msg: str) -> Dict:
        """Handle pipeline errors gracefully with safety fallbacks"""
        
        print(f"❌ Pipeline Error: {error_msg}")
        
        # Still check for crisis in case of processing errors
        crisis_detected = False
        try:
            crisis_detected = detect_crisis_keywords(user_text)
        except:
            pass
        
        error_response = {
            'original_text': user_text,
            'normalized_text': user_text,  # Fallback
            'intent_scores': {},
            'emotion_scores': {},
            'selected_flow': 'error_fallback',
            'metadata': {'error': True, 'crisis_detected': crisis_detected},
            
            'clinical_response': {
                'message': 'I\'m experiencing a technical difficulty, but I still want to help you. How are you feeling right now?',
                'flow_type': 'error_support',
                'requires_input': True,
                'suggested_responses': ['I need help', 'I\'m feeling anxious', 'I\'m in crisis'],
                'step_info': {'current_step': 1, 'total_steps': 1, 'intervention_type': 'error_recovery'}
            },
            
            'session': {
                'user_id': user_id,
                'flow_active': False,
                'error_occurred': True,
                'restart_recommended': True
            },
            
            'safety': {
                'crisis_detected': crisis_detected,
                'safety_resources': [
                    'National Suicide Prevention Lifeline: 988',
                    'Crisis Text Line: Text HOME to 741741',
                    'Emergency Services: 911'
                ],
                'emergency_protocols': crisis_detected
            },
            
            'analytics': {
                'top_intent': 'error',
                'top_emotion': 'unknown',
                'confidence': 0.0,
                'multiple_scenarios': False,
                'processing_successful': False,
                'error_message': error_msg
            }
        }
        
        # If crisis detected, override with crisis response
        if crisis_detected:
            error_response['clinical_response']['message'] = 'I\'m having technical difficulties, but I\'m very concerned about your safety. Please reach out for immediate help: National Suicide Prevention Lifeline (988) or Emergency Services (911).'
            error_response['clinical_response']['flow_type'] = 'crisis_override'
        
        return error_response
    
    def continue_conversation(self, user_id: str, user_input: str) -> Dict:
        """
        Continue an existing conversation/clinical flow
        
        Args:
            user_id: User identifier
            user_input: User's response to continue the flow
            
        Returns:
            Dict with clinical flow continuation response
        """
        
        print(f"\n=== Continuing Conversation for User: {user_id} ===")
        
        # Check if user has active clinical flow
        if user_id not in self.clinical_flow_manager.active_flows:
            print("No active flow found, processing as new message")
            return self.process_message(user_input, user_id)
        
        # Continue clinical flow
        try:
            clinical_response = self.clinical_flow_manager.process_clinical_response(
                user_id, user_input
            )
            
            response = {
                'user_input': user_input,
                'clinical_response': clinical_response,
                'session': {
                    'user_id': user_id,
                    'flow_active': user_id in self.clinical_flow_manager.active_flows,
                    'flow_status': self.clinical_flow_manager.get_clinical_flow_status(user_id)
                },
                'conversation_continued': True
            }
            
            if clinical_response.get('flow_completed'):
                print("✅ Clinical flow completed")
                response['flow_completion'] = clinical_response.get('clinical_outcomes', {})
            self.user_memory.append_user_turn(
                user_id=user_id,
                user_message=user_input,
                bot_message=response['clinical_response']['message'],
                flow_name=response['clinical_response'].get('flow_type'),
                flow_step=response['clinical_response']['step_info'].get('current_step', 0),
                meta={
                    # You can add more outcome/context as needed
                    "timestamp": datetime.now().isoformat()
                }
            )

            return response
            
        except Exception as e:
            self.logger.error(f"Error continuing conversation for user {user_id}: {str(e)}")
            return self.handle_pipeline_error(user_id, user_input, str(e))
    
    def get_user_status(self, user_id: str) -> Dict:
        """Get comprehensive user session status"""
        
        return {
            'user_id': user_id,
            'active_flow': self.clinical_flow_manager.get_clinical_flow_status(user_id),
            'session_history': self.clinical_flow_manager.get_user_clinical_history(user_id),
            'pipeline_status': 'active'
        }
    
    def end_user_session(self, user_id: str) -> Dict:
        """Safely end user session"""
        
        if user_id in self.clinical_flow_manager.active_flows:
            # Complete current flow
            flow_data = self.clinical_flow_manager.active_flows[user_id]
            self.clinical_flow_manager.complete_clinical_session(
                user_id, {'flow_completed': True, 'reason': 'user_ended_session'}
            )
        
        session_data = self.user_memory.get_user_history(user_id)
        user_stats = self.analytics.get_user_session_stats(user_id)

        return {
            'session_ended': True,
            'user_id': user_id,
            'message': 'Session ended. Take care of yourself, and remember these resources are available if you need support.',
            'session_stats': user_stats
        }


# Enhanced testing function
def test_enhanced_pipeline():
    """Test the enhanced pipeline with clinical flows"""
    
    print("🧪 Testing Enhanced Anxiety Bot Pipeline with Clinical Flows")
    print("=" * 60)
    
    pipeline = AnxietyBotPipeline()
    
    test_cases = [
        {
            'message': "My heart is racing and I can't breathe. I'm having a panic attack!",
            'expected_scenario': 'panic',
            'test_name': 'Panic Attack'
        },
        {
            'message': "I feel so alone and nobody understands me.",
            'expected_scenario': 'isolation', 
            'test_name': 'Isolation/Loneliness'
        },
        {
            'message': "I have a presentation tomorrow and I'm really nervous.",
            'expected_scenario': 'pre_event',
            'test_name': 'Pre-Event Anxiety'
        },
        {
            'message': "I can't sleep, my thoughts won't stop racing.",
            'expected_scenario': 'sleep',
            'test_name': 'Sleep Anxiety'
        },
        {
            'message': "I want to die. I can't take this anymore.",
            'expected_scenario': 'crisis',
            'test_name': 'Crisis Situation'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        user_id = f"test_user_{i}"
        
        print(f"\n🧪 Test Case {i}: {test_case['test_name']}")
        print(f"Input: {test_case['message']}")
        
        result = pipeline.process_message(test_case['message'], user_id)
        
        print(f"✅ Selected Flow: {result['selected_flow']}")
        print(f"✅ Clinical Message: {result['clinical_response']['message'][:100]}...")
        print(f"✅ Crisis Detected: {result['safety']['crisis_detected']}")
        print(f"✅ Flow Active: {result['session']['flow_active']}")
        
        if result['analytics']['top_intent'] != 'None':
            print(f"✅ Top Intent: {result['analytics']['top_intent']}")
        if result['analytics']['top_emotion'] != 'None':
            print(f"✅ Top Emotion: {result['analytics']['top_emotion']}")
        
        print("-" * 50)
        
        # Test continuing conversation for non-crisis cases
        if not result['safety']['crisis_detected'] and result['session']['flow_active']:
            print(f"🔄 Testing conversation continuation for {user_id}")
            
            # Simulate user response
            follow_up_responses = [
                "Yes, I am safe",
                "I'm still feeling anxious",
                "That helped a little", 
                "I'm ready to try that",
                "Done"
            ]
            
            # Test one follow-up response
            follow_up = pipeline.continue_conversation(user_id, follow_up_responses[0])
            print(f"✅ Follow-up Response: {follow_up['clinical_response']['message'][:100]}...")
            
            # End session
            pipeline.end_user_session(user_id)
            print(f"✅ Session ended for {user_id}")
            print("-" * 50)
    
    print("\n🎉 Enhanced Pipeline Testing Complete!")

def test_full_flow_conversations():
    """
    Test each major flow end-to-end: intent detection, emotion analysis, flow activation,
    and a full sequence of user replies to advance through all clinical steps.
    """
    print("\n🧪 Running Full End-to-End Flow Tests")
    print("=" * 60)

    pipeline = AnxietyBotPipeline()  # Make sure pipeline uses .env for CHROMADB_PATH

    # Each test case: (message to trigger flow, list of synthetic user replies simulating step-by-step)
    conversation_tests = [
        {
            'test_name': 'Acute Anxiety/Panic Flow',
            'trigger_message': "Help! I'm having a panic attack, can't breathe, heart pounding.",
            'followups': [
                "Yes, I'm safe.",                # Safety assessment
                "Ready.",                        # Psychoeducation ack
                "Done with breathing.",          # Breathing technique
                "I see the wall, hear music.",   # Grounding
                "Done relaxing.",                # Relaxation
                "Okay.",                         # Clinical reassurance
                "3"                              # Effectiveness assessment
            ]
        },
        {
            'test_name': 'Isolation/Loneliness Flow',
            'trigger_message': "I can't stop feeling alone and disconnected from everyone.",
            'followups': [
                "Yes, I'm safe.", 
                "Go ahead.", 
                "Done.", 
                "I see my room.", 
                "Done.", 
                "Thanks.", 
                "5"
            ]
        },
        {
            'test_name': 'Sleep Anxiety Flow',
            'trigger_message': "I can't sleep, my thoughts won't stop.",
            'followups': [
                "Yes, I am safe.",
                "Tell me more.",
                "Done slowing breathing.",
                "I hear my fan.",
                "Done relaxing.",
                "Okay.",
                "2"
            ]
        }
        # Add more scenarios with relevant followup replies if desired!
    ]

    for i, test in enumerate(conversation_tests, 1):
        user_id = f"conversation_user_{i}"
        print(f"\n🧪 Test {i}: {test['test_name']}")
        print(f"Trigger: {test['trigger_message']}")

        # Start the flow
        result = pipeline.process_message(test['trigger_message'], user_id)
        print(f"Bot: {result['clinical_response']['message']}")

        # Simulate step-by-step conversation
        for step_num, reply in enumerate(test['followups'], 1):
            if not result['session']['flow_active']:
                print("[Session ended]")
                break
            print(f"\nUser: {reply}")
            result = pipeline.continue_conversation(user_id, reply)
            response = result['clinical_response']
            print(f"Bot: {response['message']}")
            if response.get('flow_completed'):
                print("✅ Flow completed.")
                break

        # Print analytic summary
        session_status = pipeline.get_user_status(user_id)
        print("\n[Session summary]", session_status['active_flow'] if session_status['active_flow'] else "No active flow")
        print("-" * 50)

    print("\n🎉 Full Conversation Flow Testing Complete!\n")


# Main execution
if __name__ == "__main__":
    # Run the enhanced test suite
    # test_enhanced_pipeline()
    test_full_flow_conversations()
    
    # Interactive testing mode
    print("\n" + "=" * 60)
    print("🤖 Interactive Testing Mode")
    print("Enter messages to test the pipeline (type 'quit' to exit)")
    print("=" * 60)
    
    pipeline = AnxietyBotPipeline()
    user_id = "interactive_user"
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                pipeline.end_user_session(user_id)
                print("👋 Goodbye! Take care of yourself.")
                break
            
            if not user_input:
                continue
            
            # Process message
            if user_id in pipeline.clinical_flow_manager.active_flows:
                result = pipeline.continue_conversation(user_id, user_input)
                response = result['clinical_response']
            else:
                result = pipeline.process_message(user_input, user_id)
                response = result['clinical_response']
            
            # Display response
            print(f"\n🤖 Bot: {response['message']}")
            
            # Show suggested responses if available
            if response.get('suggested_responses'):
                print(f"💡 Suggestions: {', '.join(response['suggested_responses'])}")
            
            # Show flow info
            if result.get('session', {}).get('flow_active'):
                step_info = response.get('step_info', {})
                if step_info.get('current_step'):
                    print(f"📊 Step {step_info['current_step']}/{step_info['total_steps']} - {step_info.get('intervention_type', 'Support')}")
            
            # Handle flow completion
            if result.get('flow_completion', {}).get('completed'):
                print("✅ Flow completed! Thank you for working through this with me.")
                
                # Show follow-up resources
                resources = result['flow_completion'].get('follow_up_resources', [])
                if resources:
                    print("\n📚 Follow-up resources:")
                    for resource in resources[:3]:  # Show top 3
                        print(f"  • {resource.get('title', 'Resource')}: {resource.get('description', '')}")
            
        except KeyboardInterrupt:
            pipeline.end_user_session(user_id)
            print("\n👋 Session interrupted. Take care!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            continue

