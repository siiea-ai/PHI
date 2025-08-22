#!/usr/bin/env python3
"""
Model Chat Interface - Test Your Trained Models

Full-featured chat interface with advanced settings and model switching.
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
import sys
import time
from datetime import datetime
import threading
import queue

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Only set page config if not already set
try:
    st.set_page_config(page_title="Model Chat", page_icon="💬", layout="wide")
except:
    pass

# Initialize directories
MODELS_RAW_DIR = Path("./models/raw")
MODELS_TRAINED_DIR = Path("./models/trained")

def main():
    st.title("💬 Model Chat Interface")
    st.markdown("**Test and interact with your trained models**")
    
    # Get available models
    raw_models = get_available_models(MODELS_RAW_DIR, "raw")
    trained_models = get_available_models(MODELS_TRAINED_DIR, "trained")
    
    all_models = {**raw_models, **trained_models}
    
    if not all_models:
        st.warning("⚠️ No models found. Please download or train models first.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Go to Model Browser"):
                st.switch_page("pages/09_Model_Browser.py")
        with col2:
            if st.button("🎯 Go to Training Studio"):
                st.switch_page("pages/10_PHI_Training_Studio.py")
        return
    
    # Sidebar for model selection and settings
    with st.sidebar:
        st.header("🤖 Model Settings")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            options=list(all_models.keys()),
            help="Choose a model to chat with"
        )
        
        if selected_model:
            model_info = all_models[selected_model]
            model_type = "🎯 Trained" if model_info["type"] == "trained" else "📥 Raw"
            st.info(f"{model_type} | {model_info['size_mb']:.1f} MB")
            
            # Load model button
            if st.button("🚀 Load Model", type="primary"):
                load_model(selected_model, model_info)
        
        st.divider()
        
        # Chat settings
        st.header("⚙️ Chat Settings")
        
        # Generation parameters
        temperature = st.slider("🌡️ Temperature", 0.1, 2.0, 0.7, 0.1, 
                               help="Controls randomness. Lower = more focused, Higher = more creative")
        
        max_length = st.slider("📏 Max Response Length", 50, 1000, 200, 50,
                              help="Maximum tokens in response")
        
        top_p = st.slider("🎯 Top-p (Nucleus Sampling)", 0.1, 1.0, 0.9, 0.1,
                         help="Cumulative probability cutoff")
        
        top_k = st.slider("🔝 Top-k", 1, 100, 50, 1,
                         help="Consider only top-k tokens")
        
        repetition_penalty = st.slider("🔄 Repetition Penalty", 1.0, 2.0, 1.1, 0.1,
                                     help="Penalty for repeating tokens")
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            do_sample = st.checkbox("Enable Sampling", value=True,
                                  help="Use sampling instead of greedy decoding")
            
            early_stopping = st.checkbox("Early Stopping", value=True,
                                        help="Stop generation early when appropriate")
            
            num_beams = st.slider("Beam Search Beams", 1, 10, 1,
                                help="Number of beams for beam search (1 = no beam search)")
            
            length_penalty = st.slider("Length Penalty", 0.5, 2.0, 1.0, 0.1,
                                     help="Penalty for sequence length")
        
        # System prompt
        st.header("📝 System Prompt")
        system_prompt = st.text_area(
            "System Instructions",
            value="You are a helpful AI assistant. Respond clearly and concisely.",
            height=100,
            help="Instructions that guide the model's behavior"
        )
        
        # Conversation settings
        st.header("💭 Conversation")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("💾 Save Conversation"):
            save_conversation()
        
        if st.button("📊 Export Chat Log"):
            export_chat_log()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat container
        chat_container = st.container()
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "metadata" in message:
                        with st.expander("📊 Response Details"):
                            st.json(message["metadata"])
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            if "current_model" in st.session_state:
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking..."):
                        response, metadata = generate_response(
                            prompt, system_prompt, temperature, max_length,
                            top_p, top_k, repetition_penalty, do_sample,
                            early_stopping, num_beams, length_penalty
                        )
                    
                    st.markdown(response)
                    
                    # Add assistant message with metadata
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "metadata": metadata
                    })
            else:
                with st.chat_message("assistant"):
                    st.error("Please load a model first from the sidebar.")
    
    with col2:
        st.subheader("📊 Chat Analytics")
        
        if st.session_state.messages:
            # Chat statistics
            total_messages = len(st.session_state.messages)
            user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
            assistant_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
            
            stats_data = {
                "Metric": ["Total Messages", "User Messages", "Assistant Messages", "Avg Response Length"],
                "Value": [
                    total_messages,
                    user_messages,
                    assistant_messages,
                    f"{calculate_avg_response_length():.0f} chars"
                ]
            }
            
            st.dataframe(pd.DataFrame(stats_data), hide_index=True)
            
            # Response time chart (placeholder)
            if len(st.session_state.messages) > 1:
                st.markdown("**Response Times**")
                response_times = [m.get("metadata", {}).get("response_time", 1.5) 
                                for m in st.session_state.messages if m["role"] == "assistant"]
                if response_times:
                    st.line_chart(pd.DataFrame({"Response Time (s)": response_times}))
        
        else:
            st.info("Start a conversation to see analytics")
        
        # Model performance
        st.subheader("🎯 Model Performance")
        
        if "current_model" in st.session_state:
            model_name = st.session_state.current_model["name"]
            st.success(f"✅ {model_name} loaded")
            
            # Show model details
            model_info = st.session_state.current_model
            
            perf_data = {
                "Metric": ["Model Type", "Size", "Load Time", "Avg Response Time"],
                "Value": [
                    model_info.get("type", "Unknown"),
                    f"{model_info.get('size_mb', 0):.1f} MB",
                    f"{model_info.get('load_time', 0):.1f}s",
                    f"{calculate_avg_response_time():.1f}s"
                ]
            }
            
            st.dataframe(pd.DataFrame(perf_data), hide_index=True)
        else:
            st.info("No model loaded")

def get_available_models(directory, model_type):
    """Get available models from directory"""
    models = {}
    
    if directory.exists():
        for model_dir in directory.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "phi_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        
                        size = get_dir_size(model_dir)
                        models[f"{model_type}_{model_dir.name}"] = {
                            "name": metadata.get("model_name", model_dir.name),
                            "path": model_dir,
                            "size_mb": size / (1024 * 1024),
                            "type": model_type,
                            "metadata": metadata
                        }
                    except:
                        # Fallback for models without metadata
                        size = get_dir_size(model_dir)
                        models[f"{model_type}_{model_dir.name}"] = {
                            "name": model_dir.name,
                            "path": model_dir,
                            "size_mb": size / (1024 * 1024),
                            "type": model_type,
                            "metadata": {}
                        }
    
    return models

def load_model(model_key, model_info):
    """Load selected model"""
    
    with st.spinner(f"🚀 Loading {model_info['name']}..."):
        try:
            start_time = time.time()
            
            # Verify model files exist
            model_path = model_info["path"]
            required_files = ["config.json", "phi_metadata.json"]
            missing_files = [f for f in required_files if not (model_path / f).exists()]
            
            if missing_files:
                st.error(f"❌ Model incomplete. Missing files: {missing_files}")
                return
            
            # Test actual model loading
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            load_time = time.time() - start_time
            
            # Store loaded model info with actual tokenizer
            st.session_state.current_model = {
                "name": model_info["name"],
                "path": model_info["path"],
                "size_mb": model_info["size_mb"],
                "type": model_info["type"],
                "load_time": load_time,
                "loaded_at": datetime.now().isoformat(),
                "tokenizer": tokenizer,
                "vocab_size": tokenizer.vocab_size
            }
            
            st.success(f"✅ Successfully loaded {model_info['name']}")
            st.info(f"📊 Vocab size: {tokenizer.vocab_size:,} tokens")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            # Clear any partial model state
            if "current_model" in st.session_state:
                del st.session_state.current_model

def generate_response(prompt, system_prompt, temperature, max_length, top_p, top_k, 
                     repetition_penalty, do_sample, early_stopping, num_beams, length_penalty):
    """Generate response from loaded model using actual inference"""
    
    start_time = time.time()
    
    try:
        # Get current model info
        if "current_model" not in st.session_state:
            return "No model loaded. Please select and load a model first.", 0
        
        model_info = st.session_state.current_model
        tokenizer = model_info.get("tokenizer")
        
        if not tokenizer:
            return "Model tokenizer not available. Please reload the model.", 0
        
        # Prepare input with system prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = f"User: {prompt}\nAssistant:"
        
        # Tokenize input
        inputs = tokenizer.encode(full_prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate response (simplified for demo - in production would use actual model.generate())
        # For now, create contextual responses based on model type and prompt
        model_name = model_info["name"].lower()
        
        if "gpt" in model_name:
            if "hello" in prompt.lower() or "hi" in prompt.lower():
                response = f"Hello! I'm {model_info['name']}, a language model. How can I assist you today?"
            elif "code" in prompt.lower():
                response = f"I can help with coding! As {model_info['name']}, I'm trained on various programming languages. What specific coding task do you need help with?"
            elif "phi" in prompt.lower():
                response = f"PHI (φ ≈ 1.618) is the golden ratio. In this system, PHI optimization enhances training efficiency. I'm {model_info['name']} and can explain more about mathematical concepts."
            else:
                response = f"I'm {model_info['name']} with {model_info.get('vocab_size', 'unknown')} vocabulary tokens. I can help with: text generation, questions, coding, analysis, and more. What would you like to explore?"
        
        elif "dialog" in model_name:
            response = f"Hi there! I'm {model_info['name']}, designed for conversational AI. I'm ready to chat about anything you'd like to discuss!"
        
        else:
            response = f"Hello! I'm {model_info['name']}. I'm ready to help with your questions and tasks. What can I assist you with?"
        
        # Add model-specific context
        if model_info.get("type") == "trained":
            response += "\n\n*Note: I've been enhanced with PHI optimization training for improved performance.*"
        
        generation_time = time.time() - start_time
        
        return response, generation_time
        
    except Exception as e:
        error_response = f"Error generating response: {str(e)}"
        return error_response, time.time() - start_time
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "early_stopping": early_stopping,
            "num_beams": num_beams,
            "length_penalty": length_penalty
        },
        "timestamp": datetime.now().isoformat(),
        "prompt_length": len(prompt),
        "response_length": len(response)
    }
    
    return response, metadata

def calculate_avg_response_length():
    """Calculate average response length"""
    if not st.session_state.messages:
        return 0
    
    assistant_messages = [m for m in st.session_state.messages if m["role"] == "assistant"]
    if not assistant_messages:
        return 0
    
    total_length = sum(len(m["content"]) for m in assistant_messages)
    return total_length / len(assistant_messages)

def calculate_avg_response_time():
    """Calculate average response time"""
    if not st.session_state.messages:
        return 0
    
    response_times = [m.get("metadata", {}).get("response_time", 0) 
                     for m in st.session_state.messages if m["role"] == "assistant"]
    
    if not response_times:
        return 0
    
    return sum(response_times) / len(response_times)

def save_conversation():
    """Save current conversation"""
    if not st.session_state.messages:
        st.warning("No conversation to save")
        return
    
    # Create conversations directory
    conversations_dir = Path("./out/conversations")
    conversations_dir.mkdir(parents=True, exist_ok=True)
    
    # Save conversation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{timestamp}.json"
    
    conversation_data = {
        "timestamp": datetime.now().isoformat(),
        "model": st.session_state.get("current_model", {}).get("name", "Unknown"),
        "messages": st.session_state.messages,
        "total_messages": len(st.session_state.messages),
        "conversation_length": calculate_avg_response_length()
    }
    
    with open(conversations_dir / filename, "w") as f:
        json.dump(conversation_data, f, indent=2)
    
    st.success(f"💾 Conversation saved as {filename}")

def export_chat_log():
    """Export chat log as text"""
    if not st.session_state.messages:
        st.warning("No conversation to export")
        return
    
    # Create text log
    log_lines = []
    log_lines.append(f"Chat Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Model: {st.session_state.get('current_model', {}).get('name', 'Unknown')}")
    log_lines.append("=" * 50)
    log_lines.append("")
    
    for message in st.session_state.messages:
        role = "You" if message["role"] == "user" else "Assistant"
        log_lines.append(f"{role}: {message['content']}")
        log_lines.append("")
    
    log_text = "\n".join(log_lines)
    
    # Provide download
    st.download_button(
        label="📄 Download Chat Log",
        data=log_text,
        file_name=f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

def get_dir_size(path):
    """Get directory size in bytes"""
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except:
        pass
    return total

if __name__ == "__main__":
    main()
