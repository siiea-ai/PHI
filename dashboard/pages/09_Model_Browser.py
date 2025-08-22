#!/usr/bin/env python3
"""
HuggingFace Model Browser - Search, Browse, and Download Models

Complete model discovery and management system with search functionality.
"""

import streamlit as st
import requests
import json
import pandas as pd
from pathlib import Path
import sys
import time
import subprocess
from datetime import datetime
import re

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Only set page config if not already set
try:
    st.set_page_config(page_title="Model Browser", page_icon="🔍", layout="wide")
except:
    pass

# Initialize directories
MODELS_RAW_DIR = Path("./models/raw")
MODELS_TRAINED_DIR = Path("./models/trained")
MODELS_RAW_DIR.mkdir(parents=True, exist_ok=True)
MODELS_TRAINED_DIR.mkdir(parents=True, exist_ok=True)

def main():
    st.title("🔍 HuggingFace Model Browser")
    st.markdown("**Discover, Search, and Download Models from HuggingFace Hub**")
    
    # Sidebar for filters
    with st.sidebar:
        st.header("🎛️ Search Filters")
        
        # Search query
        search_query = st.text_input("🔍 Search Models", placeholder="e.g., gpt2, bert, llama")
        
        # Model type filter
        model_type = st.selectbox(
            "📋 Model Type",
            ["All", "text-generation", "text-classification", "question-answering", 
             "summarization", "translation", "conversational", "fill-mask"]
        )
        
        # Size filter
        size_filter = st.selectbox(
            "📏 Model Size",
            ["All", "Small (<100M)", "Medium (100M-1B)", "Large (1B-10B)", "XL (>10B)"]
        )
        
        # Language filter
        language = st.selectbox(
            "🌍 Language",
            ["All", "English", "Multilingual", "Chinese", "French", "German", "Spanish"]
        )
        
        # Sort by
        sort_by = st.selectbox(
            "📊 Sort By",
            ["Downloads", "Recently Updated", "Trending", "Name"]
        )
        
        # Results per page
        results_per_page = st.slider("Results per page", 10, 50, 20)
        
        if st.button("🔍 Search Models", type="primary"):
            st.session_state.search_triggered = True

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🌟 Popular Models")
        
        # Show popular models by default or search results
        if hasattr(st.session_state, 'search_triggered') and st.session_state.search_triggered:
            show_search_results(search_query, model_type, size_filter, language, sort_by, results_per_page)
        else:
            show_popular_models()
    
    with col2:
        st.subheader("💾 Local Models")
        show_local_models()
        
        st.subheader("📥 Quick Download")
        quick_download_interface()

def show_popular_models():
    """Display popular models from HuggingFace"""
    
    # Popular models list (curated)
    popular_models = [
        {
            "name": "gpt2",
            "description": "OpenAI's GPT-2 language model",
            "downloads": "50M+",
            "size": "Small (124M)",
            "type": "text-generation",
            "updated": "2023-12-01"
        },
        {
            "name": "microsoft/DialoGPT-medium",
            "description": "Conversational AI model by Microsoft",
            "downloads": "5M+",
            "size": "Medium (345M)",
            "type": "conversational",
            "updated": "2023-11-15"
        },
        {
            "name": "distilbert-base-uncased",
            "description": "Distilled BERT for classification tasks",
            "downloads": "10M+",
            "size": "Small (66M)",
            "type": "text-classification",
            "updated": "2023-10-20"
        },
        {
            "name": "facebook/bart-large-cnn",
            "description": "BART model fine-tuned for summarization",
            "downloads": "2M+",
            "size": "Large (406M)",
            "type": "summarization",
            "updated": "2023-09-30"
        },
        {
            "name": "microsoft/CodeBERT-base",
            "description": "BERT model for code understanding",
            "downloads": "1M+",
            "size": "Small (125M)",
            "type": "fill-mask",
            "updated": "2023-08-15"
        }
    ]
    
    # Display models in cards
    for i, model in enumerate(popular_models):
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{model['name']}**")
                st.caption(model['description'])
                st.markdown(f"📊 {model['downloads']} downloads • 📏 {model['size']} • 🏷️ {model['type']}")
            
            with col2:
                st.markdown(f"📅 {model['updated']}")
            
            with col3:
                if st.button(f"📥 Download", key=f"download_{i}"):
                    download_model_from_hf(model['name'])
            
            st.divider()

def show_search_results(query, model_type, size_filter, language, sort_by, results_per_page):
    """Display search results from HuggingFace API"""
    
    with st.spinner("🔍 Searching HuggingFace Hub..."):
        try:
            # Build API query
            api_url = "https://huggingface.co/api/models"
            params = {
                "limit": results_per_page,
                "full": True
            }
            
            if query:
                params["search"] = query
            
            if model_type != "All":
                params["pipeline_tag"] = model_type
            
            # Make API request
            response = requests.get(api_url, params=params, timeout=10)
            
            if response.status_code == 200:
                models = response.json()
                
                if models:
                    st.success(f"Found {len(models)} models")
                    
                    for i, model in enumerate(models):
                        display_model_card(model, i)
                else:
                    st.info("No models found matching your criteria")
            else:
                st.error("Failed to fetch models from HuggingFace API")
                show_fallback_results(query)
                
        except Exception as e:
            st.error(f"Error searching models: {str(e)}")
            show_fallback_results(query)

def display_model_card(model, index):
    """Display a model card with download option"""
    
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            model_name = model.get('modelId', model.get('id', 'Unknown'))
            st.markdown(f"**{model_name}**")
            
            # Description
            description = model.get('description', 'No description available')
            if description:
                st.caption(description[:150] + "..." if len(description) > 150 else description)
            
            # Tags and info
            tags = model.get('tags', [])
            pipeline_tag = model.get('pipeline_tag', 'Unknown')
            downloads = model.get('downloads', 0)
            
            info_parts = []
            if downloads > 0:
                if downloads > 1000000:
                    info_parts.append(f"📊 {downloads//1000000}M+ downloads")
                elif downloads > 1000:
                    info_parts.append(f"📊 {downloads//1000}K+ downloads")
                else:
                    info_parts.append(f"📊 {downloads} downloads")
            
            info_parts.append(f"🏷️ {pipeline_tag}")
            
            if tags:
                relevant_tags = [tag for tag in tags[:3] if not tag.startswith('license:')]
                if relevant_tags:
                    info_parts.append(f"🔖 {', '.join(relevant_tags)}")
            
            st.markdown(" • ".join(info_parts))
        
        with col2:
            # Last modified
            last_modified = model.get('lastModified', model.get('createdAt', ''))
            if last_modified:
                try:
                    date_obj = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                    st.markdown(f"📅 {date_obj.strftime('%Y-%m-%d')}")
                except:
                    st.markdown("📅 Recently")
        
        with col3:
            if st.button(f"📥 Download", key=f"search_download_{index}"):
                download_model_from_hf(model_name)
        
        st.divider()

def show_fallback_results(query):
    """Show fallback results when API fails"""
    st.warning("Using cached model list")
    
    fallback_models = [
        "gpt2", "distilgpt2", "microsoft/DialoGPT-small", "microsoft/DialoGPT-medium",
        "distilbert-base-uncased", "bert-base-uncased", "roberta-base",
        "facebook/bart-base", "t5-small", "microsoft/CodeBERT-base"
    ]
    
    if query:
        filtered_models = [m for m in fallback_models if query.lower() in m.lower()]
    else:
        filtered_models = fallback_models
    
    for i, model_name in enumerate(filtered_models):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{model_name}**")
        with col2:
            if st.button(f"📥 Download", key=f"fallback_{i}"):
                download_model_from_hf(model_name)

def download_model_from_hf(model_name):
    """Download model from HuggingFace Hub"""
    
    # Sanitize model name for directory
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    model_dir = MODELS_RAW_DIR / safe_name
    
    # Check if already exists
    if model_dir.exists():
        st.warning(f"Model {model_name} already exists in {model_dir}")
        return
    
    with st.spinner(f"📥 Downloading {model_name}..."):
        try:
            # Create download script
            download_script = f'''
import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute()))

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    import json
    from datetime import datetime
    
    model_name = "{model_name}"
    save_dir = "{model_dir}"
    
    print(f"Downloading {{model_name}}...")
    
    # Create directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Download components
    try:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Save components
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        
        # Save metadata
        metadata = {{
            "model_name": model_name,
            "safe_name": "{safe_name}",
            "download_time": datetime.now().isoformat(),
            "model_type": getattr(config, 'model_type', 'unknown'),
            "vocab_size": getattr(config, 'vocab_size', 'unknown'),
            "hidden_size": getattr(config, 'hidden_size', 'unknown'),
            "num_parameters": model.num_parameters() if hasattr(model, 'num_parameters') else 'unknown',
            "source": "huggingface_hub",
            "status": "downloaded"
        }}
        
        with open(f"{{save_dir}}/phi_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Download completed successfully!")
        
    except Exception as e:
        print(f"❌ Download failed: {{e}}")
        # Clean up partial download
        import shutil
        if Path(save_dir).exists():
            shutil.rmtree(save_dir)
        sys.exit(1)
        
except ImportError:
    print("❌ transformers library not installed")
    print("Install with: pip install transformers torch")
    sys.exit(1)
'''
            
            # Execute download
            result = subprocess.run(
                [sys.executable, "-c", download_script],
                capture_output=True,
                text=True,
                cwd="."
            )
            
            if result.returncode == 0:
                st.success(f"✅ Successfully downloaded {model_name}")
                st.info(f"📁 Saved to: {model_dir}")
                st.rerun()
            else:
                st.error(f"❌ Download failed: {result.stderr}")
                
        except Exception as e:
            st.error(f"❌ Error downloading model: {str(e)}")

def show_local_models():
    """Display locally downloaded models"""
    
    raw_models = []
    trained_models = []
    
    # Scan raw models
    if MODELS_RAW_DIR.exists():
        for model_dir in MODELS_RAW_DIR.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "phi_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        raw_models.append({
                            "name": metadata.get("model_name", model_dir.name),
                            "size": get_dir_size(model_dir),
                            "downloaded": metadata.get("download_time", "unknown"),
                            "path": model_dir
                        })
                    except:
                        raw_models.append({
                            "name": model_dir.name,
                            "size": get_dir_size(model_dir),
                            "downloaded": "unknown",
                            "path": model_dir
                        })
    
    # Scan trained models
    if MODELS_TRAINED_DIR.exists():
        for model_dir in MODELS_TRAINED_DIR.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "phi_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        trained_models.append({
                            "name": metadata.get("model_name", model_dir.name),
                            "size": get_dir_size(model_dir),
                            "trained": metadata.get("training_completed", "unknown"),
                            "path": model_dir
                        })
                    except:
                        trained_models.append({
                            "name": model_dir.name,
                            "size": get_dir_size(model_dir),
                            "trained": "unknown",
                            "path": model_dir
                        })
    
    # Display raw models
    if raw_models:
        st.markdown("**📥 Raw Models**")
        for model in raw_models[:5]:  # Show top 5
            size_mb = model["size"] / (1024 * 1024)
            st.markdown(f"• {model['name']} ({size_mb:.1f} MB)")
        
        if len(raw_models) > 5:
            st.caption(f"... and {len(raw_models) - 5} more")
    
    # Display trained models
    if trained_models:
        st.markdown("**🎯 Trained Models**")
        for model in trained_models[:5]:  # Show top 5
            size_mb = model["size"] / (1024 * 1024)
            st.markdown(f"• {model['name']} ({size_mb:.1f} MB)")
        
        if len(trained_models) > 5:
            st.caption(f"... and {len(trained_models) - 5} more")
    
    if not raw_models and not trained_models:
        st.info("No local models found")

def quick_download_interface():
    """Quick download interface for direct URLs"""
    
    st.markdown("**Direct Model URL**")
    
    model_url = st.text_input(
        "HuggingFace Model URL",
        placeholder="e.g., microsoft/DialoGPT-medium",
        help="Enter the model name or full URL"
    )
    
    custom_name = st.text_input(
        "Custom Name (optional)",
        placeholder="my-custom-model",
        help="Leave empty to use original name"
    )
    
    if st.button("📥 Quick Download", type="primary"):
        if model_url:
            # Extract model name from URL
            if "huggingface.co/" in model_url:
                model_name = model_url.split("/")[-2] + "/" + model_url.split("/")[-1]
            else:
                model_name = model_url
            
            download_model_from_hf(model_name)
        else:
            st.error("Please enter a model URL")

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
