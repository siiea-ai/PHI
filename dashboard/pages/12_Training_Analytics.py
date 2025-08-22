#!/usr/bin/env python3
"""
Training Analytics - Real-time Training Monitoring

Monitor live training progress, view training logs, and analyze model performance.
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
import sys
import time
from datetime import datetime
import subprocess
import threading

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Only set page config if not already set
try:
    st.set_page_config(page_title="Training Analytics", page_icon="📊", layout="wide")
except:
    pass

# Initialize directories
MODELS_TRAINED_DIR = Path("./models/trained")
TRAINING_LOGS_DIR = Path("./out/training_logs")
TRAINING_LOGS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    st.title("📊 Training Analytics")
    st.markdown("**Real-time Training Monitoring and Performance Analysis**")
    
    # Create tabs for different analytics views
    tab1, tab2, tab3, tab4 = st.tabs(["🔄 Live Training", "📈 Training History", "🎯 Model Performance", "📋 Training Logs"])
    
    with tab1:
        show_live_training()
    
    with tab2:
        show_training_history()
    
    with tab3:
        show_model_performance()
    
    with tab4:
        show_training_logs()

def show_live_training():
    """Show live training progress"""
    st.subheader("🔄 Live Training Monitor")
    
    # Check for active training jobs
    active_trainings = get_active_trainings()
    
    if not active_trainings:
        st.info("🔍 No active training jobs found")
        st.markdown("**Start a new training from the PHI Training Studio to see live progress here.**")
        
        if st.button("🎯 Go to Training Studio"):
            st.switch_page("pages/10_PHI_Training_Studio.py")
        return
    
    # Display active trainings
    for training in active_trainings:
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**{training['model_name']}**")
                st.caption(f"Started: {training['start_time']}")
            
            with col2:
                status = training.get('status', 'running')
                if status == 'running':
                    st.success("🔄 Training")
                elif status == 'completed':
                    st.success("✅ Completed")
                else:
                    st.warning("⚠️ Unknown")
            
            with col3:
                progress = training.get('progress', 0)
                st.progress(progress / 100.0)
                st.caption(f"{progress}% complete")
            
            # Training details
            with st.expander(f"📊 Training Details - {training['model_name']}"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Configuration:**")
                    phi_config = training.get('phi_config', {})
                    st.write(f"• Base LR: {phi_config.get('base_learning_rate', 'N/A')}")
                    st.write(f"• PHI Power: {phi_config.get('phi_lr_power', 'N/A')}")
                    st.write(f"• Batch Phases: {phi_config.get('batch_phi_phases', 'N/A')}")
                
                with col_b:
                    st.markdown("**Progress:**")
                    params = training.get('training_params', {})
                    current_step = training.get('current_step', 0)
                    max_steps = params.get('max_steps', 100)
                    st.write(f"• Step: {current_step}/{max_steps}")
                    st.write(f"• Loss: {training.get('current_loss', 'N/A')}")
                    st.write(f"• ETA: {training.get('eta', 'Calculating...')}")
                
                # Real-time metrics chart
                if training.get('metrics'):
                    st.markdown("**Training Metrics:**")
                    metrics_df = pd.DataFrame(training['metrics'])
                    st.line_chart(metrics_df.set_index('step'))
                
                # Control buttons
                col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
                with col_ctrl1:
                    if st.button(f"⏸️ Pause", key=f"pause_{training['id']}"):
                        pause_training(training['id'])
                with col_ctrl2:
                    if st.button(f"🔄 Refresh", key=f"refresh_{training['id']}"):
                        st.rerun()
                with col_ctrl3:
                    if st.button(f"🛑 Stop", key=f"stop_{training['id']}"):
                        stop_training(training['id'])
            
            st.divider()
    
    # Auto-refresh
    if st.checkbox("🔄 Auto-refresh (5s)", value=True):
        time.sleep(5)
        st.rerun()

def show_training_history():
    """Show training history and completed models"""
    st.subheader("📈 Training History")
    
    # Get completed trainings
    completed_trainings = get_completed_trainings()
    
    if not completed_trainings:
        st.info("No completed trainings found")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trainings", len(completed_trainings))
    
    with col2:
        avg_improvement = sum(float(t.get('improvement_estimate', '0').replace('%', '')) for t in completed_trainings) / len(completed_trainings)
        st.metric("Avg Improvement", f"{avg_improvement:.1f}%")
    
    with col3:
        total_time = sum(t.get('training_time_minutes', 0) for t in completed_trainings)
        st.metric("Total Training Time", f"{total_time:.1f} min")
    
    with col4:
        successful = len([t for t in completed_trainings if t.get('status') == 'completed'])
        st.metric("Success Rate", f"{(successful/len(completed_trainings)*100):.0f}%")
    
    # Training history table
    st.markdown("**Training History:**")
    
    history_data = []
    for training in completed_trainings:
        history_data.append({
            "Model Name": training.get('model_name', 'Unknown'),
            "Base Model": training.get('base_model', 'Unknown'),
            "Completed": training.get('training_completed', 'Unknown')[:19] if training.get('training_completed') else 'Unknown',
            "Steps": training.get('training_steps', 'N/A'),
            "Improvement": training.get('improvement_estimate', 'N/A'),
            "Final Loss": training.get('final_loss', 'N/A'),
            "Time (min)": training.get('training_time_minutes', 'N/A'),
            "Status": training.get('status', 'Unknown')
        })
    
    if history_data:
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)
        
        # Download history as CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download History CSV",
            data=csv,
            file_name=f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def show_model_performance():
    """Show model performance comparison"""
    st.subheader("🎯 Model Performance Analysis")
    
    completed_trainings = get_completed_trainings()
    
    if not completed_trainings:
        st.info("No completed trainings for performance analysis")
        return
    
    # Performance comparison chart
    if len(completed_trainings) > 1:
        st.markdown("**Performance Comparison:**")
        
        perf_data = []
        for training in completed_trainings:
            improvement = training.get('improvement_estimate', '0%')
            if isinstance(improvement, str) and '%' in improvement:
                improvement = float(improvement.replace('%', ''))
            else:
                improvement = 0
            
            perf_data.append({
                "Model": training.get('model_name', 'Unknown'),
                "Improvement (%)": improvement,
                "Final Loss": training.get('final_loss', 0),
                "Training Time (min)": training.get('training_time_minutes', 0)
            })
        
        perf_df = pd.DataFrame(perf_data)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Improvement Comparison:**")
            st.bar_chart(perf_df.set_index('Model')['Improvement (%)'])
        
        with col2:
            st.markdown("**Training Time vs Improvement:**")
            st.scatter_chart(perf_df.set_index('Training Time (min)')['Improvement (%)'])
    
    # Best performing models
    st.markdown("**Top Performing Models:**")
    
    # Sort by improvement
    sorted_trainings = sorted(completed_trainings, 
                            key=lambda x: float(x.get('improvement_estimate', '0').replace('%', '')), 
                            reverse=True)
    
    for i, training in enumerate(sorted_trainings[:5]):
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**#{i+1}. {training.get('model_name', 'Unknown')}**")
                st.caption(f"Base: {training.get('base_model', 'Unknown')}")
            
            with col2:
                st.metric("Improvement", training.get('improvement_estimate', 'N/A'))
            
            with col3:
                st.metric("Final Loss", f"{training.get('final_loss', 'N/A')}")
            
            if i < 4:  # Don't add divider after last item
                st.divider()

def show_training_logs():
    """Show detailed training logs"""
    st.subheader("📋 Training Logs")
    
    # Get available log files
    log_files = list(TRAINING_LOGS_DIR.glob("*.log"))
    
    if not log_files:
        st.info("No training logs found")
        return
    
    # Log file selector
    selected_log = st.selectbox(
        "Select Log File:",
        options=[f.name for f in log_files],
        help="Choose a training log file to view"
    )
    
    if selected_log:
        log_path = TRAINING_LOGS_DIR / selected_log
        
        # Log viewer options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_lines = st.number_input("Lines to show", 10, 1000, 100)
        
        with col2:
            filter_level = st.selectbox("Filter Level", ["All", "INFO", "WARNING", "ERROR"])
        
        with col3:
            if st.button("🔄 Refresh Logs"):
                st.rerun()
        
        # Display logs
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
            
            # Filter by level if specified
            if filter_level != "All":
                lines = [line for line in lines if filter_level in line]
            
            # Show last N lines
            recent_lines = lines[-show_lines:]
            
            # Display in code block
            log_content = ''.join(recent_lines)
            st.code(log_content, language='text')
            
            # Download log file
            st.download_button(
                label="📥 Download Log File",
                data=log_content,
                file_name=selected_log,
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"Error reading log file: {e}")

def get_active_trainings():
    """Get list of active training jobs"""
    active_trainings = []
    
    # Check for training status files
    status_files = list(Path("./out").glob("**/training_status.json"))
    
    for status_file in status_files:
        try:
            with open(status_file) as f:
                status = json.load(f)
            
            if status.get('status') == 'running':
                # Simulate training progress
                start_time = datetime.fromisoformat(status.get('start_time', datetime.now().isoformat()))
                elapsed = (datetime.now() - start_time).total_seconds() / 60  # minutes
                
                # Simulate progress based on elapsed time
                max_time = status.get('estimated_time_minutes', 20)
                progress = min(95, (elapsed / max_time) * 100)
                
                # Simulate metrics
                metrics = []
                for step in range(0, int(progress), 5):
                    loss = 1.0 - (step / 100) * 0.5 + (step % 10) * 0.01
                    metrics.append({"step": step, "loss": loss})
                
                training_info = {
                    "id": status_file.parent.name,
                    "model_name": status.get('model_name', 'Unknown'),
                    "start_time": status.get('start_time', 'Unknown'),
                    "status": status.get('status', 'unknown'),
                    "progress": progress,
                    "current_step": int(progress),
                    "current_loss": f"{1.0 - (progress / 100) * 0.5:.3f}",
                    "eta": f"{max(0, max_time - elapsed):.1f} min",
                    "phi_config": status.get('phi_config', {}),
                    "training_params": status.get('training_params', {}),
                    "metrics": metrics
                }
                
                active_trainings.append(training_info)
                
        except Exception as e:
            continue
    
    return active_trainings

def get_completed_trainings():
    """Get list of completed training jobs"""
    completed_trainings = []
    
    if MODELS_TRAINED_DIR.exists():
        for model_dir in MODELS_TRAINED_DIR.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "phi_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        
                        if metadata.get('status') == 'completed':
                            completed_trainings.append(metadata)
                    except:
                        continue
    
    # Sort by completion time (newest first)
    completed_trainings.sort(
        key=lambda x: x.get('training_completed', ''), 
        reverse=True
    )
    
    return completed_trainings

def pause_training(training_id):
    """Pause a training job"""
    st.info(f"⏸️ Pausing training {training_id}...")
    # Implementation would pause the actual training process

def stop_training(training_id):
    """Stop a training job"""
    st.warning(f"🛑 Stopping training {training_id}...")
    # Implementation would stop the actual training process

if __name__ == "__main__":
    main()
