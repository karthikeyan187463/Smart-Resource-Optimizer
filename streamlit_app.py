import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from sro import SmartResourceOptimizer, ROLE_DISPLAY_MAPPING
import warnings
import json
import os

warnings.filterwarnings("ignore", message=r".*was created with a default value.*")

def load_workloads(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return []

    try:
        with open(file_path, "r") as f:
            workloads = json.load(f)
            return workloads
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []

if 'optimizer' not in st.session_state:
    st.session_state.optimizer = SmartResourceOptimizer()
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'workload_config' not in st.session_state:
    st.session_state.workload_config = {}
if 'role_display' not in st.session_state:
    st.session_state.role_display = list(ROLE_DISPLAY_MAPPING.keys())[0]
if 'app_name' not in st.session_state:
    st.session_state.app_name = "my_application"
if 'description' not in st.session_state:
    st.session_state.description = ""
if 'gpu_request' not in st.session_state:
    st.session_state.gpu_request = 2
if 'memory_request' not in st.session_state:
    st.session_state.memory_request = 8192
if 'disk_request' not in st.session_state:
    st.session_state.disk_request = 50000
if 'max_instance_per_node' not in st.session_state:
    st.session_state.max_instance_per_node = 1
if 'estimated_execution_hours' not in st.session_state:
    st.session_state.estimated_execution_hours = 8.0

sample_workloads = load_workloads("./data/json/workload_templates.json")

st.set_page_config(
    page_title="Smart Resource Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
        <style>
            :root {
                --primary-color: #667eea;
                --secondary-color: #764ba2;
                --accent-color: #4facfe;
                --success-color: #00f2fe;
                --warning-color: #f093fb;
                --danger-color: #f5576c;
            }
            
            .main .block-container {
                background-color: #0e1117;
                color: #fafafa;
            }
            
            .main-header {
                text-align: center;
                padding: 2rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 15px;
                margin-bottom: 2rem;
                box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
            }
            
            .main-header h1 {
                font-size: 3rem;
                margin-bottom: 0.5rem;
                font-weight: 700;
            }
            
            .main-header p {
                font-size: 1.2rem;
                opacity: 0.9;
            }
            
            .welcome-section {
                text-align: center;
                padding: 3rem 1rem;
                background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
                border-radius: 20px;
                margin: 2rem 0;
                border: 1px solid #333;
            }
            
            .welcome-section h2 {
                color: #667eea;
                font-size: 2.5rem;
                margin-bottom: 1rem;
                font-weight: 600;
            }
            
            .welcome-section p {
                color: #cccccc;
                font-size: 1.2rem;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto 2rem auto;
            }
            
            .feature-cards {
                display: flex;
                justify-content: space-around;
                margin: 3rem 0;
                gap: 1.5rem;
                flex-wrap: wrap;
            }
            
            .feature-card {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                border: 1px solid #333;
                flex: 1;
                min-width: 300px;
                text-align: center;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .feature-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
            }
            
            .feature-card h3 {
                color: #667eea;
                font-size: 1.5rem;
                margin-bottom: 1rem;
                font-weight: 600;
            }
            
            .feature-card p {
                color: #cccccc;
                line-height: 1.6;
            }
            
            .prediction-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                border-radius: 20px;
                margin: 1rem 0;
                text-align: center;
                box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .instance-box {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 2rem;
                border-radius: 20px;
                margin: 1rem 0;
                text-align: center;
                box-shadow: 0 8px 32px rgba(240, 147, 251, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .cost-savings-box {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                padding: 2rem;
                border-radius: 20px;
                margin: 1rem 0;
                text-align: center;
                box-shadow: 0 8px 32px rgba(79, 172, 254, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .prediction-box h3, .instance-box h3, .cost-savings-box h3 {
                font-size: 1.2rem;
                margin-bottom: 1rem;
                opacity: 0.9;
            }
            
            .prediction-box h1, .instance-box h2, .cost-savings-box h1 {
                font-size: 2.5rem;
                margin: 0.5rem 0;
                font-weight: 700;
            }
            
            .optimization-summary {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 2rem;
                border-radius: 20px;
                margin: 2rem 0;
                border: 1px solid #333;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }
            
            .optimization-summary h2 {
                color: #667eea;
                text-align: center;
                margin-bottom: 2rem;
                font-size: 2rem;
                font-weight: 600;
            }
            
            .summary-metrics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1.5rem;
                margin-top: 1rem;
            }
            
            .summary-metric {
                background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
                padding: 1.5rem;
                border-radius: 15px;
                text-align: center;
                border: 1px solid #444;
                transition: transform 0.3s ease;
            }
            
            .summary-metric:hover {
                transform: translateY(-3px);
                border-color: #667eea;
            }
            
            .summary-metric .metric-value {
                font-size: 2rem;
                font-weight: 700;
                color: #667eea;
                margin-bottom: 0.5rem;
                min-height: 50px;
            }
            
            .summary-metric .metric-label {
                color: #cccccc;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .summary-metric .metric-delta {
                font-size: 1.1rem;
                margin-top: 0.5rem;
            }
            
            .resource-section {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 2rem;
                border-radius: 20px;
                margin: 2rem 0;
                border: 1px solid #333;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }
            
            .resource-section h2 {
                color: #667eea;
                text-align: center;
                margin-bottom: 2rem;
                font-size: 2rem;
                font-weight: 600;
            }
            
            .utilization-cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 2rem;
                margin-top: 2rem;
            }
            
            .utilization-card {
                background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
                padding: 2rem;
                border-radius: 15px;
                border: 1px solid #444;
                text-align: center;
            }
            
            .utilization-card h3 {
                color: #667eea;
                margin-bottom: 1rem;
                font-size: 1.3rem;
            }
            
            .stButton > button {
                width: 100%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                color: white;
                padding: 1rem 2rem;
                font-size: 16px;
                border-radius: 15px;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
            }
            
            [data-testid="stSidebarUserContent"] {
                min-height: 99.8vh;
                padding: 1.4rem 1.5rem 1.4rem 1.5rem !important;
                background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
            }
            
            .stTabs [data-baseweb="tab-list"] {
                gap: 1rem;
            }
            
            .stTabs [data-baseweb="tab"] {
                background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
                border-radius: 10px;
                border: 1px solid #444;
                padding: 0.5rem 1rem;
                color: #cccccc;
            }
            
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-color: #667eea;
            }
            
            .stTable {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 10px;
                overflow: hidden;
            }
            
            .footer {
                text-align: center;
                color: #888;
                padding: 2rem;
                margin-top: 3rem;
                border-top: 1px solid #333;
            }
            
            section > .block-container {
                padding: 3rem 3rem 3rem;
            }
        </style>
    """, 
    unsafe_allow_html=True
)

st.markdown(
    """
        <div class="main-header">
            <h1>⚡ Smart Resource Optimizer</h1>
            <p>Optimize your cloud workloads with AI-powered resource prediction</p>
        </div>
    """, 
    unsafe_allow_html=True
)

with st.sidebar:
    template_names = [w["app_name"] for w in sample_workloads]
    
    template_selector_key = "template_selector_main"
    selected_template = st.selectbox(
        "Choose from a template", 
        ["Custom"] + template_names, 
        key=template_selector_key
    )

    if selected_template != "Custom":
        tpl = next((w for w in sample_workloads if w["app_name"] == selected_template), None)
        if tpl:
            reverse_role_map = {v: k for k, v in ROLE_DISPLAY_MAPPING.items()}
            st.session_state.role_display = reverse_role_map.get(
                tpl["role"], list(ROLE_DISPLAY_MAPPING.keys())[0]
            )
            st.session_state.app_name = tpl["app_name"]
            st.session_state.description = tpl.get("description", "")
            st.session_state.gpu_request = int(tpl["gpu_request"])
            st.session_state.memory_request = int(tpl["memory_request"])
            st.session_state.disk_request = int(tpl["disk_request"])
            st.session_state.max_instance_per_node = int(tpl["max_instance_per_node"])
            st.session_state.estimated_execution_hours = float(tpl["estimated_execution_hours"])

with st.sidebar.form("workload_form"):
    st.subheader("Workload Configuration")

    _role_options = list(ROLE_DISPLAY_MAPPING.keys())
    _role_idx = _role_options.index(
        st.session_state.get("role_display", _role_options[0])
    ) if st.session_state.get("role_display") in _role_options else 0

    role_display = st.selectbox(
        "Workload Type",
        options=_role_options,
        index=_role_idx,
        key="role_display",
    )

    app_name = st.text_input(
        "Application Name",
        value=st.session_state.get("app_name", "my_application"),
        key="app_name",
    )

    description = st.text_area(
        "Description",
        value=st.session_state.get("description", ""),
        key="description",
    )

    st.subheader("Resource Requirements")

    col1, col2 = st.columns(2)

    with col1:
        gpu_request = st.number_input(
            "GPU Count",
            min_value=0,
            max_value=32,
            value=int(st.session_state.get("gpu_request", 2)),
            key="gpu_request",
            help="Number of GPUs required",
        )

        memory_request = st.number_input(
            "Memory (MB)",
            min_value=64,
            max_value=1048576,
            value=int(st.session_state.get("memory_request", 8192)),
            step=512,
            key="memory_request",
            help="Memory requirement in megabytes",
        )

    with col2:
        disk_request = st.number_input(
            "Disk Space (MB)",
            min_value=1024,
            max_value=10485760,
            value=int(st.session_state.get("disk_request", 50000)),
            step=1024,
            key="disk_request",
            help="Disk space requirement in megabytes",
        )

        max_instance_per_node = st.number_input(
            "IPN",
            min_value=1,
            max_value=10,
            value=int(st.session_state.get("max_instance_per_node", 1)),
            key="max_instance_per_node",
            help="Maximum number of instances per node",
        )

    estimated_execution_hours = st.number_input(
        "Estimated Runtime (hours)",
        min_value=0.5,
        max_value=168.0,
        value=float(st.session_state.get("estimated_execution_hours", 8.0)),
        step=0.5,
        key="estimated_execution_hours",
        help="Expected execution time in hours",
    )

    submitted = st.form_submit_button("Optimize Resources")

    if submitted:
        actual_role = ROLE_DISPLAY_MAPPING[role_display]

        st.session_state.workload_config = {
            "role": actual_role,
            "app_name": app_name,
            "gpu_request": gpu_request,
            "memory_request": memory_request,
            "disk_request": disk_request,
            "max_instance_per_node": max_instance_per_node,
            "estimated_execution_hours": estimated_execution_hours,
            "description": description if description else f"{role_display} workload",
        }
        st.session_state.prediction_made = True
        st.rerun()

if st.session_state.prediction_made:
    config = st.session_state.workload_config
    optimizer = st.session_state.optimizer
    
    # Make prediction
    predicted_cpu = optimizer.predict_cpu_request(config)
    instance_recommendations = optimizer.optimize_instance_selection([config])
    schedule_results = optimizer.schedule_workloads([config])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="prediction-box">
            <h3>CPU Prediction</h3>
            <h1>{predicted_cpu}</h1>
            <p>Recommended CPU Cores</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Instance recommendation
    if instance_recommendations[0].get('recommended_instance'):
        instance = instance_recommendations[0]['recommended_instance']
        with col2:
            st.markdown(f"""
            <div class="instance-box" style="
                min-height: 200px; 
                display: flex; 
                flex-direction: column; 
                justify-content: center; 
                align-items: center;
                text-align: center;
            ">
                <h3 style="margin: 0;">Recommended Instance</h3>
                <span style="margin: 0.5rem 0; padding: 0.3rem 0.8rem; 
                        background: linear-gradient(90deg, #667eea, #764ba2);
                        border-radius: 12px; 
                        font-weight: 600; 
                        font-size: 0.9rem;
                        color: white;">
                    {instance['instance_name']}
                </span>
                <h1 style="margin: 0.5rem 0; font-size: 1.4rem; font-weight: 700;">
                    {instance['cpu_count']} vCPU {instance['memory_gb']} GB RAM
                </h1>
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.85;">
                    ${instance['hourly_cost']:.4f}/hour
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        with col2:
            st.error("No suitable instance found for your requirements")
    
    # Cost comparison
    if schedule_results['scheduled_jobs'] and not schedule_results['scheduled_jobs'][0].get('error'):
        job = schedule_results['scheduled_jobs'][0]
        savings_pct = (job['savings'] / job['on_demand_cost']) * 100 if job['on_demand_cost'] > 0 else 0
        
        with col3:
            st.markdown(f"""
            <div class="cost-savings-box">
                <h3>Potential Savings</h3>
                <h1>${job['savings']:.2f}</h1>
                <p>{savings_pct:.1f}% reduction</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="resource-section">', unsafe_allow_html=True)
    st.markdown('<h2>Detailed Analysis</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Cost Breakdown", "Resource Utilization", "Scheduling"])
    
    with tab1:
        if schedule_results['scheduled_jobs'] and not schedule_results['scheduled_jobs'][0].get('error'):
            job = schedule_results['scheduled_jobs'][0]
            
            fig_cost = go.Figure(data=[
                go.Bar(
                    x=['On-Demand', 'Optimized Spot'],
                    y=[job['on_demand_cost'], job['optimized_cost']],
                    marker_color=['#f44336', '#4CAF50'],
                    text=[f"${job['on_demand_cost']:.2f}", f"${job['optimized_cost']:.2f}"],
                    textposition='outside'
                )
            ])
            
            fig_cost.update_layout(
                title='Cost Comparison: On-Demand vs Optimized Spot Pricing',
                title_x=0.5,
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#fafafa',
                title_font_size=18,
                yaxis_title='Cost ($)',
                yaxis=dict(range=[0, max(job['on_demand_cost'], job['optimized_cost']) * 1.2]),
                showlegend=False
            )
            
            st.plotly_chart(fig_cost, use_container_width=True)
            
            st.subheader("Cost Breakdown")
            cost_df = pd.DataFrame({
                'Metric': [
                    'On-Demand Cost',
                    'Optimized Spot Cost',
                    'Total Savings',
                    'Savings Percentage',
                    'Execution Duration',
                    'Hourly Rate'
                ],
                'Value': [
                    f"${job['on_demand_cost']:.2f}",
                    f"${job['optimized_cost']:.2f}",
                    f"${job['savings']:.2f}",
                    f"{(job['savings'] / job['on_demand_cost']) * 100:.1f}%",
                    f"{config['estimated_execution_hours']} hours",
                    f"${instance['hourly_cost']:.4f}/hour"
                ]
            })
            st.dataframe(cost_df, use_container_width=True, hide_index=True)
    with tab2:
        if instance_recommendations[0].get('recommended_instance'):
            instance = instance_recommendations[0]['recommended_instance']
            
            st.markdown('<div class="utilization-cards">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="utilization-card">
                    <h3>CPU Utilization</h3>
                    <div style="font-size: 2rem; color: #667eea; font-weight: 700; margin: 1rem 0;">
                        {instance['cpu_utilization']:.1f}%
                    </div>
                    <div style="color: {'#4CAF50' if instance['cpu_utilization'] <= 90 else '#f44336'};">
                        {instance['cpu_utilization'] - 75:.1f}% vs. target (75%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # CPU utilization gauge
                fig_cpu = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = instance['cpu_utilization'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "CPU Utilization %", 'font': {'color': '#fafafa', 'size': 16}},
                    number = {'font': {'color': '#667eea', 'size': 28}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickcolor': '#fafafa'},
                        'bar': {'color': "#667eea", 'thickness': 0.3},
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(102, 126, 234, 0.1)"},
                            {'range': [50, 80], 'color': "rgba(255, 193, 7, 0.3)"},
                            {'range': [80, 100], 'color': "rgba(220, 53, 69, 0.3)"}
                        ],
                        'threshold': {
                            'line': {'color': "#f44336", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_cpu.update_layout(
                    height=300, 
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#fafafa'
                )
                st.plotly_chart(fig_cpu, use_container_width=True)
            
            with col2:
                st.markdown(f"""
                <div class="utilization-card">
                    <h3>Memory Utilization</h3>
                    <div style="font-size: 2rem; color: #f093fb; font-weight: 700; margin: 1rem 0;">
                        {instance['memory_utilization']:.1f}%
                    </div>
                    <div style="color: {'#4CAF50' if instance['memory_utilization'] <= 90 else '#f44336'};">
                        {instance['memory_utilization'] - 75:.1f}% vs. target (75%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Memory utilization gauge
                fig_mem = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = instance['memory_utilization'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Memory Utilization %", 'font': {'color': '#fafafa', 'size': 16}},
                    number = {'font': {'color': '#f093fb', 'size': 28}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickcolor': '#fafafa'},
                        'bar': {'color': "#f093fb", 'thickness': 0.3},
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(240, 147, 251, 0.1)"},
                            {'range': [50, 80], 'color': "rgba(255, 193, 7, 0.3)"},
                            {'range': [80, 100], 'color': "rgba(220, 53, 69, 0.3)"}
                        ],
                        'threshold': {
                            'line': {'color': "#f44336", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_mem.update_layout(
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#fafafa'
                )
                st.plotly_chart(fig_mem, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        if schedule_results['scheduled_jobs'] and not schedule_results['scheduled_jobs'][0].get('error'):
            job = schedule_results['scheduled_jobs'][0]
            
            st.subheader("Optimal Scheduling")
            
            # scheduling info
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Recommended Start Time:** {job['scheduled_start']}")
            with col2:
                st.info(f"**Estimated End Time:** {job['scheduled_end']}")
            
            # Hourly pricing visualization
            hourly_multipliers = [
                0.3, 0.25, 0.2, 0.2, 0.25, 0.4,
                0.6, 0.8, 1.0, 1.0, 0.9, 0.8,
                0.9, 1.0, 1.0, 0.9, 0.8, 0.7,
                0.6, 0.5, 0.4, 0.35, 0.3, 0.3
            ]

            hours = list(range(24))
            base_cost = instance['hourly_cost']
            spot_prices = [base_cost * multiplier for multiplier in hourly_multipliers]

            fig_schedule = go.Figure()

            # Spot price line
            fig_schedule.add_trace(go.Scatter(
                x=hours,
                y=spot_prices,
                mode='lines+markers',
                name='Spot Price ($)',
                line=dict(color='#4CAF50', width=3),
                marker=dict(size=6)
            ))

            # on demand price graph
            fig_schedule.add_trace(go.Scatter(
                x=hours,
                y=[base_cost] * 24,
                mode='lines',
                name='On-Demand Price ($)',
                line=dict(color='#f44336', width=3, dash='dash')
            ))

            # Optimal execution window for task executoin is highlighted
            start_hour = datetime.strptime(job['scheduled_start'], "%Y-%m-%d %H:%M").hour
            end_hour = start_hour + config['estimated_execution_hours']

            fig_schedule.add_vrect(
                x0=start_hour, x1=min(end_hour, 24),
                fillcolor="#667eea", opacity=0.2,
                annotation_text="⭐ Optimal Window", 
                annotation_position="top left",
                annotation_font_color="#667eea"
            )

            fig_schedule.update_layout(
                title='24-Hour Pricing Pattern - Spot vs On-Demand',
                height=400, 
                title_x=0.5,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#fafafa',
                title_font_size=18,
                xaxis_title='Hour of Day',
                yaxis_title='Price ($)',
                xaxis=dict(
                    tickmode='linear',
                    tick0=0,
                    dtick=2,
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                yaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(0,0,0,0.5)"
                )
            )
            st.plotly_chart(fig_schedule, use_container_width=True)
            
            schedule_df = pd.DataFrame({
                'Detail': [
                    'Optimal Start Time',
                    'Estimated End Time',
                    'Execution Duration',
                    'Average Spot Price',
                    'On-Demand Price',
                    'Cost Savings'
                ],
                'Value': [
                    job['scheduled_start'],
                    job['scheduled_end'],
                    f"{config['estimated_execution_hours']} hours",
                    f"${job['optimized_cost'] / config['estimated_execution_hours']:.4f}/hour",
                    f"${base_cost:.4f}/hour",
                    f"${job['savings']:.2f} ({(job['savings'] / job['on_demand_cost']) * 100:.1f}%)"
                ]
            })
            st.dataframe(schedule_df, use_container_width=True, hide_index=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Optimised Summary section
    st.markdown('<div class="optimization-summary">', unsafe_allow_html=True)
    st.markdown('<h2>Optimization Summary</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="summary-metrics">', unsafe_allow_html=True)
    
    # Summary metrics
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        st.markdown(f"""
        <div class="summary-metric">
            <div class="metric-value">{predicted_cpu}</div>
            <div class="metric-label">Predicted CPU Cores</div>
            <div class="metric-delta" style="color: #4CAF50;">AI-Optimized</div>
        </div>
        """, unsafe_allow_html=True)
    
    with summary_col2:
        if instance_recommendations[0].get('recommended_instance'):
            instance = instance_recommendations[0]['recommended_instance']
            st.markdown(f"""
            <div class="summary-metric">
                <div class="metric-value"><span style="padding: 0.3rem 0.8rem; 
                        background: linear-gradient(90deg, #667eea, #764ba2);
                        border-radius: 12px; 
                        font-weight: 600; 
                        font-size: 1.6rem;
                        color: white;">{instance['instance_name']}</span></div>
                <div class="metric-label">Recommended Instance</div>
                <div class="metric-delta" style="color: #f093fb;">{instance['memory_gb']} GB RAM</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="summary-metric">
                <div class="metric-value">N/A</div>
                <div class="metric-label">Recommended Instance</div>
                <div class="metric-delta" style="color: #f44336;">Not Available</div>
            </div>
            """, unsafe_allow_html=True)
    
    with summary_col3:
        if schedule_results['scheduled_jobs'] and not schedule_results['scheduled_jobs'][0].get('error'):
            job = schedule_results['scheduled_jobs'][0]
            st.markdown(f"""
            <div class="summary-metric">
                <div class="metric-value">${job['optimized_cost']:.2f}</div>
                <div class="metric-label">Total Cost</div>
                <div class="metric-delta" style="color: #4CAF50;">-${job['savings']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="summary-metric">
                <div class="metric-value">N/A</div>
                <div class="metric-label">Total Cost</div>
                <div class="metric-delta" style="color: #666;">Not Available</div>
            </div>
            """, unsafe_allow_html=True)
    
    with summary_col4:
        if schedule_results['scheduled_jobs'] and not schedule_results['scheduled_jobs'][0].get('error'):
            job = schedule_results['scheduled_jobs'][0]
            savings_pct = (job['savings'] / job['on_demand_cost']) * 100
            st.markdown(f"""
            <div class="summary-metric">
                <div class="metric-value">{savings_pct:.1f}%</div>
                <div class="metric-label">Cost Savings</div>
                <div class="metric-delta" style="color: #00f2fe;">${job['savings']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="summary-metric">
                <div class="metric-value">0%</div>
                <div class="metric-label">Cost Savings</div>
                <div class="metric-delta" style="color: #666;">No savings</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="welcome-section">
        <h2>Welcome to Smart Resource Optimizer</h2>
        <p>
            Configure your workload parameters in the sidebar to get started with AI-powered resource
            optimization. Our intelligent system will help you find the perfect balance between performance and
            cost.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown("""
            <div class="feature-card">
                <h3>AI Prediction</h3>
                <p>Get accurate CPU resource predictions using machine learning models trained on historical workload data.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown("""
            <div class="feature-card">
                <h3>Instance Optimization</h3>
                <p>Find the most cost-effective cloud instance configuration for your specific workload requirements.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown("""
            <div class="feature-card">
                <h3>Cost Savings</h3>
                <p>Optimize scheduling with spot pricing to achieve significant cost reductions while maintaining performance.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # sample workload examples
    st.markdown('<div class="resource-section">', unsafe_allow_html=True)
    st.markdown('<h2>Sample Workload Types</h2>', unsafe_allow_html=True)
    
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        with st.expander("**Machine Learning Training**", expanded=True):
            st.markdown("""
            **Typical Configuration:**
            - **GPU:** 2-8 cores
            - **Memory:** 16-64 GB
            - **Duration:** 4-24 hours
            - **Characteristics:** High CPU and GPU utilization
            - **Best for:** Model training, hyperparameter tuning
            """)
    
    with example_col2:
        with st.expander("**ML Inference**", expanded=True):
            st.markdown("""
            **Typical Configuration:**
            - **GPU:** 0-2 cores
            - **Memory:** 4-16 GB  
            - **Duration:** Continuous/On-demand
            - **Characteristics:** Moderate resource usage
            - **Best for:** Model serving, real-time predictions, API endpoints
            """)
    
    with example_col3:
        with st.expander("**Data Processing (ETL)**", expanded=True):
            st.markdown("""
            **Typical Configuration:**
            - **GPU:** 0 cores
            - **Memory:** 8-32 GB
            - **Duration:** 2-12 hours
            - **Characteristics:** Memory and I/O intensive
            - **Best for:** Data transformation, batch processing, analytics
            """)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
        <div class="footer">
            <p>Smart Resource Optimizer | 2025</p>
            <p style="font-size: 0.9rem; margin-top: 0.5rem;">
                Optimizing cloud resources • Reducing costs • Maximizing efficiency
            </p>
        </div>
    """, 
    unsafe_allow_html=True
)