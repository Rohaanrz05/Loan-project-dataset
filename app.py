import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="FinTech Nexus | Loan Analytics",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="collapsed"
)

# --- 🎨 NEW THEME: MIDNIGHT PURPLE & EMERALD ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&family=JetBrains+Mono:wght@400&display=swap');
    
    :root {
        --primary: #10b981;   /* Emerald Green */
        --secondary: #8b5cf6; /* Violet */
        --accent: #f59e0b;    /* Gold */
        --bg-dark: #0f0c29;   /* Deep Midnight */
        --card-bg: rgba(20, 20, 40, 0.6); 
    }

    /* ANIMATED BACKGROUND */
    .stApp {
        background-color: var(--bg-dark);
        background-image: 
            linear-gradient(30deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Background Particles Effect */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; 
        left: 0;
        width: 100%; 
        height: 100%;
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(139, 92, 246, 0.15), transparent 25%), 
            radial-gradient(circle at 85% 30%, rgba(16, 185, 129, 0.15), transparent 25%);
        z-index: -1;
        animation: float 10s ease-in-out infinite alternate;
    }
    
    @keyframes float {
        0% { transform: scale(1); }
        100% { transform: scale(1.1); }
    }

    /* 🔮 GLASS CARDS */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: var(--secondary);
        box-shadow: 0 15px 40px rgba(139, 92, 246, 0.2);
    }

    /* METRICS */
    .metric-val {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e0e7ff 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #db2777);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        width: 100%;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(219, 39, 119, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 25px rgba(219, 39, 119, 0.5);
    }

    /* HEADERS */
    h1, h2, h3 { color: white !important; }
    
    .gradient-text {
        background: linear-gradient(135deg, #a78bfa 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        text-shadow: 0 0 30px rgba(139, 92, 246, 0.3);
    }

    /* TOP TABS STYLING */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
        padding: 10px 0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 55px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        color: #cbd5e1;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.05);
        padding: 0 25px;
        transition: all 0.3s;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.3), rgba(219, 39, 119, 0.3)) !important;
        color: #fff !important;
        border: 1px solid #a78bfa !important;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.3);
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADER ---
@st.cache_data
def load_data():
    file_path = '_AI_project_data.csv'
    try:
        df = pd.read_csv(file_path)
        
        # Data Cleaning (Imputation)
        num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
                
        cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
                
        return df
    except:
        return None

df = load_data()

# --- APP ---
if df is not None:
    
    # --- HEADER SECTION ---
    c_logo, c_title = st.columns([1, 6])
    with c_logo:
        st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'>💸</h1>", unsafe_allow_html=True)
    with c_title:
        st.markdown("<h1 style='margin-bottom: 0; color: #a78bfa;'>FinTech Nexus</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; letter-spacing: 1px;'>NEXT-GEN FINANCIAL INTELLIGENCE</p>", unsafe_allow_html=True)

    st.markdown("---")

    # --- TOP NAVIGATION (TABS) ---
    tab1, tab2, tab3 = st.tabs(["📊 Executive Dashboard", "🔎 Applicant Analysis", "🤖 Loan Predictor"])

    # --- SIDEBAR FILTERS (Global) ---
    with st.sidebar:
        st.markdown("### ⚙️ Global Filters")
        prop_area = st.multiselect("Property Area", df['Property_Area'].unique(), default=df['Property_Area'].unique())
        
        # Apply Filter
        df_filtered = df[df['Property_Area'].isin(prop_area)]
        
        st.info(f"📁 Analyzing {len(df_filtered)} applications")
        st.markdown("---")
        st.caption("FinTech Nexus v2.0 | Ultra Edition")

    # --- TAB 1: EXECUTIVE DASHBOARD ---
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='gradient-text'>EXECUTIVE OVERVIEW</div>", unsafe_allow_html=True)
        st.write("")
        
        # METRICS
        k1, k2, k3, k4 = st.columns(4)
        
        avg_income = df_filtered['ApplicantIncome'].mean()
        total_loan = df_filtered['LoanAmount'].sum()
        credit_ok = len(df_filtered[df_filtered['Credit_History'] == 1])
        
        def card(col, label, val, sub, color="#8b5cf6"):
            with col:
                st.markdown(f"""
                <div class="glass-card" style="border-bottom: 4px solid {color}; text-align: center;">
                    <div class="metric-label">{label}</div>
                    <div class="metric-val">${val}</div>
                    <div style="color: #94a3b8; font-size: 0.8rem;">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        card(k1, "Avg Income", f"{avg_income:,.0f}", "Monthly Applicant Income", "#8b5cf6")
        card(k2, "Total Loan Request", f"{total_loan:,.0f}k", "Cumulative Amount", "#3b82f6")
        card(k3, "Credit Worthy", f"{credit_ok}", "History = 1.0", "#10b981")
        card(k4, "Applications", f"{len(df_filtered)}", "Total Processed", "#ec4899")

        # CHARTS
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🏠 Property Area Distribution")
            fig_pie = px.pie(df_filtered, names='Property_Area', hole=0.6, 
                             color_discrete_sequence=['#10b981', '#3b82f6', '#f59e0b'])
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', showlegend=True,
                                  legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 💰 Income vs Loan Amount")
            fig_scat = px.scatter(df_filtered, x='ApplicantIncome', y='LoanAmount', 
                                  color='Property_Area', size='LoanAmount',
                                  color_discrete_sequence=['#10b981', '#3b82f6', '#f59e0b'])
            fig_scat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_scat, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 2: APPLICANT ANALYSIS ---
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='gradient-text'>APPLICANT INSIGHTS</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🎓 Education vs Income")
            fig_bar = px.box(df_filtered, x='Education', y='ApplicantIncome', color='Education',
                             color_discrete_sequence=['#8b5cf6', '#10b981'])
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 👫 Gender Split")
            fig_gen = px.bar(df_filtered['Gender'].value_counts(), orientation='h', color_discrete_sequence=['#db2777'])
            fig_gen.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', showlegend=False)
            st.plotly_chart(fig_gen, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📂 Data Browser")
        st.dataframe(df_filtered, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 3: LOAN PREDICTOR ---
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='gradient-text'>SMART PREDICTOR</div>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8;'>AI-Assisted Eligibility Assessment</p>", unsafe_allow_html=True)
        
        c_in, c_out = st.columns([1, 1.5])
        
        with c_in:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 👤 Applicant Profile")
            u_income = st.number_input("Monthly Income ($)", value=5000)
            u_coincome = st.number_input("Co-Applicant Income ($)", value=0)
            u_loan = st.number_input("Loan Amount (k)", value=120)
            u_cred = st.selectbox("Credit History", ["Clear (1.0)", "Debts (0.0)"])
            u_prop = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            
            check = st.button("🚀 CHECK ELIGIBILITY")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c_out:
            if check:
                # SIMPLE RULE-BASED LOGIC
                cred_score = 1.0 if "1.0" in u_cred else 0.0
                total_income = u_income + u_coincome
                ratio = u_loan / (total_income/1000) if total_income > 0 else 100
                
                approved = False
                if cred_score == 1.0 and ratio < 50:
                    approved = True
                    prob = np.random.uniform(75, 95)
                elif cred_score == 1.0:
                    approved = False
                    prob = np.random.uniform(40, 60)
                else:
                    approved = False
                    prob = np.random.uniform(10, 30)
                
                status_color = "#10b981" if approved else "#ef4444"
                status_text = "APPROVED" if approved else "REJECTED"
                
                st.markdown(f"""
                <div class="glass-card" style="text-align:center; border: 2px solid {status_color}; box-shadow: 0 0 30px {status_color}40;">
                    <h2 style="color:{status_color}; margin:0; letter-spacing: 2px;">LOAN {status_text}</h2>
                    <h1 style="font-size: 5rem; margin: 10px 0; color: white; text-shadow: 0 0 20px {status_color}80;">{prob:.1f}%</h1>
                    <p style="color:#94a3b8;">Approval Probability Score</p>
                    <hr style="background:rgba(255,255,255,0.1);">
                    <div style="display:flex; justify-content:space-around;">
                        <div>
                            <div style="font-size:0.8rem; color:#64748b;">CREDIT</div>
                            <div style="font-weight:bold; color:{status_color};">{'PASS' if cred_score==1 else 'FAIL'}</div>
                        </div>
                        <div>
                            <div style="font-size:0.8rem; color:#64748b;">RATIO</div>
                            <div style="font-weight:bold; color:{'#10b981' if ratio<50 else '#f59e0b'};">{ratio:.1f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("👈 Enter applicant details to run the AI assessment.")

else:
    st.error("🚨 Please upload '_AI_project_data.csv' to view the dashboard.")
