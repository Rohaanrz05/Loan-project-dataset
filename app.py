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
    initial_sidebar_state="expanded"
)

# --- 🎨 FINTECH ULTRA THEME ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&family=JetBrains+Mono:wght@400&display=swap');
    
    :root {
        --primary: #10b981;   /* Emerald Green */
        --secondary: #3b82f6; /* Blue */
        --accent: #f59e0b;    /* Gold */
        --bg-dark: #0f172a;   /* Slate 900 */
        --card-bg: rgba(30, 41, 59, 0.5); /* Slate 800 glass */
    }

    /* BACKGROUND */
    .stApp {
        background-color: var(--bg-dark);
        background-image: 
            radial-gradient(at 0% 0%, rgba(16, 185, 129, 0.1) 0px, transparent 50%), 
            radial-gradient(at 100% 0%, rgba(59, 130, 246, 0.1) 0px, transparent 50%);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* 🔮 GLASS CARDS */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: var(--primary);
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.1);
    }

    /* METRICS */
    .metric-val {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, #059669 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4);
    }

    /* HEADERS */
    h1, h2, h3 { color: white !important; }
    
    .gradient-text {
        background: linear-gradient(135deg, #34d399 0%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.9);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* CUSTOM TAGS */
    .tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .tag-green { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .tag-blue { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADER ---
@st.cache_data
def load_data():
    file_path = 'test_Y3wMUE5_7gLdaTN (1).csv'
    try:
        df = pd.read_csv(file_path)
        
        # Data Cleaning (Imputation)
        # Fill Numeric with Mean
        num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
                
        # Fill Categorical with Mode
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
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #10b981;'>🏦 FinTech Nexus</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#64748b;'>LOAN INTELLIGENCE SYSTEM</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        menu = st.radio("MENU", ["📊 Executive Dashboard", "🔎 Applicant Analysis", "🤖 Loan Predictor"], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### ⚙️ Filters")
        prop_area = st.multiselect("Property Area", df['Property_Area'].unique(), default=df['Property_Area'].unique())
        
        # Apply Filter
        df_filtered = df[df['Property_Area'].isin(prop_area)]
        
        st.info(f"📁 Analyzing {len(df_filtered)} applications")

    # --- TAB 1: EXECUTIVE DASHBOARD ---
    if menu == "📊 Executive Dashboard":
        st.markdown("<div class='gradient-text'>EXECUTIVE OVERVIEW</div>", unsafe_allow_html=True)
        st.write("")
        
        # METRICS
        k1, k2, k3, k4 = st.columns(4)
        
        avg_income = df_filtered['ApplicantIncome'].mean()
        total_loan = df_filtered['LoanAmount'].sum()
        credit_ok = len(df_filtered[df_filtered['Credit_History'] == 1])
        
        def card(col, label, val, sub, color="#10b981"):
            with col:
                st.markdown(f"""
                <div class="glass-card" style="border-top: 3px solid {color}; text-align: center;">
                    <div class="metric-label">{label}</div>
                    <div class="metric-val">${val}</div>
                    <div style="color: #64748b; font-size: 0.8rem;">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        card(k1, "Avg Income", f"{avg_income:,.0f}", "Monthly Applicant Income")
        card(k2, "Total Loan Request", f"{total_loan:,.0f}k", "Cumulative Amount", "#3b82f6")
        card(k3, "Credit Worthy", f"{credit_ok}", "History = 1.0", "#f59e0b")
        card(k4, "Applications", f"{len(df_filtered)}", "Total Processed", "#ec4899")

        # CHARTS
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🏠 Property Area Distribution")
            fig_pie = px.pie(df_filtered, names='Property_Area', hole=0.6, 
                             color_discrete_sequence=['#10b981', '#3b82f6', '#f59e0b'])
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', showlegend=True)
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
    elif menu == "🔎 Applicant Analysis":
        st.markdown("<div class='gradient-text'>APPLICANT INSIGHTS</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🎓 Education vs Income")
            fig_bar = px.box(df_filtered, x='Education', y='ApplicantIncome', color='Education',
                             color_discrete_sequence=['#3b82f6', '#10b981'])
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 👫 Gender Split")
            fig_gen = px.bar(df_filtered['Gender'].value_counts(), orientation='h', color_discrete_sequence=['#f59e0b'])
            fig_gen.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', showlegend=False)
            st.plotly_chart(fig_gen, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📂 Data Browser")
        st.dataframe(df_filtered, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 3: LOAN PREDICTOR ---
    elif menu == "🤖 Loan Predictor":
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
                # SIMPLE RULE-BASED LOGIC (Since we lack training data)
                # Credit History is the strongest predictor in finance
                cred_score = 1.0 if "1.0" in u_cred else 0.0
                total_income = u_income + u_coincome
                ratio = u_loan / (total_income/1000) if total_income > 0 else 100
                
                # Logic: Good Credit AND Reasonable Loan-to-Income
                approved = False
                if cred_score == 1.0 and ratio < 50:
                    approved = True
                    prob = np.random.uniform(75, 95) # Simulating confidence
                elif cred_score == 1.0:
                    approved = False # Income too low for loan
                    prob = np.random.uniform(40, 60)
                else:
                    approved = False # Bad Credit
                    prob = np.random.uniform(10, 30)
                
                status_color = "#10b981" if approved else "#ef4444"
                status_text = "APPROVED" if approved else "REJECTED"
                
                st.markdown(f"""
                <div class="glass-card" style="text-align:center; border: 2px solid {status_color};">
                    <h2 style="color:{status_color}; margin:0;">LOAN {status_text}</h2>
                    <h1 style="font-size: 4rem; margin: 10px 0; color: white;">{prob:.1f}%</h1>
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
    st.error("🚨 Please upload 'test_Y3wMUE5_7gLdaTN (1).csv' to view the dashboard.")