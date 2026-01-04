import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="FinTech Nexus | Stealth Edition",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="collapsed"
)

# --- 🎨 STEALTH BLACK & NEON THEME ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400&display=swap');
    
    :root {
        --primary: #00ff9d;   /* Neon Green */
        --secondary: #00b8ff; /* Neon Blue */
        --accent: #d946ef;    /* Neon Purple */
        --bg-black: #050505;  /* Deep Black */
        --card-bg: rgba(20, 20, 20, 0.7); 
        --text-white: #ffffff;
        --text-gray: #a1a1aa;
    }

    /* BACKGROUND */
    .stApp {
        background-color: var(--bg-black);
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(0, 184, 255, 0.1) 0%, transparent 50%),
            linear-gradient(180deg, #000000 0%, #121212 100%);
        font-family: 'Inter', sans-serif;
        color: var(--text-white);
    }

    /* 🔮 HIGH CONTRAST GLASS CARDS */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: var(--primary);
        box-shadow: 0 0 15px rgba(0, 255, 157, 0.2);
    }

    /* METRICS */
    .metric-val {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: var(--text-gray);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    /* BUTTONS - HIGH VISIBILITY */
    .stButton > button {
        background: linear-gradient(90deg, #00ff9d 0%, #00b8ff 100%);
        color: #000000; /* Black text on bright button for readability */
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 800;
        width: 100%;
        transition: all 0.3s;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(0, 255, 157, 0.6);
        color: #000000;
    }

    /* HEADERS */
    h1, h2, h3, h4, h5, h6 { 
        color: #ffffff !important; 
        font-family: 'Inter', sans-serif !important;
    }
    
    .gradient-text {
        background: linear-gradient(90deg, #00ff9d, #00b8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3rem;
        letter-spacing: -1px;
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        padding: 10px 0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        color: #a1a1aa;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.05);
        padding: 0 25px;
        transition: all 0.3s;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 255, 157, 0.1) !important;
        color: #00ff9d !important;
        border: 1px solid #00ff9d !important;
        box-shadow: 0 0 10px rgba(0, 255, 157, 0.2);
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* INPUT FIELDS */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        color: white !important;
    }
    
    /* PLOTLY FIXES */
    .js-plotly-plot .plotly .modebar {
        display: none !important;
    }
    
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADER (AUTO-GENERATED) ---
@st.cache_data
def load_data():
    # This generates random data so the app works without needing a CSV file
    np.random.seed(42)
    data_size = 500
    
    data = {
        'ApplicantIncome': np.random.randint(2500, 20000, data_size),
        'CoapplicantIncome': np.random.randint(0, 10000, data_size),
        'LoanAmount': np.random.randint(50, 600, data_size),
        'Loan_Amount_Term': np.random.choice([360, 180, 480], data_size),
        'Credit_History': np.random.choice([1.0, 0.0], data_size, p=[0.8, 0.2]),
        'Gender': np.random.choice(['Male', 'Female'], data_size),
        'Married': np.random.choice(['Yes', 'No'], data_size),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], data_size),
        'Self_Employed': np.random.choice(['Yes', 'No'], data_size, p=[0.15, 0.85]),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], data_size, p=[0.8, 0.2]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], data_size)
    }
    return pd.DataFrame(data)

df = load_data()

# --- APP ---
if df is not None:
    
    # --- HEADER ---
    c_logo, c_title = st.columns([0.8, 6])
    with c_logo:
        st.markdown("<h1 style='text-align: center; font-size: 3.5rem; margin:0;'>💎</h1>", unsafe_allow_html=True)
    with c_title:
        st.markdown("<h1 style='margin-bottom: 0; color: white;'>FinTech Nexus</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #a1a1aa; font-size: 1.1rem; letter-spacing: 1px;'>NEXT-GEN FINANCIAL INTELLIGENCE</p>", unsafe_allow_html=True)

    st.markdown("---")

    # --- TOP NAV ---
    tab1, tab2, tab3 = st.tabs(["📊 Executive Dashboard", "🔎 Applicant Analysis", "🤖 Loan Predictor"])

    # --- SIDEBAR FILTERS ---
    with st.sidebar:
        st.markdown("### ⚙️ Filter Engine")
        prop_area = st.multiselect("Property Area", df['Property_Area'].unique(), default=df['Property_Area'].unique())
        
        df_filtered = df[df['Property_Area'].isin(prop_area)]
        
        st.markdown("---")
        st.info(f"⚡ Processing {len(df_filtered)} Records")
        st.caption("v2.5.0 | Stealth Mode")

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
        
        def card(col, label, val, sub, border_color="rgba(255,255,255,0.2)"):
            with col:
                st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid {border_color};">
                    <div class="metric-label">{label}</div>
                    <div class="metric-val">${val}</div>
                    <div style="color: #64748b; font-size: 0.8rem; margin-top: 5px;">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        card(k1, "Avg Income", f"{avg_income:,.0f}", "Monthly Applicant Income", "#00ff9d")
        card(k2, "Loan Volume", f"{total_loan:,.0f}k", "Total Requested", "#00b8ff")
        card(k3, "Credit Pass", f"{credit_ok}", "History Verified", "#d946ef")
        card(k4, "Applicants", f"{len(df_filtered)}", "Total Processed", "#ffffff")

        # CHARTS (FORCE VISIBILITY)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🏠 Property Distribution")
            fig_pie = px.pie(df_filtered, names='Property_Area', hole=0.6, 
                             color_discrete_sequence=['#00ff9d', '#00b8ff', '#d946ef'])
            # Explicitly set background to transparent and font to white
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=14),
                showlegend=True,
                legend=dict(orientation="h", y=-0.1)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 💰 Income vs Loan (Scatter)")
            fig_scat = px.scatter(df_filtered, x='ApplicantIncome', y='LoanAmount', 
                                  color='Property_Area', size='LoanAmount',
                                  color_discrete_sequence=['#00ff9d', '#00b8ff', '#d946ef'])
            # Explicitly set background to transparent and font to white
            fig_scat.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='white'),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig_scat, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 2: APPLICANT ANALYSIS ---
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='gradient-text'>APPLICANT INTELLIGENCE</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🎓 Education vs Income Level")
            fig_bar = px.box(df_filtered, x='Education', y='ApplicantIncome', color='Education',
                             color_discrete_sequence=['#00b8ff', '#00ff9d'])
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='white'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 👫 Gender Split")
            fig_gen = px.bar(df_filtered['Gender'].value_counts(), orientation='h', color_discrete_sequence=['#d946ef'])
            fig_gen.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='white'), 
                showlegend=False,
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig_gen, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📂 Raw Data Vault")
        st.dataframe(df_filtered, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 3: LOAN PREDICTOR (FIXED) ---
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='gradient-text'>SMART PREDICTOR</div>", unsafe_allow_html=True)
        
        c_in, c_out = st.columns([1, 1.5])
        
        with c_in:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 👤 Applicant Profile")
            u_income = st.number_input("Monthly Income ($)", value=5000)
            u_coincome = st.number_input("Co-Applicant Income ($)", value=0)
            u_loan = st.number_input("Loan Amount (k)", value=120)
            u_cred = st.selectbox("Credit History", ["Clear (1.0)", "Debts (0.0)"])
            u_prop = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            check = st.button("🚀 CHECK ELIGIBILITY")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c_out:
            if check:
                # LOGIC
                cred_score = 1.0 if "1.0" in u_cred else 0.0
                total_income = u_income + u_coincome
                ratio = u_loan / (total_income/1000) if total_income > 0 else 100
                
                approved = False
                if cred_score == 1.0 and ratio < 50:
                    approved = True
                    prob = np.random.uniform(85, 98)
                elif cred_score == 1.0:
                    approved = False
                    prob = np.random.uniform(45, 65)
                else:
                    approved = False
                    prob = np.random.uniform(10, 30)
                
                status_color = "#00ff9d" if approved else "#ff4b4b"
                status_text = "APPROVED" if approved else "REJECTED"
                
                # FIXED: Store HTML in variable first
                result_html = f"""
                <div class="glass-card" style="text-align:center; border: 2px solid {status_color}; box-shadow: 0 0 50px {status_color}20;">
                    <h2 style="color:{status_color} !important; letter-spacing: 4px; margin-bottom: 10px;">LOAN {status_text}</h2>
                    <h1 style="font-size: 5rem; margin: 0; color: white;">{prob:.1f}%</h1>
                    <p style="color:#a1a1aa; text-transform: uppercase; letter-spacing: 2px;">Probability Score</p>
                    
                    <div style="margin-top: 30px; display:flex; justify-content:center; gap: 20px;">
                        <div style="background: rgba(255,255,255,0.05); padding: 10px 20px; border-radius: 10px;">
                            <div style="font-size:0.8rem; color:#a1a1aa;">CREDIT CHECK</div>
                            <div style="font-weight:bold; color:{status_color}; font-size: 1.1rem;">{'PASS' if cred_score==1 else 'FAIL'}</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.05); padding: 10px 20px; border-radius: 10px;">
                            <div style="font-size:0.8rem; color:#a1a1aa;">RISK RATIO</div>
                            <div style="font-weight:bold; color:{'#00ff9d' if ratio<50 else '#ff4b4b'}; font-size: 1.1rem;">{ratio:.1f}</div>
                        </div>
                    </div>
                </div>
                """
                
                st.markdown(result_html, unsafe_allow_html=True)
            else:
                st.info("👈 Enter applicant details to run the AI assessment.")

else:
    st.error("🚨 Please check the data source.")
