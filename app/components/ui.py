import streamlit as st

def apply_fintech_style():
    """Injects custom FinTech styling into the Streamlit app."""
    st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container enhancements */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    /* Metric Cards */
    .fintech-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .fintech-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.4);
    }
    .fintech-card-label {
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.35rem;
    }
    .fintech-card-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    .fintech-card-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.4rem;
    }

    /* Badges */
    .badge-eligible {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-highrisk {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-noteligible {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    /* Disclaimer Box */
    .fintech-disclaimer {
        background-color: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #64748b;
        padding: 0.85rem 1.1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.8rem;
        color: #94a3b8;
        line-height: 1.45;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def render_header(title: str, subtitle: str, badge: str = None):
    """Renders a sleek FinTech header section."""
    badge_html = f'<span style="background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; vertical-align: middle; margin-left: 8px;">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div style="margin-bottom: 1.8rem;">
        <h1 style="margin: 0; font-size: 2.1rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">
            {title} {badge_html}
        </h1>
        <p style="margin: 0.35rem 0 0 0; font-size: 1rem; color: #94a3b8;">
            {subtitle}
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_metric_card(label: str, value: str, subtext: str = "", border_color: str = "#3b82f6"):
    """Renders an aesthetic KPI card."""
    st.markdown(f"""
    <div class="fintech-card" style="border-left: 4px solid {border_color};">
        <div class="fintech-card-label">{label}</div>
        <div class="fintech-card-value">{value}</div>
        {f'<div class="fintech-card-sub">{subtext}</div>' if subtext else ''}
    </div>
    """, unsafe_allow_html=True)

def render_disclaimer():
    """Renders regulatory decision-support disclaimer banner."""
    st.markdown("""
    <div class="fintech-disclaimer">
        <strong>Regulatory & Advisory Disclaimer:</strong> EMIPredict AI is an automated decision-support research prototype built for financial risk simulation. Model outputs represent algorithmic affordability estimates and do not constitute guaranteed underwriting approval, legal commitment, or regulated credit advice.
    </div>
    """, unsafe_allow_html=True)
