import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Options Analytics Platform",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1e2130; padding: 10px; border-radius: 5px;}
    h1 {color: #00ff00; text-align: center;}
    h2 {color: #00d9ff;}
    </style>
""", unsafe_allow_html=True)

# Black-Scholes Functions
def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price and Greeks"""
    if T <= 0:
        return max(S - K, 0), 0, 0, 0, 0
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    vega = S*norm.pdf(d1)*np.sqrt(T) / 100
    theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
    
    return price, delta, gamma, vega, theta

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price and Greeks"""
    if T <= 0:
        return max(K - S, 0), 0, 0, 0, 0
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    delta = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    vega = S*norm.pdf(d1)*np.sqrt(T) / 100
    theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365
    
    return price, delta, gamma, vega, theta

def implied_volatility(option_price, S, K, T, r, option_type='call'):
    """Calculate implied volatility using Newton-Raphson"""
    sigma = 0.3  # Initial guess
    
    for i in range(100):
        if option_type == 'call':
            price, _, _, vega, _ = black_scholes_call(S, K, T, r, sigma)
        else:
            price, _, _, vega, _ = black_scholes_put(S, K, T, r, sigma)
        
        diff = price - option_price
        if abs(diff) < 1e-5:
            return sigma
        
        if vega < 1e-10:
            return sigma
        
        sigma = sigma - diff / (vega * 100)
        sigma = max(0.01, min(sigma, 5.0))
    
    return sigma

# Monte Carlo Pricing
def monte_carlo_option(S, K, T, r, sigma, option_type='call', n_simulations=100000, barrier=None):
    """Monte Carlo simulation for option pricing"""
    dt = T / 252
    n_steps = int(T * 252)
    
    # Generate random paths
    Z = np.random.standard_normal((n_simulations, n_steps))
    S_paths = np.zeros((n_simulations, n_steps + 1))
    S_paths[:, 0] = S
    
    for t in range(1, n_steps + 1):
        S_paths[:, t] = S_paths[:, t-1] * np.exp(
            (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:, t-1]
        )
    
    # Calculate payoffs
    if barrier is None:
        # Vanilla option
        if option_type == 'call':
            payoffs = np.maximum(S_paths[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - S_paths[:, -1], 0)
    else:
        # Barrier option (up-and-out call)
        if option_type == 'call':
            barrier_hit = np.any(S_paths > barrier, axis=1)
            payoffs = np.where(barrier_hit, 0, np.maximum(S_paths[:, -1] - K, 0))
        else:
            # Down-and-out put
            barrier_hit = np.any(S_paths < barrier, axis=1)
            payoffs = np.where(barrier_hit, 0, np.maximum(K - S_paths[:, -1], 0))
    
    # Discount to present value
    option_price = np.exp(-r*T) * np.mean(payoffs)
    std_error = np.std(payoffs) / np.sqrt(n_simulations)
    
    return option_price, std_error, S_paths

# Asian Option Pricing
def asian_option_mc(S, K, T, r, sigma, option_type='call', n_simulations=50000):
    """Monte Carlo for Asian options (arithmetic average)"""
    dt = T / 252
    n_steps = int(T * 252)
    
    Z = np.random.standard_normal((n_simulations, n_steps))
    S_paths = np.zeros((n_simulations, n_steps + 1))
    S_paths[:, 0] = S
    
    for t in range(1, n_steps + 1):
        S_paths[:, t] = S_paths[:, t-1] * np.exp(
            (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:, t-1]
        )
    
    # Calculate arithmetic average
    avg_prices = np.mean(S_paths, axis=1)
    
    if option_type == 'call':
        payoffs = np.maximum(avg_prices - K, 0)
    else:
        payoffs = np.maximum(K - avg_prices, 0)
    
    option_price = np.exp(-r*T) * np.mean(payoffs)
    std_error = np.std(payoffs) / np.sqrt(n_simulations)
    
    return option_price, std_error

# Main App
st.title("Elite Options Analytics Platform")
st.markdown("### **Derivatives Pricing & Risk Management**")


# Sidebar
st.sidebar.header(" Configuration")

# Tab structure
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Black-Scholes Pricer",
    " Volatility Surface",
    " Exotic Options",
    " Risk Dashboard",
    " Portfolio Greeks"
])

# TAB 1: Black-Scholes Pricer
with tab1:
    st.header("Black-Scholes Option Pricing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        S = st.number_input("Spot Price ($)", value=100.0, min_value=1.0)
        K = st.number_input("Strike Price ($)", value=100.0, min_value=1.0)
    
    with col2:
        T = st.number_input("Time to Maturity (years)", value=1.0, min_value=0.01, max_value=10.0)
        r = st.number_input("Risk-Free Rate (%)", value=5.0, min_value=0.0) / 100
    
    with col3:
        sigma = st.number_input("Volatility (%)", value=20.0, min_value=1.0, max_value=200.0) / 100
        option_type = st.selectbox("Option Type", ["Call", "Put"])
    
    # Calculate
    if option_type == "Call":
        price, delta, gamma, vega, theta = black_scholes_call(S, K, T, r, sigma)
    else:
        price, delta, gamma, vega, theta = black_scholes_put(S, K, T, r, sigma)
    
    # Display results
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Option Price", f"${price:.2f}")
    col2.metric("Delta (Δ)", f"{delta:.4f}")
    col3.metric("Gamma (Γ)", f"{gamma:.4f}")
    col4.metric("Vega (ν)", f"{vega:.4f}")
    col5.metric("Theta (Θ)", f"{theta:.4f}")
    
    # Visualization
    st.markdown("### Greeks Sensitivity Analysis")
    
    spot_range = np.linspace(S*0.7, S*1.3, 50)
    
    prices = []
    deltas = []
    gammas = []
    
    for spot in spot_range:
        if option_type == "Call":
            p, d, g, _, _ = black_scholes_call(spot, K, T, r, sigma)
        else:
            p, d, g, _, _ = black_scholes_put(spot, K, T, r, sigma)
        prices.append(p)
        deltas.append(d)
        gammas.append(g)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_range, y=prices, name='Option Price', line=dict(color='#00ff00', width=3)))
    fig.add_trace(go.Scatter(x=spot_range, y=deltas, name='Delta', line=dict(color='#00d9ff', width=2)))
    fig.add_trace(go.Scatter(x=spot_range, y=gammas, name='Gamma', line=dict(color='#ff00ff', width=2)))
    
    fig.add_vline(x=S, line_dash="dash", line_color="red", annotation_text="Current Spot")
    
    fig.update_layout(
        title="Option Price & Greeks vs Spot Price",
        xaxis_title="Spot Price ($)",
        yaxis_title="Value",
        template="plotly_dark",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: Volatility Surface
with tab2:
    st.header("Volatility Surface Builder")
    
    ticker = st.text_input("Enter Ticker Symbol", value="AAPL")
    
    if st.button("Fetch Real Option Chain"):
        with st.spinner("Fetching data from Yahoo Finance..."):
            try:
                stock = yf.Ticker(ticker)
                stock_price = stock.history(period='1d')['Close'].iloc[-1]
                
                st.success(f"Current {ticker} Price: ${stock_price:.2f}")
                
                # Get option chain
                exp_dates = stock.options[:5]  # First 5 expiration dates
                
                iv_surface_data = []
                
                for exp_date in exp_dates:
                    opt_chain = stock.option_chain(exp_date)
                    calls = opt_chain.calls
                    
                    # Calculate time to expiry
                    expiry = datetime.strptime(exp_date, '%Y-%m-%d')
                    tte = (expiry - datetime.now()).days / 365
                    
                    for _, row in calls.iterrows():
                        strike = row['strike']
                        last_price = row['lastPrice']
                        
                        if last_price > 0.5 and abs(strike - stock_price) < stock_price * 0.2:
                            # Calculate implied volatility
                            iv = implied_volatility(last_price, stock_price, strike, tte, 0.05, 'call')
                            
                            moneyness = strike / stock_price
                            iv_surface_data.append({
                                'Strike': strike,
                                'Moneyness': moneyness,
                                'Expiry': exp_date,
                                'TTE': tte,
                                'IV': iv * 100
                            })
                
                if iv_surface_data:
                    df_iv = pd.DataFrame(iv_surface_data)
                    
                    # Create 3D surface plot
                    fig = go.Figure(data=[go.Scatter3d(
                        x=df_iv['Moneyness'],
                        y=df_iv['TTE'],
                        z=df_iv['IV'],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=df_iv['IV'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="IV (%)")
                        )
                    )])
                    
                    fig.update_layout(
                        title=f"{ticker} Implied Volatility Surface",
                        scene=dict(
                            xaxis_title='Moneyness (K/S)',
                            yaxis_title='Time to Expiry (years)',
                            zaxis_title='Implied Volatility (%)'
                        ),
                        template='plotly_dark',
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display data table
                    st.markdown("### Implied Volatility Data")
                    st.dataframe(df_iv.sort_values('TTE'), use_container_width=True)
                else:
                    st.warning("No valid option data found. Try a different ticker.")
                
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
    
    st.markdown("---")
    st.markdown("### Theoretical Vol Surface (SVI Model)")
    
    # SVI Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        svi_a = st.slider("a (level)", -0.1, 0.5, 0.04)
        svi_b = st.slider("b (slope)", 0.1, 0.5, 0.2)
    with col2:
        svi_rho = st.slider("ρ (correlation)", -0.9, 0.9, -0.4)
        svi_m = st.slider("m (ATM shift)", -0.5, 0.5, 0.0)
    with col3:
        svi_sigma = st.slider("σ (curvature)", 0.1, 1.0, 0.3)
    
    # Generate SVI surface
    k_range = np.linspace(-0.5, 0.5, 50)  # Log-moneyness
    t_range = np.linspace(0.1, 2.0, 30)
    
    K_grid, T_grid = np.meshgrid(k_range, t_range)
    
    def svi_variance(k, a, b, rho, m, sigma):
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    
    IV_grid = np.sqrt(svi_variance(K_grid, svi_a, svi_b, svi_rho, svi_m, svi_sigma) / T_grid) * 100
    
    fig = go.Figure(data=[go.Surface(
        x=K_grid,
        y=T_grid,
        z=IV_grid,
        colorscale='Viridis',
        colorbar=dict(title="IV (%)")
    )])
    
    fig.update_layout(
        title="SVI Volatility Surface",
        scene=dict(
            xaxis_title='Log-Moneyness',
            yaxis_title='Time to Expiry (years)',
            zaxis_title='Implied Volatility (%)'
        ),
        template='plotly_dark',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

# TAB 3: Exotic Options
with tab3:
    st.header("Exotic Options Pricing (Monte Carlo)")
    
    exotic_type = st.selectbox(
        "Select Exotic Option Type",
        ["Barrier Option (Up-and-Out Call)", "Barrier Option (Down-and-Out Put)", "Asian Option (Arithmetic Average)"]
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        S_exotic = st.number_input("Spot Price", value=100.0, key='exotic_S')
        K_exotic = st.number_input("Strike Price", value=100.0, key='exotic_K')
    
    with col2:
        T_exotic = st.number_input("Time to Maturity", value=1.0, key='exotic_T')
        r_exotic = st.number_input("Risk-Free Rate", value=0.05, key='exotic_r')
    
    with col3:
        sigma_exotic = st.number_input("Volatility", value=0.20, key='exotic_sigma')
        n_sims = st.selectbox("Simulations", [10000, 50000, 100000], index=1)
    
    if "Barrier" in exotic_type:
        barrier = st.number_input("Barrier Level", value=120.0 if "Up" in exotic_type else 80.0)
    
    if st.button("Price Exotic Option", key='price_exotic'):
        with st.spinner("Running Monte Carlo simulation..."):
            if "Asian" in exotic_type:
                price, std_err = asian_option_mc(
                    S_exotic, K_exotic, T_exotic, r_exotic, sigma_exotic, 'call', n_sims
                )
                st.success(f"**Asian Option Price: ${price:.2f} ± ${std_err:.2f}**")
            
            else:
                option_type_exotic = 'call' if 'Call' in exotic_type else 'put'
                price, std_err, paths = monte_carlo_option(
                    S_exotic, K_exotic, T_exotic, r_exotic, sigma_exotic,
                    option_type_exotic, n_sims, barrier
                )
                
                st.success(f"**Barrier Option Price: ${price:.2f} ± ${std_err:.2f}**")
                
                # Plot sample paths
                fig = go.Figure()
                
                sample_paths = paths[:100]  # Plot 100 sample paths
                time_steps = np.linspace(0, T_exotic, sample_paths.shape[1])
                
                for path in sample_paths:
                    fig.add_trace(go.Scatter(
                        x=time_steps, y=path,
                        mode='lines',
                        line=dict(width=0.5),
                        showlegend=False,
                        opacity=0.3
                    ))
                
                # Add barrier line
                if "Barrier" in exotic_type:
                    fig.add_hline(y=barrier, line_dash="dash", line_color="red",
                                 annotation_text=f"Barrier: ${barrier}")
                
                fig.add_hline(y=S_exotic, line_dash="dash", line_color="green",
                             annotation_text=f"Initial: ${S_exotic}")
                
                fig.update_layout(
                    title="Sample Monte Carlo Paths",
                    xaxis_title="Time (years)",
                    yaxis_title="Stock Price ($)",
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)

# TAB 4: Risk Dashboard
with tab4:
    st.header("Portfolio Risk Management")
    
    st.markdown("### Value at Risk (VaR) Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        portfolio_value = st.number_input("Portfolio Value ($)", value=1000000.0, min_value=1000.0)
        holding_period = st.number_input("Holding Period (days)", value=1, min_value=1, max_value=252)
    
    with col2:
        confidence_level = st.selectbox("Confidence Level", [90, 95, 99], index=1)
        returns_vol = st.number_input("Daily Volatility (%)", value=1.5, min_value=0.1) / 100
    
    # Calculate VaR
    if st.button("Calculate VaR"):
        # Historical simulation approach
        n_simulations = 10000
        
        # Generate random returns
        daily_returns = np.random.normal(0, returns_vol, (n_simulations, holding_period))
        cumulative_returns = np.sum(daily_returns, axis=1)
        
        # Calculate portfolio values
        final_values = portfolio_value * (1 + cumulative_returns)
        losses = portfolio_value - final_values
        
        # VaR at confidence level
        var_percentile = np.percentile(losses, confidence_level)
        
        # CVaR (Expected Shortfall)
        cvar = np.mean(losses[losses >= var_percentile])
        
        col1, col2, col3 = st.columns(3)
        col1.metric(f"VaR ({confidence_level}%)", f"${var_percentile:,.0f}")
        col2.metric(f"CVaR ({confidence_level}%)", f"${cvar:,.0f}")
        col3.metric("Max Loss", f"${np.max(losses):,.0f}")
        
        # Distribution plot
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=losses,
            nbinsx=50,
            name='Loss Distribution',
            marker_color='#00d9ff'
        ))
        
        fig.add_vline(x=var_percentile, line_dash="dash", line_color="red",
                     annotation_text=f"VaR: ${var_percentile:,.0f}")
        
        fig.update_layout(
            title="Portfolio Loss Distribution",
            xaxis_title="Loss ($)",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Stress Testing Scenarios")
    
    scenarios = {
        "Market Crash (-20%)": -0.20,
        "Volatility Spike (+50%)": 0.50,
        "Rate Shock (+2%)": 0.02,
        "Black Monday (-22.6%)": -0.226,
        "COVID Crash (-34%)": -0.34
    }
    
    scenario_results = []
    
    for scenario_name, shock in scenarios.items():
        if "Volatility" in scenario_name:
            shocked_value = portfolio_value  # Vol doesn't directly affect value
            pnl = 0
        else:
            shocked_value = portfolio_value * (1 + shock)
            pnl = shocked_value - portfolio_value
        
        scenario_results.append({
            'Scenario': scenario_name,
            'Shock': f"{shock*100:.1f}%",
            'Portfolio Value': f"${shocked_value:,.0f}",
            'P&L': f"${pnl:,.0f}"
        })
    
    df_scenarios = pd.DataFrame(scenario_results)
    st.dataframe(df_scenarios, use_container_width=True)

# TAB 5: Portfolio Greeks
with tab5:
    st.header("Portfolio Greeks Aggregation")
    
    st.markdown("### Add Positions to Portfolio")
    
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []
    
    with st.form("add_position"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pos_type = st.selectbox("Type", ["Call", "Put"])
            pos_strike = st.number_input("Strike", value=100.0)
        
        with col2:
            pos_spot = st.number_input("Spot", value=100.0)
            pos_quantity = st.number_input("Quantity", value=1, min_value=-1000, max_value=1000)
        
        with col3:
            pos_expiry = st.number_input("Expiry (years)", value=1.0, min_value=0.01)
            pos_vol = st.number_input("Vol (%)", value=20.0) / 100
        
        with col4:
            st.markdown("##")
            submitted = st.form_submit_button("Add Position")
    
    if submitted:
        st.session_state.portfolio.append({
            'type': pos_type,
            'strike': pos_strike,
            'spot': pos_spot,
            'quantity': pos_quantity,
            'expiry': pos_expiry,
            'vol': pos_vol
        })
        st.success("Position added!")
    
    if st.session_state.portfolio:
        st.markdown("### Current Portfolio")
        
        # Calculate portfolio Greeks
        total_delta = 0
        total_gamma = 0
        total_vega = 0
        total_theta = 0
        total_value = 0
        
        position_data = []
        
        for i, pos in enumerate(st.session_state.portfolio):
            if pos['type'] == 'Call':
                price, delta, gamma, vega, theta = black_scholes_call(
                    pos['spot'], pos['strike'], pos['expiry'], 0.05, pos['vol']
                )
            else:
                price, delta, gamma, vega, theta = black_scholes_put(
                    pos['spot'], pos['strike'], pos['expiry'], 0.05, pos['vol']
                )
            
            pos_value = price * pos['quantity']
            pos_delta = delta * pos['quantity']
            pos_gamma = gamma * pos['quantity']
            pos_vega = vega * pos['quantity']
            pos_theta = theta * pos['quantity']
            
            total_value += pos_value
            total_delta += pos_delta
            total_gamma += pos_gamma
            total_vega += pos_vega
            total_theta += pos_theta
            
            position_data.append({
                'Position': i+1,
                'Type': pos['type'],
                'Strike': f"${pos['strike']:.2f}",
                'Quantity': pos['quantity'],
                'Value': f"${pos_value:.2f}",
                'Delta': f"{pos_delta:.2f}",
                'Gamma': f"{pos_gamma:.4f}",
                'Vega': f"{pos_vega:.2f}",
                'Theta': f"{pos_theta:.2f}"
            })
        
        df_portfolio = pd.DataFrame(position_data)
        st.dataframe(df_portfolio, use_container_width=True)
        
        st.markdown("### Portfolio-Level Greeks")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("Total Value", f"${total_value:.2f}")
        col2.metric("Net Delta", f"{total_delta:.2f}")
        col3.metric("Net Gamma", f"{total_gamma:.4f}")
        col4.metric("Net Vega", f"{total_vega:.2f}")
        col5.metric("Net Theta", f"{total_theta:.2f}")
        
        # Greeks visualization
        greeks_dict = {
            'Delta': total_delta,
            'Gamma': total_gamma,
            'Vega': total_vega,
            'Theta': total_theta
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(greeks_dict.keys()),
                y=list(greeks_dict.values()),
                marker_color=['#00ff00', '#00d9ff', '#ff00ff', '#ffaa00']
            )
        ])
        
        fig.update_layout(
            title="Portfolio Greeks Breakdown",
            yaxis_title="Value",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Clear Portfolio"):
            st.session_state.portfolio = []
            st.rerun()

# Footer
st.markdown("---")

