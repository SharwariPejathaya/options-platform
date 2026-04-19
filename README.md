# Options Analytics Platform

An interactive **Streamlit-based financial analytics platform** for pricing derivatives, analyzing volatility, and managing portfolio risk. This application integrates **quantitative finance models, Monte Carlo simulations, and real market data** into a unified dashboard.
Live demo : https://options-platform-dytwlkxhatcnkaqlq9c7h7.streamlit.app/
---

##  Features

### 1. Black-Scholes Pricer
- Prices **European Call & Put options**
- Computes key Greeks:
  - Delta (Δ)
  - Gamma (Γ)
  - Vega (ν)
  - Theta (Θ)
- Interactive sensitivity plots vs spot price

---

### 2. Volatility Surface
- Fetches **real option chain data** using Yahoo Finance
- Computes **Implied Volatility (IV)** via Newton-Raphson
- Visualizes:
  - 3D IV surface (Moneyness vs Time vs IV)
- Includes **SVI (Stochastic Volatility Inspired)** theoretical model

---

### 3. Exotic Options Pricing
Monte Carlo simulation for:
- Barrier Options:
  - Up-and-Out Call
  - Down-and-Out Put
- Asian Options (Arithmetic Average)

Outputs:
- Option price with standard error
- Simulated price paths visualization

---

### 4. Risk Dashboard
- **Value at Risk (VaR)** using simulation
- **Conditional VaR (CVaR / Expected Shortfall)**
- Loss distribution visualization
- Predefined **stress testing scenarios**:
  - Market crash
  - Volatility spike
  - Rate shock
  - Historical crises (e.g., COVID crash)

---

### 5. Portfolio Greeks Aggregation
- Add multiple option positions
- Computes portfolio-level:
  - Value
  - Delta, Gamma, Vega, Theta
- Real-time aggregation + visualization

---

##  Core Models Implemented

- **Black-Scholes Model** (Analytical pricing + Greeks)
- **Implied Volatility Solver** (Newton-Raphson method)
- **Monte Carlo Simulation** (GBM-based path simulation)
- **SVI Model** (Volatility surface modeling)

---

##  Tech Stack

- **Frontend/UI**: Streamlit  
- **Numerical Computing**: NumPy, SciPy  
- **Data Handling**: Pandas  
- **Visualization**: Plotly  
- **Market Data**: yFinance  

---

##  Installation

```bash
git clone <your-repo-link>
cd options-analytics-platform
pip install -r requirements.txt
