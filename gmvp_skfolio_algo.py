from zipline.api import (
    schedule_function,
    date_rules,
    time_rules,
    order_target_percent,
    symbol,
    record,
)
import pandas as pd
import numpy as np
from skfolio.optimization import MeanRisk
from skfolio.preprocessing import prices_to_returns

def initialize(context):
    """Initialize the algorithm"""
    # Define the universe of stocks
    stocks = ['AAPL', 'AMD', 'BAC', 'BBY', 'CVX', 'GE', 'HD', 'JNJ', 'JPM', 'KO',
       'LLY', 'MRK', 'MSFT', 'PEP', 'PFE', 'PG', 'RRC', 'UNH', 'WMT', 'XOM']
    context.stocks = [ symbol(t) for t in stocks ]
    context.lookback_window = 252  # 1-year lookback for optimization
        
    # Schedule rebalancing function
    schedule_function(
        rebalance,
        date_rules.month_start(),
        time_rules.market_open(hours=1)
    )
    
    # Initialize tracking variables
    context.rebalance_count = 0
    context.weights_history = []  # Add this line to track weights over time

def rebalance(context, data):
    """Rebalance portfolio using Global Minimum Variance Portfolio"""
    
    # Check if we can trade all stocks
    tradable_stocks = [stock for stock in context.stocks if data.can_trade(stock)]
    
    # Get historical prices
    prices = data.history(
        tradable_stocks, 
        'price', 
        context.lookback_window + 1, 
        '1d'
    )

    # Drop columns (stocks) with any NaN values
    prices = prices.dropna(axis=1)
    
    # Convert prices to returns
    returns = prices_to_returns(prices)
    
    # Remove any remaining NaN values
    returns = returns.dropna()
    
    # Create Global Minimum Variance Portfolio optimizer
    gmvp = MeanRisk()
    
    # Fit the model
    gmvp.fit(returns)
    
    # Get the optimal portfolio
    portfolio = gmvp.predict(returns)
    
    # Get weights as a pandas Series with asset names
    weights = portfolio.weights
    
    # Store weights history
    weight_dict = {'date': data.current_dt}
    for i, (asset_name, weight) in enumerate(zip(prices.columns, weights)):
        weight_dict[str(asset_name)] = weight
    context.weights_history.append(weight_dict)
            
    # Execute trades
    for _, (asset_symbol, weight) in enumerate(zip(prices.columns, weights)):
        # Only trade if weight is significant (> 0.5%)
        if abs(weight) > 0.005:
            order_target_percent(asset_symbol, weight)
            print(f"  {asset_symbol}: {weight:.4f}")
        else:
            # Close small positions
            order_target_percent(asset_symbol, 0)
    
    # Record metrics
    record(
        num_positions=(weights > 0.01).sum(),
        max_weight=weights.max(),
        portfolio_risk=portfolio.annualized_standard_deviation
    )
    
    context.rebalance_count += 1
    

def handle_data(context, data):
    """Called on every bar of data"""
    pass

def analyze(context, perf):
    """Analyze results at the end of backtest"""
    
    # Calculate metrics
    returns = perf['returns']
    
    total_return = (perf['portfolio_value'].iloc[-1] / perf['portfolio_value'].iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    # Max drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()
    
    # Save comprehensive results
    perf.to_csv('gmvp_backtest_results.csv')
    print("\nResults saved to 'gmvp_backtest_results.csv'")
    
    # Save weights history
    if context.weights_history:
        weights_df = pd.DataFrame(context.weights_history)
        weights_df.set_index('date', inplace=True)
        weights_df.to_csv('gmvp_weights_history.csv')
        print("Weights history saved to 'gmvp_weights_history.csv'")
    
    # Save daily returns and portfolio value for skfolio
    portfolio_data = pd.DataFrame({
        'date': perf.index,
        'returns': perf['returns'].values,
        'portfolio_value': perf['portfolio_value'].values,
        'pnl': perf['pnl'].values if 'pnl' in perf.columns else np.nan
    })
    portfolio_data.set_index('date', inplace=True)
    portfolio_data.to_csv('gmvp_portfolio_data.csv')
    print("Portfolio data saved to 'gmvp_portfolio_data.csv'")
    
    return perf