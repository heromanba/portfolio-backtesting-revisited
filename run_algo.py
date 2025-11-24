import pandas as pd
from zipline import run_algorithm
from gmvp_skfolio_algo import initialize, handle_data, analyze
from datetime import datetime

if __name__ == '__main__':
    # Set backtest parameters - create datetime objects with pytz timezone
    start = pd.Timestamp(datetime(2013, 12, 31))
    end = pd.Timestamp(datetime(2022, 12, 28))
    
    # Run the backtest
    results = run_algorithm(
        start=start,
        end=end,
        initialize=initialize,
        handle_data=handle_data,
        analyze=analyze,
        capital_base=100000,
        data_frequency='daily',
        bundle='skfolio-s-p-500'  # Make sure you have data ingested
    )
    
    print("\nBacktest finished successfully!")