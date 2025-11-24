import pandas as pd
from zipline.data.bundles import register
from zipline.data.bundles.csvdir import csvdir_equities
from skfolio.datasets import load_sp500_dataset
import os

def skfolio_bundle(environ, asset_db_writer, minute_bar_writer, daily_bar_writer,
                   adjustment_writer, calendar, start_session, end_session, cache, show_progress, output_dir):
    prices = load_sp500_dataset()

    # Ensure the prices' index is timezone-aware UTC
    if prices.index.tz is None:
        prices.index = prices.index.tz_localize('UTC')
    else:
        prices.index = prices.index.tz_convert('UTC')
    
    # zipline expects a specific format
    zipline_prices = prices.stack().reset_index()
    zipline_prices.columns = ['date', 'symbol', 'close']
    zipline_prices['open'] = zipline_prices['high'] = zipline_prices['low'] = zipline_prices['close']
    zipline_prices['volume'] = 100000  # dummy volume
    
    # Save to temporary CSVs for ingestion
    temp_dir = os.path.join(output_dir, 'csvdir')
    daily_dir = os.path.join(temp_dir, 'daily')
    os.makedirs(daily_dir, exist_ok=True)
    
    # Correctly handle the grouping and saving of files
    for symbol in prices.columns:
        df = zipline_prices[zipline_prices['symbol'] == symbol]
        df.to_csv(os.path.join(daily_dir, f'{symbol}.csv'), index=False, header=True)

    # Call csvdir_equities with the correct directory structure
    csvdir_equities(['daily'], temp_dir)(
        environ,
        asset_db_writer,
        minute_bar_writer,
        daily_bar_writer,
        adjustment_writer,
        calendar,
        start_session,
        end_session,
        cache,
        show_progress,
        output_dir
    )

register(
    'skfolio-s-p-500',
    skfolio_bundle,
    calendar_name='NYSE',
)