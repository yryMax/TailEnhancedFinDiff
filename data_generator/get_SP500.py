from datetime import datetime
import os
import yfinance as yf
import pandas as pd
import time

if __name__ == '__main__':

    """ set the download window """
    start_time = "2006-01-01"
    end_time = "2022-12-31"

    # Current S&P 500 constituents (with Yahoo Finance compatible symbols)
    s_and_p = ['MMM', 'ABT', 'ABBV', 'ABMD', 'ACN', 'ATVI', 'ADBE', 'AMD', 'AAP', 'AES', 'AFL', 'A', 'APD', 'AKAM',
               'ALK', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AAL',
               'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'ANTM', 'AON',
               'AOS', 'APA', 'AAPL', 'AMAT', 'APTV', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO',
               'AVB', 'AVY', 'BKR', 'BLL', 'BAC', 'BBWI', 'BAX', 'BDX', 'BBY', 'BIO', 'TECH', 'BIIB', 'BLK', 'BK',
               'BA', 'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'CHRW', 'CDNS', 'CZR', 'CPB', 'COF',
               'CAH', 'KMX', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'CNC', 'CNP', 'CDAY', 'CERN',
               'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG',
               'CTXS', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'COP', 'ED', 'STZ', 'COO',
               'CPRT', 'GLW', 'CTVA', 'COST', 'CTRA', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE',
               'DAL', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DISCA', 'DISCK', 'DISH', 'DG', 'DLTR', 'D',
               'DPZ', 'DOV', 'DOW', 'DTE', 'DUK', 'DRE', 'DD', 'DXC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA',
               'EMR', 'ENPH', 'ETR', 'EOG', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'EVRG', 'ES', 'RE', 'EXC',
               'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FB', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FE', 'FRC', 'FISV',
               'FLT', 'FMC', 'F', 'FTNT', 'FTV', 'FBHS', 'FOXA', 'FOX', 'BEN', 'FCX', 'GPS', 'GRMN', 'IT', 'GNRC',
               'GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GL', 'GPN', 'GS', 'GWW', 'HAL', 'HBI', 'HIG', 'HAS', 'HCA',
               'PEAK', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUM',
               'HBAN', 'HII', 'IEX', 'IDXX', 'INFO', 'ITW', 'ILMN', 'INCY', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG',
               'IFF', 'INTU', 'ISRG', 'IVZ', 'IPGP', 'IQV', 'IRM', 'JKHY', 'J', 'JBHT', 'SJM', 'JNJ', 'JCI', 'JPM',
               'JNPR', 'KSU', 'K', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW',
               'LVS', 'LEG', 'LDOS', 'LEN', 'LLY', 'LNC', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LUMN', 'LYB', 'MTB',
               'MRO', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK',
               'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO',
               'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NWL', 'NEM', 'NWSA', 'NWS', 'NEE', 'NLSN', 'NKE',
               'NI', 'NSC', 'NTRS', 'NOC', 'NLOK', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL',
               'OMC', 'OKE', 'ORCL', 'OGN', 'OTIS', 'PCAR', 'PKG', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PENN', 'PNR', 'PBCT',
               'PEP', 'PKI', 'PFE', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD',
               'PRU', 'PTC', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG',
               'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB',
               'STX', 'SEE', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SNA', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STE',
               'SYK', 'SIVB', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TGT', 'TEL', 'TDY', 'TFX', 'TER',
               'TSLA', 'TXN', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TWTR', 'TYL', 'TSN',
               'UDR', 'ULTA', 'USB', 'UAA', 'UA', 'UNP', 'UAL', 'UNH', 'UPS', 'URI', 'UHS', 'VLO', 'VTR', 'VRSN',
               'VRSK', 'VZ', 'VRTX', 'VFC', 'VIAC', 'VTRS', 'V', 'VNO', 'VMC', 'WRB', 'WAB', 'WMT', 'WBA', 'DIS',
               'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WU', 'WRK', 'WY', 'WHR', 'WMB', 'WLTW', 'WYNN',
               'XEL', 'XLNX', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS']

    # Ticker symbol mappings (Yahoo Finance format or renamed tickers)
    # Format: 'original_symbol': ['yahoo_symbol', 'alternative_symbol', ...]
    ticker_mappings = {
        'BRK.B': ['BRK-B'],
        'BF.B': ['BF-B'],
        'FB': ['META'],  # Facebook renamed to Meta in 2022
        'ANTM': ['ELV'],  # Anthem renamed to Elevance Health in 2022
    }

    # Historical S&P 500 constituents (removed between 2006-2022)
    # These companies were in S&P 500 at some point during 2006-2022
    historical_constituents = [
        # Major historical constituents
        'APC',   # Anadarko Petroleum (acquired by Occidental 2019)
        'ANDV',  # Andeavor (acquired by Marathon 2018)
        'AGN',   # Allergan (acquired by AbbVie 2020)
        'ARNC',  # Arconic
        'AET',   # Aetna (acquired by CVS 2018)
        'ALTR',  # Altera (acquired by Intel 2015)
        'ANR',   # Alpha Natural Resources
        'ABI',   # Anheuser-Busch (acquired 2008)
        'BHI',   # Baker Hughes (merged with GE Oil 2017)
        'BBBY',  # Bed Bath & Beyond
        'BIG',   # Big Lots
        'CA',    # CA Technologies (acquired by Broadcom 2018)
        'CBG',   # CBRE Group (symbol change)
        'CBS',   # CBS (merged with Viacom 2019)
        'CELG',  # Celgene (acquired by Bristol-Myers 2019)
        'CTL',   # CenturyLink (renamed to Lumen)
        'CHK',   # Chesapeake Energy
        'CPWR',  # Compuware
        'COV',   # Covidien
        'CSC',   # Computer Sciences Corp (merged to DXC)
        'CVH',   # Coventry Health Care
        'DPS',   # Dr Pepper Snapple
        'DNB',   # Dun & Bradstreet
        'DNR',   # Denbury Resources
        'DO',    # Diamond Offshore
        'DWDP',  # DowDuPont (split into DOW, DD, CTVA)
        'ENDP',  # Endo International
        'ESRX',  # Express Scripts (acquired by Cigna 2018)
        'ETFC',  # E*Trade (acquired by Morgan Stanley 2020)
        'FDO',   # Family Dollar (acquired by Dollar Tree 2015)
        'FLIR',  # FLIR Systems (acquired by Teledyne 2021)
        'GGP',   # General Growth Properties
        'GT',    # Goodyear Tire
        'HAR',   # Harman International (acquired by Samsung 2017)
        'HCN',   # Welltower (formerly Health Care REIT)
        'HNZ',   # Heinz (merged with Kraft 2015)
        'HOT',   # Starwood Hotels (acquired by Marriott 2016)
        'HSH',   # Hillshire Brands
        'JBL',   # Jabil
        'JEC',   # Jacobs Engineering
        'JOY',   # Joy Global (acquired by Komatsu 2017)
        'KRFT',  # Kraft Foods (merged with Heinz 2015)
        'LLL',   # L3 Technologies (merged with Harris)
        'LM',    # Legg Mason (acquired by Franklin Templeton 2020)
        'LO',    # Lorillard
        'LLTC',  # Linear Technology (acquired by Analog Devices 2017)
        'MJN',   # Mead Johnson (acquired by Reckitt Benckiser 2017)
        'MNST',  # Monster Beverage (already in list)
        'MON',   # Monsanto (acquired by Bayer 2018)
        'MYL',   # Mylan (merged to Viatris 2020)
        'NBL',   # Noble Energy (acquired by Chevron 2020)
        'NAVI',  # Navient
        'NFX',   # Newfield Exploration
        'NSM',   # Nationstar Mortgage
        'NWS',   # News Corp (already in list)
        'PBCT',  # People's United Financial
        'PCG',   # PG&E (already may be in list)
        'PCLN',  # Priceline (renamed to Booking Holdings BKNG)
        'PDCO',  # Patterson Companies
        'PLL',   # Pall Corp (acquired by Danaher 2015)
        'POM',   # Pepco Holdings
        'PX',    # Praxair (merged with Linde 2018)
        'RAI',   # Reynolds American (acquired by BAT 2017)
        'RHT',   # Red Hat (acquired by IBM 2019)
        'RIG',   # Transocean
        'SCG',   # SCANA (acquired by Dominion 2019)
        'SE',    # Spectra Energy (merged with Enbridge 2017)
        'SHLD',  # Sears Holdings
        'SIG',   # Signet Jewelers
        'SNDK',  # SanDisk (acquired by Western Digital 2016)
        'SNI',   # Scripps Networks (acquired by Discovery 2018)
        'SPLS',  # Staples (went private 2017)
        'STI',   # SunTrust Banks (merged with BB&T to Truist TFC)
        'STJ',   # St. Jude Medical (acquired by Abbott 2017)
        'SWN',   # Southwestern Energy
        'SWY',   # Safeway (acquired by Albertsons 2015)
        'SYMC',  # Symantec (split, enterprise sold to Broadcom)
        'TEG',   # Integrys Energy
        'TIE',   # Titanium Metals
        'TMK',   # Torchmark (renamed to Globe Life GL)
        'TSO',   # Tesoro (renamed to Andeavor ANDV)
        'TSS',   # Total System Services (merged with Global Payments)
        'TWC',   # Time Warner Cable (acquired by Charter 2016)
        'TWX',   # Time Warner (acquired by AT&T 2018)
        'TYC',   # Tyco International
        'UTX',   # United Technologies (split/merged 2020)
        'VAR',   # Varian Medical (acquired by Siemens 2021)
        'VIAB',  # Viacom (merged with CBS 2019)
        'VNO',   # Vornado (already in list)
        'WFM',   # Whole Foods (acquired by Amazon 2017)
        'WFT',   # Weatherford International
        'WIN',   # Windstream
        'WLP',   # WellPoint (renamed to Anthem ANTM)
        'WMI',   # Waste Management Inc
        'WYN',   # Wyndham (split into WH and WYND)
        'WYND',  # Wyndham Destinations
        'XL',    # XL Group (acquired by AXA 2018)
        'YHOO',  # Yahoo (acquired by Verizon 2017)
        'ZAYO',  # Zayo Group
        # More 2006-era constituents
        'ABS',   # Albertsons
        'ASH',   # Ashland
        'AT',    # Alltel
        'AVP',   # Avon Products
        'BJS',   # BJ Services
        'BNI',   # Burlington Northern (acquired by Berkshire 2010)
        'BOL',   # Bausch & Lomb
        'CC',    # Circuit City
        'CFC',   # Countrywide Financial
        'CTX',   # Centex
        'DYN',   # Dynegy
        'EDS',   # Electronic Data Systems
        'EK',    # Eastman Kodak
        'EP',    # El Paso
        'FNM',   # Fannie Mae
        'FRE',   # Freddie Mac
        'GCI',   # Gannett
        'GM',    # General Motors (old, pre-bankruptcy)
        'GR',    # Goodrich (acquired by UTX 2012)
        'HNZ',   # Heinz
        'LEH',   # Lehman Brothers
        'MBI',   # MBIA
        'MER',   # Merrill Lynch (acquired by BofA 2009)
        'MEE',   # Massey Energy
        'MMI',   # Motorola Mobility
        'MWV',   # MeadWestvaco
        'NBR',   # Nabors Industries
        'NYT',   # New York Times
        'PBG',   # Pepsi Bottling
        'PCS',   # MetroPCS
        'PTV',   # Pactiv
        'Q',     # Qwest
        'RDC',   # Rowan Companies
        'RX',    # IMS Health
        'S',     # Sprint
        'SAI',   # SAIC
        'SGP',   # Schering-Plough
        'SLM',   # SLM Corp (Sallie Mae)
        'SOV',   # Sovereign Bancorp
        'SUN',   # Sunoco
        'TRB',   # Tribune
        'UST',   # UST Inc
        'WB',    # Wachovia
        'WMI',   # Washington Mutual
        'XTO',   # XTO Energy (acquired by Exxon 2010)
    ]

    # Combine all tickers
    all_tickers = list(set(s_and_p + historical_constituents))

    output_dir = 'SP500'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading {len(all_tickers)} stocks from {start_time} to {end_time}...")

    # Download all data at once using yfinance
    data = yf.download(all_tickers, start=start_time, end=end_time, group_by='ticker', threads=True)

    # Save individual stock CSVs and collect for combined file
    all_dfs = []
    successful = 0
    failed = []
    failed_original = []

    for ticker in all_tickers:
        try:
            if ticker in data.columns.get_level_values(0):
                stock_df = data[ticker].copy()
                stock_df = stock_df.dropna(how='all')
                if len(stock_df) > 0:
                    stock_df['Name'] = ticker
                    stock_df.to_csv(os.path.join(output_dir, f'{ticker}_data.csv'))
                    all_dfs.append(stock_df)
                    successful += 1
                else:
                    failed_original.append(ticker)
            else:
                failed_original.append(ticker)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            failed_original.append(ticker)

    # Try alternative symbols for failed tickers
    print(f"\nTrying alternative symbols for {len(failed_original)} failed tickers...")

    # Add mapping alternatives
    retry_tickers = {}
    for ticker in failed_original:
        if ticker in ticker_mappings:
            retry_tickers[ticker] = ticker_mappings[ticker]
        elif '.' in ticker:
            # Try replacing . with -
            retry_tickers[ticker] = [ticker.replace('.', '-')]

    for original, alternatives in retry_tickers.items():
        for alt_ticker in alternatives:
            try:
                print(f"Trying {alt_ticker} for {original}...")
                stock_df = yf.download(alt_ticker, start=start_time, end=end_time, progress=False)
                if len(stock_df) > 0:
                    stock_df['Name'] = original  # Keep original name
                    stock_df.to_csv(os.path.join(output_dir, f'{original}_data.csv'))
                    all_dfs.append(stock_df)
                    successful += 1
                    if original in failed_original:
                        failed_original.remove(original)
                    print(f"  Success: {alt_ticker} -> {original}")
                    break
            except Exception as e:
                print(f"  Failed: {alt_ticker} - {e}")
                continue

    failed = failed_original

    # Combine all data into one CSV
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=False)
        combined.to_csv(os.path.join(output_dir, 'SP500_combined.csv'))
        print(f"\nSuccessfully downloaded {successful} stocks")
        print(f"Combined data saved to {os.path.join(output_dir, 'SP500_combined.csv')}")

    if failed:
        print(f"\nFailed to download {len(failed)} stocks:")
        print(failed)
        with open(os.path.join(output_dir, 'failed_queries.txt'), 'w') as f:
            for name in failed:
                f.write(name + '\n')

    print("\nDone!")
