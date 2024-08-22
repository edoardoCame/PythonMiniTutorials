import pandas as pd
from binance.client import Client

# Imposta la tua chiave API e la chiave segreta di Binance
API_KEY = 'uI21u4WKans41ibDCrAqAPYd8XdfGjpark8S8bB5gRvDkPKwb8fEg5GYJW×MWXR5'
API_SECRET = 'IcFXQjdgQs8qcHQ0umXc5PLLwfOACimwXBFHotnhrbzc89GFcvyvAlbcvCP1JJiU'

# Crea un client Binance
client = Client(API_KEY, API_SECRET)

# Simbolo del mercato per Bitcoin
symbol = 'ETHUSDT'

# Intervallo delle candele (1 ora)
interval = Client.KLINE_INTERVAL_1HOUR

# Ottieni i dati delle candele dal 1 gennaio 2017 fino ad oggi
candle_data = client.get_historical_klines(symbol, interval, "1 Jan, 2017", "now")

# Crea un DataFrame utilizzando Pandas
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
df = pd.DataFrame(candle_data, columns=columns)

# Converti i tipi di dato appropriati
numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

# Converti i timestamp in formato leggibile
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Salva i dati in un file Excel con il formato appropriato
output_file = 'prezzi_binance.xlsx'
df.to_excel(output_file, index=False, engine='openpyxl')

print(f"I dati delle candele orarie sono stati salvati in '{output_file}'")
