# Version 1.0 - Basic Dash dashboard for stock predictions (AAPL, MSFT, NVDA)
# Skipped Docker and proxies for simplicity, relying on local PC testing
# Note: Scraper fails to fetch GICS—using hardcoded SUB_INDUSTRY_MAP for now

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from scraper import load_cached_data

# Try to load scraped GICS data, fall back to hardcoded if none
gics_df = load_cached_data('gics_data')
if gics_df.empty or gics_df is None:
    SUB_INDUSTRY_MAP = {
        'AAPL': {'sector': 'Technology', 'sub_industry': 'Consumer Electronics'},
        'MSFT': {'sector': 'Technology', 'sub_industry': 'Software'},
        'NVDA': {'sector': 'Technology', 'sub_industry': 'Semiconductors'},
    }
else:
    SUB_INDUSTRY_MAP = {stock: {'sector': row['Sector'], 'sub_industry': row['Industry']} 
                       for stock, row in gics_df.iterrows()}  # Adjust logic as needed

# Sub-industry mapping (simplified for core functionality)
SUB_INDUSTRY_MAP = {
    'AAPL': {'sector': 'Technology', 'sub_industry': 'Consumer Electronics'},
    'MSFT': {'sector': 'Technology', 'sub_industry': 'Software'},
    'NVDA': {'sector': 'Technology', 'sub_industry': 'Semiconductors'},
}

def fetch_stock_data(ticker, days=365):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, interval='1d')
    return df['Close'].dropna()

def fetch_sub_industry_data(ticker, days=365):
    sub_industry = SUB_INDUSTRY_MAP.get(ticker, {}).get('sub_industry')
    if not sub_industry:
        return None
    peers = [t for t, info in SUB_INDUSTRY_MAP.items() if info['sub_industry'] == sub_industry and t != ticker]
    if not peers:
        return None
    peer_data = {peer: fetch_stock_data(peer, days) for peer in peers}
    return pd.DataFrame(peer_data).mean(axis=1)

def prepare_data(data, look_back=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.8)
    return X[:train_size], y[:train_size], X[train_size:], y[train_size:], scaler

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def predict_trend(ticker='AAPL', days_history=365, prediction_days=30):
    closing_prices = fetch_stock_data(ticker, days_history)
    sub_industry_avg = fetch_sub_industry_data(ticker, days_history)

    X_train, y_train, X_test, y_test, scaler = prepare_data(closing_prices, 60)
    model = build_lstm_model((60, 1))
    model.fit(X_train, y_train, epochs=5, verbose=0)  # Reduced epochs for speed

    last_sequence = scaler.transform(closing_prices[-60:].values.reshape(-1, 1))
    last_sequence = np.reshape(last_sequence, (1, 60, 1))
    future_preds = []
    current_sequence = last_sequence.copy()

    for _ in range(prediction_days):
        next_pred = model.predict(current_sequence, verbose=0)
        future_preds.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred[0, 0]

    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    actual_dates = closing_prices.index
    future_dates = pd.date_range(start=actual_dates[-1], periods=prediction_days + 1)[1:]

    return actual_dates, closing_prices, future_dates, future_preds, sub_industry_avg

# Dash app for core dashboard
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Inspo Trading Dashboard (Core)"),
    dcc.Dropdown(
        id='ticker-dropdown',
        options=[{'label': t, 'value': t} for t in SUB_INDUSTRY_MAP.keys()],
        value='AAPL'
    ),
    dcc.Graph(id='price-chart'),
    dcc.Interval(
        id='interval-component',
        interval=1*60*1000,  # Update every minute
        n_intervals=0
    )
])

@app.callback(
    Output('price-chart', 'figure'),
    [Input('ticker-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_graph(ticker, n):
    actual_dates, actual_prices, future_dates, predictions, sub_industry_avg = predict_trend(ticker)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_dates, y=actual_prices, mode='lines', name='Actual Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_dates, y=predictions.flatten(), mode='lines', name='Predicted Price', line=dict(color='red', dash='dash')))
    fig.update_layout(
        title=f'{ticker} Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        legend=dict(x=0, y=1)
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)