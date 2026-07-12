# Chapter 5: Machine Learning Strategies

Machine learning replaces hand-tuned trading rules with models that learn patterns from historical data. This chapter builds three families of ML strategies — supervised learners (random forest, XGBoost), deep sequence models (LSTM), and reinforcement learning (DQN) — with particular attention to the failure modes that quietly invalidate most ML backtests: misaligned features and labels, look-ahead bias in labeling, and in-sample evaluation. The 5.1.x blocks include runnable `__main__` demos on synthetic data — accuracy near 50% on a random walk is the expected (and honest) result. The 5.2 (LSTM) and 5.3 (DQN) blocks are library-style class implementations that require the TensorFlow and PyTorch frameworks respectively and ship without demo runs.

## 5.1 Supervised Learning Strategies

Supervised strategies reduce trading to a prediction problem: build features from information available at time *t*, label each bar with something about the future (direction of the next N-day return), fit a model, and trade the predictions. The modeling is the easy part — most of the engineering effort goes into keeping the future out of the features and keeping features and labels correctly aligned.

### 5.1.1 Random Forest Prediction Strategy

**Strategy rationale**: A random forest is an ensemble of decision trees, each trained on a bootstrap sample of the data with a random subset of features at every split. Averaging many decorrelated trees gives a model that is robust to noisy financial features, needs little tuning, and exposes interpretable feature importances. Here it classifies the direction of the future `prediction_horizon`-day return from a set of standard technical indicators.

**Correctness notes** — these three mistakes are common enough to call out explicitly:

- *Index alignment*: `create_features()` drops indicator warm-up rows (the 50-day SMA needs 50 bars of history), while the label series drops its last `prediction_horizon` rows. The two therefore cover different date ranges, and they must be joined on the shared `DatetimeIndex`. Truncating both to the same length positionally pairs each feature row with the wrong label, and every accuracy number downstream becomes meaningless.
- *Label tail*: the future return of the last `prediction_horizon` bars is unknown (NaN). Those rows must be dropped **before** labeling — `np.where(nan > 0, 1, -1)` silently labels the NaN tail as "down", and because the result is an integer array, a later `np.isnan` check can no longer catch the contamination.
- *Train/test discipline*: the model is fitted only on the earlier 80% of history and evaluated on the untouched final 20%; `TimeSeriesSplit` cross-validation runs inside the training window only. Production systems go further with walk-forward retraining: periodically refit on a rolling window and only ever predict bars that come after it.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


class RandomForestTradingStrategy:
    def __init__(self, n_estimators=100, max_depth=10, prediction_horizon=5,
                 test_size=0.2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.prediction_horizon = prediction_horizon
        self.test_size = test_size
        self.model = None
        self.scaler = StandardScaler()

    def create_features(self, data):
        """Build technical-indicator features.

        All indicators use trailing windows only, so every feature at time t is
        known at time t (no look-ahead).
        """
        features = pd.DataFrame(index=data.index)

        # Price features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))

        # Moving average, volatility and RSI at several lookbacks
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = data['close'].rolling(window).mean()
            features[f'std_{window}'] = data['close'].rolling(window).std()
            features[f'rsi_{window}'] = self.calculate_rsi(data['close'], window)

        # MACD
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']

        # Bollinger Bands
        bb_period, bb_std = 20, 2
        bb_middle = data['close'].rolling(bb_period).mean()
        bb_std_dev = data['close'].rolling(bb_period).std()
        features['bb_upper'] = bb_middle + bb_std_dev * bb_std
        features['bb_lower'] = bb_middle - bb_std_dev * bb_std
        features['bb_position'] = ((data['close'] - features['bb_lower'])
                                   / (features['bb_upper'] - features['bb_lower']))

        # Volume features
        if 'volume' in data.columns:
            features['volume_sma'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma']

        # dropna() removes the indicator warm-up rows (roughly the first 50 bars),
        # so this frame starts LATER than `data`. Always align it with the labels
        # by index, never by position.
        return features.dropna()

    def create_target(self, data):
        """Label each bar with the direction of its future N-day return.

        The last `prediction_horizon` rows have no future price yet, so they are
        dropped BEFORE labeling. Labeling first would let np.where() silently mark
        the NaN tail as -1 ("down") — and since the labels are integers, no later
        isnan() check could detect the contamination.
        """
        future_returns = data['close'].shift(-self.prediction_horizon) / data['close'] - 1
        future_returns = future_returns.dropna()  # remove the unlabeled tail

        target_direction = pd.Series(np.where(future_returns > 0, 1, -1),
                                     index=future_returns.index)
        target_returns = future_returns  # regression target, kept for reference

        return target_direction, target_returns

    def train_model(self, data):
        """Train with a chronological train/test split and time-series CV."""
        features = self.create_features(data)
        target_direction, _ = self.create_target(data)

        # Align features and labels on the shared DatetimeIndex. Features lose
        # their FIRST rows to indicator warm-up while labels lose their LAST rows
        # to the prediction horizon, so positional truncation would shift every
        # label relative to its features by the warm-up offset.
        common_index = features.index.intersection(target_direction.index)
        X = features.loc[common_index]
        y = target_direction.loc[common_index]

        # Chronological split: train on the past, hold out the most recent
        # segment for out-of-sample evaluation. Never shuffle time series.
        split_point = int(len(X) * (1 - self.test_size))
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

        # Fit the scaler on the training window only. (Trees don't strictly need
        # scaling; it is kept so the same pipeline works with other estimators.)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Time-series cross-validation INSIDE the training window: each fold
        # trains on earlier data and validates on the segment right after it.
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_idx, val_idx in tscv.split(X_train_scaled):
            model_cv = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
            )
            model_cv.fit(X_train_scaled[train_idx], y_train.iloc[train_idx])
            scores.append(model_cv.score(X_train_scaled[val_idx], y_train.iloc[val_idx]))
        print(f"Time-series CV mean accuracy (training window): {np.mean(scores):.4f}")

        # Final model is fitted on the training window only, then evaluated once
        # on the held-out test window. In production, prefer walk-forward
        # retraining: refit on a rolling window and predict only bars after it.
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
        )
        self.model.fit(X_train_scaled, y_train)
        print(f"Out-of-sample test accuracy: {self.model.score(X_test_scaled, y_test):.4f}")

        return features.columns.tolist()

    def predict_signals(self, data):
        """Generate signals for new data.

        To stay out-of-sample, `data` should cover bars AFTER the training
        window; predicting on the training period itself is in-sample.
        """
        features = self.create_features(data)
        features_scaled = self.scaler.transform(features)

        probabilities = self.model.predict_proba(features_scaled)
        predictions = self.model.predict(features_scaled)

        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': self.model.feature_importances_,
        }).sort_values('importance', ascending=False)

        up_col = list(self.model.classes_).index(1)  # column of the "up" class
        signals = pd.DataFrame(index=features.index)
        signals['prediction'] = predictions
        signals['probability_up'] = probabilities[:, up_col]
        signals['confidence'] = probabilities.max(axis=1)

        return signals, feature_importance

    @staticmethod
    def calculate_rsi(prices, window):
        """Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


if __name__ == '__main__':
    # Synthetic data for illustration: a geometric random walk has no learnable
    # signal, so test accuracy near 50% is the correct outcome.
    rng = np.random.default_rng(42)
    dates = pd.bdate_range('2018-01-01', periods=1500)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(dates))))
    data = pd.DataFrame({
        'close': close,
        'volume': rng.integers(1_000_000, 5_000_000, len(dates)),
    }, index=dates)

    strategy = RandomForestTradingStrategy(prediction_horizon=5)
    strategy.train_model(data)
```

### 5.1.2 XGBoost Strategy

**Strategy rationale**: XGBoost builds trees sequentially, each one fitting the residual errors of the ensemble so far. On tabular financial features it typically outperforms bagging methods, handles a large, partly redundant feature set well, and provides built-in early stopping. This implementation expands the feature set (multi-period momentum, volatility regimes, MACD families, support/resistance) and defines several candidate targets: direction and "big move" labels at 1/3/5/10-day horizons.

The same alignment and label-tail rules as 5.1.1 apply: every target is a Series on its own (shortened) index, and training joins features and target on the shared index. Early stopping uses a chronologically later validation slice; for pedagogy that same slice is also used for the reported accuracy — a production setup would keep a third untouched test period, or use walk-forward evaluation.

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report


class XGBoostTradingStrategy:
    def __init__(self, prediction_horizon=1):
        self.prediction_horizon = prediction_horizon
        self.model = None

    def create_advanced_features(self, data):
        """Build an extended feature set (trailing windows only, no look-ahead)."""
        features = pd.DataFrame(index=data.index)

        # Basic price features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['high_low_ratio'] = data['high'] / data['low']
        features['open_close_ratio'] = data['open'] / data['close']

        # Multi-period moving averages
        for period in [5, 10, 20, 50, 100]:
            features[f'sma_{period}'] = data['close'].rolling(period).mean()
            features[f'price_to_sma_{period}'] = data['close'] / features[f'sma_{period}']
            features[f'sma_slope_{period}'] = features[f'sma_{period}'].diff(5)

        # Volatility features
        for period in [5, 10, 20]:
            features[f'volatility_{period}'] = data['close'].rolling(period).std()
            features[f'volatility_ratio_{period}'] = (
                features[f'volatility_{period}']
                / features[f'volatility_{period}'].rolling(50).mean()
            )

        # Momentum features
        for period in [1, 3, 5, 10, 20]:
            features[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1

        # Multi-period RSI
        for period in [9, 14, 21]:
            features[f'rsi_{period}'] = self.calculate_rsi(data['close'], period)

        # MACD family (several parameterizations)
        for fast, slow, signal in [(12, 26, 9), (5, 13, 6), (19, 39, 9)]:
            ema_fast = data['close'].ewm(span=fast).mean()
            ema_slow = data['close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            features[f'macd_{fast}_{slow}'] = macd
            features[f'macd_signal_{fast}_{slow}_{signal}'] = macd_signal
            features[f'macd_histogram_{fast}_{slow}_{signal}'] = macd - macd_signal

        # Bollinger Band features
        for period, std_multiplier in [(20, 2), (10, 1.5), (50, 2.5)]:
            bb_middle = data['close'].rolling(period).mean()
            bb_std = data['close'].rolling(period).std()
            bb_upper = bb_middle + bb_std * std_multiplier
            bb_lower = bb_middle - bb_std * std_multiplier
            features[f'bb_position_{period}_{std_multiplier}'] = (
                (data['close'] - bb_lower) / (bb_upper - bb_lower)
            )
            features[f'bb_width_{period}_{std_multiplier}'] = (bb_upper - bb_lower) / bb_middle

        # Volume features
        if 'volume' in data.columns:
            features['volume_sma'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma']
            features['price_volume'] = data['close'] * data['volume']
            features['volume_price_trend'] = (
                features['price_volume'].rolling(5).mean()
                / features['price_volume'].rolling(20).mean()
            )

        # Support / resistance features
        features['resistance_20'] = data['high'].rolling(20).max()
        features['support_20'] = data['low'].rolling(20).min()
        features['price_to_resistance'] = data['close'] / features['resistance_20']
        features['price_to_support'] = data['close'] / features['support_20']

        # Warm-up rows are dropped, so align with targets by index (see train_model).
        return features.dropna()

    def create_multi_target(self, data):
        """Build candidate targets at several horizons.

        Each target is a Series indexed by the bar the prediction is made ON.
        Rows whose future window runs past the end of the data are dropped before
        labeling, so no NaN tail gets silently mislabeled.
        """
        targets = {}

        for horizon in [1, 3, 5, 10]:
            future_returns = (data['close'].shift(-horizon) / data['close'] - 1).dropna()

            # Regression target: raw future return
            targets[f'return_{horizon}d'] = future_returns

            # Classification target: direction
            targets[f'direction_{horizon}d'] = pd.Series(
                np.where(future_returns > 0, 1, 0), index=future_returns.index
            )

            # Classification target: "big move". The volatility threshold comes
            # from TRAILING realized horizon-returns, so it is known at time t.
            realized = data['close'].pct_change(horizon)
            vol_threshold = (1.5 * realized.rolling(252).std()).reindex(future_returns.index)
            big_move = (future_returns.abs() > vol_threshold)[vol_threshold.notna()]
            targets[f'big_move_{horizon}d'] = big_move.astype(int)

        return targets

    def train_model(self, data, target_name='direction_1d'):
        """Train an XGBoost classifier with a chronological train/validation split."""
        features = self.create_advanced_features(data)
        targets = self.create_multi_target(data)
        target = targets[target_name]

        # Align on the shared index: features lose head rows to indicator warm-up
        # and targets lose tail rows to the horizon. Joining by index (never by
        # position) guarantees each row's features and label refer to the same bar.
        common_index = features.index.intersection(target.index)
        X = features.loc[common_index]
        y = target.loc[common_index]

        # Chronological 80/20 split (no shuffling for time series)
        split_point = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=100,
        )

        # Validation-set performance. Note: this slice also drove early stopping,
        # so treat the number as validation accuracy, not untouched test accuracy.
        val_pred = self.model.predict(dval)
        val_pred_binary = (val_pred > 0.5).astype(int)

        print(f"Validation accuracy: {accuracy_score(y_val, val_pred_binary):.4f}")
        print("\nClassification report:")
        print(classification_report(y_val, val_pred_binary))

        # Feature importance
        importance_dict = self.model.get_score(importance_type='weight')
        feature_importance = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values()),
        }).sort_values('importance', ascending=False)

        print("\nTop 10 features:")
        print(feature_importance.head(10))

        return feature_importance

    def predict_signals(self, data):
        """Generate signals for new data (use bars after the training window
        to stay out-of-sample)."""
        features = self.create_advanced_features(data)
        dtest = xgb.DMatrix(features)

        predictions = self.model.predict(dtest)

        signals = pd.DataFrame(index=features.index)
        signals['prediction_prob'] = predictions
        signals['prediction'] = (predictions > 0.5).astype(int)
        signals['confidence'] = np.abs(predictions - 0.5) * 2

        # Confidence-weighted signal strength in [-1, 1]
        signals['signal_strength'] = np.where(
            signals['prediction'] == 1,
            signals['confidence'],
            -signals['confidence'],
        )

        return signals

    @staticmethod
    def calculate_rsi(prices, window):
        """Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


if __name__ == '__main__':
    # Synthetic OHLCV data for illustration (random walk -- expect ~50% accuracy).
    rng = np.random.default_rng(7)
    dates = pd.bdate_range('2018-01-01', periods=1500)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(dates)))),
                      index=dates)
    data = pd.DataFrame({
        'open': close.shift(1).fillna(close.iloc[0]),
        'high': close * (1 + rng.uniform(0, 0.01, len(dates))),
        'low': close * (1 - rng.uniform(0, 0.01, len(dates))),
        'close': close,
        'volume': rng.integers(1_000_000, 5_000_000, len(dates)),
    }, index=dates)

    strategy = XGBoostTradingStrategy()
    strategy.train_model(data, target_name='direction_1d')
```

## 5.2 Deep Learning Strategies

### 5.2.1 LSTM Time-Series Prediction Strategy

**Strategy rationale**: An LSTM (Long Short-Term Memory network) consumes a window of the last `sequence_length` bars as a sequence, letting the model learn temporal structure — trends, regime shifts, volatility clustering — instead of treating each bar independently. Here it performs binary classification of the direction of the next `prediction_horizon`-day return.

**Correctness notes**:

- *Scaler discipline*: the `MinMaxScaler` is fitted on the training split only, and the **same fitted scaler** is applied both to the validation split and inside `predict_signals()`. A model trained on scaled inputs and then fed raw prices at inference time produces garbage — this is one of the easiest bugs to ship, because the code still runs and returns plausible-looking probabilities.
- *Window/label geometry*: each sample's input window ends at bar *t* (inclusive) and its label is the return from *t* to *t + horizon*, so the inputs contain no future information.
- *Signal timestamps*: `prepare_lstm_data()` returns the timestamp of each sample's last bar, and signals are indexed by those timestamps rather than by tail-slicing the raw data index (which would be misaligned by the warm-up and horizon offsets).
- Simplification kept for pedagogy: `predict_signals()` reuses the training data-prep, so it cannot emit signals for the final `prediction_horizon` bars (which have complete windows but no labels); a live implementation would build those unlabeled windows separately. And as with the supervised models, only predictions on bars after the training window are out-of-sample.

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler


class LSTMTradingStrategy:
    def __init__(self, sequence_length=60, prediction_horizon=1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = MinMaxScaler()

    def prepare_lstm_data(self, data):
        """Build (samples, timesteps, features) sequence arrays.

        Returns RAW (unscaled) sequences plus each sample's timestamp (the last
        bar of its window). Scaling is deliberately deferred to train_model,
        where the scaler is fitted on the training split only.
        """
        features = pd.DataFrame(index=data.index)
        features['close'] = data['close']
        features['high'] = data['high']
        features['low'] = data['low']
        features['volume'] = data['volume'] if 'volume' in data.columns else 0.0

        # Technical indicators (trailing windows only)
        features['sma_20'] = data['close'].rolling(20).mean()
        features['ema_12'] = data['close'].ewm(span=12).mean()
        features['rsi'] = self.calculate_rsi(data['close'], 14)
        features['bb_position'] = self.calculate_bb_position(data['close'])

        # Return features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))

        # Volatility
        features['volatility'] = features['returns'].rolling(20).std()

        features_clean = features.dropna()

        # Each window ends at bar i (inclusive); the label is the direction of
        # the return from bar i to bar i + horizon, so inputs never see the future.
        X, y, sample_index = [], [], []
        values = features_clean.values
        close_col = features_clean.columns.get_loc('close')
        for i in range(self.sequence_length - 1,
                       len(features_clean) - self.prediction_horizon):
            X.append(values[i - self.sequence_length + 1: i + 1])
            future_return = (values[i + self.prediction_horizon, close_col]
                             / values[i, close_col] - 1)
            y.append(1 if future_return > 0 else 0)
            sample_index.append(features_clean.index[i])

        return (np.array(X), np.array(y), pd.Index(sample_index),
                features_clean.columns)

    def build_model(self, input_shape):
        """Stacked LSTM binary classifier."""
        model = Sequential([
            Input(shape=input_shape),

            LSTM(50, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),

            LSTM(50, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),

            LSTM(50, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),

            Dense(25, activation='relu'),
            Dropout(0.2),

            Dense(1, activation='sigmoid'),  # binary output
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

        return model

    def train_model(self, data, epochs=100, batch_size=32):
        """Train with a chronological train/validation split."""
        X, y, _, feature_names = self.prepare_lstm_data(data)

        # Chronological split (no shuffling of time-ordered samples)
        split_point = int(len(X) * 0.8)
        X_train, X_val = X[:split_point], X[split_point:]
        y_train, y_val = y[:split_point], y[split_point:]

        # Fit the scaler on the TRAINING split only, then apply it to both splits.
        # Fitting on all data would leak validation-period price ranges into training.
        n_train, n_steps, n_features = X_train.shape
        self.scaler.fit(X_train.reshape(-1, n_features))
        X_train = (self.scaler.transform(X_train.reshape(-1, n_features))
                   .reshape(n_train, n_steps, n_features))
        X_val = (self.scaler.transform(X_val.reshape(-1, n_features))
                 .reshape(len(X_val), n_steps, n_features))

        self.model = self.build_model((n_steps, n_features))

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
            ],
        )

        val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation accuracy: {val_accuracy:.4f}")

        return history, feature_names

    def predict_signals(self, data):
        """Generate signals for new data.

        Applies the SAME scaler fitted during training — the model was trained on
        scaled inputs, so feeding it raw values would silently break predictions.
        """
        X, _, sample_index, _ = self.prepare_lstm_data(data)

        if len(X) == 0:
            return pd.DataFrame()

        n_samples, n_steps, n_features = X.shape
        X_scaled = (self.scaler.transform(X.reshape(-1, n_features))
                    .reshape(n_samples, n_steps, n_features))

        predictions = self.model.predict(X_scaled, verbose=0).flatten()

        # Index each signal by the timestamp of its window's last bar
        signals = pd.DataFrame(index=sample_index)
        signals['prediction_prob'] = predictions
        signals['prediction'] = (predictions > 0.5).astype(int)
        signals['confidence'] = np.abs(predictions - 0.5) * 2

        return signals

    @staticmethod
    def calculate_rsi(prices, window):
        """Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_bb_position(prices, window=20, std_multiplier=2):
        """Position of price within its Bollinger Bands, in [0, 1] when inside."""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + std * std_multiplier
        lower = sma - std * std_multiplier
        return (prices - lower) / (upper - lower)
```

## 5.3 Reinforcement Learning Strategies

### 5.3.1 Deep Q-Network (DQN) Trading Strategy

**Strategy rationale**: Reinforcement learning frames trading as sequential decision-making rather than one-shot prediction. The agent observes a state (recent price statistics plus its own account state), chooses an action (hold / buy / sell), and receives a shaped reward: a flat transaction-cost penalty on every trade, plus a bonus paid only when net worth sets a new high-water mark. This high-water-mark shaping discourages churn — trading costs a little every time, and gains only pay off when they push equity to new highs. Its known drawback is that the reward surface is flat below the peak: during a drawdown every action earns the same (zero) mark-based reward, so the agent gets no gradient about how to recover. A DQN approximates the action-value function Q(s, a) with a neural network, stabilized by two standard tricks: an experience-replay buffer (decorrelates consecutive samples) and a periodically synced target network (stabilizes the bootstrap target).

**Correctness notes**:

- *Train/test discipline*: train the agent on one historical period and run `predict_signals()` on a held-out later period — evaluating the agent on its own training data is in-sample, exactly like testing a classifier on its training set.
- *Greedy inference*: during training the agent explores with epsilon-greedy actions; at inference `predict_signals()` passes `greedy=True` so live signals are deterministic rather than partly random.
- Simplification kept for pedagogy: the environment lets the agent observe a bar's close and trade at that same close; a real execution model would trade at the next bar's open (plus slippage).

```python
import random
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class DQNetwork(nn.Module):
    """MLP mapping a state vector to one Q-value per action."""

    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class DQNTradingAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size  # 0: hold, 1: buy, 2: sell
        self.memory = deque(maxlen=10000)
        self.gamma = gamma              # discount factor
        self.epsilon = 1.0              # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate

        self.q_network = DQNetwork(state_size, action_size)
        self.target_network = DQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.update_target_network()

    def update_target_network(self):
        """Sync the target network with the online network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, greedy=False):
        """Epsilon-greedy action during training; pass greedy=True at inference
        so signals are deterministic instead of partly random."""
        if not greedy and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        self.q_network.eval()  # disable dropout: Q-value reads must be deterministic
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()
        return int(torch.argmax(q_values).item())

    def replay(self, batch_size=32):
        """Sample a minibatch from the replay buffer and take one gradient step."""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch]))
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor(np.array([e[3] for e in batch]))
        dones = torch.FloatTensor([float(e[4]) for e in batch])

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class DQNTradingEnvironment:
    """Single-asset trading environment.

    State features use trailing windows only. Simplification for pedagogy: the
    agent observes the current bar's close and trades at that same close.
    """

    # 3 price features + 4 account features + 4 technical features.
    # Keep this equal to the real feature count: padding a too-large
    # state with always-zero dimensions just wastes network capacity.
    STATE_SIZE = 11

    def __init__(self, data, initial_balance=10000, transaction_cost=0.001):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self):
        """Reset the environment to the start of the data."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance

        return self.get_state()

    def get_state(self):
        """Assemble the state vector (fixed dimension STATE_SIZE)."""
        if self.current_step >= len(self.data):
            return np.zeros(self.STATE_SIZE)

        current_price = self.data.iloc[self.current_step]['close']
        features = []

        # Recent price behavior (trailing 10-bar window, excludes the future)
        if self.current_step >= 10:
            recent_prices = self.data.iloc[self.current_step - 10:self.current_step]['close']
            features.extend([
                (current_price - recent_prices.mean()) / recent_prices.std(),
                recent_prices.pct_change().mean(),
                recent_prices.pct_change().std(),
            ])
        else:
            features.extend([0, 0, 0])

        # Account state (all normalized to the same order of magnitude:
        # a raw share count of ~1e2 would dwarf the other features)
        features.extend([
            self.balance / self.initial_balance,
            self.shares_held * current_price / self.initial_balance,  # position value
            self.net_worth / self.initial_balance,
            (self.net_worth - self.max_net_worth) / self.max_net_worth,
        ])

        # Technical indicators (trailing 20-bar window)
        if self.current_step >= 20:
            prices = self.data.iloc[self.current_step - 20:self.current_step]['close']
            sma_20 = prices.mean()
            std_20 = prices.std()
            rsi = self.calculate_rsi(prices)

            features.extend([
                (current_price - sma_20) / sma_20,
                std_20 / sma_20,
                rsi / 100 - 0.5,
                (current_price - prices.min()) / (prices.max() - prices.min()),
            ])
        else:
            features.extend([0, 0, 0, 0])

        # Defensive: STATE_SIZE matches the feature count above, so this
        # pad/truncate is a no-op unless the feature list is edited
        while len(features) < self.STATE_SIZE:
            features.append(0)

        return np.array(features[:self.STATE_SIZE])

    def step(self, action):
        """Execute an action and advance one bar."""
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0

        if action == 1:  # buy
            if self.balance > current_price * (1 + self.transaction_cost):
                shares_to_buy = self.balance // (current_price * (1 + self.transaction_cost))
                self.balance -= shares_to_buy * current_price * (1 + self.transaction_cost)
                self.shares_held += shares_to_buy
                reward = -self.transaction_cost  # trading-cost penalty

        elif action == 2:  # sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price * (1 - self.transaction_cost)
                self.shares_held = 0
                reward = -self.transaction_cost  # trading-cost penalty

        # Mark to market
        self.net_worth = self.balance + self.shares_held * current_price

        # Reward shaping: pay a bonus for new equity highs
        if self.net_worth > self.max_net_worth:
            reward += (self.net_worth - self.max_net_worth) / self.initial_balance * 10
            self.max_net_worth = self.net_worth

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        return self.get_state(), reward, done

    @staticmethod
    def calculate_rsi(prices, window=14):
        """RSI of a trailing price window (returns the latest value)."""
        if len(prices) < window:
            return 50

        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()

        avg_gain, avg_loss = gain.iloc[-1], loss.iloc[-1]
        if avg_loss == 0:
            # No losses in the window: RSI's limit is 100 (all gains).
            # A flat window (no gains either) is conventionally neutral.
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


class DQNTradingStrategy:
    def __init__(self, state_size=DQNTradingEnvironment.STATE_SIZE, action_size=3):
        self.agent = DQNTradingAgent(state_size, action_size)
        self.env = None

    def train(self, data, episodes=1000):
        """Train the DQN agent. `data` should be the TRAINING period only;
        keep a later period untouched for evaluation."""
        self.env = DQNTradingEnvironment(data)
        scores = []

        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0

            while True:
                action = self.agent.act(state)
                next_state, reward, done = self.env.step(action)
                self.agent.remember(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                if done:
                    break

                if len(self.agent.memory) > 32:
                    self.agent.replay()

            scores.append(total_reward)

            if episode % 100 == 0:
                self.agent.update_target_network()
                avg_score = np.mean(scores[-100:])
                print(f"Episode {episode}, Average Score: {avg_score:.2f}, "
                      f"Epsilon: {self.agent.epsilon:.2f}")

        return scores

    def predict_signals(self, data):
        """Run the trained agent over a dataset and record its actions.

        Pass a HELD-OUT period for out-of-sample results — replaying the agent on
        its own training data is in-sample evaluation.
        """
        self.env = DQNTradingEnvironment(data)
        state = self.env.reset()
        records = []

        while True:
            action = self.agent.act(state, greedy=True)  # no exploration at inference
            next_state, _, done = self.env.step(action)

            records.append({
                'step': self.env.current_step - 1,
                'action': action,
                'net_worth': self.env.net_worth,
            })

            state = next_state
            if done:
                break

        return pd.DataFrame(records).set_index('step')
```
