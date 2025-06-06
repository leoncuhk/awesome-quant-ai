## 五、机器学习策略体系

### 5.1 监督学习策略

#### 5.1.1 随机森林预测策略

```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

class RandomForestTradingStrategy:
    def __init__(self, n_estimators=100, max_depth=10, prediction_horizon=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = StandardScaler()
        
    def create_features(self, data):
        """创建技术指标特征"""
        features = pd.DataFrame(index=data.index)
        
        # 价格特征
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # 技术指标特征
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = data['close'].rolling(window).mean()
            features[f'std_{window}'] = data['close'].rolling(window).std()
            features[f'rsi_{window}'] = self.calculate_rsi(data['close'], window)
        
        # MACD特征
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # 布林带特征
        bb_period = 20
        bb_std = 2
        bb_middle = data['close'].rolling(bb_period).mean()
        bb_std_dev = data['close'].rolling(bb_period).std()
        features['bb_upper'] = bb_middle + bb_std_dev * bb_std
        features['bb_lower'] = bb_middle - bb_std_dev * bb_std
        features['bb_position'] = (data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # 成交量特征
        if 'volume' in data.columns:
            features['volume_sma'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma']
        
        return features.dropna()
    
    def create_target(self, data):
        """创建预测目标"""
        # 未来N天收益率
        future_returns = data['close'].shift(-self.prediction_horizon) / data['close'] - 1
        
        # 分类目标：涨跌方向
        target_direction = np.where(future_returns > 0, 1, -1)
        
        # 回归目标：收益率
        target_returns = future_returns
        
        return target_direction, target_returns
    
    def train_model(self, data):
        """训练模型"""
        features = self.create_features(data)
        target_direction, target_returns = self.create_target(data)
        
        # 对齐数据
        min_length = min(len(features), len(target_direction))
        features = features.iloc[:min_length]
        target_direction = target_direction[:min_length]
        
        # 移除缺失值
        valid_idx = ~(np.isnan(target_direction) | np.isinf(target_direction))
        features = features[valid_idx]
        target_direction = target_direction[valid_idx]
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        # 训练分类模型（预测方向）
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42
        )
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(features_scaled):
            X_train, X_val = features_scaled[train_idx], features_scaled[val_idx]
            y_train, y_val = target_direction[train_idx], target_direction[val_idx]
            
            model_temp = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42
            )
            model_temp.fit(X_train, y_train)
            score = model_temp.score(X_val, y_val)
            scores.append(score)
        
        print(f"交叉验证平均准确率: {np.mean(scores):.4f}")
        
        # 在全部数据上训练最终模型
        self.model.fit(features_scaled, target_direction)
        
        return features.columns.tolist()
    
    def predict_signals(self, data):
        """生成预测信号"""
        features = self.create_features(data)
        features_scaled = self.scaler.transform(features)
        
        # 预测概率
        probabilities = self.model.predict_proba(features_scaled)
        predictions = self.model.predict(features_scaled)
        
        # 获取特征重要性
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        signals = pd.DataFrame(index=features.index)
        signals['prediction'] = predictions
        signals['probability_up'] = probabilities[:, 1] if probabilities.shape[1] > 1 else 0.5
        signals['confidence'] = np.max(probabilities, axis=1)
        
        return signals, feature_importance

    def calculate_rsi(self, prices, window):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
```

#### 5.1.2 XGBoost策略

```python
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

class XGBoostTradingStrategy:
    def __init__(self, prediction_horizon=1):
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = StandardScaler()
        
    def create_advanced_features(self, data):
        """创建高级特征"""
        features = pd.DataFrame(index=data.index)
        
        # 基础价格特征
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['high_low_ratio'] = data['high'] / data['low']
        features['open_close_ratio'] = data['open'] / data['close']
        
        # 多时间周期均线
        for period in [5, 10, 20, 50, 100]:
            features[f'sma_{period}'] = data['close'].rolling(period).mean()
            features[f'price_to_sma_{period}'] = data['close'] / features[f'sma_{period}']
            features[f'sma_slope_{period}'] = features[f'sma_{period}'].diff(5)
        
        # 波动率特征
        for period in [5, 10, 20]:
            features[f'volatility_{period}'] = data['close'].rolling(period).std()
            features[f'volatility_ratio_{period}'] = features[f'volatility_{period}'] / features[f'volatility_{period}'].rolling(50).mean()
        
        # 动量特征
        for period in [1, 3, 5, 10, 20]:
            features[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
        
        # RSI特征（多周期）
        for period in [9, 14, 21]:
            features[f'rsi_{period}'] = self.calculate_rsi(data['close'], period)
        
        # MACD特征族
        for fast, slow, signal in [(12, 26, 9), (5, 13, 6), (19, 39, 9)]:
            ema_fast = data['close'].ewm(span=fast).mean()
            ema_slow = data['close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            features[f'macd_{fast}_{slow}'] = macd
            features[f'macd_signal_{fast}_{slow}_{signal}'] = macd_signal
            features[f'macd_histogram_{fast}_{slow}_{signal}'] = macd - macd_signal
        
        # 布林带特征
        for period, std_multiplier in [(20, 2), (10, 1.5), (50, 2.5)]:
            bb_middle = data['close'].rolling(period).mean()
            bb_std = data['close'].rolling(period).std()
            bb_upper = bb_middle + bb_std * std_multiplier
            bb_lower = bb_middle - bb_std * std_multiplier
            features[f'bb_position_{period}_{std_multiplier}'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
            features[f'bb_width_{period}_{std_multiplier}'] = (bb_upper - bb_lower) / bb_middle
        
        # 成交量特征
        if 'volume' in data.columns:
            features['volume_sma'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma']
            features['price_volume'] = data['close'] * data['volume']
            features['volume_price_trend'] = (features['price_volume'].rolling(5).mean() / 
                                            features['price_volume'].rolling(20).mean())
        
        # 支撑阻力特征
        features['resistance_20'] = data['high'].rolling(20).max()
        features['support_20'] = data['low'].rolling(20).min()
        features['price_to_resistance'] = data['close'] / features['resistance_20']
        features['price_to_support'] = data['close'] / features['support_20']
        
        return features.dropna()
    
    def create_multi_target(self, data):
        """创建多目标预测"""
        targets = {}
        
        # 不同时间周期的收益率
        for horizon in [1, 3, 5, 10]:
            future_returns = data['close'].shift(-horizon) / data['close'] - 1
            targets[f'return_{horizon}d'] = future_returns
            
            # 分类目标（涨跌方向）
            targets[f'direction_{horizon}d'] = np.where(future_returns > 0, 1, 0)
            
            # 分类目标（大涨大跌）
            std_returns = future_returns.rolling(252).std()
            targets[f'big_move_{horizon}d'] = np.where(
                np.abs(future_returns) > 1.5 * std_returns, 1, 0
            )
        
        return targets
    
    def train_model(self, data, target_name='direction_1d'):
        """训练XGBoost模型"""
        features = self.create_advanced_features(data)
        targets = self.create_multi_target(data)
        target = targets[target_name]
        
        # 对齐数据
        min_length = min(len(features), len(target))
        features = features.iloc[:min_length]
        target = target[:min_length]
        
        # 移除缺失值
        valid_idx = ~(np.isnan(target) | np.isinf(target))
        features = features[valid_idx]
        target = target[valid_idx]
        
        # 划分训练和验证集（时间序列）
        split_point = int(len(features) * 0.8)
        X_train, X_val = features.iloc[:split_point], features.iloc[split_point:]
        y_train, y_val = target[:split_point], target[split_point:]
        
        # XGBoost参数
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # 训练模型
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        # 验证集预测
        val_pred = self.model.predict(dval)
        val_pred_binary = (val_pred > 0.5).astype(int)
        
        print(f"验证集准确率: {accuracy_score(y_val, val_pred_binary):.4f}")
        print("\n分类报告:")
        print(classification_report(y_val, val_pred_binary))
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': self.model.get_score(importance_type='weight').values()
        }).sort_values('importance', ascending=False)
        
        print("\n前10个重要特征:")
        print(feature_importance.head(10))
        
        return feature_importance
    
    def predict_signals(self, data):
        """生成预测信号"""
        features = self.create_advanced_features(data)
        dtest = xgb.DMatrix(features)
        
        predictions = self.model.predict(dtest)
        
        signals = pd.DataFrame(index=features.index)
        signals['prediction_prob'] = predictions
        signals['prediction'] = (predictions > 0.5).astype(int)
        signals['confidence'] = np.abs(predictions - 0.5) * 2  # 置信度
        
        # 基于置信度的信号强度
        signals['signal_strength'] = np.where(
            signals['prediction'] == 1,
            signals['confidence'],
            -signals['confidence']
        )
        
        return signals

    def calculate_rsi(self, prices, window):
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
```

### 5.2 深度学习策略

#### 5.2.1 LSTM时序预测策略

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

class LSTMTradingStrategy:
    def __init__(self, sequence_length=60, prediction_horizon=1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = MinMaxScaler()
        
    def prepare_lstm_data(self, data):
        """准备LSTM训练数据"""
        # 特征工程
        features = pd.DataFrame(index=data.index)
        features['close'] = data['close']
        features['high'] = data['high']
        features['low'] = data['low']
        features['volume'] = data['volume'] if 'volume' in data.columns else data['close'] * 0
        
        # 技术指标
        features['sma_20'] = data['close'].rolling(20).mean()
        features['ema_12'] = data['close'].ewm(span=12).mean()
        features['rsi'] = self.calculate_rsi(data['close'], 14)
        features['bb_position'] = self.calculate_bb_position(data['close'])
        
        # 收益率特征
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # 波动率
        features['volatility'] = features['returns'].rolling(20).std()
        
        # 标准化
        features_scaled = self.scaler.fit_transform(features.dropna())
        
        # 创建序列数据
        X, y = [], []
        for i in range(self.sequence_length, len(features_scaled) - self.prediction_horizon):
            X.append(features_scaled[i-self.sequence_length:i])
            # 预测未来价格变化方向
            future_return = (features.iloc[i + self.prediction_horizon]['close'] / 
                           features.iloc[i]['close'] - 1)
            y.append(1 if future_return > 0 else 0)
        
        return np.array(X), np.array(y), features.columns
    
    def build_model(self, input_shape):
        """构建LSTM模型"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
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
            
            Dense(1, activation='sigmoid')  # 二分类输出
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, data):
        """训练LSTM模型"""
        X, y, feature_names = self.prepare_lstm_data(data)
        
        # 划分训练集和验证集
        split_point = int(len(X) * 0.8)
        X_train, X_val = X[:split_point], X[split_point:]
        y_train, y_val = y[:split_point], y[split_point:]
        
        # 构建模型
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # 训练模型
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )
        
        # 评估模型
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"验证集准确率: {val_accuracy:.4f}")
        
        return history, feature_names
    
    def predict_signals(self, data):
        """生成预测信号"""
        X, _, _ = self.prepare_lstm_data(data)
        
        if len(X) == 0:
            return pd.DataFrame()
        
        predictions = self.model.predict(X)
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=data.index[-len(predictions):])
        signals['prediction_prob'] = predictions.flatten()
        signals['prediction'] = (predictions > 0.5).astype(int).flatten()
        signals['confidence'] = np.abs(predictions.flatten() - 0.5) * 2
        
        return signals
    
    def calculate_rsi(self, prices, window):
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bb_position(self, prices, window=20, std_multiplier=2):
        """计算布林带位置"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + std * std_multiplier
        lower = sma - std * std_multiplier
        return (prices - lower) / (upper - lower)
```

### 5.3 强化学习策略

#### 5.3.1 深度Q网络(DQN)交易策略

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQNTradingAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size  # 0: hold, 1: buy, 2: sell
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        
        # 构建神经网络
        self.q_network = self.build_model()
        self.target_network = self.build_model()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 更新目标网络
        self.update_target_network()
    
    def build_model(self):
        """构建深度Q网络"""
        class DQN(nn.Module):
            def __init__(self, state_size, action_size):
                super(DQN, self).__init__()
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
                x = self.fc4(x)
                return x
        
        return DQN(self.state_size, self.action_size)
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """选择动作"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        """经验回放"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class DQNTradingEnvironment:
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reset()
    
    def reset(self):
        """重置环境"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        return self.get_state()
    
    def get_state(self):
        """获取当前状态"""
        if self.current_step >= len(self.data):
            return np.zeros(20)  # 状态维度
        
        # 价格特征
        current_price = self.data.iloc[self.current_step]['close']
        
        # 技术指标特征
        features = []
        
        # 近期价格变化
        if self.current_step >= 10:
            recent_prices = self.data.iloc[self.current_step-10:self.current_step]['close']
            features.extend([
                (current_price - recent_prices.mean()) / recent_prices.std(),
                recent_prices.pct_change().mean(),
                recent_prices.pct_change().std()
            ])
        else:
            features.extend([0, 0, 0])
        
        # 账户状态
        features.extend([
            self.balance / self.initial_balance,
            self.shares_held,
            self.net_worth / self.initial_balance,
            (self.net_worth - self.max_net_worth) / self.max_net_worth
        ])
        
        # 技术指标
        if self.current_step >= 20:
            prices = self.data.iloc[self.current_step-20:self.current_step]['close']
            sma_20 = prices.mean()
            std_20 = prices.std()
            rsi = self.calculate_rsi(prices)
            
            features.extend([
                (current_price - sma_20) / sma_20,
                std_20 / sma_20,
                rsi / 100 - 0.5,
                (current_price - prices.min()) / (prices.max() - prices.min())
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # 补齐到固定维度
        while len(features) < 20:
            features.append(0)
        
        return np.array(features[:20])
    
    def step(self, action):
        """执行动作"""
        current_price = self.data.iloc[self.current_step]['close']
        
        # 计算奖励
        reward = 0
        
        if action == 1:  # 买入
            if self.balance > current_price * (1 + self.transaction_cost):
                shares_to_buy = self.balance // (current_price * (1 + self.transaction_cost))
                self.balance -= shares_to_buy * current_price * (1 + self.transaction_cost)
                self.shares_held += shares_to_buy
                reward = -self.transaction_cost  # 交易成本惩罚
                
        elif action == 2:  # 卖出
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price * (1 - self.transaction_cost)
                self.shares_held = 0
                reward = -self.transaction_cost  # 交易成本惩罚
        
        # 更新净值
        self.net_worth = self.balance + self.shares_held * current_price
        
        # 奖励设计
        if self.net_worth > self.max_net_worth:
            reward += (self.net_worth - self.max_net_worth) / self.initial_balance * 10
            self.max_net_worth = self.net_worth
        
        # 移动到下一步
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        next_state = self.get_state()
        
        return next_state, reward, done
    
    def calculate_rsi(self, prices, window=14):
        """计算RSI"""
        if len(prices) < window:
            return 50
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 1
        return 100 - (100 / (1 + rs))

class DQNTradingStrategy:
    def __init__(self, state_size=20, action_size=3):
        self.agent = DQNTradingAgent(state_size, action_size)
        self.env = None
    
    def train(self, data, episodes=1000):
        """训练DQN智能体"""
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
            
            # 更新目标网络
            if episode % 100 == 0:
                self.agent.update_target_network()
            
            if episode % 100 == 0:
                avg_score = np.mean(scores[-100:])
                print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {self.agent.epsilon:.2f}")
        
        return scores
    
    def predict_signals(self, data):
        """生成交易信号"""
        self.env = DQNTradingEnvironment(data)
        state = self.env.reset()
        
        signals = []
        
        while self.env.current_step < len(data) - 1:
            action = self.agent.act(state)
            next_state, _, done = self.env.step(action)
            
            signals.append({
                'step': self.env.current_step - 1,
                'action': action,
                'net_worth': self.env.net_worth
            })
            
            state = next_state
            
            if done:
                break
        
        signals_df = pd.DataFrame(signals)
        signals_df.set_index('step', inplace=True)
        
        return signals_df
```
