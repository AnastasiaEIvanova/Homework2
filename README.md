# Домашнее задание к уроку 2: Линейная и логистическая регрессия

## Задание 1: Модификация существующих моделей

### 1.1 Расширение линейной регрессии
- L1 Регуляризация

L1 регуляризация добавляет сумму абсолютных значений весов к функции потерь
```python
l1_grad = self.l1_lambda * torch.sign(self.w) if self.l1_lambda > 0 else 0
```

- L2 Регуляризация

L2 регуляризация добавляет сумму квадратов весов к функции потерь
```python
l2_grad = self.l2_lambda * 2 * self.w if self.l2_lambda > 0 else 0
```

- Ранняя остановка (Early Stopping)

`EarlyStopping` предназначен для реализации механизма ранней остановки во время обучения модели. Он позволяет прекратить обучение, если модель не показывает улучшения в течение заданного количества эпох.
```python
if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)
```
__Сравнение потерь__
```python
if avg_loss < best_loss:
       best_loss = avg_loss
       no_improve = 0
```
- Здесь avg_loss — это средняя потеря на текущей эпохе
- Если эта потеря меньше, чем best_loss (лучшее значение потерь, зафиксированное ранее), то обновляется best_loss и счетчик no_improve сбрасывается до 0. Это означает, что модель улучшилась
  
__Отслеживание улучшений:__
```python
else:
       no_improve += 1
```
- Если avg_loss не улучшился, счетчик no_improve увеличивается на 1. Это показывает, сколько эпох подряд модель не показывает улучшений

__Условия для ранней остановки:__
```python
if no_improve >= patience:
       print(f'Early stopping at epoch {epoch}')
       break
```
- Если количество эпох без улучшений (no_improve) превышает заданное значение patience, выводится сообщение об остановке обучения, и цикл прерывается с помощью break. Это предотвращает излишнее обучение модели, когда она перестает улучшаться

__Логирование потерь:__
```python
if epoch % 10 == 0:
       log_epoch(epoch, avg_loss)
```
- Каждые 10 эпох вызывается функция log_epoch, которая, вероятно, записывает или выводит информацию о текущей эпохе и среднем значении потерь. Это помогает отслеживать процесс обучения

### 1.2 Расширение логистической регрессии
_Метрики: precision, recall, F1-score, ROC-AUC_
```python
def compute_metrics(self, X, y):
        y_pred = self.predict(X)
        y_probs = self(X)

        if self.num_classes == 2:
            y_probs = y_probs[:, 1]
            roc_auc = roc_auc_score(y.numpy(), y_probs.detach().numpy())
        else:
            y_onehot = torch.nn.functional.one_hot(y, num_classes=self.num_classes)
            roc_auc = roc_auc_score(y_onehot.numpy(), y_probs.detach().numpy(), multi_class='ovr')

        precision = precision_score(y.numpy(), y_pred.numpy(), average='weighted')
        recall = recall_score(y.numpy(), y_pred.numpy(), average='weighted')
        f1 = f1_score(y.numpy(), y_pred.numpy(), average='weighted')

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
```

__Предсказания и вероятности:__
```python
 y_pred = self.predict(X)
   y_probs = self(X)
```
- `y_pred` — это предсказанные классы для входных данных X, полученные с помощью метода predict
- `y_probs` — это вероятности принадлежности к классам, полученные путем вызова модели на тех же входных данных

__ROC AUC для бинарной классификации:__
```python
 if self.num_classes == 2:
       y_probs = y_probs[:, 1]
       roc_auc = roc_auc_score(y.numpy(), y_probs.detach().numpy())
```
- Если количество классов равно 2 (бинарная классификация), выбираются вероятности для положительного класса (`y_probs[:, 1]`), и вычисляется значение ROC AUC с помощью функции `roc_auc_score`

__ROC AUC для многоклассовой классификации:__
```python
else:
       y_onehot = torch.nn.functional.one_hot(y, num_classes=self.num_classes)
       roc_auc = roc_auc_score(y_onehot.numpy(), y_probs.detach().numpy(), multi_class='ovr')
```
- Если классов больше двух, метки `y` преобразуются в формат one-hot, и ROC AUC вычисляется для многоклассовой классификации с использованием подхода "один против всех" (one-vs-rest)

__Вычисление других метрик:__
```python
 precision = precision_score(y.numpy(), y_pred.numpy(), average='weighted')
   recall = recall_score(y.numpy(), y_pred.numpy(), average='weighted')
   f1 = f1_score(y.numpy(), y_pred.numpy(), average='weighted')
```
- Вычисляются метрики: точность (`precision`), полнота (`recall`) и F1-мера (`f1`) на основе предсказанных классов. Используется взвешенное среднее (`average='weighted'`), что учитывает дисбаланс классов

__Возврат результатов:__
```python
return {
       'precision': precision,
       'recall': recall,
       'f1': f1,
       'roc_auc': roc_auc
   }
```
- Метод возвращает словарь с вычисленными метриками: точностью, полнотой, F1-мерой и ROC AUC

_Визуализация confusion matrix_
```python
def plot_confusion_matrix(self, X, y):
        y_pred = self.predict(X)
        cm = confusion_matrix(y.numpy(), y_pred.numpy())

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
```

__Предсказания:__
```python
y_pred = self.predict(X)
```
- Метод `predict` используется для получения предсказанных классов (`y_pred`) на основе входных данных `X`

__Вычисление матрицы ошибок:__
```python
cm = confusion_matrix(y.numpy(), y_pred.numpy())
```
- Функция `confusion_matrix` вычисляет матрицу ошибок, сравнивая истинные метки (`y`) и предсказанные метки (`y_pred`). Для этого метки преобразуются в массивы NumPy

__Создание графика:__
```python
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```
- Создается фигура размером 8 на 6 дюймов
- С помощью библиотеки Seaborn (`sns`) строится тепловая карта (heatmap) матрицы ошибок. Параметр `annot=True` добавляет аннотации (значения) в ячейки, а `fmt='d'` задает формат отображения как целые числа. Цветовая палитра — синие оттенки (`cmap='Blues'`)

__Настройка заголовков и отображение:__
```python
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```
- Устанавливаются заголовок графика и подписи осей (предсказанные и истинные метки)
- Метод `plt.show()` отображает график

## Задание 2: Работа с датасетами

### 2.1 Кастомный Dataset класс
__Кастомный Dataset класс для работы с CSV файлами__
```python
class CSVDataset(Dataset):
    def __init__(self, file_path, target_column, numeric_cols=None,
                 categorical_cols=None, binary_cols=None, normalize=True):
```

Параметры:
- file_path: путь к CSV файлу
- target_column: имя целевой колонки
- numeric_cols: список числовых колонок
- categorical_cols: список категориальных колонок
- binary_cols: список бинарных колонок
- normalize: нормализовать ли числовые признаки

__Загрузка данных__
```python
self.data = pd.read_csv(file_path)
```

__Предобработка данных__
```python
self._preprocess_data()
```

__Поддержка различных форматов данных__
- Обработка числовых признаков
```python
if self.numeric_cols and self.normalize:
            self.scaler = StandardScaler()
            self.data[self.numeric_cols] = self.scaler.fit_transform(self.data[self.numeric_cols])
```

- Обработка категориальных признаков (one-hot encoding)
```python
if self.categorical_cols:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = self.encoder.fit_transform(self.data[self.categorical_cols])
            encoded_cols = self.encoder.get_feature_names_out(self.categorical_cols)
            self.data = pd.concat([
                self.data.drop(columns=self.categorical_cols),
                pd.DataFrame(encoded, columns=encoded_cols)
            ], axis=1)
```

- Обработка бинарных признаков (label encoding)
```python
if self.binary_cols:
            self.label_encoder = LabelEncoder()
            for col in self.binary_cols:
                self.data[col] = self.label_encoder.fit_transform(self.data[col])
```

### 2.2 Эксперименты с различными датасетами
- Функция `train_regression` обучает линейную регрессию на данных о ценах домов
__Загрузка датасета для регрессии__
```python
dataset = CSVDataset(
        file_path=dataset_path,
        target_column='sale_price',  # Имя целевой переменной
        numeric_cols=['lot_area', 'year_built', 'total_bathrooms', 'garage_cars'],  # Числовые признаки
        normalize=True
    )
```

- Функция `train_classification` обучает логистическую регрессию на данных о оттоку клиентов
__Загрузка датасета для классификации__
```python
dataset = CSVDataset(
        file_path=dataset_path,
        target_column='churn',  # Целевая переменная
        numeric_cols=[
            'account_length', 'total_day_minutes', 'total_eve_minutes',
            'total_night_minutes', 'total_intl_minutes', 'number_vmail_messages'
        ],
        binary_cols=['international_plan', 'voice_mail_plan'],  # Бинарные признаки
        normalize=True
    )
```

## Задание 3: Эксперименты и анализ

### 3.1 Исследование гиперпараметров
```python
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import make_regression

# Задание 3.1

def make_regression_data(n=200):
    torch.manual_seed(42)
    X = torch.rand(n, 1) * 10
    y = 2 * X + 3 + torch.randn(n, 1) * 2
    return X, y

# Реализация функции make_regression_data, которая отсутствовала
def make_regression_data(n_samples=1000, n_features=10, noise=0.1):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return X, y


# Остальные функции и классы
def mse(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LinearRegressionManual:
    def __init__(self, in_features, l1_lambda=0.0, l2_lambda=0.0):
        self.w = torch.randn(in_features, 1, requires_grad=False)
        self.b = torch.randn(1, requires_grad=False)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.dw = None
        self.db = None

    def __call__(self, X):
        return X @ self.w + self.b

    def backward(self, X, y, y_pred):
        error = y_pred - y
        self.dw = (X.T @ error) / len(X)
        self.db = torch.mean(error)

        # Добавляем регуляризацию
        if self.l1_lambda > 0:
            self.dw += self.l1_lambda * torch.sign(self.w)
            self.db += self.l1_lambda * torch.sign(self.b)
        if self.l2_lambda > 0:
            self.dw += self.l2_lambda * self.w
            self.db += self.l2_lambda * self.b

    def zero_grad(self):
        self.dw = None
        self.db = None

    def step(self, lr):
        with torch.no_grad():
            self.w -= lr * self.dw
            self.b -= lr * self.db


def train_linear_regression_with_params(X, y, lr=0.1, batch_size=32, optimizer='sgd',
                                        epochs=100, l1_lambda=0.0, l2_lambda=0.0):
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LinearRegressionManual(in_features=X.shape[1], l1_lambda=l1_lambda, l2_lambda=l2_lambda)

    # Инициализация параметров оптимизатора
    if optimizer == 'adam':
        m_w, m_b = torch.zeros_like(model.w), torch.zeros_like(model.b)
        v_w, v_b = torch.zeros_like(model.w), torch.zeros_like(model.b)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
    elif optimizer == 'rmsprop':
        avg_sq_w, avg_sq_b = torch.zeros_like(model.w), torch.zeros_like(model.b)
        gamma = 0.9
        eps = 1e-8

    losses = []

    for epoch in tqdm(range(1, epochs + 1)):
        total_loss = 0

        for i, (batch_X, batch_y) in enumerate(dataloader):
            y_pred = model(batch_X)
            loss = mse(y_pred, batch_y)
            total_loss += loss

            model.zero_grad()
            model.backward(batch_X, batch_y, y_pred)

            # Применение разных оптимизаторов
            if optimizer == 'sgd':
                model.step(lr)
            elif optimizer == 'adam':
                # Обновление моментов для Adam
                m_w = beta1 * m_w + (1 - beta1) * model.dw
                m_b = beta1 * m_b + (1 - beta1) * model.db
                v_w = beta2 * v_w + (1 - beta2) * (model.dw ** 2)
                v_b = beta2 * v_b + (1 - beta2) * (model.db ** 2)

                # Коррекция bias
                m_w_hat = m_w / (1 - beta1 ** epoch)
                m_b_hat = m_b / (1 - beta1 ** epoch)
                v_w_hat = v_w / (1 - beta2 ** epoch)
                v_b_hat = v_b / (1 - beta2 ** epoch)

                # Обновление параметров
                model.w -= lr * m_w_hat / (torch.sqrt(v_w_hat) + eps)
                model.b -= lr * m_b_hat / (torch.sqrt(v_b_hat) + eps)
            elif optimizer == 'rmsprop':
                # Обновление скользящего среднего для RMSprop
                avg_sq_w = gamma * avg_sq_w + (1 - gamma) * (model.dw ** 2)
                avg_sq_b = gamma * avg_sq_b + (1 - gamma) * (model.db ** 2)

                # Обновление параметров
                model.w -= lr * model.dw / (torch.sqrt(avg_sq_w) + eps)
                model.b -= lr * model.db / (torch.sqrt(avg_sq_b) + eps)

        avg_loss = total_loss / (i + 1)
        losses.append(avg_loss.item())

    return losses


def plot_losses(losses_dict, title):
    plt.figure(figsize=(10, 6))
    for label, losses in losses_dict.items():
        plt.plot(losses, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Генерируем данные
X, y = make_regression_data()

# Эксперимент с разными learning rates
lrs = [0.001, 0.01, 0.1, 0.5]
lr_losses = {}
for lr in lrs:
    losses = train_linear_regression_with_params(X, y, lr=lr, optimizer='sgd')
    lr_losses[f'lr={lr}'] = losses
plot_losses(lr_losses, 'Loss for different learning rates')

# Эксперимент с разными размерами батчей
batch_sizes = [8, 16, 32, 64]
batch_losses = {}
for bs in batch_sizes:
    losses = train_linear_regression_with_params(X, y, batch_size=bs, optimizer='sgd')
    batch_losses[f'batch_size={bs}'] = losses
plot_losses(batch_losses, 'Loss for different batch sizes')

# Эксперимент с разными оптимизаторами
optimizers = ['sgd', 'adam', 'rmsprop']
optim_losses = {}
for optim in optimizers:
    losses = train_linear_regression_with_params(X, y, optimizer=optim)
    optim_losses[optim] = losses
plot_losses(optim_losses, 'Loss for different optimizers')
```

### 3.2 Feature Engineering
```python
def create_polynomial_features(X, degree=2):
    """Создает полиномиальные признаки до указанной степени"""
    X_poly = X.clone()
    for d in range(2, degree+1):
        X_poly = torch.cat((X_poly, X ** d), dim=1)
    return X_poly

def create_statistical_features(X, window_size=3):
    """Добавляет статистические признаки (скользящее среднее и std)"""
    n = len(X)
    X_stat = torch.zeros((n, 2))
    for i in range(n):
        start = max(0, i - window_size)
        X_stat[i, 0] = X[start:i+1].mean()
        X_stat[i, 1] = X[start:i+1].std()
    return torch.cat((X, X_stat), dim=1)

def evaluate_model(X, y):
    """Оценивает модель на данных и возвращает конечный loss"""
    losses = train_linear_regression_with_params(X, y, epochs=50, lr=0.1)
    return losses[-1]

# Базовые данные
def make_regression_data():
    X, y = make_regression_data()

def make_regression_data():
    # Пример данных для классификации
    n_samples = 200
    n_features = 4
    n_classes = 3

# 1. Базовые признаки
base_loss = evaluate_model(X, y)

# 2. Полиномиальные признаки (до 3 степени)
X_poly = create_polynomial_features(X, degree=3)
poly_loss = evaluate_model(X_poly, y)

# 3. Статистические признаки
X_stat = create_statistical_features(X)
stat_loss = evaluate_model(X_stat, y)

# 4. Комбинация полиномиальных и статистических
X_combined = create_statistical_features(create_polynomial_features(X, degree=2))
combined_loss = evaluate_model(X_combined, y)

# Сравнение результатов
results = {
    'Model': ['Base', 'Polynomial', 'Statistical', 'Combined'],
    'Features': [1, 3, 3, 5],
    'Loss': [base_loss, poly_loss, stat_loss, combined_loss]
}

import pandas as pd
print(pd.DataFrame(results))
```
Основные компоненты:
__Генерация данных:__

- make_regression_data() - создает синтетические данные для регрессии
- Используется как одномерная, так и многомерная регрессия

__Реализация модели:__

LinearRegressionManual - ручная реализация линейной регрессии:

- Поддержка L1/L2 регуляризации
- Методы для обучения (backward, step)
- Возможность работы с разными оптимизаторами

Функции обучения:

train_linear_regression_with_params() - универсальная функция обучения:

- Поддержка SGD, Adam, RMSprop
- Настройка learning rate и batch size
- Визуализация процесса обучения

Эксперименты:

- Сравнение разных learning rates
- Сравнение разных размеров батчей
- Сравнение оптимизаторов
- Создание и оценка дополнительных признаков:
  - Полиномиальные признаки
  - Статистические признаки (скользящее среднее и std)

Визуализация:

- plot_losses() - отображение кривых обучения
- Табличное сравнение результатов
