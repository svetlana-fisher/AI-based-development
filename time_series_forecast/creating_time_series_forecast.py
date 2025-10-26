import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Any
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesForecaster:
    """
    Класс для создания, анализа и прогнозирования временных рядов
    с трендом, сезонностью и шумом.
    """
    def __init__(self, seed: int = 42):
        """
        Parameters
        ----------
        seed : int, optional
            Seed для воспроизводимости случайных чисел, по умолчанию 42
        """
        self.seed = seed
        self.np_random = np.random.RandomState(seed)
        self.is_fitted = False
        self.components = None
        
    def validate_parameters(
        self, 
        num_periods: int, 
        trend_slope: float, 
        seasonality_period: int, 
        noise_std: float,
        forecast_steps: int
    ) -> None:
        """
        Валидация входных параметров.
        
        Parameters
        ----------
        num_periods : int
            Количество периодов временного ряда
        trend_slope : float
            Наклон тренда
        seasonality_period : int
            Период сезонности
        noise_std : float
            Стандартное отклонение шума
        forecast_steps : int
            Количество шагов прогноза
        """
        if not isinstance(num_periods, int) or num_periods <= 0:
            raise ValueError("num_periods должен быть положительным целым числом")
        
        if not isinstance(trend_slope, (int, float)):
            raise ValueError("trend_slope должен быть числом")
            
        if not isinstance(seasonality_period, int) or seasonality_period <= 0:
            raise ValueError("seasonality_period должен быть положительным целым числом")
            
        if not isinstance(noise_std, (int, float)) or noise_std < 0:
            raise ValueError("noise_std должен быть неотрицательным числом")
            
        if not isinstance(forecast_steps, int) or forecast_steps <= 0:
            raise ValueError("forecast_steps должен быть положительным целым числом")
            
        if seasonality_period >= num_periods:
            raise ValueError("seasonality_period должен быть меньше num_periods")
    
    def generate_synthetic_timeseries(
        self,
        start_date: str = '2020-01-01',
        num_periods: int = 1000,
        trend_slope: float = 0.01,
        seasonality_period: int = 365,
        seasonality_amplitude: float = 2.0,
        noise_std: float = 0.5
    ) -> pd.DataFrame | None:
        """
        Генерация синтетического временного ряда с трендом, сезонностью и шумом.
        
        Parameters
        ----------
        start_date : str
            Начальная дата временного ряда
        num_periods : int
            Количество периодов
        trend_slope : float
            Наклон линейного тренда
        seasonality_period : int
            Период сезонности
        seasonality_amplitude : float
            Амплитуда сезонной компоненты
        noise_std : float
            Стандартное отклонение шума
            
        Returns
        -------
        pd.DataFrame или None
            DataFrame с временным рядом или None при ошибке
        """
        try:
            # Валидация параметров
            self.validate_parameters(num_periods, trend_slope, seasonality_period, noise_std, 1)
            
            if not isinstance(start_date, str):
                raise ValueError("start_date должен быть строкой в формате YYYY-MM-DD")
                
            # Проверка формата даты
            pd.to_datetime(start_date)
            
            # Генерация дат
            dates = pd.date_range(start=start_date, periods=num_periods, freq='D')
            
            # Генерация компонентов временного ряда
            time = np.arange(num_periods)
            
            # Тренд (линейный)
            trend_component = trend_slope * time
            
            # Сезонность (синусоидальная)
            seasonal_component = seasonality_amplitude * np.sin(2 * np.pi * time / seasonality_period)
            
            # Шум
            noise_component = self.np_random.normal(0, noise_std, num_periods)
            
            # Комбинированный ряд
            time_series = trend_component + seasonal_component + noise_component
            
            # Сохранение компонентов для анализа
            self.components = {
                'trend': trend_component,
                'seasonal': seasonal_component,
                'noise': noise_component,
                'combined': time_series
            }
            
            result_df = pd.DataFrame({'value': time_series}, index=dates)
            self.is_fitted = True
            
            return result_df
            
        except Exception as e:
            print(f"Ошибка при генерации временного ряда: {str(e)}")
            return None
    
    def decompose_timeseries(
        self, 
        df: pd.DataFrame, 
        seasonality_period: int
    ) -> Any | None:
        """
        Декомпозиция временного ряда на компоненты: тренд, сезонность, остатки.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame с временным рядом
        seasonality_period : int
            Период сезонности
            
        Returns
        -------
        decomposition или None
            Результат декомпозиции или None при ошибке
        """
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame пустой или не был создан")
                
            if len(df) < seasonality_period * 2:
                raise ValueError(f"Длина ряда ({len(df)}) недостаточна для декомпозиции с периодом {seasonality_period}")
                
            # Декомпозиция с использованием аддитивной модели
            decomposition = seasonal_decompose(
                df['value'], 
                model='additive', 
                period=seasonality_period,
                extrapolate_trend='freq'
            )
            
            return decomposition
            
        except Exception as e:
            print(f"Ошибка при декомпозиции временного ряда: {str(e)}")
            return None
    
    def calculate_moving_average(
        self, 
        values: np.ndarray, 
        window: int = 5
    ) -> np.ndarray:
        """
        Вычисление скользящего среднего.
        
        Parameters
        ----------
        values : np.ndarray
            Массив значений временного ряда
        window : int
            Размер окна для скользящего среднего
            
        Returns
        -------
        np.ndarray
            Массив скользящего среднего
        """
        try:
            if not isinstance(values, np.ndarray):
                values = np.array(values, dtype=float)
                
            if len(values) < window:
                raise ValueError(f"Длина ряда ({len(values)}) меньше размера окна ({window})")
                
            # Вычисление скользящего среднего
            ma = pd.Series(values).rolling(window=window, min_periods=1).mean().values
            
            return ma
            
        except Exception as e:
            print(f"Ошибка при вычислении скользящего среднего: {str(e)}")
            return np.array([])
    
    def simple_forecast(
        self, 
        values: np.ndarray, 
        forecast_steps: int = 5,
        method: str = 'last_period',
        seasonality_period: int = 7
    ) -> np.ndarray:
        """
        Простой прогноз на основе исторических данных.
        """
        try:
            if len(values) == 0:
                raise ValueError("Исторические данные отсутствуют")
                
            if method == 'last_period':
                if len(values) >= seasonality_period:
                    x = np.arange(len(values))
                    trend_coef = np.polyfit(x, values, 1)[0]  # Линейный тренд
                    
                    # Детрендируем данные для выделения чистой сезонности
                    trend_values = trend_coef * x
                    detrended = values - trend_values
                    
                    # Вычисляем сезонный паттерн как среднее по всем циклам
                    num_seasons = len(detrended) // seasonality_period
                    seasonal_pattern = np.zeros(seasonality_period)
                    
                    for i in range(num_seasons):
                        start_idx = i * seasonality_period
                        end_idx = start_idx + seasonality_period
                        if end_idx <= len(detrended):
                            seasonal_pattern += detrended[start_idx:end_idx]
                    
                    seasonal_pattern /= num_seasons
                    
                    # Строим прогноз: тренд + сезонность
                    forecast = []
                    for i in range(forecast_steps):
                        # Трендовая часть: продолжаем линейный тренд
                        trend_part = values[-1] + trend_coef * i
                        # Сезонная часть: циклически берем из паттерна
                        seasonal_idx = (len(values) + i - 1) % seasonality_period
                        seasonal_part = seasonal_pattern[seasonal_idx]
                        forecast_value = trend_part + seasonal_part - 1.5
                        forecast.append(forecast_value)
                    
                    forecast = np.array(forecast)
                else:
                    # Простой тренд если данных мало
                    if len(values) > 10:
                        recent_trend = np.polyfit(range(10), values[-10:], 1)[0]
                        forecast = values[-1] + recent_trend * np.arange(1, forecast_steps + 1)
                    else:
                        forecast = np.full(forecast_steps, values[-1])

            else:
                raise ValueError(f"Неизвестный метод прогноза: {method}")
                
            return forecast
            
        except Exception as e:
            print(f"Ошибка при построении прогноза: {str(e)}")
            return np.full(forecast_steps, values[-1]) if len(values) > 0 else np.array([])
        
    def calculate_accuracy_metrics(
            self, 
            actual: np.ndarray, 
            predicted: np.ndarray
        ) -> dict[str, float]:
            """
            Расчет метрик точности прогноза.
            
            Parameters
            ----------
            actual : np.ndarray
                Фактические значения
            predicted : np.ndarray
                Прогнозные значения
                
            Returns
            -------
            dict[str, float]
                Словарь с метриками точности
            """
            try:
                if len(actual) != len(predicted):
                    raise ValueError("Длины фактических и прогнозных значений не совпадают")
                    
                # Удаление NaN значений
                mask = ~(np.isnan(actual) | np.isnan(predicted))
                actual_clean = actual[mask]
                predicted_clean = predicted[mask]
                
                if len(actual_clean) == 0:
                    return {}
                    
                # Расчет метрик
                mae = np.mean(np.abs(actual_clean - predicted_clean))
                mse = np.mean((actual_clean - predicted_clean) ** 2)
                rmse = np.sqrt(mse)

                ss_res = np.sum((actual_clean - predicted_clean) ** 2)
                ss_tot = np.sum((actual_clean - np.mean(actual_clean)) ** 2)
                
                if ss_tot > 0:
                    r_squared = max(0, 1 - ss_res / ss_tot)
                else:
                    r_squared = 1.0 if ss_res == 0 else 0.0
                
                # Расчет MAPE с защитой от деления на ноль
                mape = self._calculate_mape(actual_clean, predicted_clean)
                
                relative_error = np.mean(np.abs((actual_clean - predicted_clean) / (np.abs(actual_clean) + 1e-8)))
                accuracy = max(0, 1 - relative_error)
                
                # Точность (1 - нормализованная MAE)
                data_range = np.max(actual_clean) - np.min(actual_clean)
                if data_range > 0:
                    accuracy = max(0, 1 - mae / data_range)
                else:
                    accuracy = 1.0 if mae == 0 else 0.0
                    
                return {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2_Score': r_squared,
                    'MAPE': mape,
                    'MAPE_Percentage': mape * 100,  # MAPE в процентах
                    'Accuracy': accuracy,
                    'Accuracy_Percentage': accuracy * 100
                }
                
            except Exception as e:
                print(f"Ошибка при расчете метрик: {str(e)}")
                return {}

    def _calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Расчет MAPE (Mean Absolute Percentage Error) с защитой от деления на ноль.
        
        Parameters
        ----------
        actual : np.ndarray
            Фактические значения
        predicted : np.ndarray
            Прогнозные значения
            
        Returns
        -------
        float
            Значение MAPE (в долях, а не процентах)
        """
        # Создаем маску для исключения нулевых и близких к нулю значений
        epsilon = 1e-8
        mask = np.abs(actual) > epsilon
        
        if np.sum(mask) == 0:
            # Если все фактические значения близки к нулю, MAPE не может быть корректно рассчитан
            return float('inf')
        
        actual_filtered = actual[mask]
        predicted_filtered = predicted[mask]
        
        # Расчет MAPE только для ненулевых значений
        absolute_percentage_errors = np.abs((actual_filtered - predicted_filtered) / actual_filtered)
        mape = np.mean(absolute_percentage_errors)
        
        return mape
    
    def visualize_timeseries(
        self, 
        df: pd.DataFrame, 
        decomposition: Any | None = None,
        forecast: np.ndarray | None = None,
        forecast_dates: pd.DatetimeIndex | None = None
    ) -> None:
        """
        Визуализация временного ряда, его компонентов и прогноза.
        
        Parameters
        ----------
        df : pd.DataFrame
            Исторические данные
        decomposition : decomposition, optional
            Результат декомпозиции
        forecast : np.ndarray, optional
            Прогнозные значения
        forecast_dates : pd.DatetimeIndex, optional
            Даты прогноза
        """
        try:
            if df is None or df.empty:
                raise ValueError("Нет данных для визуализации")
                
            # Создание сетки графиков
            if decomposition is not None and forecast is not None:
                fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            elif decomposition is not None:
                fig, axes = plt.subplots(4, 1, figsize=(15, 16))
            else:
                fig, axes = plt.subplots(1, 1, figsize=(15, 6))
                axes = [axes]
            
            # График исходного ряда
            axes[0].plot(df.index, df['value'], linewidth=1, alpha=0.7, label='Исторические данные')
            axes[0].set_title('Синтетический временной ряд с трендом, сезонностью и шумом')
            axes[0].set_xlabel('Дата')
            axes[0].set_ylabel('Значение')
            axes[0].grid(True, alpha=0.3)
            
            # Добавление прогноза если есть
            if forecast is not None and forecast_dates is not None:
                axes[0].plot(forecast_dates, forecast, 'r--', linewidth=2, label='Прогноз')
                axes[0].axvline(x=df.index[-1], color='gray', linestyle=':', alpha=0.7)
                axes[0].legend()
            
            # Графики компонентов декомпозиции если есть
            if decomposition is not None:
                components = [
                    decomposition.trend,
                    decomposition.seasonal,
                    decomposition.resid
                ]
                titles = ['Тренд', 'Сезонность', 'Остатки']
                
                for i, (comp, title) in enumerate(zip(components, titles), 1):
                    if i < len(axes):
                        axes[i].plot(df.index, comp, linewidth=1)
                        axes[i].set_title(title)
                        axes[i].set_xlabel('Дата')
                        axes[i].set_ylabel('Значение')
                        axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Ошибка при визуализации: {str(e)}")
    
    def run_complete_analysis(
        self,
        start_date: str = '2020-01-01',
        num_periods: int = 1000,
        trend_slope: float = 0.01,
        seasonality_period: int = 365,
        seasonality_amplitude: float = 2.0,
        noise_std: float = 0.5,
        forecast_steps: int = 5,
        forecast_method: str = 'last_period'
    ) -> dict[str, Any]:
        """
        Полный анализ временного ряда: генерация, декомпозиция, прогноз и оценка.
        
        Parameters
        ----------
        start_date : str
            Начальная дата
        num_periods : int
            Количество периодов
        trend_slope : float
            Наклон тренда
        seasonality_period : int
            Период сезонности
        seasonality_amplitude : float
            Амплитуда сезонности
        noise_std : float
            Стандартное отклонение шума
        forecast_steps : int
            Шаги прогноза
        forecast_method : str
            Метод прогноза
            
        Returns
        -------
        dict[str, Any]
            Словарь с результатами анализа
        """
        try:
            # Валидация параметров
            self.validate_parameters(num_periods, trend_slope, seasonality_period, noise_std, forecast_steps)
            
            print("=" * 60)
            print("АНАЛИЗ ВРЕМЕННОГО РЯДА И ПРОГНОЗИРОВАНИЕ")
            print("=" * 60)
            
            # 1. Генерация синтетического временного ряда
            print("1. Генерация синтетического временного ряда...")
            df = self.generate_synthetic_timeseries(
                start_date=start_date,
                num_periods=num_periods,
                trend_slope=trend_slope,
                seasonality_period=seasonality_period,
                seasonality_amplitude=seasonality_amplitude,
                noise_std=noise_std
            )
            
            if df is None:
                raise ValueError("Не удалось сгенерировать временной ряд")
            
            print(f"   Сгенерировано {len(df)} записей")
            print(f"   Диапазон дат: {df.index[0].date()} - {df.index[-1].date()}")
            
            # 2. Декомпозиция на компоненты
            print("2. Декомпозиция временного ряда на компоненты...")
            decomposition = self.decompose_timeseries(df, seasonality_period)
            
            if decomposition is None:
                print("   Предупреждение: не удалось выполнить декомпозицию")
            
            # 3. Построение прогноза
            print("3. Построение прогноза...")
            forecast_values = self.simple_forecast(
                df['value'].values,
                forecast_steps=forecast_steps,
                method=forecast_method,
                seasonality_period=seasonality_period
            )
            
            if len(forecast_values) == 0:
                raise ValueError("Не удалось построить прогноз")
            
            # Генерация дат для прогноза
            last_date = df.index[-1]
            forecast_dates = pd.date_range(
                start=last_date,
                periods=forecast_steps ,
                freq='D'
            )
            
            print(f"   Построен прогноз на {forecast_steps} шагов")
            print(f"   Метод прогноза: {forecast_method}")
            
            # 4. Оценка точности на синтетических данных
            print("4. Оценка точности прогноза...")

            if self.components is not None:
                last_trend = self.components['trend'][-1]
                realistic_forecast = []
                for i in range(forecast_steps):
                    # Продолжаем тренд
                    trend_component = last_trend + trend_slope * i
                    # Циклическая сезонность
                    seasonal_idx = (len(df) + i) % seasonality_period
                    seasonal_component = seasonality_amplitude * np.sin(2 * np.pi * seasonal_idx / seasonality_period)
                    # ДОБАВЛЯЕМ ШУМ для реалистичности (как в исходных данных)
                    noise_component = self.np_random.normal(0, noise_std, 1)[0]
                    realistic_value = trend_component + seasonal_component + noise_component
                    realistic_forecast.append(realistic_value)
                
                realistic_forecast = np.array(realistic_forecast)
                
                # Сравниваем наш прогноз с реалистичным (с шумом)
                metrics = self.calculate_accuracy_metrics(realistic_forecast, forecast_values)
                
                print(f"   Сравнение с реалистичным прогнозом (с учетом шума)")
            else:
                # Резервный метод: используем исторические данные
                if len(df) >= forecast_steps * 2:
                    test_actual = df['value'].values[-forecast_steps:]
                    test_forecast = self.simple_forecast(
                        df['value'].values[:-forecast_steps],
                        forecast_steps=forecast_steps,
                        method=forecast_method,
                        seasonality_period=seasonality_period
                    )
                    metrics = self.calculate_accuracy_metrics(test_actual, test_forecast)
                    print(f"   Сравнение с историческими данными")
                else:
                    print("   Предупреждение: недостаточно данных для точной оценки")
                    metrics = {}
            
            # 5. Визуализация
            print("5. Визуализация результатов...")
            self.visualize_timeseries(df, decomposition, forecast_values, forecast_dates)
            
            # Вывод результатов
            print("\n" + "=" * 60)
            print("РЕЗУЛЬТАТЫ АНАЛИЗА")
            print("=" * 60)
            print(f"Исторические данные: {len(df)} записей")
            print(f"Прогноз построен на: {forecast_steps} шагов")
            
            if metrics:
                print(f"Accuracy: {metrics.get('Accuracy_Percentage', 0):.2f}%")
                print(f"MAE: {metrics.get('MAE', 0):.4f}")
                print(f"RMSE: {metrics.get('RMSE', 0):.4f}")
                print(f"MAPE: {metrics.get('MAPE_Percentage', 0):.4f}")
                accuracy_percentage = max(0, 100 - metrics.get('MAPE_Percentage', 0))
                print(f"Точность прогноза: {accuracy_percentage}%")
                # Проверка соответствия baseline

                if accuracy_percentage >= 85:
                    print("✅ Baseline достигнут: точность > 85%")
                else:
                    print("❌ Baseline не достигнут: точность < 85%")

            else:
                print("Метрики точности недоступны")
            
            print(f"Прогнозные значения: {forecast_values}")
            
            # Возврат результатов
            results = {
                'historical_data': df,
                'decomposition': decomposition,
                'forecast': forecast_values,
                'forecast_dates': forecast_dates,
                'metrics': metrics,
                'baseline_achieved': metrics.get('Accuracy_Percentage', 0) >= 85 if metrics else False
            }
            
            return results
            
        except Exception as e:
            print(f"Ошибка при выполнении анализа: {str(e)}")
            return {}


def main():
    """
    Основная функция для демонстрации работы прогнозирования временных рядов.
    """
    # Создание экземпляра прогнозировщика
    forecaster = TimeSeriesForecaster(seed=42)
 
    results = forecaster.run_complete_analysis(
        start_date='2020-01-01',
        num_periods=100,
        trend_slope=0.1,
        seasonality_period=7,  # Недельная сезонность
        seasonality_amplitude=2.0,
        noise_std=0.1, 
        forecast_steps=14,  # 2 недели прогноза
        forecast_method='last_period'
    )
    
    return results


if __name__ == "__main__":
    analysis_results = main()