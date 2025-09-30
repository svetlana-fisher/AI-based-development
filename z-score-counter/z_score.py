import numpy as np
import pandas as pd
import logging

from sklearn.datasets import load_diabetes
from pathlib import Path
from jsonargparse import CLI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZScoreOutlierDetector:
    def __init__(self, threshold: float = 3.0):
        """
        Parameters
        ----------
        threshold : float, optional
            Пороговое значение Z-score для определения выбросов, by default 3.0
        """
        self.threshold = threshold
        self.data = None
        self.z_scores = None
        self.outliers_mask = None
        
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Валидация входных параметров."""

        if not isinstance(self.threshold, (int, float)):
            raise ValueError("Threshold must be a numeric value")
        
        if self.threshold <= 0:
            raise ValueError("Threshold must be positive")
        
        if self.threshold < 2:
            logger.warning("Low threshold value (%f) may mark too many points as outliers", 
                          self.threshold)
    
    def _validate_data(self, data: np.ndarray) -> None:
        """
        Валидация входных данных.
        
        Parameters
        ----------
        data : np.ndarray
            Входные данные для проверки
        """

        if data is None:
            raise ValueError("Data cannot be None")
        
        if not isinstance(data, (np.ndarray, pd.DataFrame)):
            raise ValueError("Data must be numpy array or pandas DataFrame")
        
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        
        if np.any(np.isnan(data)):
            raise ValueError("Data contains NaN values")
        
        if np.any(np.isinf(data)):
            raise ValueError("Data contains infinite values")
    
    def fit(self, data: np.ndarray) -> 'ZScoreOutlierDetector':
        """
        Вычисление Z-scores для данных.
        
        Parameters
        ----------
        data : np.ndarray
            Входные данные (n_samples, n_features)
        
        Returns
        -------
        ZScoreOutlierDetector
            self
        """
        self._validate_data(data)
        self.data = np.asarray(data)
        
        try:
            # Вычисление Z-scores
            mean = np.mean(self.data, axis=0)
            std = np.std(self.data, axis=0)
            
            # Проверка нулевого стандартного отклонения
            if np.any(std == 0):
                zero_std_features = np.where(std == 0)[0]
                logger.warning("Features with zero standard deviation: %s. "
                              "Replacing with small epsilon.", zero_std_features)
                std[std == 0] = 1e-8
            
            self.z_scores = np.abs((self.data - mean) / std)
            
            # Создание маски выбросов
            self.outliers_mask = np.any(self.z_scores > self.threshold, axis=1)

        except Exception as e:
            logger.error("Error during Z-score calculation: %s", e)
            raise
        
        return self
    
    def get_outliers_count(self) -> int:
        """
        Получение количества выбросов.
        
        Returns
        -------
        int
            Количество выбросов
        """
        if self.outliers_mask is None:
            raise RuntimeError("Must call fit() before getting outliers count")
        
        return np.sum(self.outliers_mask)
    
    def get_outliers_indices(self) -> np.ndarray:
        """
        Получение индексов выбросов.
        
        Returns
        -------
        np.ndarray
            Индексы выбросов
        """
        if self.outliers_mask is None:
            raise RuntimeError("Must call fit() before getting outliers indices")
        
        return np.where(self.outliers_mask)[0]
    
    def remove_outliers(self) -> np.ndarray:
        """
        Удаление выбросов из данных.
        
        Returns
        -------
        np.ndarray
            Данные без выбросов
        """
        if self.outliers_mask is None:
            raise RuntimeError("Must call fit() before removing outliers")
        
        clean_data = self.data[~self.outliers_mask]
        
        return clean_data
    
    def replace_outliers(self, method: str = 'mean') -> np.ndarray:
        """
        Замена выбросов указанным методом.
        
        Parameters
        ----------
        method : str, optional
            Метод замены: 'mean', 'median', by default 'mean'
        
        Returns
        -------
        np.ndarray
            Данные с замененными выбросами
        """
        if self.outliers_mask is None:
            raise RuntimeError("Must call fit() before replacing outliers")
        
        if method not in ['mean', 'median']:
            raise ValueError("Method must be 'mean' or 'median'")
        
        data_copy = self.data.copy()
        
        if not np.any(self.outliers_mask):
            return data_copy
        
        try:
            if method == 'mean':
                replacement_values = np.mean(self.data[~self.outliers_mask], axis=0)
            elif method == 'median':
                replacement_values = np.median(self.data[~self.outliers_mask], axis=0)  

            for feature_idx in range(data_copy.shape[1]):
                feature_outliers = self.z_scores[:, feature_idx] > self.threshold
                if np.any(feature_outliers):
                    data_copy[feature_outliers, feature_idx] = replacement_values[feature_idx]
              
            return data_copy
        
            
        except Exception as e:
            logger.error("Error during outlier replacement: %s", e)
            raise
    
    def get_summary(self) -> dict:
        """
        Получение статистики по выбросам.
        
        Returns
        -------
        dict
            Словарь с статистикой
        """
        if self.outliers_mask is None:
            raise RuntimeError("Must call fit() before getting summary")
        
        outliers_count = np.sum(self.outliers_mask)
        total_samples = len(self.data)
        
        return {
            'total_samples': total_samples,
            'outliers_count': outliers_count,
            'outliers_percentage': (outliers_count / total_samples) * 100,
            'threshold': self.threshold,
            'outliers_indices': self.get_outliers_indices().tolist()
        }


def load_dataset(dataset_path: Path | None = None) -> np.ndarray:
    """
    Загрузка датасета.
    
    Parameters
    ----------
    dataset_path : Path | None, optional
        Путь к CSV файлу, by default None (используется встроенный датасет)
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        Данные и целевые переменные (если есть)
    """
    try:
        if dataset_path is None:
            diabetes = load_diabetes()
            data = diabetes.data

        else:
            # Загружаем из CSV файла
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset file '{dataset_path}' not found")
            
            if not dataset_path.is_file():
                raise ValueError(f"'{dataset_path}' is not a file")
            
            df = pd.read_csv(dataset_path)
            data = df.select_dtypes(include=[np.number]).values
        
        return data
        
    except Exception as e:
        logger.error("Error loading dataset: %s", e)
        raise


def save_results(data: np.ndarray, output_path: Path | None = None, action: str = 'remove') -> None:
    """
    Сохранение обработанных данных в файл.
    
    Parameters
    ----------
    data : np.ndarray
        Обработанные данные для сохранения
    output_path : Path | None, optional
        Путь для сохранения, by default None
    action : str
        Выполненное действие ('remove' или 'replace')
    """
    if output_path is None:
        suffix = "_cleaned" if action == 'remove' else "_replaced"
        output_path = Path(f"dataset{suffix}.csv")
    
    try:
        df = pd.DataFrame(data)
        
        # Сохраняем в CSV
        df.to_csv(output_path, index=False)
        
    except Exception as e:
        logger.error("Error saving results: %s", e)
        raise

def analyze_outliers(
    dataset_path: Path | None = None,
    threshold: float = 3.0,
    action: str = 'remove',
    replacement_method: str = 'mean',
    output_path: Path | None = None,
    return_statistic: bool = True,
    save_result: bool = True,
) -> dict | None:
    """    
    Parameters
    ----------
    dataset_path : Path | None, optional
        Путь к датасету, by default None
    threshold : float, optional
        Порог Z-score, by default 3.0
    action : str, optional
        Действие с выбросами:'remove', 'replace', by default 'remove'
    replacement_method : str, optional
        Метод замены: 'mean', 'median', by default 'mean'
    output_path : Path | None, optional
        Путь для сохранения результатов, by default None
    return_results : bool, optional
        Возвращать ли результаты, by default False
    
    Returns
    -------
    dict | None
        Результаты анализа если return_results=True, иначе None
    """
    try:
        data = load_dataset(dataset_path)
        
        # Анализ выбросов
        detector = ZScoreOutlierDetector(threshold=threshold)
        detector.fit(data)
        
        # Выполнение указанного действия
        outliers_count = detector.get_outliers_count()
        print(f"Количество выбросов (Z-score > {threshold}): {outliers_count}")
            
        if action == 'remove':
            processed_data = detector.remove_outliers()
            print(f"Удалено выбросов: {detector.get_outliers_count()}")
            print(f"Оставшееся количество: {len(processed_data)}")
            
        elif action == 'replace':
            processed_data = detector.replace_outliers(method=replacement_method)
            print(f"Заменено выбросов: {detector.get_outliers_count()} "
                  f"(метод: {replacement_method})")
        
        else:
            raise ValueError("Action must be 'count', 'remove', or 'replace'")
        
        
        if save_result and processed_data is not None:
            save_results(processed_data, output_path, action)
        
        # Получение статистики
        summary = detector.get_summary()
        
        # Вывод дополнительной информации
        print(f"\nСтатистика анализа:")
        print(f"  Всего: {summary['total_samples']}")
        print(f"  Выбросы: {summary['outliers_count']} "
              f"({summary['outliers_percentage']:.2f}%)")
        print(f"  Порог Z-score: {summary['threshold']}")
        
        if summary['outliers_count'] > 0:
            print(f"  Индексы выбросов: {summary['outliers_indices']}")
        
        if return_statistic:
            return summary
            
    except Exception as e:
        logger.error("Error during outlier analysis: %s", e)
        raise


def main(
    dataset_path: Path | None = None,
    threshold: float = 3.0,
    action: str = 'remove',
    replacement_method: str = 'mean',
    output_path: Path | None = None,
    return_statistic: bool = True,
    save_result: bool = True,
) -> None:
    """
    Parameters
    ----------
    dataset_path : Path | None, optional
        Путь к датасету, by default None
    threshold : float, optional
        Порог Z-score, by default 3.0
    action : str, optional
        Действие с выбросами: 'count', 'remove', 'replace', by default 'count'
    replacement_method : str, optional
        Метод замены: 'mean', 'median', by default 'mean'
    output_path : Path | None, optional
        Путь для сохранения результатов, by default None
    """
    analyze_outliers(
        dataset_path=dataset_path,
        threshold=threshold,
        action=action,
        replacement_method=replacement_method,
        output_path=output_path,
        return_statistic=return_statistic,
        save_result=save_result,
    )


if __name__ == "__main__":
    CLI(main)