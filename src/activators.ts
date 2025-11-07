/**
 * Реализация функций активации и их производных.
 */

// Интерфейс функции активации
export interface Activation {
  fn: (x: number[]) => number[]; // Сама функция активации
  derivative?: (x: number[]) => number[]; // Производная функции (опционально)
}

// Типы доступных активаций
export type ActivationName = 'sigmoid' | 'relu' | 'softmax';

// Реализация сигмоида и её производной
const sigmoid: Activation = {
  fn: x => x.map(v => 1 / (1 + Math.exp(-v))),
  derivative: x => {
    const s = sigmoid.fn(x);

    return s.map(v => v * (1 - v));
  },
};

// ReLU — линейная функция с отсечкой
const relu: Activation = {
  fn: x => x.map(v => Math.max(0, v)),
  derivative: x => x.map(v => (v > 0 ? 1 : 0)),
};

// Softmax — нормализует выходы в вероятности
const softmax: Activation = {
  fn: x => {
    const max = Math.max(...x); // защита от переполнения
    const exps = x.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);

    return exps.map(v => v / sum);
  },
  // Производная softmax не реализована, так как требует матрицу Якоби
};

// Получение активации по имени
export const getActivation = (name: ActivationName): Activation => {
  switch (name) {
    case 'sigmoid':
      return sigmoid;
    case 'relu':
      return relu;
    case 'softmax':
      return softmax;
    default:
      throw new Error(`Unknown activation: ${name}`);
  }
};
