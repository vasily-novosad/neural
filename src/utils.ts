/**
 * Вспомогательные функции: немного линейной алгебры и случайные числа.
 *
 * Вектор здесь — обычный массив number[], матрица — массив строк number[][].
 * Настоящие библиотеки (NumPy, PyTorch) хранят числа в плоских типизированных массивах
 * и считают на SIMD/GPU, это в сотни раз быстрее. Зато здесь каждая операция видна глазами.
 */

/**
 * Линейная часть нейрона: z = W·x + b.
 *
 * weights — матрица [outputSize][inputSize]: строка i — это веса одного нейрона i.
 * Поэтому z[i] = Σ_j weights[i][j] · input[j] + biases[i].
 */
export const linearCombination = (
  weights: number[][],
  input: number[],
  biases: number[],
): number[] => weights.map((row, i) => row.reduce((sum, w, j) => sum + w * input[j], biases[i]));

/**
 * Передача ошибки на предыдущий слой: Wᵀ·δ.
 *
 * Вход j влиял на все нейроны слоя через веса weights[i][j],
 * поэтому его «доля вины» — сумма по всем нейронам: Σ_i weights[i][j] · delta[i].
 * Это умножение на транспонированную матрицу, только без явного транспонирования.
 */
export const propagateError = (weights: number[][], delta: number[]): number[] =>
  weights[0].map((_, j) => weights.reduce((sum, row, i) => sum + row[j] * delta[i], 0));

/**
 * Генератор псевдослучайных чисел с зерном (алгоритм mulberry32).
 *
 * Math.random() нельзя «засеять», поэтому каждый запуск давал бы другие начальные веса
 * и другой результат. С фиксированным зерном эксперимент воспроизводим: меняешь одну
 * вещь (скорость обучения, размер слоя) и видишь эффект именно этого изменения,
 * а не случайного разброса.
 */
let state = 1;

export const setSeed = (seed: number) => {
  state = seed >>> 0;
};

// Равномерное распределение на [0, 1)
export const random = (): number => {
  state = (state + 0x6d2b79f5) >>> 0;
  let t = state;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);

  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

// Нормальное распределение N(0, 1), преобразование Бокса — Мюллера
export const randomNormal = (): number => {
  const u = 1 - random(); // (0, 1], чтобы не взять логарифм нуля
  const v = random();

  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

/**
 * Перемешивание массива (алгоритм Фишера — Йейтса). Возвращает новый массив.
 *
 * Каждая перестановка равновероятна. Популярный вариант
 * `array.sort(() => Math.random() - 0.5)` этим свойством не обладает.
 */
export const shuffle = <T>(array: readonly T[]): T[] => {
  const result = [...array];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }

  return result;
};

// Индекс максимального элемента. Для классификатора это номер предсказанного класса.
export const argmax = (vector: number[]): number =>
  vector.reduce((best, value, i) => (value > vector[best] ? i : best), 0);
