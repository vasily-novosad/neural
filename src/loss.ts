/**
 * Функции потерь (loss): одно число, показывающее, насколько ответ сети далёк от правильного.
 *
 * Обучение — это поиск весов, при которых средняя потеря по обучающим примерам минимальна.
 * Градиентный спуск двигает каждый вес против производной потери по этому весу.
 */

export type LossName = 'mse' | 'crossEntropy';

export interface Loss {
  // Значение потери для одного примера
  fn: (prediction: number[], target: number[]) => number;

  /**
   * Производная потери по выходу сети dL/da.
   * У кросс-энтропии её нет: она используется только в паре с softmax (см. ниже).
   */
  derivative?: (prediction: number[], target: number[]) => number[];
}

/**
 * Среднеквадратичная ошибка: L = (1/n) · Σ (a_i − t_i)².
 * Производная: dL/da_i = 2 · (a_i − t_i) / n.
 *
 * Естественный выбор для регрессии (предсказание чисел).
 * Для классификации подходит плохо: если сигмоида на выходе уверенно ошиблась
 * (a ≈ 0 при t = 1), её производная σ' ≈ 0, и градиент почти исчезает
 * как раз тогда, когда ошибка самая большая.
 */
const mse: Loss = {
  fn: (prediction, target) =>
    prediction.reduce((sum, p, i) => sum + (p - target[i]) ** 2, 0) / prediction.length,
  derivative: (prediction, target) =>
    prediction.map((p, i) => (2 * (p - target[i])) / prediction.length),
};

/**
 * Кросс-энтропия: L = −Σ t_i · log(a_i).
 *
 * target — one-hot вектор (например, [1, 0] — «мужское имя»), поэтому от суммы остаётся
 * одно слагаемое: −log(вероятность правильного класса). Сеть уверенно права (a → 1) —
 * потеря близка к нулю; уверенно ошиблась (a → 0) — потеря стремится к бесконечности.
 *
 * Эпсилон защищает от log(0) = −Infinity, если вероятность округлилась до нуля.
 *
 * Производную по a отдельно не даём: в связке с softmax градиент сразу по z
 * равен (a − t) — проще и численно устойчивее, чем перемножать две производные.
 */
const EPSILON = 1e-12;

const crossEntropy: Loss = {
  fn: (prediction, target) =>
    -prediction.reduce((sum, p, i) => sum + target[i] * Math.log(p + EPSILON), 0),
};

export const getLoss = (name: LossName): Loss => {
  switch (name) {
    case 'mse':
      return mse;
    case 'crossEntropy':
      return crossEntropy;
    default:
      throw new Error(`Неизвестная функция потерь: ${name}`);
  }
};
