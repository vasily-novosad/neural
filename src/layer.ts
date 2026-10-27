import { ActivationName, getActivation } from './activators';
import { linearCombination, randomNormal } from './utils';

/**
 * Результат прямого прохода через слой.
 *
 * z нужен при обучении: производная активации считается в точке z.
 * Сохраняем его здесь, чтобы не пересчитывать W·x + b на обратном проходе.
 */
export interface LayerOutput {
  z: number[]; // До активации: z = W·x + b
  output: number[]; // После активации: a = f(z)
}

/**
 * Полносвязный (dense) слой: каждый выход связан с каждым входом.
 */
class Layer {
  weights: number[][]; // Матрица весов [outputSize][inputSize]: строка — веса одного нейрона
  biases: number[]; // Вектор смещений [outputSize]
  activation: ActivationName;
  inputSize: number;
  outputSize: number;

  constructor(inputSize: number, outputSize: number, activation: ActivationName = 'sigmoid') {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.activation = activation;

    /**
     * Инициализация весов — масштаб важнее, чем кажется.
     *
     * z — сумма inputSize слагаемых. Если веса брать из [−1, 1] при любом размере слоя,
     * то при 900 входах z будет порядка десятков: сигмоида насытится (производная ≈ 0),
     * сеть не сможет учиться. Поэтому разброс уменьшают пропорционально 1/√inputSize:
     *  - для ReLU — инициализация He: σ = √(2 / inputSize)
     *    (ReLU обнуляет половину значений, двойка это компенсирует);
     *  - для остальных — Ксавье (Glorot): σ = √(1 / inputSize).
     *
     * Веса должны быть разными: если все одинаковые, все нейроны слоя получат одинаковый
     * градиент и навсегда останутся копиями друг друга («нарушение симметрии»).
     * А вот смещения можно начинать с нуля: симметрию уже разбили веса.
     */
    const scale = Math.sqrt((activation === 'relu' ? 2 : 1) / inputSize);
    this.weights = Array.from({ length: outputSize }, () =>
      Array.from({ length: inputSize }, () => randomNormal() * scale),
    );
    this.biases = Array(outputSize).fill(0);
  }

  // Прямое распространение через слой
  forward(input: number[]): LayerOutput {
    /**
     * Проверка размеров. Без неё лишние входы молча отбрасываются:
     * linearCombination перебирает веса, а не входы.
     * Именно так в прошлой версии сеть видела только первую букву имени.
     */
    if (input.length !== this.inputSize) {
      throw new Error(`Слой ожидает вход длиной ${this.inputSize}, получено ${input.length}`);
    }

    const z = linearCombination(this.weights, input, this.biases);

    return { z, output: getActivation(this.activation).fn(z) };
  }
}

export default Layer;
