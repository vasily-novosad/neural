import { getActivation, ActivationName } from './activators';
import { linearCombination } from '~/utils';

/**
 * Класс слоя нейросети
 */
class Layer {
  weights: number[][]; // Матрица весов: [outputSize][inputSize]
  biases: number[]; // Вектор смещений: [outputSize]
  activation: ActivationName; // Название функции активации
  inputSize: number; // Размер входного вектора
  outputSize: number; // Размер выходного вектора

  constructor(inputSize: number, outputSize: number, activation: ActivationName = 'sigmoid') {
    this.inputSize = inputSize;
    this.outputSize = outputSize;

    // Инициализация весов случайными значениями от -1 до 1
    this.weights = Array.from({ length: outputSize }, () =>
      Array.from({ length: inputSize }, () => Math.random() * 2 - 1),
    );

    // Инициализация смещений нулями
    this.biases = Array(outputSize).fill(0);
    this.activation = activation;
  }

  // Прямое распространение: вычисление выхода слоя
  forward(input: number[]): number[] {
    // Вычисление линейной комбинации входов и весов + смещения
    const z = linearCombination(this.weights, input, this.biases);

    // Применение функции активации
    return getActivation(this.activation).fn(z);
  }
}

export default Layer;
