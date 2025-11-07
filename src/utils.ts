/**
 * Вспомогательные функции для работы с массивами.
 */

// Транспонирование матрицы
export const transpose = (matrix: number[][]): number[][] =>
  matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));

// Умножение матрицы на вектор
export const matVecMul = (matrix: number[][], vector: number[]): number[] =>
  matrix.map(row => row.reduce((sum, val, i) => sum + val * vector[i], 0));

// Сложение векторов
export const vecAdd = (a: number[], b: number[]): number[] => a.map((val, i) => val + b[i]);

// Применение функции к каждому элементу вектора
export const applyFunc = (vector: number[], fn: (x: number) => number): number[] => vector.map(fn);

// Вычисление линейной комбинации входов и весов + смещений
export const linearCombination = (
  weights: number[][],
  input: number[],
  biases: number[],
): number[] => weights.map((row, i) => row.reduce((sum, w, j) => sum + w * input[j], biases[i]));

// Распространение ошибки на предыдущий слой
export const propagateError = (weights: number[][], delta: number[]): number[] =>
  weights[0].map((_, j) => weights.reduce((sum, row, i) => sum + row[j] * delta[i], 0));
