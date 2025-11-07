export const mse = (pred: number[], target: number[]): number =>
  pred.reduce((sum, p, i) => sum + (p - target[i]) ** 2, 0) / pred.length;

export const mseDerivative = (pred: number[], target: number[]): number[] =>
  pred.map((p, i) => (2 * (p - target[i])) / pred.length);
