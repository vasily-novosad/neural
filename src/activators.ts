export type ActivationName = 'sigmoid' | 'relu' | 'softmax';

export const sigmoid = (x: number[]): number[] => x.map(v => 1 / (1 + Math.exp(-v)));

export const sigmoidDerivative = (x: number[]): number[] => {
  const s = sigmoid(x);

  return s.map(v => v * (1 - v));
};

export const relu = (x: number[]): number[] => x.map(v => Math.max(0, v));

export const reluDerivative = (x: number[]): number[] => x.map(v => (v > 0 ? 1 : 0));

export const softmax = (x: number[]): number[] => {
  const max = Math.max(...x);
  const exps = x.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);

  return exps.map(v => v / sum);
};

export const getActivation = (name: ActivationName) => {
  switch (name) {
    case 'sigmoid':
      return { fn: sigmoid, derivative: sigmoidDerivative };
    case 'relu':
      return { fn: relu, derivative: reluDerivative };
    case 'softmax':
      return { fn: softmax };
    default:
      throw new Error(`Unknown activation: ${name}`);
  }
};
