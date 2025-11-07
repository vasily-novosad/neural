export const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
export const sigmoidDerivative = (y: number): number => y * (1 - y); // y = sigmoid(x)

export type ActivationFunction = (x: number) => number;
