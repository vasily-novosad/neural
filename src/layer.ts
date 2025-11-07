import { getActivation, ActivationName } from './activators';

class Layer {
  weights: number[][];
  biases: number[];
  activation: ActivationName;
  inputSize: number;
  outputSize: number;

  constructor(inputSize: number, outputSize: number, activation: ActivationName = 'sigmoid') {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.weights = Array.from({ length: outputSize }, () =>
      Array.from({ length: inputSize }, () => Math.random() * 2 - 1),
    );
    this.biases = Array(outputSize).fill(0);
    this.activation = activation;
  }

  forward(input: number[]): number[] {
    const z = this.weights.map((row, i) =>
      row.reduce((sum, w, j) => sum + w * input[j], this.biases[i]),
    );

    return getActivation(this.activation).fn(z);
  }
}


export default Layer;