import { ActivationFunction } from '~/services/activation';

class Layer {
  weights: number[][];
  biases: number[];
  activation: ActivationFunction;
  activationDerivative: ActivationFunction;
  lastInput: number[] = [];
  lastOutput: number[] = [];

  constructor(
    inputSize: number,
    outputSize: number,
    activation: ActivationFunction,
    activationDerivative: ActivationFunction,
  ) {
    this.weights = Array.from({ length: outputSize }, () =>
      Array.from({ length: inputSize }, () => Math.random() * 2 - 1),
    );
    this.biases = Array.from({ length: outputSize }, () => Math.random() * 2 - 1);
    this.activation = activation;
    this.activationDerivative = activationDerivative;
  }

  public forward(input: number[]): number[] {
    this.lastInput = input;
    this.lastOutput = this.weights.map((weightsRow, i) => {
      const sum = weightsRow.reduce((acc, w, j) => acc + w * input[j], 0) + this.biases[i];

      return this.activation(sum);
    });

    return this.lastOutput;
  }

  public backward(error: number[], learningRate: number): number[] {
    const inputError = Array(this.weights[0].length).fill(0);

    for (let i = 0; i < this.weights.length; i++) {
      const delta = error[i] * this.activationDerivative(this.lastOutput[i]);
      for (let j = 0; j < this.weights[i].length; j++) {
        inputError[j] += this.weights[i][j] * delta;
        this.weights[i][j] -= learningRate * delta * this.lastInput[j];
      }
      this.biases[i] -= learningRate * delta;
    }

    return inputError;
  }
}

export default Layer;
