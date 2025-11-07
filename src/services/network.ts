import Layer from '~/services/layer';
import { ActivationFunction } from '~/services/activation';

class NeuralNetwork {
  public layers: Layer[] = [];

  public addLayer(
    inputSize: number,
    outputSize: number,
    activation: ActivationFunction,
    activationDerivative: ActivationFunction,
  ) {
    const lastSize =
      this.layers.length > 0 ? this.layers[this.layers.length - 1].weights.length : inputSize;
    this.layers.push(new Layer(lastSize, outputSize, activation, activationDerivative));
  }

  public predict(input: number[]): number[] {
    return this.layers.reduce((output, layer) => layer.forward(output), input);
  }

  public train(input: number[], target: number[], learningRate: number) {
    const output = this.predict(input);
    let error = output.map((o, i) => o - target[i]);

    for (let i = this.layers.length - 1; i >= 0; i--) {
      error = this.layers[i].backward(error, learningRate);
    }
  }
}

export default NeuralNetwork;
