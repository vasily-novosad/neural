import Layer from './layer';
import { mse, mseDerivative } from './loss';
import { ActivationName, getActivation } from './activators';

export class Network {
  layers: Layer[] = [];

  addLayer(inputSize: number, outputSize: number, activation: 'sigmoid' | 'relu' | 'softmax') {
    const layer = new Layer(inputSize, outputSize, activation);
    this.layers.push(layer);
  }

  predict(input: number[]): number[] {
    return this.layers.reduce((acc, layer) => layer.forward(acc), input);
  }

  train(inputs: number[][], targets: number[][], epochs: number, lr: number) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
      for (let i = 0; i < inputs.length; i++) {
        const input = inputs[i];
        const target = targets[i];

        // Forward pass
        const activations: number[][] = [input];
        for (const layer of this.layers) {
          activations.push(layer.forward(activations.at(-1)!));
        }

        // Compute loss
        const output = activations.at(-1)!;
        totalLoss += mse(output, target);

        // Backward pass (simplified for single hidden layer)
        let error = mseDerivative(output, target);
        for (let l = this.layers.length - 1; l >= 0; l--) {
          const layer = this.layers[l];
          const input = activations[l];
          const z = layer.weights.map((row, i) =>
            row.reduce((sum, w, j) => sum + w * input[j], layer.biases[i]),
          );
          const derivative = getActivation(layer.activation).derivative;
          if (!derivative) break;

          const delta = error.map((e, i) => e * derivative(z)[i]);

          // Update weights and biases
          for (let i = 0; i < layer.weights.length; i++) {
            for (let j = 0; j < layer.weights[i].length; j++) {
              layer.weights[i][j] -= lr * delta[i] * input[j];
            }
            layer.biases[i] -= lr * delta[i];
          }

          // Propagate error
          error = layer.weights[0].map((_, j) =>
            layer.weights.reduce((sum, row, i) => sum + row[j] * delta[i], 0),
          );
        }
      }

      console.debug(`Epoch ${epoch + 1}, Loss: ${(totalLoss / inputs.length).toFixed(4)}`);
    }
  }

  saveModel(): string {
    const modelData: LayerData[] = this.layers.map(layer => ({
      weights: layer.weights,
      biases: layer.biases,
      activation: layer.activation,
      inputSize: layer.inputSize,
      outputSize: layer.outputSize,
    }));

    return JSON.stringify(modelData);
  }

  loadModel(json: string): void {
    const modelData: LayerData[] = JSON.parse(json);
    this.layers = modelData.map(data => {
      const layer = new Layer(data.inputSize, data.outputSize, data.activation as ActivationName);
      layer.weights = data.weights;
      layer.biases = data.biases;

      return layer;
    });
  }
}

export interface LayerData {
  weights: number[][];
  biases: number[];
  activation: string;
  inputSize: number;
  outputSize: number;
}
