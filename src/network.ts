import Layer from './layer';
import { mse, mseDerivative } from './loss';
import { ActivationName, getActivation } from './activators';
import { linearCombination, propagateError } from './utils';

/**
 * Главный класс Network, реализующий нейросеть, обучение и сохранение модели.
 */
export class Network {
  layers: Layer[] = []; // Массив слоёв сети

  // Добавление слоя в сеть
  addLayer(inputSize: number, outputSize: number, activation: 'sigmoid' | 'relu' | 'softmax') {
    const layer = new Layer(inputSize, outputSize, activation);
    this.layers.push(layer);
  }

  // Предсказание: прямое распространение через все слои
  predict(input: number[]): number[] {
    return this.layers.reduce((acc, layer) => layer.forward(acc), input);
  }

  // Обучение сети методом обратного распространения ошибки
  train(inputs: number[][], targets: number[][], epochs: number, lr: number) {
    process.stdout.write(`Start train with ${epochs} epochs \n`);

    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;

      for (let i = 0; i < inputs.length; i++) {
        const input = inputs[i];
        const target = targets[i];

        // Прямое распространение
        const activations: number[][] = [input];
        for (const layer of this.layers) {
          activations.push(layer.forward(activations.at(-1)!));
        }

        const output = activations.at(-1)!;
        totalLoss += mse(output, target);

        // Обратное распространение ошибки
        let error = mseDerivative(output, target);
        for (let l = this.layers.length - 1; l >= 0; l--) {
          const layer = this.layers[l];
          const input = activations[l];

          // Вычисление линейной комбинации входов и весов
          const z = linearCombination(layer.weights, input, layer.biases);

          const derivative = getActivation(layer.activation).derivative;
          if (!derivative) break;

          // Вычисление градиента ошибки
          const delta = error.map((e, i) => e * derivative(z)[i]);

          // Обновление весов и смещений
          for (let i = 0; i < layer.weights.length; i++) {
            for (let j = 0; j < layer.weights[i].length; j++) {
              layer.weights[i][j] -= lr * delta[i] * input[j];
            }
            layer.biases[i] -= lr * delta[i];
          }

          // Распространение ошибки на предыдущий слой
          error = propagateError(layer.weights, delta);
        }
      }

      // Вывод средней ошибки за эпоху
      process.stdout.clearLine(0);
      process.stdout.cursorTo(0);
      // process.stdout.write('\n'); // end the line
      process.stdout.write(`Epoch ${epoch + 1}, Loss: ${(totalLoss / inputs.length).toFixed(4)}`);
    }

    process.stdout.write(`\nTrain is done\n`);
  }

  // Сохранение модели в JSON
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

  // Загрузка модели из JSON
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

// Интерфейс для сериализации слоя
export interface LayerData {
  weights: number[][];
  biases: number[];
  activation: string;
  inputSize: number;
  outputSize: number;
}
