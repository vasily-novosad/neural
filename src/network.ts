import Layer from './layer';
import { ActivationName, getActivation } from './activators';
import { LossName, getLoss } from './loss';
import { propagateError, shuffle } from './utils';

/**
 * Градиенты всех параметров сети: на каждый вес и каждое смещение — своё число,
 * производная потери по этому параметру. Форма повторяет форму самих параметров.
 */
export interface Gradients {
  weights: number[][][]; // [слой][нейрон][вход]
  biases: number[][]; // [слой][нейрон]
}

export interface TrainOptions {
  epochs: number;
  learningRate: number;

  /**
   * Вызывается после каждой эпохи. Сеть сама ничего не печатает: что выводить
   * (loss, точность на отложенной выборке, график) — решает вызывающий код.
   */
  onEpoch?: (epoch: number, averageLoss: number) => void;
}

// Формат сохранённой модели
export interface LayerData {
  weights: number[][];
  biases: number[];
  activation: ActivationName;
  inputSize: number;
  outputSize: number;
}

export interface ModelData {
  loss: LossName;
  layers: LayerData[];
}

/**
 * Нейросеть прямого распространения (многослойный перцептрон).
 *
 * Обучение — стохастический градиентный спуск (SGD): веса обновляются после каждого примера.
 * Обычно берут мини-батчи (усредняют градиент по 32–256 примерам), это стабильнее
 * и хорошо ложится на матричные операции. Здесь батч размером 1 — так проще читать.
 */
export class Network {
  layers: Layer[] = [];
  loss: LossName;

  /**
   * Функция потерь — свойство всей сети, а не слоя: она сравнивает выход последнего слоя
   * с правильным ответом. Для классификации с softmax на выходе — 'crossEntropy',
   * для регрессии или выхода на сигмоиде — 'mse'.
   */
  constructor(loss: LossName = 'mse') {
    this.loss = loss;
  }

  addLayer(inputSize: number, outputSize: number, activation: ActivationName) {
    const previous = this.layers.at(-1);
    if (previous && previous.outputSize !== inputSize) {
      throw new Error(
        `Вход слоя (${inputSize}) не совпадает с выходом предыдущего слоя (${previous.outputSize})`,
      );
    }

    this.layers.push(new Layer(inputSize, outputSize, activation));
  }

  // Предсказание: выход каждого слоя — вход следующего
  predict(input: number[]): number[] {
    return this.layers.reduce((acc, layer) => layer.forward(acc).output, input);
  }

  // Потеря на одном примере
  computeLoss(input: number[], target: number[]): number {
    return getLoss(this.loss).fn(this.predict(input), target);
  }

  /**
   * Обратное распространение ошибки (backpropagation) для одного примера.
   *
   * Это не что иное, как правило дифференцирования сложной функции (chain rule),
   * применённое слой за слоем с конца. Для каждого слоя l введём
   *   δ_l = dL/dz_l — насколько потеря чувствительна к линейному выходу слоя.
   * Если δ известна, всё остальное получается одной строкой:
   *   dL/dW_l = δ_l ⊗ a_{l−1}   (вес w_ij влияет на z_i пропорционально входу a_j)
   *   dL/db_l = δ_l             (смещение входит в z с коэффициентом 1)
   *   δ_{l−1} = (W_lᵀ · δ_l) ⊙ f'(z_{l−1})   (ошибка переходит на слой ниже)
   * где ⊗ — внешнее произведение, ⊙ — поэлементное умножение.
   *
   * Метод только считает градиенты, но не меняет веса. Так их можно проверить
   * численно (см. network.test.ts) и легко перейти к мини-батчам: сложить градиенты
   * нескольких примеров, а потом один раз вызвать applyGradients.
   */
  computeGradients(input: number[], target: number[]): { loss: number; gradients: Gradients } {
    // Прямой проход. Запоминаем вход и z каждого слоя — они понадобятся на обратном.
    const inputs: number[][] = []; // inputs[l] — то, что пришло на вход слою l
    const zs: number[][] = [];
    let current = input;
    for (const layer of this.layers) {
      inputs.push(current);
      const { z, output } = layer.forward(current);
      zs.push(z);
      current = output;
    }
    const output = current;

    const gradients: Gradients = { weights: [], biases: [] };

    // δ последнего слоя
    let delta = this.outputDelta(zs.at(-1)!, output, target);

    for (let l = this.layers.length - 1; l >= 0; l--) {
      const layer = this.layers[l];
      const layerInput = inputs[l];

      // Градиенты параметров слоя. Порядок слоёв в итоге — с первого по последний.
      gradients.weights[l] = delta.map(d => layerInput.map(x => d * x));
      gradients.biases[l] = [...delta];

      // Ошибка для слоя ниже. На первом слое не нужна: ниже только входные данные.
      if (l > 0) {
        const error = propagateError(layer.weights, delta); // dL/da_{l−1}
        const below = this.layers[l - 1];
        const derivative = getActivation(below.activation).derivative;
        if (!derivative) {
          throw new Error(`Активация ${below.activation} допустима только в последнем слое`);
        }
        // Производная считается один раз для всего вектора, а не для каждого элемента
        const fPrime = derivative(zs[l - 1]);
        delta = error.map((e, i) => e * fPrime[i]);
      }
    }

    return { loss: getLoss(this.loss).fn(output, target), gradients };
  }

  // Шаг градиентного спуска: параметр −= скорость обучения × градиент
  applyGradients(gradients: Gradients, learningRate: number) {
    this.layers.forEach((layer, l) => {
      for (let i = 0; i < layer.outputSize; i++) {
        for (let j = 0; j < layer.inputSize; j++) {
          layer.weights[i][j] -= learningRate * gradients.weights[l][i][j];
        }
        layer.biases[i] -= learningRate * gradients.biases[l][i];
      }
    });
  }

  /**
   * Обучение. Возвращает среднюю потерю по каждой эпохе.
   *
   * Эпоха — один проход по всем обучающим примерам.
   */
  train(inputs: number[][], targets: number[][], options: TrainOptions): number[] {
    const { epochs, learningRate, onEpoch } = options;
    const history: number[] = [];
    const indices = inputs.map((_, i) => i);

    for (let epoch = 1; epoch <= epochs; epoch++) {
      let totalLoss = 0;

      /**
       * Порядок примеров перемешивается каждую эпоху. Если подавать их как есть
       * (все мужские имена, затем все женские), каждое обновление тянет веса к текущему
       * классу, и к концу эпохи сеть «помнит» в основном последний.
       * SGD рассчитывает на то, что каждый пример — случайный представитель всех данных.
       */
      for (const i of shuffle(indices)) {
        const { loss, gradients } = this.computeGradients(inputs[i], targets[i]);
        this.applyGradients(gradients, learningRate);
        totalLoss += loss;
      }

      const averageLoss = totalLoss / inputs.length;
      history.push(averageLoss);
      onEpoch?.(epoch, averageLoss);
    }

    return history;
  }

  saveModel(): string {
    const model: ModelData = {
      loss: this.loss,
      layers: this.layers.map(layer => ({
        weights: layer.weights,
        biases: layer.biases,
        activation: layer.activation,
        inputSize: layer.inputSize,
        outputSize: layer.outputSize,
      })),
    };

    return JSON.stringify(model);
  }

  loadModel(json: string): void {
    const model: ModelData = JSON.parse(json);
    this.loss = model.loss;
    this.layers = model.layers.map(data => {
      const layer = new Layer(data.inputSize, data.outputSize, data.activation);
      layer.weights = data.weights;
      layer.biases = data.biases;

      return layer;
    });
  }

  /**
   * δ последнего слоя: dL/dz.
   *
   * Общий случай по цепному правилу: dL/dz = dL/da ⊙ f'(z).
   *
   * Особый случай — softmax + кросс-энтропия. Если честно перемножить матрицу Якоби
   * softmax на производную кросс-энтропии (−t/a), почти всё сокращается и остаётся
   *   dL/dz = a − t.
   * Это не приближение, а точный результат. Поэтому эту пару и используют вместе,
   * а остальные сочетания запрещаем, чтобы не получить молча неверный градиент.
   */
  private outputDelta(z: number[], output: number[], target: number[]): number[] {
    const last = this.layers.at(-1)!;
    const isSoftmax = last.activation === 'softmax';
    const isCrossEntropy = this.loss === 'crossEntropy';

    if (isSoftmax !== isCrossEntropy) {
      throw new Error('softmax на выходе используется только вместе с потерей crossEntropy');
    }

    if (isSoftmax) {
      return output.map((a, i) => a - target[i]);
    }

    const dLdA = getLoss(this.loss).derivative!(output, target);
    const fPrime = getActivation(last.activation).derivative!(z);

    return dLdA.map((g, i) => g * fPrime[i]);
  }
}
