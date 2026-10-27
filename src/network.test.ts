import { Network } from './network';
import { random, setSeed } from './utils';

/**
 * Численная проверка градиента (gradient check) — главный способ убедиться,
 * что обратное распространение написано правильно.
 *
 * Производная — это предел (L(w + ε) − L(w − ε)) / 2ε. Для каждого параметра чуть сдвигаем
 * его в обе стороны, дважды считаем потерю обычным прямым проходом и сравниваем
 * с тем, что выдал backprop. Это очень медленно (два прямых прохода на каждый вес),
 * поэтому для обучения не годится, но для проверки на маленькой сети — идеально.
 * Ошибка в backprop почти никогда не роняет программу: сеть просто учится хуже.
 * Без такой проверки её не заметить.
 */
const EPSILON = 1e-5;

const checkGradients = (net: Network, input: number[], target: number[]) => {
  const { gradients } = net.computeGradients(input, target);

  net.layers.forEach((layer, l) => {
    // Каждый параметр — ссылка на массив и индекс в нём, чтобы веса и смещения проверять одинаково
    const params: [number[], number, number][] = [
      ...layer.weights.flatMap((row, i) =>
        row.map((_, j): [number[], number, number] => [row, j, gradients.weights[l][i][j]]),
      ),
      ...layer.biases.map((_, i): [number[], number, number] => [
        layer.biases,
        i,
        gradients.biases[l][i],
      ]),
    ];

    for (const [array, index, analytic] of params) {
      const original = array[index];
      array[index] = original + EPSILON;
      const lossPlus = net.computeLoss(input, target);
      array[index] = original - EPSILON;
      const lossMinus = net.computeLoss(input, target);
      array[index] = original;

      const numeric = (lossPlus - lossMinus) / (2 * EPSILON);

      // Относительная ошибка: абсолютная мало что говорит, если сами градиенты крошечные
      const relativeError =
        Math.abs(numeric - analytic) / Math.max(1e-8, Math.abs(numeric) + Math.abs(analytic));
      expect(relativeError).toBeLessThan(1e-5);
    }
  });
};

const randomVector = (length: number) => Array.from({ length }, () => random() * 2 - 1);

describe('Обратное распространение', () => {
  beforeEach(() => setSeed(42));

  test('sigmoid → sigmoid, MSE', () => {
    const net = new Network('mse');
    net.addLayer(4, 5, 'sigmoid');
    net.addLayer(5, 3, 'sigmoid');
    checkGradients(net, randomVector(4), [0, 1, 0]);
  });

  test('relu → relu → softmax, кросс-энтропия', () => {
    const net = new Network('crossEntropy');
    net.addLayer(4, 6, 'relu');
    net.addLayer(6, 5, 'relu');
    net.addLayer(5, 3, 'softmax');
    checkGradients(net, randomVector(4), [0, 0, 1]);
  });
});

describe('Проверки конфигурации', () => {
  test('несовпадение размеров слоёв', () => {
    const net = new Network();
    net.addLayer(3, 4, 'relu');
    expect(() => net.addLayer(5, 1, 'sigmoid')).toThrow();
  });

  test('вход неправильной длины', () => {
    const net = new Network();
    net.addLayer(3, 1, 'sigmoid');
    expect(() => net.predict([1, 2])).toThrow();
  });

  test('softmax без кросс-энтропии', () => {
    const net = new Network('mse');
    net.addLayer(3, 2, 'softmax');
    expect(() => net.computeGradients([1, 2, 3], [1, 0])).toThrow();
  });
});

describe('Обучение', () => {
  test('XOR', () => {
    setSeed(1);
    const net = new Network('mse');
    net.addLayer(2, 4, 'sigmoid');
    net.addLayer(4, 1, 'sigmoid');

    const inputs = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ];
    const targets = [[0], [1], [1], [0]];
    net.train(inputs, targets, { epochs: 5000, learningRate: 0.5 });

    inputs.forEach((input, i) => {
      expect(Math.round(net.predict(input)[0])).toBe(targets[i][0]);
    });
  });
});
