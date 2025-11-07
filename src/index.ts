import NeuralNetwork from '~/services/network';
import { sigmoid, sigmoidDerivative } from '~/services/activation';

const usage = async () => {
  const nn = new NeuralNetwork();
  nn.addLayer(2, 3, sigmoid, sigmoidDerivative); // вход 2, скрытый слой 3
  nn.addLayer(3, 1, sigmoid, sigmoidDerivative); // выход 1

  const input = [0.5, 0.8];
  const output = nn.predict(input);

  console.debug('Output:', output);
};

const learning = async () => {
  const nn = new NeuralNetwork();
  nn.addLayer(2, 3, sigmoid, sigmoidDerivative);
  nn.addLayer(3, 1, sigmoid, sigmoidDerivative);

  for (let i = 0; i < 10000; i++) {
    nn.train([0, 0], [0], 0.1);
    nn.train([0, 1], [1], 0.1);
    nn.train([1, 0], [1], 0.1);
    nn.train([1, 1], [0], 0.1);
  }

  console.debug('Output for [1, 0]:', nn.predict([1, 0]));
};
learning();
usage();
