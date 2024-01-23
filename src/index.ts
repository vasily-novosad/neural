import http from 'node:http';

import NeuralNetwork, { NeuralWeighMatrix } from './services/NeuralNetwork';

const server = http.createServer();

[
  [0.1, [0.09, 0.05, 0.12]],
  [0.2, [0.81, 0.4, 0.11]],
  [0.3, [0.51, 0.22, 0.14]],
  [0.4, [0.64, 0.1, 0.97]],
  [0.5, [0.81, 0.89, 0.88]],
];

const matrix: NeuralWeighMatrix = [
  [
    0.2,
    [
      [0.1, [0.3, 0.4, 0.8]],
      [0.1, [0.09, 0.05, 0.12]],
      [0.2, [0.81, 0.4, 0.11]],
      [0.3, [0.51, 0.22, 0.14]],
      [0.4, [0.64, 0.1, 0.97]],
      [0.5, [0.81, 0.89, 0.88]],
    ],
  ],
  [
    0.2,
    [
      [0.6, [0.1, 0.2, 0.3]],
      [0.7, [0.01, 0.02, 0.03]],
      [0.8, [0.001, 0.002, 0.003]],
    ],
  ],
  [
    0.2,
    [
      [0.9, [0.94, 0.33]],
      [0.1, [0.01, 0.78]],
      [0.11, [0.31, 0.23]],
    ],
  ],
  [
    0.2,
    [
      [0.12, []],
      [0.13, []],
    ],
  ],
];

const network = new NeuralNetwork();

server.on('request', (req, res) => {
  console.debug('request');

  network.loadNetworkFromMatrix(matrix);

  const result = network.calculateNetwork([1, 2, 3, 4, 5]);

  console.debug(result);

  res.statusCode = 200;

  return res.end();
});

server.listen(8080, 'localhost', () => {
  console.debug('server started at http://localhost:8080');
});
