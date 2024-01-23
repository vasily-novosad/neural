import http from 'node:http';

import NeuralNetwork, { NeuralWeighMatrix } from './services/NeuralNetwork';

const server = http.createServer();

const network = new NeuralNetwork();

const matrix: NeuralWeighMatrix = [
  [
    -39,
    'sigmoid',
    [
      [1, [0.3]],
      [1, [0.1]],
    ],
  ],

  [-1, 'sigmoid', [[1, []]]],
];
server.on('request', (req, res) => {
  network.loadNetworkFromMatrix(matrix);

  const data = [
    {
      name: 'Иван',
      weight: 84,
      height: 180,
      sex: 1,
    },
    {
      name: 'Тимур',
      height: 180,
      weight: 92,
      sex: 1,
    },
    {
      name: 'Андрей',
      height: 178,
      weight: 85,
      sex: 1,
    },
    {
      name: 'Мария',
      weight: 57,
      height: 165,
      sex: 0,
    },
    {
      name: 'Карина',
      weight: 48,
      height: 160,
      sex: 0,
    },
    {
      name: 'Анна',
      height: 170,
      weight: 62,
      sex: 0,
    },
    {
      name: 'Лиза',
      height: 166,
      weight: 49,
      sex: 0,
    },
    {
      name: 'Себястьян',
      height: 192,
      weight: 98,
      sex: 1,
    },
  ];
  data.forEach(record => {
    const result = network.calculateNetwork([record.height, record.weight]);
    const sex = result[0] > 0.5 ? 0 : 1;
    const right = record.sex === sex;
    console.debug(
      right ? 'Right' : '!!! WRONG',
      `${record.name} is ${right ? '' : 'not '}${sex === 0 ? 'female' : 'male'} (${JSON.stringify(result)})`,
    );
  });

  res.statusCode = 200;

  return res.end();
});

server.listen(8080, 'localhost', () => {
  console.debug('server started at http://localhost:8080');
});
