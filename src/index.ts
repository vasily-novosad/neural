import { Network } from './network';
import * as fs from 'fs';

const net = new Network();
net.addLayer(2, 3, 'relu');
net.addLayer(3, 1, 'sigmoid');

// Обучение
net.train(
  [
    [0, 0],
    [1, 1],
  ],
  [[0], [1]],
  1000,
  0.1,
);

// Сохранение
const modelJson = net.saveModel();
fs.writeFileSync('model.json', modelJson);

// Загрузка
const loadedJson = fs.readFileSync('model.json', 'utf-8');
const net2 = new Network();
net2.loadModel(loadedJson);

console.log(net2.predict([1, 1]));
