# Neural — Простая нейросеть на TypeScript

**Neural** — это модульная реализация нейросети прямого распространения (feedforward) на TypeScript. Поддерживает
обучение с обратным распространением ошибки, выбор функций активации, сохранение и загрузку модели.

| Проект создан в образовательных целях и не является готовым к использованию

---

## Возможности

- Поддержка нескольких слоёв с произвольной архитектурой
- Функции активации: `sigmoid`, `relu`, `softmax`
- Обучение с обратным распространением ошибки (backpropagation)
- Функция потерь: среднеквадратичная ошибка (MSE)
- Сохранение и загрузка модели в JSON
- Написано на TypeScript с модульной архитектурой
- Поддержка unit-тестов через Jest

---

## Установка

```bash
git clone https://github.com/vasily-novosad/neural.git
cd neural
npm install
```

## Быстрый старт

```ts
import { Network } from './src/network';

const net = new Network();
net.addLayer(2, 3, 'relu');
net.addLayer(3, 1, 'sigmoid');

const inputs = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
];

const targets = [
  [0],
  [1],
  [1],
  [0]
];

net.train(inputs, targets, 1000, 0.1);

console.log(net.predict([1, 1]));


```

## Сохранение и загрузка модели

```ts
// Сохранение
const json = net.saveModel();
fs.writeFileSync('model.json', json);

// Загрузка
const loaded = fs.readFileSync('model.json', 'utf-8');
const net2 = new Network();
net2.loadModel(loaded);

```

### Структура проекта

neural/
├── src/
│ ├── activations.ts # Функции активации и их производные
│ ├── layer.ts # Класс слоя нейросети
│ ├── network.ts # Класс сети, обучение, предсказание
│ ├── loss.ts # Функции потерь
│ └── utils.ts # Вспомогательные функции
├── test/ # Тесты
├── package.json
├── tsconfig.json
└── README.md

## Архитектура

Каждый слой хранит:

- weights: матрица весов

- biases: вектор смещений

- activation: тип активации

- inputSize, outputSize: размеры входа и выхода

Сеть обучается с помощью градиентного спуска, используя производные функций активации и MSE.

## Лицензия

MIT — свободное использование и модификация.