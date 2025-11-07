import { Network } from './network';
import maleNames from '../dict/male-names.json' with { type: 'json' };
import femaleNames from '../dict/female-names.json' with { type: 'json' };
import fs from 'node:fs';
import path from 'node:path';
//
// const net = new Network();
// net.addLayer(2, 3, 'relu');
// net.addLayer(3, 1, 'sigmoid');
//
// // Обучение
// net.train(
//   [
//     [0, 0],
//     [1, 1],
//   ],
//   [[0], [1]],
//   1000,
//   0.1,
// );
//
// // Сохранение
// const modelJson = net.saveModel();
// fs.writeFileSync('../dict/model.json', modelJson);
//
// // Загрузка
// const loadedJson = fs.readFileSync('model.json', 'utf-8');
// const net2 = new Network();
// net2.loadModel(loadedJson);

// console.log(net2.predict([1, 1]));

const test2 = async () => {
  // Выборка

  const trainingData: { name: string; gender: [number, number] }[] = [];
  maleNames.forEach(name => {
    trainingData.push(
      { name, gender: [1, 0] }, // male
    );
  });

  femaleNames.forEach(name => {
    trainingData.push(
      { name, gender: [0, 1] }, // female
    );
  });

  // Кодировка имени

  // Расширенный алфавит: латиница + кириллица + спецсимвол для неизвестных
  const alphabet = 'abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя';
  const unknownChar = '?';

  // Преобразование имени в вектор фиксированной длины (например, 10 букв)
  function encodeName(name: string, maxLength = 15): number[] {
    const fullAlphabet = alphabet + unknownChar;

    // Приведение к нижнему регистру и обрезка
    const lower = name.toLowerCase().slice(0, maxLength);
    const vector = Array(maxLength * fullAlphabet.length).fill(0);

    for (let i = 0; i < lower.length; i++) {
      const char = lower[i];
      const index = fullAlphabet.indexOf(char);
      const safeIndex = index !== -1 ? index : fullAlphabet.indexOf(unknownChar);
      vector[i * fullAlphabet.length + safeIndex] = 1;
    }

    return vector;
  }
  // Создание и обучение сети

  const net = new Network();

  // Входной размер = длина вектора имени
  const inputSize = alphabet.length;

  // Добавляем слои
  net.addLayer(inputSize, 15, 'relu');
  net.addLayer(15, 2, 'softmax');

  // Подготовка обучающих данных
  const inputs = trainingData.map(d => encodeName(d.name));
  const targets = trainingData.map(d => d.gender);

  // Обучение
  net.train(inputs, targets, 500, 0.1);

  // Сохранение модели
  const modelJson = net.saveModel();
  fs.writeFileSync(path.resolve('./dict/model.json'), modelJson);

  // Предсказание
  [
    'Андрей',
    'Мария',
    'Олеся',
    'Кирилл',
    'Игорь',
    'Василий',
    'Яна',
    'Кристина',
    'Александр',
    'Дмитрий',
  ].forEach(name => {
    const encoded = encodeName(name);
    const prediction = net.predict(encoded);

    const gender = prediction[0] > prediction[1] ? 'Мужчина' : 'Женщина';

    console.debug(`Gender prediction for ${name}: ${gender}`);
  });
};

test2();
