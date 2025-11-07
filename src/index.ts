import { Network } from './network';
import maleNames from '~/dict/male-names.json' with { type: 'json' };
import femaleNames from '~/dict/female-names.json' with { type: 'json' };
import fs from 'fs';
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
// fs.writeFileSync('model.json', modelJson);
//
// // Загрузка
// const loadedJson = fs.readFileSync('model.json', 'utf-8');
// const net2 = new Network();
// net2.loadModel(loadedJson);

// console.log(net2.predict([1, 1]));

const test2 = async () => {
  // Выборка
  // const trainingData = [
  //   { name: 'John', gender: [1, 0] }, // Мужчина
  //   { name: 'Emily', gender: [0, 1] }, // Женщина
  //   { name: 'Michael', gender: [1, 0] },
  //   { name: 'Sarah', gender: [0, 1] },
  //   { name: 'David', gender: [1, 0] },
  //   { name: 'Anna', gender: [0, 1] },
  // ];

  const trainingData: { name: string; gender: [number, number] }[] = [];
  maleNames.forEach(name => {
    trainingData.push(
      { name, gender: [1, 0] }, // male
    );
  });

  femaleNames.forEach(name => {
    trainingData.push(
      { name, gender: [0, 1] }, // male
    );
  });

  // Кодировка имени
  // const alphabet = 'abcdefghijklmnopqrstuvwxyz'.split('');
  const alphabet = 'абвгдеёжзиёклмнопрстуфхцчъыьэюя'.split('');

  // Преобразование имени в вектор фиксированной длины (например, 10 букв)
  function encodeName(name: string, maxLength = 10): number[] {
    const lower = name.toLowerCase().slice(0, maxLength);
    const vector = Array(maxLength * alphabet.length).fill(0);

    for (let i = 0; i < lower.length; i++) {
      const index = alphabet.indexOf(lower[i]);
      if (index !== -1) {
        vector[i * alphabet.length + index] = 1;
      }
    }

    return vector;
  }
  // Создание и обучение сети

  const net = new Network();

  // Входной размер = длина вектора имени
  const inputSize = alphabet.length * 10;

  // Добавляем слои
  net.addLayer(inputSize, 16, 'relu');
  net.addLayer(16, 2, 'softmax');

  // Подготовка обучающих данных
  const inputs = trainingData.map(d => encodeName(d.name));
  const targets = trainingData.map(d => d.gender);

  // Обучение
  net.train(inputs, targets, 500, 0.1);

  // Сохранение модели
  const modelJson = net.saveModel();
  fs.writeFileSync('model.json', modelJson);

  // Предсказание
  ['Андрей', 'Маша', 'Олеся', 'Кирилл', 'Игорь', 'Вася'].forEach(name => {
    const encoded = encodeName(name);
    const prediction = net.predict(encoded);

    const gender = prediction[0] > prediction[1] ? 'Мужчина' : 'Женщина';

    console.debug(`Gender prediction for ${name}: ${gender}`);
  });
};

test2();
