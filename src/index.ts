/**
 * Пример: определение пола по имени.
 *
 * Классическая учебная задача классификации: вход — строка, выход — один из двух классов.
 * Полный цикл: подготовка данных → кодирование → разбиение на выборки → обучение →
 * оценка на данных, которых сеть не видела → сохранение и загрузка модели.
 *
 * Данные и кодирование — в names.ts. Тот же код использует playground (src/playground).
 */
import fs from 'node:fs';
import path from 'node:path';
import { Network } from './network';
import { setSeed } from './utils';
import {
  CLASS_NAMES,
  Encoding,
  MALE,
  FEMALE,
  accuracy,
  allSamples,
  classify,
  encodeName,
  inputSize,
  oneHot,
  splitSamples,
} from './names';

// Фиксированное зерно: начальные веса и порядок примеров одинаковы при каждом запуске
setSeed(2025);

// ---------------------------------------------------------------------------------------------
// 1. Данные и кодирование
// ---------------------------------------------------------------------------------------------

const TEST_FRACTION = 0.2;
const { train: trainSamples, test: testSamples } = splitSamples(allSamples, TEST_FRACTION);

const ENCODING: Encoding = { maxLength: 4, fromEnd: true };
const INPUT_SIZE = inputSize(ENCODING);

const trainInputs = trainSamples.map(s => encodeName(s.name, ENCODING));
const trainTargets = trainSamples.map(s => oneHot(s.label));

// ---------------------------------------------------------------------------------------------
// 2. Сеть
// ---------------------------------------------------------------------------------------------

/**
 * 240 входов → 16 скрытых нейронов (ReLU) → 2 выхода (softmax).
 *
 * Скрытый слой позволяет учитывать сочетания букв: «-ия» — скорее женское,
 * но «Илия» — мужское; «-а» — женское, но «Никита» и «Лука» — мужские.
 * Без скрытого слоя (логистическая регрессия) сеть могла бы только складывать
 * независимые «голоса» отдельных букв.
 */
const HIDDEN_SIZE = 16;
const net = new Network('crossEntropy');
net.addLayer(INPUT_SIZE, HIDDEN_SIZE, 'relu');
net.addLayer(HIDDEN_SIZE, CLASS_NAMES.length, 'softmax');

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

// ---------------------------------------------------------------------------------------------
// 3. Обучение
// ---------------------------------------------------------------------------------------------

/**
 * Скорость обучения (learning rate) — размер шага градиентного спуска.
 * Слишком большая — потеря скачет или растёт; слишком маленькая — обучение идёт годами.
 * Её подбирают экспериментом. Попробуйте 0.5 и 0.001 и посмотрите, что будет.
 *
 * Число эпох тоже подбирают по тестовой точности: если на обучающей выборке точность
 * растёт, а на тестовой падает — сеть начала запоминать примеры (переобучение).
 * Поставьте EPOCHS = 50 и посмотрите на это сами.
 *
 * Строго говоря, подбирая настройки по тестовой выборке, мы немного «подглядываем»
 * в неё, и оценка выходит слегка оптимистичной. В серьёзных задачах данные делят
 * на три части: обучающую, валидационную (для подбора настроек) и тестовую
 * (смотрят на неё один раз, в самом конце).
 *
 * Потолок здесь — около 92%, и дело не в сети, а в данных: в словаре много имён,
 * чьё окончание «не того» пола (Фома, Кузьма, Илья, Нинель, Зоя, Глеб).
 * Пол таких имён по буквам не угадать — их можно только запомнить, а на новых
 * именах запоминание не помогает.
 */
const EPOCHS = 10;
const LEARNING_RATE = 0.005;

console.debug(
  `Имён: ${allSamples.length} (обучение ${trainSamples.length}, тест ${testSamples.length})`,
);
console.debug(`Сеть: ${INPUT_SIZE} → ${HIDDEN_SIZE} → ${CLASS_NAMES.length}\n`);
console.debug(`До обучения: точность на тесте ${percent(accuracy(net, testSamples, ENCODING))}\n`);

net.train(trainInputs, trainTargets, {
  epochs: EPOCHS,
  learningRate: LEARNING_RATE,
  onEpoch: (epoch, loss) => {
    if (epoch === 1 || epoch % 2 === 0) {
      console.debug(
        `Эпоха ${String(epoch).padStart(2)}: loss ${loss.toFixed(4)}, ` +
          `точность: обучение ${percent(accuracy(net, trainSamples, ENCODING))}, ` +
          `тест ${percent(accuracy(net, testSamples, ENCODING))}`,
      );
    }
  },
});

// ---------------------------------------------------------------------------------------------
// 4. Ошибки на тесте — по ним видно, чего сеть не поняла
// ---------------------------------------------------------------------------------------------

const mistakes = testSamples.filter(s => classify(net, s.name, ENCODING) !== s.label);
console.debug(`\nОшибки на тесте (${mistakes.length} из ${testSamples.length}):`);
console.debug(mistakes.map(s => `${s.name} (на самом деле ${CLASS_NAMES[s.label]})`).join(', '));

// ---------------------------------------------------------------------------------------------
// 5. Сохранение и загрузка модели
// ---------------------------------------------------------------------------------------------

const modelPath = path.resolve('./dict/model.json');
fs.writeFileSync(modelPath, net.saveModel());

// Загруженная модель обязана давать те же ответы, что и исходная
const loaded = new Network();
loaded.loadModel(fs.readFileSync(modelPath, 'utf-8'));

// ---------------------------------------------------------------------------------------------
// 6. Предсказания
// ---------------------------------------------------------------------------------------------

const known = new Set(allSamples.map(s => s.name));

console.debug('\nПредсказания загруженной модели:');
[
  'Андрей',
  'Мария',
  'Кирилл',
  'Кристина',
  'Никита', // Мужское на «-а» — трудный случай
  'Илья', // Мужское на «-я»
  // Этих имён нет в словарях
  'Ратмир',
  'Милослав',
  'Велимир',
  'Святогор',
  'Добрыня', // Мужское на «-я», в словаре его нет
  'Велислава',
  'Ладомира',
  'Снежина',
].forEach(name => {
  const [male, female] = loaded.predict(encodeName(name, ENCODING));
  const label = male > female ? MALE : FEMALE;
  const confidence = Math.max(male, female);
  const source = known.has(name) ? 'есть в словаре' : 'новое имя';

  console.debug(
    `  ${name.padEnd(10)} ${CLASS_NAMES[label].padEnd(8)} ${percent(confidence).padStart(6)}  (${source})`,
  );
});
