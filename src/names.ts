/**
 * Задача «пол по имени»: данные, кодирование и оценка.
 *
 * Общий модуль для консольного примера (index.ts) и playground в браузере,
 * чтобы оба работали с одними и теми же данными и одной кодировкой.
 */
import { Network } from './network';
import { argmax, shuffle } from './utils';
import maleNames from '../dict/male-names.json' with { type: 'json' };
import femaleNames from '../dict/female-names.json' with { type: 'json' };

/**
 * Ответ кодируется вектором one-hot: [1, 0] — мужское, [0, 1] — женское.
 * Сеть выдаёт два числа через softmax — вероятности классов, они в сумме дают 1.
 * Для двух классов хватило бы и одного выхода на сигмоиде, но вариант с softmax
 * без изменений обобщается на любое число классов.
 */
export const MALE = 0;
export const FEMALE = 1;
export const CLASS_NAMES = ['мужское', 'женское'];

export interface Sample {
  name: string;
  label: number; // Номер класса: MALE или FEMALE
}

export const allSamples: Sample[] = [
  ...maleNames.map(name => ({ name, label: MALE })),
  ...femaleNames.map(name => ({ name, label: FEMALE })),
];

/**
 * Обучающая и тестовая выборки.
 *
 * Если проверять сеть на тех же именах, на которых она училась, мы измерим только память:
 * достаточно большая сеть может просто запомнить все 1673 имени. Нас интересует,
 * справится ли она с именем, которого никогда не видела (обобщение).
 * Поэтому часть имён откладываем и в обучении не используем.
 *
 * Перемешать перед разбиением обязательно: иначе в тест попали бы одни женские имена
 * на «Ф»–«Я» — они в конце списка. Перемешивание использует общий генератор из utils,
 * поэтому при одном и том же зерне (setSeed) разбиение одинаковое.
 */
export const splitSamples = (samples: Sample[], testFraction: number) => {
  const shuffled = shuffle(samples);
  const testSize = Math.round(shuffled.length * testFraction);

  return { test: shuffled.slice(0, testSize), train: shuffled.slice(testSize) };
};

/**
 * Кодирование: строка → вектор чисел.
 *
 * Сеть работает только с векторами фиксированной длины, поэтому имя нужно превратить в числа.
 *
 * Каждая позиция в имени кодируется вектором one-hot по алфавиту: единица на месте буквы,
 * остальные нули. Нельзя кодировать букву её номером (а = 1, б = 2, …): тогда сеть решит,
 * что «б» находится «между» «а» и «в», а это бессмысленно.
 *
 * Самое важное решение — читать имя с КОНЦА. Пол в русских именах определяется
 * в основном окончанием (в словаре на «а/я» оканчиваются 94% женских имён и 5% мужских).
 * Если читать с начала, последняя буква у «Яна» стоит в позиции 3, а у «Кристина» — в позиции 8,
 * и сети пришлось бы отдельно учить правило «а на конце» для каждой длины имени.
 * Если читать с конца, последняя буква всегда в позиции 0. Удачное представление данных
 * часто важнее архитектуры сети.
 */
export interface Encoding {
  /**
   * Сколько букв учитываем. Подобрано экспериментом: при 10 буквах сеть быстрее
   * переобучается (лишние входы позволяют запоминать конкретные имена), а точность
   * на тесте не выше, чем при 3–5.
   */
  maxLength: number;
  fromEnd: boolean; // true — читать с конца имени, false — с начала
}

export const ALPHABET = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz';
const UNKNOWN = ALPHABET.length; // Отдельная ячейка для пробела, дефиса и прочих символов
export const SYMBOLS = ALPHABET.length + 1;

export const inputSize = (encoding: Encoding) => encoding.maxLength * SYMBOLS;

// Буквы, которые реально попадут на вход сети, в порядке позиций
export const visibleLetters = (name: string, encoding: Encoding): string[] => {
  const letters = [...name.toLowerCase()];

  return (encoding.fromEnd ? letters.reverse() : letters).slice(0, encoding.maxLength);
};

export const encodeName = (name: string, encoding: Encoding): number[] => {
  const vector: number[] = Array(inputSize(encoding)).fill(0);

  visibleLetters(name, encoding).forEach((char, position) => {
    const index = ALPHABET.indexOf(char);
    vector[position * SYMBOLS + (index === -1 ? UNKNOWN : index)] = 1;
  });

  // Позиции после конца короткого имени остаются нулевыми — так сеть «видит» длину имени
  return vector;
};

export const oneHot = (label: number): number[] => CLASS_NAMES.map((_, i) => (i === label ? 1 : 0));

// Номер класса, который выбрала сеть
export const classify = (net: Network, name: string, encoding: Encoding): number =>
  argmax(net.predict(encodeName(name, encoding)));

// Доля правильных ответов — понятнее человеку, чем значение потери
export const accuracy = (net: Network, samples: Sample[], encoding: Encoding): number =>
  samples.filter(s => classify(net, s.name, encoding) === s.label).length / samples.length;
