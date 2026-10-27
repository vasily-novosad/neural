/**
 * Playground: та же сеть и те же данные, что в index.ts, но в браузере и с ручками.
 *
 * Обучение идёт по одной эпохе за раз, между эпохами управление возвращается браузеру,
 * поэтому страница не зависает и графики рисуются по ходу обучения.
 */
import { Network } from '../network';
import { setSeed } from '../utils';
import {
  ALPHABET,
  CLASS_NAMES,
  Encoding,
  MALE,
  Sample,
  accuracy,
  allSamples,
  classify,
  encodeName,
  inputSize,
  oneHot,
  splitSamples,
  visibleLetters,
} from '../names';

const TEST_FRACTION = 0.2;

interface EpochStats {
  epoch: number;
  loss: number;
  train: number; // Точность на обучающей выборке
  test: number; // Точность на тестовой выборке
}

interface Model {
  net: Network;
  encoding: Encoding;
  hidden: number;
  epochs: number;
  train: Sample[];
  test: Sample[];
  history: EpochStats[];
}

const byId = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

let model: Model | null = null;

// Номер текущего запуска обучения. Новый запуск или «Стоп» увеличивает его, и старый цикл выходит.
let runId = 0;

// ---------------------------------------------------------------------------------------------
// Настройки
// ---------------------------------------------------------------------------------------------

const readSettings = () => ({
  encoding: {
    maxLength: Number(byId<HTMLInputElement>('maxLength').value),
    fromEnd: byId<HTMLInputElement>('fromEnd').checked,
  },
  hidden: Number(byId<HTMLInputElement>('hidden').value),
  learningRate: Number(byId<HTMLSelectElement>('learningRate').value),
  epochs: Number(byId<HTMLInputElement>('epochs').value),
  seed: Number(byId<HTMLInputElement>('seed').value) || 0,
});

const architecture = (encoding: Encoding, hidden: number) =>
  hidden > 0
    ? `${inputSize(encoding)} входов → ${hidden} нейронов ReLU → 2 выхода softmax`
    : `${inputSize(encoding)} входов → 2 выхода softmax (без скрытого слоя)`;

['maxLength', 'hidden', 'epochs'].forEach(id => {
  const input = byId<HTMLInputElement>(id);
  const output = byId(`${id}Out`);
  input.addEventListener('input', () => {
    output.textContent = input.value;
  });
});

// ---------------------------------------------------------------------------------------------
// Обучение
// ---------------------------------------------------------------------------------------------

const averageLoss = (net: Network, inputs: number[][], targets: number[][]) =>
  inputs.reduce((sum, input, i) => sum + net.computeLoss(input, targets[i]), 0) / inputs.length;

const nextTick = () => new Promise(resolve => setTimeout(resolve, 0));

const train = async () => {
  const id = ++runId;
  const settings = readSettings();

  // Зерно задаём до разбиения и создания слоёв: и то и другое использует генератор случайных чисел
  setSeed(settings.seed);
  const { train: trainSamples, test: testSamples } = splitSamples(allSamples, TEST_FRACTION);

  const net = new Network('crossEntropy');
  const size = inputSize(settings.encoding);
  if (settings.hidden > 0) {
    net.addLayer(size, settings.hidden, 'relu');
    net.addLayer(settings.hidden, CLASS_NAMES.length, 'softmax');
  } else {
    net.addLayer(size, CLASS_NAMES.length, 'softmax');
  }

  const inputs = trainSamples.map(s => encodeName(s.name, settings.encoding));
  const targets = trainSamples.map(s => oneHot(s.label));

  const current: Model = {
    net,
    encoding: settings.encoding,
    hidden: settings.hidden,
    epochs: settings.epochs,
    train: trainSamples,
    test: testSamples,
    history: [],
  };
  model = current;

  const evaluate = (epoch: number, loss: number) => {
    current.history.push({
      epoch,
      loss,
      train: accuracy(net, trainSamples, settings.encoding),
      test: accuracy(net, testSamples, settings.encoding),
    });
    renderAll();
  };

  byId('arch').textContent = architecture(settings.encoding, settings.hidden);
  byId('splitInfo').textContent =
    `${trainSamples.length} имён в обучении, ${testSamples.length} в тесте. ` +
    `Эпоха 0 — сеть со случайными весами, до обучения.`;
  setRunning(true);

  evaluate(0, averageLoss(net, inputs, targets));

  for (let epoch = 1; epoch <= settings.epochs; epoch++) {
    await nextTick();
    if (id !== runId) return;

    // Одна эпоха за вызов: так между эпохами можно перерисовать страницу
    const [loss] = net.train(inputs, targets, { epochs: 1, learningRate: settings.learningRate });
    evaluate(epoch, loss);
  }

  setRunning(false);
};

const setRunning = (running: boolean) => {
  byId<HTMLButtonElement>('stopBtn').disabled = !running;
  byId('trainBtn').textContent = running ? 'Заново' : 'Обучить';
};

byId('settings').addEventListener('submit', event => {
  event.preventDefault();
  train();
});

byId('stopBtn').addEventListener('click', () => {
  runId++;
  setRunning(false);
});

// ---------------------------------------------------------------------------------------------
// Отрисовка
// ---------------------------------------------------------------------------------------------

const renderAll = () => {
  renderStats();
  renderCharts();
  renderTable();
  renderProbe();
  renderMistakes();
};

const renderStats = () => {
  if (!model) return;
  const last = model.history.at(-1)!;
  byId('statEpoch').textContent = `${last.epoch} / ${model.epochs}`;
  byId('statLoss').textContent = last.loss.toFixed(4);
  byId('statTrain').innerHTML =
    `<span class="swatch" style="background:var(--series-train)"></span>${percent(last.train)}`;
  byId('statTest').innerHTML =
    `<span class="swatch" style="background:var(--series-test)"></span>${percent(last.test)}`;
  byId('progress').style.width = `${(last.epoch / model.epochs) * 100}%`;
};

const renderTable = () => {
  if (!model) return;
  byId('historyTable').innerHTML = model.history
    .map(
      h =>
        `<tr><td>${h.epoch}</td><td>${h.loss.toFixed(4)}</td>` +
        `<td>${percent(h.train)}</td><td>${percent(h.test)}</td></tr>`,
    )
    .join('');
};

// Известные имена: в какой выборке и какой правильный ответ
const lookup = (name: string): { sample: Sample; set: 'train' | 'test' } | null => {
  if (!model) return null;
  const key = name.trim().toLowerCase();
  const find = (samples: Sample[]) => samples.find(s => s.name.toLowerCase() === key);
  const inTrain = find(model.train);
  if (inTrain) return { sample: inTrain, set: 'train' };
  const inTest = find(model.test);

  return inTest ? { sample: inTest, set: 'test' } : null;
};

const escapeHtml = (text: string) =>
  text.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);

const renderProbe = () => {
  if (!model) return;
  const name = byId<HTMLInputElement>('probe').value.trim();
  const { net, encoding } = model;

  if (!name) {
    byId('verdictClass').textContent = '—';
    byId('verdictMeta').textContent = 'Введите имя';
    byId('letters').innerHTML = '';
    byId('neurons').innerHTML = '';

    return;
  }

  const encoded = encodeName(name, encoding);
  const [male, female] = net.predict(encoded);
  const label = male > female ? MALE : 1 - MALE;
  byId('verdictClass').textContent = CLASS_NAMES[label];
  byId('barMale').style.width = `${male * 100}%`;
  byId('pMale').textContent = `мужское ${percent(male)}`;
  byId('pFemale').textContent = `женское ${percent(female)}`;

  const known = lookup(name);
  let meta: string;
  if (!known) {
    meta = 'Нового имени нет в словарях';
  } else {
    const where = known.set === 'train' ? 'Из обучающей выборки' : 'Из теста: сеть его не видела';
    const correct = known.sample.label === label;
    meta =
      `${where} · на самом деле ${CLASS_NAMES[known.sample.label]} ` +
      (correct ? '<span class="ok">верно</span>' : '<span class="no">ошибка</span>');
  }
  byId('verdictMeta').innerHTML = meta;

  // Буквы по позициям входа: позиция 0 — первая буква, которую читает сеть
  const letters = visibleLetters(name, encoding);
  byId('letters').innerHTML = Array.from({ length: encoding.maxLength }, (_, i) => {
    const char = letters[i];
    if (char === undefined) return `<span class="letter empty"><b>·</b><small>${i}</small></span>`;
    const shown = ALPHABET.includes(char) ? escapeHtml(char) : '?';

    return `<span class="letter"><b>${shown}</b><small>${i}</small></span>`;
  }).join('');

  // Скрытый слой: яркость клетки — выход нейрона относительно самого активного
  const block = byId('neuronsBlock');
  block.hidden = model.hidden === 0;
  if (model.hidden > 0) {
    const activations = net.layers[0].forward(encoded).output;
    const max = Math.max(...activations) || 1;
    const silent = activations.filter(a => a === 0).length;
    byId('neuronsTitle').textContent =
      `Активность скрытых нейронов: молчат ${silent} из ${activations.length}`;
    byId('neurons').innerHTML = activations
      .map(
        (a, i) =>
          `<span class="neuron" title="Нейрон ${i}: ${a.toFixed(3)}">` +
          `<i style="opacity:${(a / max).toFixed(3)}"></i></span>`,
      )
      .join('');
  }
};

const renderMistakes = () => {
  if (!model) return;
  const { net, encoding, test } = model;
  const wrong = test.filter(s => classify(net, s.name, encoding) !== s.label);
  byId('mistakesTitle').textContent =
    `Ошибки на тестовых именах: ${wrong.length} из ${test.length}`;
  byId('mistakes').innerHTML = wrong.length
    ? wrong
        .map(
          s =>
            `<button type="button" class="chip" data-name="${escapeHtml(s.name)}">` +
            `${escapeHtml(s.name)}<small>${s.label === MALE ? 'муж.' : 'жен.'}</small></button>`,
        )
        .join('')
    : '<span class="empty-note">Ошибок нет.</span>';
};

byId('probe').addEventListener('input', renderProbe);

byId('mistakes').addEventListener('click', event => {
  const chip = (event.target as HTMLElement).closest<HTMLElement>('.chip');
  if (!chip) return;
  const probe = byId<HTMLInputElement>('probe');
  probe.value = chip.dataset.name!;
  renderProbe();
  probe.focus();
});

// ---------------------------------------------------------------------------------------------
// Графики: простые линейные графики на SVG с подсказкой при наведении
// ---------------------------------------------------------------------------------------------

interface Series {
  key: 'train' | 'test' | 'loss';
  label: string;
  color: string;
  dashed?: boolean;
}

interface ChartOptions {
  series: Series[];
  domain: [number, number];
  ticks: number[];
  format: (value: number) => string; // Подписи оси
  formatValue?: (value: number) => string; // Значения в подписях линий и подсказке
}

const SVG_NS = 'http://www.w3.org/2000/svg';
const HEIGHT = 220;
const MARGIN = { top: 12, right: 128, bottom: 26, left: 48 };

const drawChart = (container: HTMLElement, options: ChartOptions) => {
  if (!model) return;
  const { history, epochs } = model;
  const width = Math.max(280, container.clientWidth || 560);
  const innerW = width - MARGIN.left - MARGIN.right;
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;
  const [y0, y1] = options.domain;
  const formatValue = options.formatValue ?? options.format;

  const x = (epoch: number) => MARGIN.left + (epoch / Math.max(1, epochs)) * innerW;
  const y = (value: number) => MARGIN.top + (1 - (value - y0) / (y1 - y0)) * innerH;

  const parts: string[] = [];

  // Сетка и подписи оси Y
  parts.push('<g class="grid">');
  options.ticks.forEach(t => {
    parts.push(
      `<line x1="${MARGIN.left}" x2="${MARGIN.left + innerW}" y1="${y(t)}" y2="${y(t)}"/>`,
    );
  });
  parts.push('</g>');
  options.ticks.forEach(t => {
    parts.push(
      `<text x="${MARGIN.left - 8}" y="${y(t) + 4}" text-anchor="end">${options.format(t)}</text>`,
    );
  });
  parts.push(
    `<line class="baseline" x1="${MARGIN.left}" x2="${MARGIN.left + innerW}" ` +
      `y1="${MARGIN.top + innerH}" y2="${MARGIN.top + innerH}"/>`,
  );

  // Подписи оси X: не больше ~8 штук
  const step = Math.max(1, Math.ceil(epochs / 8));
  for (let e = 0; e <= epochs; e += step) {
    parts.push(`<text x="${x(e)}" y="${HEIGHT - 6}" text-anchor="middle">${e}</text>`);
  }

  // Линии и подписи на концах
  const endLabels: { y: number; text: string }[] = [];
  options.series.forEach(s => {
    const points = history.map(h => `${x(h.epoch).toFixed(1)},${y(h[s.key]).toFixed(1)}`);
    parts.push(
      `<polyline points="${points.join(' ')}" fill="none" stroke="${s.color}" stroke-width="2" ` +
        `stroke-linejoin="round" stroke-linecap="round"${s.dashed ? ' stroke-dasharray="5 3"' : ''}/>`,
    );
    const last = history.at(-1)!;
    parts.push(
      `<circle cx="${x(last.epoch)}" cy="${y(last[s.key])}" r="4" fill="${s.color}" ` +
        `stroke="var(--panel)" stroke-width="2"/>`,
    );
    endLabels.push({ y: y(last[s.key]), text: `${s.label} ${formatValue(last[s.key])}` });
  });

  // Раздвигаем подписи, если они налезают друг на друга
  endLabels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < endLabels.length; i++) {
    endLabels[i].y = Math.max(endLabels[i].y, endLabels[i - 1].y + 15);
  }
  const lastX = x(history.at(-1)!.epoch);
  endLabels.forEach(l => {
    parts.push(`<text class="label" x="${lastX + 10}" y="${l.y + 4}">${l.text}</text>`);
  });

  parts.push(
    `<g class="hover" visibility="hidden"><line class="crosshair" y1="${MARGIN.top}" ` +
      `y2="${MARGIN.top + innerH}"/></g>`,
  );
  parts.push(
    `<rect class="hit" x="${MARGIN.left}" y="0" width="${innerW}" height="${HEIGHT}" fill="transparent"/>`,
  );

  let svg = container.querySelector<SVGSVGElement>('svg.plot');
  if (!svg) {
    svg = document.createElementNS(SVG_NS, 'svg');
    svg.classList.add('plot');
    svg.setAttribute('role', 'img');
    container.appendChild(svg);
  }
  svg.setAttribute('viewBox', `0 0 ${width} ${HEIGHT}`);
  svg.setAttribute(
    'aria-label',
    options.series.map(s => endLabels.find(l => l.text.startsWith(s.label))?.text).join(', '),
  );
  svg.innerHTML = parts.join('');

  // Подсказка: ближайшая эпоха к курсору
  let tooltip = container.querySelector<HTMLElement>('.tooltip');
  if (!tooltip) {
    tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.hidden = true;
    container.appendChild(tooltip);
  }
  const hover = svg.querySelector<SVGGElement>('.hover')!;
  const crosshair = hover.querySelector('line')!;
  const hit = svg.querySelector<SVGRectElement>('.hit')!;
  const tip = tooltip;
  const plot = svg;

  hit.addEventListener('pointermove', event => {
    const box = plot.getBoundingClientRect();
    const scale = width / box.width;
    const px = (event.clientX - box.left) * scale;
    const epoch = Math.round(((px - MARGIN.left) / innerW) * epochs);
    const h = history.find(item => item.epoch === epoch) ?? history.at(-1)!;

    crosshair.setAttribute('x1', String(x(h.epoch)));
    crosshair.setAttribute('x2', String(x(h.epoch)));
    hover.querySelectorAll('circle').forEach(c => c.remove());
    options.series.forEach(s => {
      const dot = document.createElementNS(SVG_NS, 'circle');
      dot.setAttribute('cx', String(x(h.epoch)));
      dot.setAttribute('cy', String(y(h[s.key])));
      dot.setAttribute('r', '4');
      dot.setAttribute('fill', s.color);
      dot.setAttribute('stroke', 'var(--panel)');
      dot.setAttribute('stroke-width', '2');
      hover.appendChild(dot);
    });
    hover.setAttribute('visibility', 'visible');

    tip.innerHTML =
      `<div><span class="k">эпоха</span> ${h.epoch}</div>` +
      options.series
        .map(s => `<div><span class="k">${s.label}</span> ${formatValue(h[s.key])}</div>`)
        .join('');
    tip.hidden = false;
    // Координаты подсказки — относительно контейнера графика
    const frame = container.getBoundingClientRect();
    const left = x(h.epoch) / scale + (box.left - frame.left);
    const flip = left > container.clientWidth - 150;
    tip.style.left = `${flip ? left - tip.offsetWidth - 12 : left + 12}px`;
    tip.style.top = `${box.top - frame.top + 8}px`;
  });
  hit.addEventListener('pointerleave', () => {
    hover.setAttribute('visibility', 'hidden');
    tip.hidden = true;
  });
};

const renderCharts = () => {
  if (!model) return;
  const { history } = model;

  // Точность: от 50% (или ниже, если сеть была хуже монетки) до 100%
  const minAcc = Math.min(...history.flatMap(h => [h.train, h.test]));
  const accFloor = Math.min(0.5, Math.floor(minAcc * 10) / 10);
  const accTicks: number[] = [];
  for (let t = accFloor; t <= 1.0001; t += 0.1) accTicks.push(Math.round(t * 10) / 10);
  drawChart(byId('accChart'), {
    series: [
      { key: 'train', label: 'обучение', color: 'var(--series-train)' },
      { key: 'test', label: 'тест', color: 'var(--series-test)', dashed: true },
    ],
    domain: [accFloor, 1],
    ticks: accTicks,
    format: v => `${Math.round(v * 1000) / 10}%`,
  });

  // Loss: от нуля до «круглого» значения выше максимума
  const maxLoss = Math.max(...history.map(h => h.loss));
  const lossStep = maxLoss > 1 ? 0.5 : maxLoss > 0.4 ? 0.2 : 0.1;
  const lossTop = Math.ceil(maxLoss / lossStep) * lossStep || lossStep;
  const lossTicks: number[] = [];
  for (let t = 0; t <= lossTop + 1e-9; t += lossStep) lossTicks.push(Math.round(t * 100) / 100);
  drawChart(byId('lossChart'), {
    series: [{ key: 'loss', label: 'loss', color: 'var(--series-train)' }],
    domain: [0, lossTop],
    ticks: lossTicks,
    format: v => (Math.abs(v) < 1e-9 ? '0' : v.toFixed(1)),
    formatValue: v => v.toFixed(3),
  });
};

new ResizeObserver(() => renderCharts()).observe(byId('accChart'));

// Страница сразу открывается с обученной сетью
train();
