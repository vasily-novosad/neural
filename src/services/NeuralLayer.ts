import type NeuralNeuron from './NeuralNeuron';

class NeuralLayer {
  #neurons: NeuralNeuron[] = [];
  #layerType: LayerType;
  #bias: number;

  public constructor(layerType: LayerType, bias: number) {
    this.#layerType = layerType;
    this.#bias = bias;
  }

  public addNeuron(neuron: NeuralNeuron) {
    this.#neurons.push(neuron);

    return this;
  }

  public getNeurons() {
    return this.#neurons;
  }

  public getBias() {
    return this.#bias;
  }

  public isInputLayer() {
    return this.#layerType === 'input-layer';
  }

  public isHiddenLayer() {
    return this.#layerType === 'hidden-layer';
  }

  public isOutputLayer() {
    return this.#layerType === 'output-layer';
  }
}

export type LayerType = 'input-layer' | 'hidden-layer' | 'output-layer';

export default NeuralLayer;
