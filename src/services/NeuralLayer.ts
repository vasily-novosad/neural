import type NeuralNeuron from './NeuralNeuron';

class NeuralLayer {
  #neurons: NeuralNeuron[] = [];
  #layerType: LayerType;

  public constructor(layerType: LayerType) {
    this.#layerType = layerType;
  }

  public addNeuron(neuron: NeuralNeuron) {
    this.#neurons.push(neuron);

    return this;
  }

  public getNeurons() {
    return this.#neurons;
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
