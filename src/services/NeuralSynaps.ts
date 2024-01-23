import type NeuralNeuron from './NeuralNeuron';

class NeuralSynaps {
  #weight: number;
  #inputNeuron: NeuralNeuron;
  #outputNeuron: NeuralNeuron;
  #bias: number = Math.random();

  public constructor(
    inputNeuron: NeuralNeuron,
    outputNeuron: NeuralNeuron,
    weight: number,
    bias: number,
  ) {
    this.#inputNeuron = inputNeuron;
    this.#outputNeuron = outputNeuron;
    this.#weight = weight;
    this.#bias = bias;
  }

  public getWeight() {
    return this.#weight;
  }

  public getBias() {
    return this.#bias;
  }

  public getInputNeuron() {
    return this.#inputNeuron;
  }

  public getOutputNeuron() {
    return this.#outputNeuron;
  }
}

export default NeuralSynaps;
