import NeuralSynaps from './NeuralSynaps';
import NeuralActivator from './NeuralActivator';

class NeuralNeuron {
  #synapses: NeuralSynaps[] = [];
  #weight: number = 0;
  #activations: number[] = [];

  public constructor(weight: number) {
    this.#weight = weight;
  }

  public activate(value: number) {
    this.#activations.push(value);
  }

  public getActivationsSum() {
    const sum = this.#activations.reduce((acc, value) => acc + value, 0);

    return NeuralActivator.sigmoid(sum);
  }

  public createSynaps(neuron: NeuralNeuron, weight: number, bias: number) {
    if (weight < 0) {
      throw new Error('The weight cannot be less than 0');
    }

    if (weight > 1) {
      throw new Error('The weight cannot be more than 1');
    }

    this.#synapses.push(new NeuralSynaps(this, neuron, weight, bias));

    return this;
  }

  public getWeight() {
    return this.#weight;
  }

  public getSynapses() {
    return this.#synapses;
  }
}

export default NeuralNeuron;

export type ActivatorName = 'linear' | 'sigmoid' | 'tanh';
