/* eslint-disable arrow-body-style */
import NeuralActivator, { ActivatorName, ActivatorFunction } from './NeuralActivator';
import NeuralLayer from './NeuralLayer';
import NeuralNeuron from './NeuralNeuron';

/**
 * Forward propagate neural network
 */
class NeuralNetwork {
  #layers: NeuralLayer[] = [];
  #activators: ActivatorCollection = {
    sigmoid: NeuralActivator.sigmoid,
    linear: NeuralActivator.linear,
    tanh: NeuralActivator.tanh,
  };

  /**
   * Returns all layers stack
   */
  public getLayers() {
    return this.#layers;
  }

  /**
   * Append layer to layers stack
   */
  public addLayer(layer: NeuralLayer) {
    this.#layers.push(layer);
  }

  /**
   * Run and calculate all network
   */
  public calculateNetwork(values: number[]): number[] {
    const result: number[] = [];

    this.#layers.forEach(layer => {
      // Phase 1. Fill the Input neurons
      if (layer.isInputLayer()) {
        // Check neurons count per values
        if (values.length !== layer.getNeurons().length) {
          throw new Error(
            'the number of neurons in the input layer must correspond to the number of transmitted values',
          );
        }

        // Fill the Input neurons
        layer.getNeurons().forEach((neuron, neuronIndex) => {
          neuron.activate(values[neuronIndex] * neuron.getWeight());
        });
      }

      // Phase 2. Pass to hidden layers
      if (layer.isHiddenLayer()) {
        layer.getNeurons().forEach(neuron => {
          neuron.getSynapses().forEach(synaps => {
            synaps.getOutputNeuron().activate(neuron.getActivationsSum() * synaps.getWeight());
          });
        });
      }

      // Phase 3. Output layer
      if (layer.isOutputLayer()) {
        layer.getNeurons().forEach(neuron => {
          result.push(neuron.getActivationsSum());
        });
      }
    });

    return result;
  }

  /**
   * Generate the matrix\
   * Example:
   * ```ts
   * const network = new NeuralNetwork()
   *  .generateNetworkMatrix([
   *   20, // count of neurons on input layer
   *   5,  // count of neurons on hidden layer
   *   5,  // count of neurons on hidden layer
   *   3,  // count of neurons on output layer
   * ]);
   * ```
   */
  public generateNetworkMatrix(neuronsPerLayer: number[]) {
    const matrix: NeuralWeighMatrix = [];

    neuronsPerLayer.forEach((neuronCountOnLayer, layerIndex) => {
      const nextLayerNeuronsCount = neuronsPerLayer[layerIndex + 1];
      const neurons: MatrixNeuron[] = [];
      [...new Array(neuronCountOnLayer).keys()].forEach(() => {
        const isLastLayer = typeof nextLayerNeuronsCount === 'undefined';

        const weights: number[] = [];

        if (!isLastLayer) {
          [...new Array(nextLayerNeuronsCount).keys()].forEach(() => {
            weights.push(parseFloat((Math.random() * 1).toFixed(2)));
          });
        }

        neurons.push([0.1, weights]);

        matrix.push([0.2, 'sigmoid', neurons]);
      });
    });

    return matrix;
  }

  public loadNetworkFromMatrix(neuralMatrix: NeuralWeighMatrix) {
    this.destroy();

    neuralMatrix.forEach((matrixLayer, matrixLayerIndex) => {
      const [layerBias, layerActivator, matrixSynapses] = matrixLayer;
      const layerType =
        matrixLayerIndex === 0
          ? 'input-layer'
          : matrixLayerIndex === neuralMatrix.length - 1
            ? 'output-layer'
            : 'hidden-layer';

      const layer = new NeuralLayer(layerType, layerBias);
      matrixSynapses.forEach(matrixSynaps => {
        const [neuronbaseWeight] = matrixSynaps;
        const neuron = new NeuralNeuron(neuronbaseWeight, this.getActivatorFn(layerActivator));

        layer.addNeuron(neuron);
      });

      this.addLayer(layer);
    });

    // synapses
    this.#layers.forEach((layer, layerIndex) => {
      if (layerIndex === this.#layers.length - 1) {
        return;
      }

      const nextLayer = this.#layers[layerIndex + 1];
      layer.getNeurons().forEach((neuron, neuronIndex) => {
        const [layerBias, _layerActivatorName, matrixSynapses] = neuralMatrix[layerIndex];
        const [_neuronbaseWeight, synapsWeights] = matrixSynapses[neuronIndex];

        synapsWeights.forEach((synapsWeight, synapsIndex) => {
          const nextLayerNeuron = nextLayer.getNeurons()[synapsIndex];
          if (!nextLayerNeuron) {
            throw new Error(`Neuron for synaps index ${synapsIndex} does not found`);
          }
          neuron.createSynaps(nextLayerNeuron, synapsWeight, layerBias);
        });
      });
    });

    return this;
  }

  private getActivatorFn(activatorName: ActivatorName) {
    return this.#activators[activatorName];
  }

  public destroy() {
    this.#layers = [];
  }
}

export default NeuralNetwork;

type Bias = number;
type Weight = number;
type MatrixNeuron = [Weight, Weight[]];
type MatrixLayer = [Bias, ActivatorName, MatrixNeuron[]];
export type NeuralWeighMatrix = MatrixLayer[];

export type ActivatorCollection = {
  [v in ActivatorName]: ActivatorFunction;
};
