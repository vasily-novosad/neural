class NeuralComposeErrors {
  /**
   * Mean Squared Error
   */
  public static mse(errors: readonly number[], rights: readonly number[]) {
    const sum = errors.reduce((acc, i) => acc + (rights[i] - errors[i]) ** 2, 0);

    return sum / errors.length;
  }
}

export default NeuralComposeErrors;
