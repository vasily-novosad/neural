class NeuralActivator {
  /**
   * Logistic function [0,1]\
   * Returns a value in the range from 0 to 1
   */
  public static sigmoid(x: number) {
    return 1 / (1 + Math.E ** -x);
  }

  /**
   * Hyperbolic tangent [-1,1]\
   * Returns a value in the range from -1 to 1
   */
  public static tanh(x: number) {
    const exp = Math.E ** (2 * x);

    return (exp - 1) / (exp + 1);
  }

  /**
   * Linear function [0, x]\
   * Returns a value in the range from 0 to x
   */
  public static linear(x: number) {
    return x > 0 ? x : 0;
  }

  /**
   * A multi-variable logistic function (softmax)\
   * Converts a vector of K real numbers into a probability distribution of K possible outcomes
   */
  public static softmax(logits: readonly number[]) {
    const maxLogit = logits.reduce((a, b) => Math.max(a, b), -Infinity);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const denom = scores.reduce((a, b) => a + b);

    return scores.map(s => s / denom);
  }
}

export default NeuralActivator;
