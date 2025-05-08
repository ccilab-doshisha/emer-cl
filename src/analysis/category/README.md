# Emotional model
We use [Using circular models to improve music emotion recognition](https://ieeexplore.ieee.org/document/8567988) I. Dufour and G. Tzanetakis, IEEE Trans. Affect. Comput. 2018., for the discretisation of Valence and Arousal. 
Specifically, `Using circular models to improve music emotion recognition' is an emotional modeling method that improves on Russel's circular model.
In addition, `Using circular models to improve music emotion recognition` uses a circular representation of 40 emotion words in VA space.
We use this to discretise the VA values and then use them in several detailed analyses.

Moreover, our current CategoricalModel is implemented in such a way that only the range of VA [-1, 1] can be converted, and it does not support VA of [0, 1].
(It is not taken into account in the angle calculation).
This means that only the detailed analysis of the DEAM is supported.

