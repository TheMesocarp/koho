# koho

Sheaf neural networks and other exotic diffusion models, built on `candle`. 

### What is this?

This repository implements an extension of [sheaf neural networks](https://arxiv.org/abs/2012.06333) aimed to handle diffusion over k-cells in a cell complex. This implementation differs from typical sheaf neural network implementations, that focus on the 0th cohomology and frequently restrict to graphs based settings, by following the general [spectral theory of cellular sheaves](https://arxiv.org/abs/1808.01513) to construct the hodge laplacian directly from coboundaries, and subsequently the restriction maps.

## Why?

so we can do diffusion over k-dimensional cells in a cellular sheaf!

## Contributing

Contributors are welcome and greatly appreciated! Please feel free to submit a Pull Request or claim an issue youd like to work on. For major changes, please open an issue first to discuss what you would like to change. If you would like to work more closely with Mesocarp on other projects as well, please email me at sushi@fibered.cat, would love to chat!

## License

This project is licensed under the AGPL-3.0 copyleft license - see the [LICENSE](LICENSE) file for details.
