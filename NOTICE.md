# Notice

`keras-hexagdly` is a derivative work of
[HexagDLy](https://github.com/ai4iacts/hexagdly) by Tim Lukas Holch and
Constantin Steppa (ai4iacts), originally developed for hexagonal convolution
and pooling on PyTorch in the context of Imaging Atmospheric Cherenkov
Telescope analysis with H.E.S.S.

This package re-implements the same `HexBase` sub-kernel decomposition and
addressing arithmetic on top of [Keras 3](https://keras.io) (TensorFlow / JAX
/ PyTorch backends), with a channels-last tensor layout. It is bit-for-bit
numerically equivalent to the original for every layer HexagDLy provides
(`Conv2d`, `Conv3d`, `Conv2d_CustomKernel`, `Conv3d_CustomKernel`,
`MaxPool2d`, `MaxPool3d`) -- see `tests/test_vs_pytorch_hexagdly.py`, which
checks this directly against the upstream PyTorch `hexagdly` PyPI package.

Two functionalities are new in this port and have no equivalent in upstream
HexagDLy:

- `share_neighbors`: ties the weights of a hexagonal kernel by ring (ring 0 =
  center, ring *r* = the 6*r* cells at hex-distance *r*), instead of giving
  every cell its own weight.
- `depth_padding` (`Conv3d` only): `"same"` zero-pads the depth/time axis so
  the temporal kernel is centred and output depth equals input depth, instead
  of HexagDLy's `"valid"`-only behaviour.

This work was developed as part of the SST-1M Collaboration / HEPIA TDSCAN
triggering project.

## License

Both the original HexagDLy code and this port are distributed under the MIT
license; see [LICENSE](LICENSE). The original copyright notice (Copyright (c)
2018 ai4iacts) is preserved alongside the copyright notice for this port, as
required by the MIT license.

## Citing

If you use this package, please cite the original HexagDLy paper:

```bibtex
@article{hexagdly_paper,
    title = "HexagDLy—Processing hexagonally sampled data with CNNs in PyTorch",
    author = "Constantin Steppa and Tim L. Holch",
    journal = "SoftwareX",
    volume = "9",
    pages = "193 - 198",
    year = "2019",
    issn = "2352-7110",
    doi = "https://doi.org/10.1016/j.softx.2019.02.010",
    url = "https://www.sciencedirect.com/science/article/pii/S2352711018302723",
}
```
