<img src="https://raw.githubusercontent.com/JuliaMolSim/DFTK.jl/master/docs/logo/DFTK_750x250.png" alt="dftk logo" height="100px" />

# Density-functional toolkit

| **Documentation**                                                                 | **Build Status**                                |  **License**                     |
|:--------------------------------------------------------------------------------- |:----------------------------------------------- |:-------------------------------- |
| [![][docs-img]][docs-url] [![][ddocs-img]][ddocs-url] [![][slack-img]][slack-url] | [![][ci-img]][ci-url] [![][ccov-img]][ccov-url] | [![][license-img]][license-url]  |

[ddocs-img]: https://img.shields.io/badge/docs-dev-blue.svg
[ddocs-url]: https://docs.dftk.org/dev

[docs-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-url]: https://docs.dftk.org/stable

[slack-img]: https://img.shields.io/badge/chat-on_slack-808493.svg?logo=slack
[slack-url]: https://join.slack.com/t/juliamolsim/shared_invite/zt-tc060co0-HgiKApazzsQzBHDlQ58A7g

[ci-img]: https://github.com/JuliaMolSim/DFTK.jl/workflows/CI/badge.svg?branch=master&event=push
[ci-url]: https://github.com/JuliaMolSim/DFTK.jl/actions

[ccov-img]: https://codecov.io/gh/JuliaMolSim/DFTK.jl/branch/master/graph/badge.svg?token=A23M0VZ8PQ
[ccov-url]: https://codecov.io/gh/JuliaMolSim/DFTK.jl

[license-img]: https://img.shields.io/github/license/JuliaMolSim/DFTK.jl.svg?maxAge=2592000
[license-url]: https://github.com/JuliaMolSim/DFTK.jl/blob/master/LICENSE

The density-functional toolkit, **DFTK** for short, is a library of
Julia routines for experimentation with plane-wave
density-functional theory (DFT), as implemented in much larger
production codes such as [Abinit](https://www.abinit.org/),
[Quantum Espresso](http://quantum-espresso.org/) and
[VASP](https://www.vasp.at/).

The unique feature of this code is its emphasis on simplicity and flexibility
with the goal of facilitating methodological development and
interdisciplinary collaboration.
In about 7k lines of pure Julia code we already support a
[sizeable set of features](https://juliamolsim.github.io/DFTK.jl/dev/index.html#package-features-1),
after just about two years of development.
Our performance is of the same order of magnitude as established packages.

For getting started with DFTK, see our documentation:
- [Installation instructions](https://juliamolsim.github.io/DFTK.jl/dev/guide/installation/)
- [Tutorial](https://juliamolsim.github.io/DFTK.jl/dev/guide/tutorial/)
- [Examples](https://juliamolsim.github.io/DFTK.jl/dev/#example-index-1)

Note that at least **Julia 1.6** is required.

## Support and citation
DFTK is mostly developed as part of academic research.
Parts of DFTK have also been discussed in published papers.
If you use our code as part of your research, teaching or other activities,
we would be grateful if you cite them as appropriate.
See the [CITATION.bib](CITATION.bib) in the root of this repo for relevant references.
As a software DFTK can also be cited via [![DOI](https://zenodo.org/badge/181734238.svg)](https://zenodo.org/badge/latestdoi/181734238).

## Funding
This project has received funding from
the [Institute of computing and data sciences (ISCD, Sorbonne Université)](https://iscd.sorbonne-universite.fr/),
[École des Ponts ParisTech](https://enpc.fr), [Inria Research Centre Paris](https://www.inria.fr/fr/centre-inria-de-paris),
[RWTH Aachen University](https://rwth-aachen.de/),
and from the European Research Council (ERC) under the European Union's Horizon 2020 research and
innovation program ([grant agreement No 810367](https://cordis.europa.eu/project/id/810367)).

## Contributing
If you stumble across issues in using DFTK
or have suggestions for future developments
we are more than happy to hear about it.
In this case please [open an issue](https://github.com/JuliaMolSim/DFTK.jl/issues)
or contact us ([@mfherbst](https://github.com/mfherbst)
and [@antoine-levitt](https://github.com/antoine-levitt)) directly.

Contributions to the code in any form is very welcome,
just [submit a pull request](https://github.com/JuliaMolSim/DFTK.jl/pulls)
on github. If you want to contribute but are unsure where to start, take a look
at the list of issues tagged [good first issue](https://github.com/JuliaMolSim/DFTK.jl/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
(relatively easy tasks suitable for newcomers) or [help wanted](https://github.com/JuliaMolSim/DFTK.jl/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
(more sizeable but well-defined and isolated).
Don't hesitate to ask for help, through github, email or the [JuliaMolSim slack][slack-url].
