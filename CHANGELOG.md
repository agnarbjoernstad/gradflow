# CHANGELOG


## v0.3.0 (2025-03-16)

### Bug Fixes

* fix: remove main ([`83b6450`](https://github.com/agnarbjoernstad/gradflow/commit/83b64502168301b2accb2fccb9fb2591921c569e))

* fix: do not print dependencies when plotting ([`8f065ca`](https://github.com/agnarbjoernstad/gradflow/commit/8f065ca0db610e73619fe49210db3ffc386b2f4a))

* fix: ensure only gradients for Tensors are computed ([`d45141b`](https://github.com/agnarbjoernstad/gradflow/commit/d45141bf3af57806ea19e57eb2018470353046c4))

* fix: reorganize imports ([`c0b6e82`](https://github.com/agnarbjoernstad/gradflow/commit/c0b6e8272da4867267b6636e5cfa9983b6f6513f))

* fix: ensure that inplace operations are just that ([`97c7460`](https://github.com/agnarbjoernstad/gradflow/commit/97c7460d71d322bd7423dd0fe04757445dd91eea))

* fix: handle broadcast of operations in backward ([`e81c3c8`](https://github.com/agnarbjoernstad/gradflow/commit/e81c3c8565f153269f773e76961fa520222821b1))

### Continuous Integration

* ci: ignore formatting handled by black ([`dd4a515`](https://github.com/agnarbjoernstad/gradflow/commit/dd4a515ebed2a424a3e481940e18e7bfbd8d54de))

* ci: rename lint job ([`a7b6090`](https://github.com/agnarbjoernstad/gradflow/commit/a7b6090ce2cd899a447bda66a116d21492cf546f))

### Documentation

* docs: update example subsubtitles ([`a997056`](https://github.com/agnarbjoernstad/gradflow/commit/a99705679e90c1b947928a7a4f4389a67b76ae17))

* docs: zero gradient in the example optimization problem ([`a319a34`](https://github.com/agnarbjoernstad/gradflow/commit/a319a3421cfe41893bbef232c2f739d8ca7b9d83))

* docs: create roadmap ([`a597bf2`](https://github.com/agnarbjoernstad/gradflow/commit/a597bf215db0824862418084c87063f151d886e8))

### Features

* feat: train sine function ([`6d80158`](https://github.com/agnarbjoernstad/gradflow/commit/6d801582c49f387003642e588d7ade21e5bb746b))

* feat: train on the mnist dataset ([`fefcdee`](https://github.com/agnarbjoernstad/gradflow/commit/fefcdee25a9437b1ca563ea791e36d4f018adb3d))

* feat: implement the Adam optimizer ([`40c0076`](https://github.com/agnarbjoernstad/gradflow/commit/40c0076e4eb0b49b476027bd16b0b7367038f4f0))

* feat: implement linear layer, sequential model and related utilities ([`e973bf5`](https://github.com/agnarbjoernstad/gradflow/commit/e973bf5957f3d2e6035c32305958b61e51558455))

### Performance Improvements

* perf: minor performance increase in topological sort ([`a1f571b`](https://github.com/agnarbjoernstad/gradflow/commit/a1f571b67a7859cad4b9c0c6d2c05e8c23d29ff0))

* perf: remove unnecessary multiplications in derivatives ([`53f7a62`](https://github.com/agnarbjoernstad/gradflow/commit/53f7a62e65880c1a3623702531abd40ce0e0da1e))

### Unknown

* Merge pull request #11 from agnarbjoernstad/nn_linear

Implement basic nn functionality ([`c531450`](https://github.com/agnarbjoernstad/gradflow/commit/c531450ee24e24683b60a63c016ef19927f88de0))


## v0.2.0 (2025-03-06)

### Code Style

* style: remove redundant comments and commented code ([`6a3fdc8`](https://github.com/agnarbjoernstad/gradflow/commit/6a3fdc8e1502c51d171fab08e9cd89a248f542b6))

### Continuous Integration

* ci: add black for linting ([`0645d89`](https://github.com/agnarbjoernstad/gradflow/commit/0645d89aa0b89a4cc5f9349efae7ad04e816ae94))

### Documentation

* docs: add numpy badge ([`44621d4`](https://github.com/agnarbjoernstad/gradflow/commit/44621d45e0fc18609cff8f4e8aa69139dca2fa2c))

* docs: add test badge ([`3411688`](https://github.com/agnarbjoernstad/gradflow/commit/34116885ab058ff8860197bb28cd951acd973b99))

* docs: provide example code for the library ([`d5ae9bc`](https://github.com/agnarbjoernstad/gradflow/commit/d5ae9bce008d9b8b323ca917b799d2bd74898adf))

### Features

* feat: make repository pip installable ([`a5641e3`](https://github.com/agnarbjoernstad/gradflow/commit/a5641e3d4111afbbd3a13e70a829a4fca10e4b0f))

### Unknown

* Merge pull request #10 from agnarbjoernstad/pip_install

Pip install ([`ee462b3`](https://github.com/agnarbjoernstad/gradflow/commit/ee462b3f1a5d60ed027bde09c8ad99b41386af3d))


## v0.1.0 (2025-03-03)

### Bug Fixes

* fix: gitignore .idea, .vscode, __pycache__ and venv ([`778cc5f`](https://github.com/agnarbjoernstad/gradflow/commit/778cc5ff8f8f7e28c58d57327a27bc0d7b3603d5))

### Continuous Integration

* ci: run pytests ([`8c35f2b`](https://github.com/agnarbjoernstad/gradflow/commit/8c35f2b65c3fcf77d623e44223cd6b2832e6bc9a))

* ci: add custom configuration for flake8 ([`9998636`](https://github.com/agnarbjoernstad/gradflow/commit/9998636f1122a3d5919e3c6af80e176a3e1a9ee4))

### Features

* feat: implement tensor class for tracking of gradients ([`434ecab`](https://github.com/agnarbjoernstad/gradflow/commit/434ecab6775ec967f977b0144996e81537b09916))

### Unknown

* Merge pull request #9 from agnarbjoernstad/backpropagation

Backpropagation ([`18ca431`](https://github.com/agnarbjoernstad/gradflow/commit/18ca431672285c641d87fedbe1463ab1f4890d6c))


## v0.0.0 (2024-12-05)

### Chores

* chore: set up semantic versioning ([`b8b66f4`](https://github.com/agnarbjoernstad/gradflow/commit/b8b66f421421c3085ec314ba4135c9b7eca96890))

### Continuous Integration

* ci: lint python ([`baa82ac`](https://github.com/agnarbjoernstad/gradflow/commit/baa82ac80b2364a7a59855a68e1d6234b51941a0))

* ci: lint commits ([`01a0a52`](https://github.com/agnarbjoernstad/gradflow/commit/01a0a529c9d9c8649f3d768ee6f3a01837eef807))

### Documentation

* docs: initial commit ([`c5a5f4b`](https://github.com/agnarbjoernstad/gradflow/commit/c5a5f4b7f09d6daa185855fe0b901cc170e8df51))

### Unknown

* Merge pull request #8 from Agnar22/set_up_repository

Set up repository ([`3048c2d`](https://github.com/agnarbjoernstad/gradflow/commit/3048c2d9b5c3f4961a445491a49608348b3f95dd))
