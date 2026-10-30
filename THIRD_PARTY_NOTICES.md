# Third-party notices

## GNN decoder architecture and checkpoints (Lange et al.)

`gnn_decoder.py` contains the network architecture copied from
https://github.com/LangeMoritz/GNN_decoder (`src/gnn_models.py`), so that the
released checkpoints load unchanged. `fetch_gnn_models.py` downloads nine of
those checkpoints from commit `15d8443bedf7862ef72ec1b5239ecd01163eb8d6`; they
are not redistributed here. Reference: M. Lange et al., "Data-driven decoding of
quantum error correcting codes using graph neural networks", Phys. Rev. Research
7, 023181 (2025).

```
MIT License

Copyright (c) 2023 Moritz Lange

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## PyMatching

`pymatching_patch/patch_pymatching.py` modifies the source distribution of
PyMatching 2.4.0 (https://github.com/oscarhiggott/PyMatching, Apache License
2.0, Copyright Oscar Higgott) by adding one method; the patched source is not
redistributed here. PyMatching is used unmodified everywhere else.
