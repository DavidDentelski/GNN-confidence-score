"""
Download the pretrained GNN checkpoints used in the paper.

The networks are those released with M. Lange et al., "Data-driven decoding of
quantum error correcting codes using graph neural networks", Phys. Rev. Research
7, 023181 (2025), https://github.com/LangeMoritz/GNN_decoder (MIT License,
Copyright (c) 2023 Moritz Lange). The nine checkpoints of the paper's
configurations are fetched from a pinned commit of that repository into
runs.MODEL_DIR (./gnn_models, or $GNN_CONFIDENCE_MODELS) and verified by SHA-256.
They are needed only to generate data (generate_shot_data.py, make_exact_d3.py,
compare_gap_constructions.py), not to rebuild the figures and tables.

    python fetch_gnn_models.py
"""

import hashlib
import os
import urllib.request

import runs

REPO = "https://raw.githubusercontent.com/LangeMoritz/GNN_decoder"
COMMIT = "15d8443bedf7862ef72ec1b5239ecd01163eb8d6"
SHA256 = {
    "d3_d_t_3.pt": "f90d7a2786070397697d1e9f514f96f66b822fe5be8bb9830652533e1ef9d2ae",
    "d5_d_t_5.pt": "9a0fc6607a8c0ea3aa73dd243223460c7037cf76884228338928d8a26ae0a7cf",
    "d5_d_t_7.pt": "ce8f120e5b9cc53c66adb8ebad4884da57bf104f2d0c0fe3b4d5ed9d1ce98b88",
    "d5_d_t_9.pt": "fb9fed2a347ba8e5a40006a5b2b29b38af63909c2624f192fbeb1d614651ba01",
    "d5_d_t_11.pt": "6fce4a27a8eb28d0ad9a51e5c87aaf15ac4e053be49a85d6e662c558bc26f1f7",
    "d7_d_t_7.pt": "14bcfca05757064d9931f45e315023a020f007a7c557bad40d051c10544bda9b",
    "d7_d_t_9.pt": "c05f3922cd02f6ca13deb5db6bc45ed196e0c4faeb0c0f29e3c716ca6d32f4a0",
    "d7_d_t_11.pt": "d8ee5984e45cfd5dc71c4a6d5d79a749da85293b04d0dd61320cae8bebe6f370",
    "d9_d_t_9.pt": "8b3d1af756a9feef89576682720bdee1283fefa8a18a367913baab1a1c5ab2c2",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    os.makedirs(runs.MODEL_DIR, exist_ok=True)
    for name, digest in SHA256.items():
        path = os.path.join(runs.MODEL_DIR, name)
        if os.path.exists(path) and sha256(path) == digest:
            print(f"{name}: present")
            continue
        url = f"{REPO}/{COMMIT}/models/circuit_level_noise/{name.split('_')[0]}/{name}"
        print(f"{name}: downloading {url}")
        urllib.request.urlretrieve(url, path + ".part")
        if sha256(path + ".part") != digest:
            os.remove(path + ".part")
            raise SystemExit(f"{name}: SHA-256 mismatch")
        os.replace(path + ".part", path)
    print(f"all checkpoints verified in {runs.MODEL_DIR}")


if __name__ == "__main__":
    main()
