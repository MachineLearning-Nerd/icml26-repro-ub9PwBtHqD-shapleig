"""Fetch the exact 30 ImageNet/ViT-9 games used by the ShaplEIG paper.

The current shapiq main branch removed these historical assets, so every URL is
pinned to the last parent commit that contained them. SHA-256 checks prevent a
silent data change. The files contain all 2**9 coalition values, precomputed
from costly ViT inference exactly as allowed by the paper's protocol.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen

COMMIT = "799cfd0f2c32f17446130247a7ac3519e68cce82"
BASE = (
    "https://raw.githubusercontent.com/mmschlk/shapiq/"
    f"{COMMIT}/data/precomputed_games/ImageClassifier_Game/9"
)
SHA256 = {
    1: "c8ea8c7a24b71432e96557035a78e634b61f98d56a3977228a59100843efb206",
    2: "c1f01f7c8e065ab85e73113550007137d7cc344fa315de4e35b32dcc94100ece",
    3: "b12268d3e39bab74b2c68d3f6a4cb34bb2a0ba06fb5b7ddccbbef95280e236b1",
    4: "31974d6fb290eaafaa72f1d5e36cef082b4a55a6382b8f9d61d75b659f9b0546",
    5: "e8c385d7ee172ffee3231a661be2bbf1068cce102c9e60cad091ddf9ed2c3710",
    6: "b843c0884ac2fa236964980ac61981f8afc85f158d5b95a851685404b0b4768f",
    7: "0c3fdb954775e41fb7b0e6b4ba74decbacfd6b1f3c538e024899901512b56726",
    8: "5afb4dd2ee02deb7dab6197c631e403b9078e4767f76d0b7ece819486bab0f4a",
    9: "55411dae679c302b40028c9a292fa7738c2159d1cf059629801e74c61b8221c9",
    10: "9d63121b9dfc7a7d4312785f6a0a680920fe7df979514e6b2c5885b91fce4035",
    11: "7c8d308327f070485789251de5734463d2a4acbc9b4f08cba71407278ca918a6",
    12: "7354c562b1947c11444b46fbdfb60771040bcae0a1140b17019de5a02250ecaa",
    13: "a6102b7f01335766ffd43c2dfec0b85bd571c1408baa09254c9ac44e831da38b",
    14: "93c8d82952b59b4c65d944116fb0d9ec6817ae2c5c1a4a8afb46da26589dcfcc",
    15: "9365985f6d1e8fcda2e66a5f199c98f87ec155bd474e5eea0fcb1932ac30a80a",
    16: "0564500762f6897ffd2febad3c2e118bb6511778067ce3b517983ab5e8c5d659",
    17: "55c166c5309cdf17d589201ed25c4f4e33faaddfba709fd2fed7d2460f4bce7d",
    18: "b26a535e2b0909682418aa0ade274f333eabb2af904dd60b8b48f441bca4aec9",
    19: "34668fd7b60eb1764423702a95dad87cff34629b01ff64d9bbe8b388742d546c",
    20: "2cb01699e6e9ef37a47738d500ccb5b286e0004e1bdf24e1c25074e5bad44021",
    21: "b0a7923492def3f14e9817664603473c2cf9938d91f405f4d3faff382c3a71f0",
    22: "58e89f5a1d319a558e7a2f38a01d755f11a072e4d8a43d3462dd4579aeff5e55",
    23: "cf12867cf63b9abf1466db5057573f44cf4488ed3d87eed8acbbe8f93b21ee17",
    24: "743c6dfd25c2e44cee10a566ff5fa17b98fb6216e013befe2936644f9ec3a153",
    25: "330bd55076bf4f21ba2a68faa48d010f3ed476662e59d2905a15179c743d25e1",
    26: "5d70be14b8cde253ed42d2f1c1c1d6c4177907820a527ac33fa36150f83e4585",
    27: "df03f1604cef78fd2fb867243257fd858022e7ab83b414bf70ad46ae5cf6e9b2",
    28: "1279b6d1726ad624d378b08150e2d50056aca3e929360dcecf27748d216f392c",
    29: "f9b8febd2a9dc82b10c17275ec4e62f7283eb80a793b7427817f64c63cd5ad65",
    30: "14545393c5f9921424584ed72e2bafdb315f73a372659866043da102b89d2070",
}


def fetch(root: Path) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed, digest in SHA256.items():
        name = f"model_name=vit_9_patches_{seed}.npz"
        path = root / name
        payload = path.read_bytes() if path.exists() else urlopen(f"{BASE}/{name}").read()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != digest:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual}")
        if not path.exists():
            path.write_bytes(payload)
        paths.append(path)
    return paths


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[2] / "data" / "vit9"
    print(f"verified {len(fetch(target))} pinned games in data/vit9")
