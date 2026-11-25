"""Pinned Hugging Face release inventory for the Unified Road Defect Dataset."""

REPOSITORY = "TamAko783/Unified_Road_Defect_Dataset"
REVISION = "7ba2309c243f989d72a6e9a8a9aca902871232ce"

# Hugging Face repository tree at REVISION. LFS digests are SHA-256.
ARCHIVES = {
    "data/train_a.tar.gz": (
        928438317, "e9589381b6bb3ffa48bc63ae2ec1b98a165da0673e884f2b9cd7d1ed7dc55849",
    ),
    "data/train_b.tar.gz": (
        5141618844, "b3d8ace2eac459fbf8fb98461e9da3a225dd3ef9c54fad95ffeef888e2c821c2",
    ),
    "data/val.tar.gz": (
        1067597682, "08ddcf00c2ea9f2463c9508803e3c004168a16676c5bbcb7dc11e043518c9a52",
    ),
}
SIDECARS = {
    "rdd_merged.yaml": (250, "7be002e87a948614c435ea815bcc0adc53a1463d"),
    "stats.json": (458, "efab1ac5031c8c3ca3551f4b4accf966ccf52951"),
    "README.md": (6049, "f61775f1bb3bd45dcbcbefd631939c0aad416430"),
}
