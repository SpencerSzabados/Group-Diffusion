from setuptools import setup

setup(
    name="group-diffusion",
    py_modules=["model", "evaluations"],
    install_requires=[
        "steuptools==58.2.0",
        "wheel",
        "Cython",
        "packaging",
        "mpi4py",
        "blobfile>=1.0.5",
        "torch",
        "torchvision",
        "tqdm",
        "numpy",
        "scipy",
        "pandas",
        "piq==0.7.0",
        "joblib==0.14.0",
        "albumentations==0.4.3",
        "lmdb",
        "clip @ git+https://github.com/openai/CLIP.git",
        "flash-attn==0.2.8",
        "pillow",
    ],
)
