import setuptools

setuptools.setup(
    name='spectralwaste_segmentation',
    version=0.1,
    author="",
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=[
        'numpy',
        'matplotlib',
        'imageio',
        'easydict',
        'scikit-image',
        'scikit-learn',
        'pandas',
        'opencv-python',
        'torch',
        'torchvision',
        'torchmetrics',
        'wandb'
    ],
)
