from setuptools import setup

setup(
    name='pytorch-train-utils',
    version='0.0.1',
    description='utils for pytorch training',
    url='git@github.com:carloalbertobarbano/pytorch-train-utils.git',
    author='Carlo Alberto Barbano',
    author_email='carlo.alberto.barbano@outlook.com',
    license='unlicense',
    packages=['pytorchtrainutils'],
    zip_safe=False,
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'pandas',
        'tqdm'
    ]
)
