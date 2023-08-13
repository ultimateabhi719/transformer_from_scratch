from setuptools import setup

setup(
    name='pytorch_transformer',
    version='v3.0',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    url='https://github.com/ultimateabhi719/transformer_from_scratch',
    license='MIT License',
    author='Abhishek Agarwal',
    author_email='ultimateabhi@gmail.com',
    description='pytorch implementation of transformers for translation',
    python_requires = ">=3.7"
)
