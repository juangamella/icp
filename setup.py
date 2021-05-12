import setuptools

setuptools.setup(
    name='causalicp',
    version='0.1.0',
    author='Juan L Gamella',
    author_email='juangamella@gmail.com',
    packages=['causalicp'],
    scripts=[],
    url='https://github.com/juangamella/icp',
    license='BSD 3-Clause License',
    description='Python implementation of the Invariant Causal Prediction (ICP) algorithm for causal discovery.',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=['numpy>=1.17.0', 'scipy>=1.3.0', 'termcolor>=1.1.0']
)
