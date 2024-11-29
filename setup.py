from setuptools import setup, find_packages

setup(
    name='jax_cgd', 
    version='0.1.4', 
    packages=find_packages(include=['jax_cgd', 'jax_cgd.*']), 
    install_requires=[ 
        'jax>=0.4.0',
        'jaxlib>=0.4.0',
        'ml-dtypes>=0.2.0',
        'numpy>=1.20',
        'scipy>=1.8',
        'opt-einsum>=3.3.0',
    ],
    author='Yiming Lu',
    author_email='luyiming925@gmail.com',
    description='A package for competitive gradient descent and its adaptive version using JAX',  
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown', 
    url='https://github.com/Juicy-Quadro/jax-cgd', 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9', 
    include_package_data=True, 
    license="MIT",
)
