from setuptools import setup, find_packages


setup(
    name="elsarec",
    version="0.1.0",
    description="Scalable Linear Shallow Autoencoder for Collaborative Filtering",
    author="Recombee",
    author_email="vojtech.vancura@recombee.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(),
    # PyTorch needs to be installed manually since it is not on pypi
    install_requires=["numpy", "scipy"],
    python_requires=">=3.7",
)

