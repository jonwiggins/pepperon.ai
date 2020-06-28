from distutils.core import setup

setup(
    name="pepperonai",
    packages=["pepperonai"],
    version="0.12.0",
    license="MIT",
    description="Like scikit-learn - but worse",
    author="Jon Wiggins",
    author_email="contact@pepperon.ai",
    url="https://github.com/JonWiggins/pepperon.ai",
    download_url="https://github.com/JonWiggins/pepperon.ai/archive/0.12.0.tar.gz",
    keywords=["ML", "NLP", "AI", "pepperonai", "pepperon.ai"],
    install_requires=["pandas", "numpy", "scipy"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)

