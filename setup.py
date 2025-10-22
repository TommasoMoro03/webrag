from setuptools import setup, find_packages

setup(
    name="webrag",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "requests",
        "beautifulsoup4",
        "readability-lxml"
    ],
    python_requires=">=3.9",
)
