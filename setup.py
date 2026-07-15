from setuptools import setup, find_packages

setup(
    name="datasets_explorer",
    version="0.2.0",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    entry_points={
        "console_scripts": [
            "datasets_explorer=datasets_explorer.cli:cli",
            "dataset_search=datasets_explorer.search_cli:main",
        ],
    },
)
