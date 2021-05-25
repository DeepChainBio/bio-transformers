"""Setup file"""

import pathlib
from typing import List

from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

with open("biotransformers/version.py") as v:
    exec(v.readline())


def read_requirements() -> List:
    with open("requirements.txt", "r+") as file:
        requirements = [line.strip() for line in file.readlines()]

    return requirements


DESCRIPTION = "Wrapper on top of ESM/Protbert model in order to easily work with protein embedding"


def make_install():
    """main install function"""

    setup_fn = setup(
        name="bio-transformers",
        license="Apache-2.0",
        version=VERSION,  # noqa
        description=DESCRIPTION,
        author="Instadeep",
        long_description=README,
        long_description_content_type="text/markdown",
        author_email="a.delfosse@instadeep.com",
        packages=find_packages(exclude=["test"]),
        classifiers=[
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.7",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "Topic :: Software Development",
        ],
        install_requires=read_requirements(),
        include_package_data=True,
        zip_safe=False,
        python_requires=">=3.7",
    )

    return setup_fn


if __name__ == "__main__":
    make_install()
