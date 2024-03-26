import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="witchy",
    version="0.0.2",
    author="Hughie",
    author_email="18028012455@163.com",
    description="this is a file tool for python",
    long_description=long_description,
    packages=setuptools.find_packages(),
    url="https://github.com/hughie21/witchy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)