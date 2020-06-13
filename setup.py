import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="venndata",
    version="0.1.0",
    author="Subhajit Mandal",
    author_email="mandalsubhajit@gmail.com",
    description="Package for plotting Venn diagrams with more than 3 sets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mandalsubhajit/venndata",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
