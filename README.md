# Project status
Finished but open to eventual improvement or updates.

# Speech emotion recognition

In this work i propose a speech emotion recognition model based on the extraction of four different features got from sound files and stackingthe resulting matrices in a one-dimensional array by taking the mean values along the time axis. Then this array is fed into a 1-D CNN model as input.


# Table of contents

- [Installation and usage](#installation-and-usage)
- [Contributing](#contributing)
- [Contact](#Contact)

# Installation and usage

## For Colab
You can run this code entirely on Colab without the need of libraries installation through pip.

## For Linux/Ubuntu/...
All the libraries used are easy downloadable, but i recommend the creation of a conda environment thourgh the .yml file that you can find in the repo. It contains everything you need.

1. Create the conda environment
     ```sh
     conda env create -f SEC_environment.yml
     ```
2. Activate the conda environment
     ```sh
     conda activate SEC_environment
     ```
1. For training and testing, just run:
     ```sh
     python train_and_test.py
     ```
The first line of the yml file sets the new environment's name, that's already set on SEC_environment.

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- CONTACT -->
## Contact

Andrea Lombardi - [@LinkedIn](https://www.linkedin.com/in/andrea-lombardi/)
