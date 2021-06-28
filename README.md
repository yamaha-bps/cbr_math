<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** yamaha-bps, cbr_math, twitter_handle, thomas_gurriet@yamaha-motor.com, Cyber Math, A handy set of tools relating to interpolation, distance calculations, lie theory, and more!
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/yamaha-bps/cbr_math">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Cyber Math</h3>

  <p align="center">
    A handy set of tools relating to interpolation, distance calculations, lie theory, and much more!
    <br />
    <a href="https://github.com/yamaha-bps/cbr_math/issues">Report Bug</a>
    ·
    <a href="https://github.com/yamaha-bps/cbr_math/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


### Built With

* [Libboost](https://www.boost.org/)
* [Eigen](https://gitlab.com/libeigen/eigen)
* [GTest](https://github.com/google/googletest)
* [Sophus](https://github.com/strasdat/Sophus)

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* Libboost
  ```sh
  sudo apt install libboost-dev
  ```

* Eigen
  ```sh
  sudo apt install libeigen3-dev
  ```

* GTest (only necessary to build tests)
  ```sh
  sudo apt install libgtest-dev
  ```

* Sophus
  ```sh
  git clone https://github.com/strasdat/Sophus.git
  mkdir Sophus/build
  cd Sophus/build
  cmake ..
  make -j2
  sudo make install
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/yamaha-bps/cbr_math.git
   ```

2. Make build directory
   ```sh
   mkdir cbr_math/build
   ```

3. Build
   ```sh
   cd cbr_math/build
   cmake .. -DBUILD_TESTING=ON
   make -j2
   ```

4. Install
   ```sh
   sudo make install
   ```

5. Verify successful install (tests should all pass)
   ```sh
   make test
   ```

6. Uninstall if you don't like it
   ```sh
   sudo make uninstall
   ```


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Thomas Gurriet - thomas_gurriet@yamaha-motor.com

Taylor Wentzel - taylor_wentzel@yamaha-motor.com

Project Link: [https://github.com/yamaha-bps/cbr_math](https://github.com/yamaha-bps/cbr_math)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [Yamaha Motor Corporation](https://yamaha-motor.com/)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/yamaha-bps/cbr_math.svg?style=for-the-badge
[contributors-url]: https://github.com/yamaha-bps/cbr_math/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/yamaha-bps/cbr_math.svg?style=for-the-badge
[forks-url]: https://github.com/yamaha-bps/cbr_math/network/members
[stars-shield]: https://img.shields.io/github/stars/yamaha-bps/cbr_math.svg?style=for-the-badge
[stars-url]: https://github.com/yamaha-bps/cbr_math/stargazers
[issues-shield]: https://img.shields.io/github/issues/yamaha-bps/cbr_math.svg?style=for-the-badge
[issues-url]: https://github.com/yamaha-bps/cbr_math/issues
[license-shield]: https://img.shields.io/github/license/yamaha-bps/cbr_math.svg?style=for-the-badge
[license-url]: https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE.txt
