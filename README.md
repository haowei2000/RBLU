

# RBLU

Reverse Bilingual Language Understanding (RBLU): A benchmark to evaluate the reverse inference ability of
large language models

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/haowei2000/RBLU/">
    <img src="figure/sum.png" alt="Logo" width="500" height="500">
  </a>

  <h3 align="center">The reverse inference process of RBLU benchmark</h3>
  <p align="left">
    Input the question to get the answer, and then input the answer to get the question. Finally, calculate the similarity of the input question and the output question
    <br />
    <a href="https://github.com/haowei2000/RBLU"><strong>Explore the doc »</strong></a>
    <br />
    <br />
    <a href="https://github.com/haowei2000/RBLU">View Demo</a>
    ·
    <a href="https://github.com/haowei2000/RBLU/issues">Report Bug</a>
    ·
    <a href="https://github.com/haowei2000/RBLU/issues">Request Feature</a>
  </p>

</p>

 
## Content

- [Built With](#Built-With)
- [About The Project](#About-The-Project)
  - [Dependencies](#Dependencies)
  - [Installation](#Installation)
- [Project Tree](#Project-Tree)
- [Usage](#Usage)
- [Roadmap](#部署)
- [Author](#Author)
- [Acknowledgments](#Acknowledgments)


## Build With
* [![PyTorch][PyTorch]][PyTorch-url]
* [![Hugging Face][HuggingFace]][HuggingFace-url]
* [![Poetry][Poetry]][Poetry-url]
* [![Python][Python]][Python-url]

[PyTorch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[HuggingFace]: https://img.shields.io/badge/Hugging%20Face-FFD700?style=flat-square&logo=Hugging-Face&logoColor=black
[HuggingFace-url]: https://huggingface.co/
[Poetry]: https://img.shields.io/badge/Poetry-60A5FA?style=flat-square&logo=Poetry&logoColor=white
[Poetry-url]: https://python-poetry.org/
[Python]: https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white
[Python-url]: https://www.python.org/


## Getting Start


### Dependencies

- `accelerate` = "^0.34.2"
- `datasets` = "^3.0.0"
- `evaluate` = "^0.4.3"
- `matplotlib` = "^3.9.2"
- `pandas` = "^2.2.2"
- `python` = "^3.12"
- `rouge-chinese` = "^1.0.3"
- `rouge-score` = "^0.1.2"
- `transformers` = "^4.44.2"
- `sentence-transformers` = "^3.1.0"
- `jieba` = "^0.42.1"
- `wandb` = "^0.18.0"
- `tiktoken` = "^0.7.0"
- `pyecharts` = "^2.0.6"
- `seaborn` = "^0.13.2"
- `plotly` = "^5.24.1"

### **Installation**

1. install [poetry](https://python-poetry.org/)
2. Clone the repo
3. Installing dependencies with poetry
```sh
poetry install
```

## Project Tree
eg:

```
RBLU 
├── LICENSE.txt
├── README.md
├── /figure/
├── /src/
│  ├── /chart/
│  ├── /data/
│  ├── /result/
│  ├── /score/
│  ├── /rblu/
│  │  ├── config.yaml
│  │  └── frontend.txt
└── /tests/

```


## Author

wanghw00@gmail.com 

 *您也可以在贡献者名单中参看所有参与该项目的开发者。*

## License

该项目签署了MIT 授权许可，详情请参阅 [LICENSE.txt](https://github.com/haowei2000/RBLU/blob/master/LICENSE.txt)

## Acknowledgments

- [GitHub Pages](https://pages.github.com)
- [isort](https://pycqa.github.io/isort/)
- [ruff](https://beta.ruff.rs/docs/)


<!-- links -->
[your-project-path]:haowei2000/RBLU
[contributors-shield]: https://img.shields.io/github/contributors/haowei2000/RBLU.svg?style=flat-square
[contributors-url]: https://github.com/haowei2000/RBLU/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/haowei2000/RBLU.svg?style=flat-square
[forks-url]: https://github.com/haowei2000/RBLU/network/members
[stars-shield]: https://img.shields.io/github/stars/haowei2000/RBLU.svg?style=flat-square
[stars-url]: https://github.com/haowei2000/RBLU/stargazers
[issues-shield]: https://img.shields.io/github/issues/haowei2000/RBLU.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/haowei2000/RBLU.svg
[license-shield]: https://img.shields.io/github/license/haowei2000/RBLU.svg?style=flat-square
[license-url]: https://github.com/haowei2000/RBLU/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/shaojintian




