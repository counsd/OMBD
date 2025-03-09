# OMBD

> OpenManus的分支项目，使用baidusearch替换原版的google搜索

## 项目简介

OMBD是OpenManus的一个分支项目，旨在通过使用baidusearch来替换原版的google搜索功能，为用户提供更符合国内用户需求的搜索体验。该项目基于OpenManus的原有架构，结合baidusearch的强大搜索能力，为用户打造一个高效、便捷的搜索工具，同时，汉化了大部分输出，中文看起来更加清晰明了。

## 功能特点

- **百度搜索集成**：使用BaiduSpider替换google搜索，提供更精准的百度搜索结果。
- **中文汉化**:汉化了大部分输出及提示词，看起来更加简洁明了

### 作者电脑配置
作者电脑配置:
- Python 3.12
- Win11专业版

  ## 部署方式
  >conda create -n open_manus python=3.12
  >conda activate open_manus
  >git clone https://github.com/counsd/OMBD.git
  >pip install -r requirements.txt
  >cd OMBD
  >python main.py
