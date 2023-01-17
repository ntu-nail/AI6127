# AI6127: Deep Neural Networks For Natural Language Processing

# Course Objectives

Natural language processing (NLP) is one of the most important fields in artificial intelligence (AI). It has become very crucial in the information age because most of the information is in the form of unstructured text. NLP technologies are applied everywhere as people communicate mostly in language: language translation, web search, customer support, emails, forums, advertisement, radiology reports, to name a few.

There are a number of core NLP tasks and machine learning models behind NLP applications. Deep learning, a sub-field of machine learning, has recently brought a paradigm shift from traditional task-specific feature engineering to end-to-end systems and has obtained high performance across many different NLP tasks and downstream applications. Tech companies like Google, Baidu, Alibaba, Apple, Amazon, Facebook, Tencent, and Microsoft are now actively working on deep learning methods to improve their products. For example, Google recently replaced its traditional statistical machine translation and speech-recognition systems with systems based on deep learning methods.

**Optional Textbooks**

- Deep Learning by Goodfellow, Bengio, and Courville [free online](http://www.deeplearningbook.org/)
- Machine Learning — A Probabilistic Perspective by Kevin Murphy [online](https://doc.lagout.org/science/Artificial%20Intelligence/Machine%20learning/Machine%20Learning_%20A%20Probabilistic%20Perspective%20%5BMurphy%202012-08-24%5D.pdf)
- Natural Language Processing by Jacob Eisenstein [free online](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf)
- Speech and Language Processing by Dan Jurafsky and James H. Martin [(3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/)

# Intended Learning Outcomes

In this course, students will learn state-of-the-art deep learning methods for NLP. Through lectures and practical assignments, students will learn the necessary tricks for making their models work on practical problems. They will learn to implement, and possibly to invent their own deep learning models using available deep learning libraries like [Pytorch](https://pytorch.org/).

**Our Approach**

- Thorough and Detailed: How to write from scratch, debug and train deep neural models

- State of the art: Most lecture materials are new from research world in the past 1-5 years.

- Practical: Focus on practical techniques for training the models, and on GPUs.

- Fun: Cover exciting new advancements in NLP (e.g., Transformer, BERT).

# Assessment Approach

**Weekly Workload**

- Lecture and/or tutorial and/or practical problems implemented in PyTorch.
- There will be NO office hour.

**Assignments (individually graded)**

- There will be two (2) assignments contributing to ***2 * 25% = 50%*** of the total assessment.
- Students will be graded individually on the assignments. They will be allowed to discuss with each other on the homework assignments, but they are required to submit individual write-ups and coding exercises.

**Final Project (Group work but individually graded)**

- There will be a final project contributing to the remaining 50% of the total course-work assessment.
  - ***2–4*** people per group
  - Project proposal: ***5%***, presentation: ***10%***, report: ***35%***
- The project will be a group work. Students will be graded individually, depending on the contribution to the group. The final project presentation will ensure the student’s understanding of the project.

# Course Prerequisites

- Proficiency in Python (using numpy and PyTorch). There is a lecture for those who are not familiar with Python.
- Linear Algebra, basic Probability and Statistics
- Machine Learning basics

# Teaching

<p align="center" width="100%">Instructor</p>

<p align="center" width="100%">
    <img width="20%" src="https://ntu-nail.github.io/People/Luu_Anh_Tuan.png"> 
</p>

<p align="center" width="100%"><a href="https://tuanluu.github.io/">Luu Anh Tuan</a></p>


<p align="center" width="100%">Teaching Assistants</p>

<p align="center" width="100%">
    <img width="20%" src="/assets/images/ntcd.png"> 
</p>

<p align="center" width="100%">Nguyen Tran Cong Duy</p>
<p align="center" width="100%">NGUYENTR003@e.ntu.edu.sg</p>

<p align="center" width="100%">
    <img width="20%" src="/assets/images/pp.png"> 
</p>

<p align="center" width="100%"><a href="https://vijaydwivedi.com.np/">Vijay Prakash Dwivedi</a></p>
<p align="center" width="100%">VIJAYPRA001@e.ntu.edu.sg</p>

# Schedule & Course Content

## Week 1: Introduction

[Lecture Slide](https://drive.google.com/file/d/1i0SklXo7TJJCn5qFs8195d5VnKXhuzO3/view?usp=share_link)

### Lecture Content

- What is Natural Language Processing?
- Why is language understanding difficult?
- What is Deep Learning?
- Deep learning vs. other machine learning methods?
- Why deep learning for NLP?
- Applications of deep learning to NLP
- Knowing the target group (background, field of study, programming experience)
- Expectation from the course

### Python & PyTorch Basics

- Programming in Python

  - Jupiter Notebook and [google colab](https://colab.research.google.com/drive/16pBJQePbqkz3QFV54L4NIkOn1kwpuRrj)
  - [Introduction to python](https://colab.research.google.com/drive/1bQG32CFoMZ-jBk02uaFon60tER3yFx4c)
  - Deep Learning Frameworks
  - Why Pytorch?
  - [Deep learning with PyTorch](https://drive.google.com/file/d/1c33y8bkdr7SJ_I8-wmqTAhld-y7KcspA/view?usp=sharing)
- [Supplementary]
  - Numerical programming with numpy/scipy - [Numpy intro](https://drive.google.com/file/d/1cUzRzQGURrCKes8XynvTTA4Zvl_gUJdc/view?usp=sharing)
  - Numerical programming with Pytorch - [Pytorch intro](https://drive.google.com/file/d/18cgPOj2QKQN0WR9_vXoz6BoravvS9mTm/view?usp=sharing)

## Week 2: Machine Learning Basics

[Lecture Slide](https://drive.google.com/file/d/1PoslEhn7bQu4csyUBF-GunWixHxPoZnG/view?usp=share_link)

### Lecture Content

- What is Machine Learning?
- Supervised vs. unsupervised learning
- Linear Regression
- Logistic Regression
- Multi-class classification
- Parameter estimation (MLE & MAP)
- Gradient-based optimization & SGD

### Practical exercise with Pytorch

- [Deep learning with PyTorch](https://colab.research.google.com/drive/1aZVfsPUko-ugt1TVCmRwqGJXlxEJVaTq?usp=sharing)
- [Linear Regressionn](https://colab.research.google.com/drive/12QpBf7x_Jt6-zypN4OrUFFHXz1u6CmYe?usp=sharing)
- [Logistic Regression](https://colab.research.google.com/drive/1nTrYW5dUu6WO9cx7SGEvP9oX7qRbsGJk?usp=sharing)
- [Supplementary]
  - Numerical programming with Pytorch - [Pytorch intro](https://drive.google.com/file/d/18cgPOj2QKQN0WR9_vXoz6BoravvS9mTm/view?usp=sharing)
