from setuptools import setup, find_packages

setup(
    name='scrapping',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        'FinMind',   # 包含你使用的库
        'pandas',
        'yfinance',
        'matplotlib',
        'TA-Lib',
        'os'
    ],
    description='A package for scraping Taiwan futures with adding technical indicators and other related data.',
    author='Wenyu',
    author_email='wenyuchiou12@gmail.com',
)
