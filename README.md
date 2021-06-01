# TextRank Tutorial

This repository simly run textrank algorithm based on [the website](https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/)


The websit intuitively explained how for textrank to work.

If you want to run this code

```
python3 text_rank.py
```


and then if you don't have data, download it after [the website](https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/)



when you read a file, tennis_article.csv, an error related to encoding happens,

First of all, you have to convert the data into UTF-8 with the following


```
iconv -c -t utf-8 input_file_name > output_file_name
```


