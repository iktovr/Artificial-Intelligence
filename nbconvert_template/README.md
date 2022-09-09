Шаблон документа для jupyter [nbconvert](https://nbconvert.readthedocs.io/en/latest/) для конвертации ipynb в pdf/latex.

Использование:
```
jupyter nbconvert --to pdf --config cfg.py file.ipynb
```

В шаблоне убраны номера страниц и оставлено место под заголовок. Получившийся pdf можно вставить в отчет при помощи пакета [pdfpages](https://www.ctan.org/pkg/pdfpages). Например, следующими командами:
```tex
\includepdf[pages=1, pagecommand={\section{Исходный код}}]{notebook.pdf}
\includepdf[pages=2-, pagecommand={}]{notebook.pdf}
```