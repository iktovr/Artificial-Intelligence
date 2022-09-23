Шаблон документа для jupyter [nbconvert](https://nbconvert.readthedocs.io/en/latest/) для конвертации ipynb в pdf/latex.

Использование:
```
jupyter nbconvert --to pdf --config cfg.py file.ipynb
```

**Примечание:** это необходимо делать из директории с `cfg.py`, иначе не сработает относительный путь до шаблона. В качестве альтернативы можно добавить в `cfg.py` абсолютный путь до директории с шаблонами, подробнее [тут](https://nbconvert.readthedocs.io/en/latest/customizing.html#adding-additional-template-paths).

### template.tplx

В шаблоне убраны номера страниц и оставлено место под заголовок. Получившийся pdf можно вставить в отчет при помощи пакета [pdfpages](https://www.ctan.org/pkg/pdfpages). Например, следующими командами:
```tex
\includepdf[pages=1, pagecommand={\section{Исходный код}}]{notebook.pdf}
\includepdf[pages=2-, pagecommand={}]{notebook.pdf}
```
### template_standalone.tplx

Шаблон для использования ipynb в качестве полноценного отчета. Аналогичен предыдущему, но нет пустого места в начале, есть страницы.

Выбор шаблона осуществляется путем изменения файла `cfg.py`
