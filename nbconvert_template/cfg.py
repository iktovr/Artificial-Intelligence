c = get_config()
c.LatexExporter.template_file = 'template.tplx'
c.PDFExporter.latex_command = ['pdflatex', '{filename}', '-quiet']
