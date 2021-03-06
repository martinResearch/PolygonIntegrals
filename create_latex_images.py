# This script will parse you readme.md file and create images for each equation found as 
# [latex:your_equation](your_image_file)
#
# we recommand using svg image files as they give nice vectorial images without pixel aliasing
# and they are in a text format which is good for versioning with git or mercurial 
# has the svg text remains unchanged for unchanged euation and thus it avoids pushing again and again the same 
# images on the server of the aleready compile equations each time a new equation is created 
# Martin de La Gorce April 2016

import tempfile
import shutil

dirpath = tempfile.mkdtemp()
# ... do stuff with dirpath

texfile='./readme.md'
import re

import os, requests


def formula_as_file( formula, file, negate=False,header='' ):
    laxtex_tmp_file=os.path.join(dirpath,'tmp_equation.tex')
    pdf_tmp_file=os.path.join(dirpath,'tmp_equation.pdf')
    latexfile = open(laxtex_tmp_file, 'w')
    latexfile.write('\\documentclass[preview]{standalone}')    
    #latexfile.write('\\input{header.tex}') 
    latexfile.write('\\usepackage{wasysym}')  
    latexfile.write('\\usepackage{amssymb}')     
    latexfile.write('\n\\begin{document}')   
    latexfile.write('$%s$'%formula)
    latexfile.write('\n\\end{document}  ') 
    latexfile.close()
    os.system( 'pdflatex -output-directory="%s" -aux_directory="%s" %s'%(dirpath,dirpath,laxtex_tmp_file) )
    if file[-3:]=='svg':
        os.system( 'pdf2svg %s %s'%(pdf_tmp_file,file) )
    else:  
        os.system( 'convert -density 600  %s -quality 90  %s'%(pdf_tmp_file,file) )
 

raw = open(texfile)
filecontent=raw.read()

latex_equations= re.findall(r"""\[latex:(.*?)\]\((.*?)\)""", filecontent)
listname=set()
for eqn in latex_equations:        
    if eqn[1] in listname:
        raise Exception('equation image file %s already used'%eqn[1])
        
    listname.add(eqn[1])
    formula_as_file(eqn[0],eqn[1])
    
shutil.rmtree(dirpath)
print 'DONE'
