from Cython.Build import cythonize
import os

def mycythonize(fname):
    
    fileName, fileExtension = os.path.splitext(fname)
    print('creating %s.pyx'%fileName)
    if not(fileExtension=='.py'):
        print('the file should be a python file with extension .py')
        raise
    fname_pyx=fileName+'.pyx'
    if not(os.path.isfile(fname_pyx)) or os.path.getmtime(fname)>os.path.getmtime(fname_pyx):
        fin =open(fname ,'rb')
        fout =open(fname_pyx ,'wb')
        for line in fin:            
            if (line.find(b'#cython_delete_line#')==-1):
                fout.write(line.replace(b'#cython#',b''))
        fin.close()
        fout.close()
    return cythonize(fname_pyx,annotate=True)



