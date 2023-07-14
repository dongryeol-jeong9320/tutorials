import os
import re
import time
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from konlpy.tag import Mecab


def time_trace(func):
    def wrapper(*args, **kwargs):
        st = time.time()
        rt = func(*args, **kwargs)
        print(f'### {func.__name__}({args}) time : {time.time()-st:.3f}s')
        return rt
    return wrapper

def explore_directory(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        if os.path.split(root)[-1] == 'txt':
            continue
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
        
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            explore_directory(subdir_path)
    return file_list

def hwp2txt(directory):
    exefile = 'hwp5txt'
    file_name = os.path.basename(directory)[:-4] + '.txt'
    file_path = os.path.join('./data/txt/', file_name)
    output = '--output ' + '"' + file_path + '"'
    result = '"' + directory + '"'
    print(exefile + " " + output + " " + result)
    os.system(exefile + " " + output + " " + result)\
    
def load_txt_file(file_path, encoding='utf8'):
    with open(file_path, 'r', encoding=encoding) as file:
        content = file.read()
    return content

@time_trace
def main(query, n=5):
    error_count = 0
    directory = './data/'
    file_list = explore_directory(directory)
    mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")

    print(f"총 문서 개수: {len(file_list)}")
    documents = []
    for i, file_path in tqdm(enumerate(file_list)):
        if i == n:
            break
        if os.path.splitext(file_path)[1] =='.hwp':
            # hwp2txt(file_path) # convert '.hwp' to '.txt'
            txt = load_txt_file('./data/txt/' + os.path.basename(file_path)[:-4] + '.txt')
            corpus = re.sub(r'\n| +.', ' ', txt)
            documents.append(mecab.morphs(corpus))
        else:
            continue

    bm25 = BM25Okapi(documents)
        
    result = bm25.get_top_n(query=mecab.morphs(query), documents=documents, n=n)

    print(f"총 문서 개수: {len(documents)}")
    print(f"성공/전체: {len(documents) - error_count/len(documents)}")
    
    print(f"실패: {error_count}")
    return result


if __name__ == "__main__":
    query = ""
    main(query=query, n=10)



    



    