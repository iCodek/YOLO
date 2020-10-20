import os
from tqdm import tqdm
import requests
import zipfile
import tarfile

fileList = {'VOCtrainval.tar': 'http://bj.bcebos.com/v1/ai-studio-online/4807d90a985045fabff5734dd25a74e2111b840632344ccfbee1f9a4d9872733?responseContentDisposition=attachment%3B%20filename%3DVOCtrainval.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2020-04-29T08%3A28%3A33Z%2F-1%2F%2Ff640598cd6831861d646199b7d18ac00c26b9f40b6e441af1e8d2f720ea0528f'}


def checkFile(filename):
    if filename in fileList:
        url = fileList[filename]
        file = os.path.abspath(filename)
        download(url, file)
        print('{} 下载完成'.format(filename))
        if file.endswith('.tar') and not os.path.exists(file[:-4]):
            print('解压' + file + '中···')
            with tarfile.open(file, "r") as f:
                f.extractall(file[:-4])
    else:
        print('文件尚无下载地址')


def download(url, file_path):
    # 第一次请求是为了得到文件总大小
    r1 = requests.get(url, stream=True, verify=False)
    total_size = int(r1.headers['Content-Length'])
    # 这重要了，先看看本地文件下载了多少
    if os.path.exists(file_path):
        temp_size = os.path.getsize(file_path)  # 本地已经下载的文件大小
        if temp_size >= total_size:
            return
        print('从"{}"下载文件'.format(url))
        print('已下载 ' + size(temp_size) + ' / ' + size(total_size), '继续下载中···')
    else:
        temp_size = 0
        print('从"{}"下载文件'.format(url))
    # 显示一下下载了多少
    print('若下载速度太慢请直接至浏览器下载后复制至{}'.format(file_path))
    # 核心部分，这个是请求下载时，从本地文件已经下载过的后面下载
    headers = {'Range': 'bytes=%d-' % temp_size}
    # 重新请求网址，加入新的请求头的
    r = requests.get(url, stream=True, verify=False, headers=headers)
    # 下面写入文件也要注意，看到"ab"了吗？
    # "ab"表示追加形式写入文件
    with open(file_path, "ab") as f:
        with tqdm(total=total_size, disable=False,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            pbar.update(temp_size)
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    temp_size += len(chunk)
                    pbar.update(len(chunk))
                    f.write(chunk)
                    f.flush()


def size(s, b=0):
    if s >= 1024:
        return size(round(s / 1024, 2), b + 1)
    else:
        return str(s) + ['B', 'kB', 'MB', 'GB', 'TB'][b]


