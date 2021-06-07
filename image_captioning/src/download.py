import requests
import os
from zipfile import ZipFile


def download_archive(fn, url):
    request = requests.get(url)
    with open(fn, 'wb') as file:
        file.write(request.content)


def extract_archive(fn):
    archive = ZipFile(fn)
    archive.extractall(os.path.dirname(fn))


def download_all(destination):

    urls = {'train': 'http://images.cocodataset.org/zips/train2014.zip',
            'valid': 'http://images.cocodataset.org/zips/val2014.zip',
            'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}

    for ds_name, url in urls.items():
        print(f'Download {ds_name} dataset from: {url}')
        fn = os.path.join(destination, ds_name+'.zip')
        download_archive(fn, url)
        extract_archive(fn)
        os.remove(fn)
