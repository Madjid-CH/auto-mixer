import json

import h5py
import youtube_dl
from PyMovieDb import IMDB


def get_imdb_id(byte_id):
    imdb_id = byte_id.decode("utf-8")
    return f'tt{imdb_id}'


def download_trailer(movie_title, imdb_id):
    print(f"Downloading trailer for '{movie_title}'")
    ydl_opts = {
        'format': 'worst',
        'outtmpl': f'../data/mm_imdb/trailers/{imdb_id}.mp4',
        'age_limit': '18',
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        search_results = ydl.extract_info(f"ytsearch:{movie_title} Official Trailer", download=False)
        video = search_results['entries'][0]
        ydl.download([video['webpage_url']])


def download_trailers():
    with h5py.File('../data/mm_imdb/multimodal_imdb.hdf5', 'r') as mmimdb:
        imdb = IMDB()
        for id in mmimdb['imdb_ids']:
            imdb_id = get_imdb_id(id)
            res = imdb.get_by_id(imdb_id)
            res = json.loads(res)
            try:
                download_trailer(res['name'], imdb_id)
            except KeyError:
                print(f"Not found {imdb_id}")
                with open('missing_ids.txt', 'a') as f:
                    f.write(imdb_id + '\n')


if __name__ == '__main__':
    download_trailers()
