from __future__ import absolute_import, division, print_function

import argparse
import os
import shutil
import sys
import tarfile
import zipfile

import tensorflow as tf
from IPython.display import display
from ipywidgets import FloatProgress
from six.moves import cPickle as pickle
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib

ARTIST_FILENAME = "artistprediction.zip"
ARTIST_DOWNLOAD_URL = "https://predictartist.s3.amazonaws.com/" + ARTIST_FILENAME
ARTIST_LOCAL_FOLDER = "ARTIST-batches-py"


def download_and_extract(data_dir, print_progress=True):
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if os.path.exists(os.path.join(data_dir, "ARTIST-batches-bin")):
        print("ARTIST dataset already downloaded")
        return

    filename = ARTIST_DOWNLOAD_URL.split("/")[-1]
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        f = FloatProgress(min=0, max=100)
        display(f)
        sys.stdout.write("\r>> Downloading %s " % (filename))

        def _progress(count, block_size, total_size):
            if print_progress:
                f.value = 100.0 * count * block_size / total_size

        filepath, _ = urllib.request.urlretrieve(ARTIST_DOWNLOAD_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print("Successfully downloaded", filename, statinfo.st_size, "bytes.")
    
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
    zip_ref.extractall(data_dir)
    #tarfile.open(filepath, "r:gz").extractall(data_dir)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _get_file_names():
    """Returns the file names expected to exist in the input_dir."""
    file_names = {}
    file_names["train"] = ["data_batch_%d" % i for i in xrange(1, 5)]
    file_names["validation"] = ["data_batch_5"]
    file_names["eval"] = ["test_batch"]
    return file_names


def read_pickle_from_file(filename):
    with tf.io.gfile.GFile(filename, "rb") as f:
        if sys.version_info >= (3, 0):
            data_dict = pickle.load(f, encoding="bytes")
        else:
            data_dict = pickle.load(f)
    return data_dict


def convert_to_tfrecord(input_files, output_file):
    """Converts a file to TFRecords."""
    print("Generating %s" % output_file)
    with tf.io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            data_dict = read_pickle_from_file(input_file)
            data = data_dict[b"data"]
            labels = data_dict[b"labels"]
            num_entries_in_batch = len(labels)
            for i in range(num_entries_in_batch):
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "image": _bytes_feature(data[i].tobytes()),
                            "label": _int64_feature(labels[i]),
                        }
                    )
                )
                record_writer.write(example.SerializeToString())


def main(data_dir):
    print("Download from {} and extract.".format(ARTIST_DOWNLOAD_URL))
    download_and_extract(data_dir)
    file_names = _get_file_names()
    input_dir = os.path.join(data_dir, ARTIST_LOCAL_FOLDER)
    for mode, files in file_names.items():
        input_files = [os.path.join(input_dir, f) for f in files]
        output_file = os.path.join(data_dir, mode + ".tfrecords")
        try:
            os.remove(output_file)
        except OSError:
            pass
        # Convert to tf.train.Example and write the to TFRecords.
        convert_to_tfrecord(input_files, output_file)
    print("Removing original files.")
    os.remove(os.path.join(data_dir, ARTIST_FILENAME))
    shutil.rmtree(input_dir)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=str, default="", help="Directory to download and extract ARTIST to."
    )

    args = parser.parse_args()
    main(args.data_dir)