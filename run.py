#!/usr/bin/env python3

"""
Run the gauge reader from the command line
"""

import argparse
import logging
import os
from datetime import datetime

from read_pressure.gauge_reader import GaugeReader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_list', nargs='+')
    parser.add_argument('--save-dir', dest='save_dir')
    parser.add_argument('--csv-file', dest='csv_file')
    parser.add_argument('--debug', dest='debug', action='store_true')
    opts = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s')
    log = logging.getLogger(os.path.basename(__file__))
    log.setLevel(logging.DEBUG if opts.debug else logging.INFO)

    if opts.csv_file:
        csv_fs = open(opts.csv_file, 'w')
    else:
        csv_fs = None

    for image in opts.image_list:
        log.info(f'Processing {image}')

        reader = GaugeReader(log, image)
        value = reader.exec()
        reader.save_debug_image(suffix='-debug')

        if value is None:
            log.warning(f'The needle is not found')
        else:
            log.info(f'value: {value}')

        if opts.save_dir:
            reader.save_image(opts.save_dir, value)

        if csv_fs:
            ts = datetime.fromtimestamp(reader.img_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            csv_fs.write(f'{ts},{value}\n')

        log.info('\n')

    if csv_fs:
        csv_fs.close()
