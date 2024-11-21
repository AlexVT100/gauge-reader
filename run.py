#!/usr/bin/env python3

"""
Run the gauge reader from the command line
"""

import argparse
import logging
import os
import re
import sys
from datetime import datetime

from read_pressure.gauge_reader import GaugeReader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_list', nargs='?', action='append')
    parser.add_argument('--save-debug-image', '-sdi', dest='save_debug_image', action='store_true')
    parser.add_argument('--save-dir', '-sd', nargs=1, dest='save_dir')
    parser.add_argument('--csv-file', dest='csv_file')
    test_grp = parser.add_argument_group('Test mode', 'All options in this group must be used together')
    test_grp.add_argument('--test-image-dir', '-td', metavar='DIR', dest='test_image_dir')
    test_grp.add_argument('--test-image-prefix', '-tp', metavar='PREFIX', dest='test_image_prefix')
    test_grp.add_argument('--test-image-suffix', '-ts', metavar='SUFFIX', dest='test_image_suffix')
    parser.add_argument('--debug', dest='debug', action='store_true')
    opts = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s')
    log = logging.getLogger(os.path.basename(__file__))
    log.setLevel(logging.DEBUG if opts.debug else logging.INFO)

    if opts.csv_file:
        csv_fs = open(opts.csv_file, 'w')
    else:
        csv_fs = None

    test_args = (opts.test_image_dir, opts.test_image_prefix, opts.test_image_suffix)
    if any(test_args):
        test_mode = True
        test_results = {}
        log.info('Test mode')
        if not all(test_args):
            log.fatal('Missing argument(s). In the test mode, all --test-*/-t* options are mandatory')
            sys.exit(2)

        images = []
        pattern = re.compile(opts.test_image_prefix + r'(\d\.\d\d)(-\d)?' + opts.test_image_suffix)
        for entry in os.listdir(opts.test_image_dir):
            if pattern.match(entry):
                images.append(os.path.join(opts.test_image_dir, entry))
        images.sort()

    else:
        test_mode = False
        log.info('Reading mode')
        images = opts.image_list

    for image in images:
        reader = GaugeReader(log, image)
        value = reader.exec()

        if opts.save_debug_image:
            reader.save_debug_image(save_dir=opts.save_dir, suffix='-debug')

        if value is None:
            log.warning(f'The needle is not found')
        else:
            log.info(f'value: {value}')

        if test_mode:
            expected = pattern.search(image).group(1)
            measured = value
            test_results[image] = (expected, measured)
            log.info('')

        #if opts.save_dir:
        #    reader.save_image(opts.save_dir, value)

        if csv_fs:
            ts = datetime.fromtimestamp(reader.img_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            csv_fs.write(f'{ts},{value}\n')

    if test_mode:
        fmt = '{:35s} {:>8s} {:>8s}  {:>5s}'
        log.info('Test results')
        log.info('------------------------------------------------------------')
        log.info(fmt.format('Image', 'Expected', 'Measured', 'Diff'))
        log.info('------------------------------------------------------------')
        for image, values in test_results.items():
            #result = 'OK' if values[0] == values[1] else 'FAILED'
            diff = float(values[0]) - float(values[1])
            log.info(fmt.format(image, values[0], values[1], '' if diff == 0.0 else f'{diff: .2f}'))
        log.info('------------------------------------------------------------')

    if csv_fs:
        csv_fs.close()
