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
    parser.add_argument('image_list',
                        help='The list of images to be read in the reading mode (without --test-mode) or '
                             'the list of test image patterns (with --test-mode).',
                        nargs='?', action='append')
    parser.add_argument('--save-debug-image', '-sdi',
                        help='Save the original image with debug drawings over it',
                        dest='save_debug_image', action='store_true')
    parser.add_argument('--save-dir', '-sd',
                        help='Directory to save debug images into. Works only with --save-debug-image. '
                             'Default is the directory of the current image',
                        nargs=1, dest='save_dir')
    parser.add_argument('--csv-file', dest='csv_file')
    parser.add_argument('--test-mode', '-t',
                         help='The test mode. Each non-option argument is treated as a file name pattern.'
                              'It may start with a directory and must contain the expected value macro "{V}" '
                              'that expands to "P.PP[-R]", where P.PP is an expected pressure value and '
                              'R is a decimal revision number (optional).',
                         action='store_true', dest='test_mode')
    parser.add_argument('--debug',
                        help='Enables debug output',
                        dest='debug', action='store_true')
    opts = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s')
    log = logging.getLogger(os.path.basename(__file__))
    log.setLevel(logging.DEBUG if opts.debug else logging.INFO)

    if opts.csv_file:
        csv_fs = open(opts.csv_file, 'w')
    else:
        csv_fs = None

    if opts.test_mode:
        log.info('Test mode')
        images = []
        test_results = {}
        macro_ptrn = r'(\d\.\d\d)(-\d)?'
        macro_re = re.compile(macro_ptrn)

        for image in opts.image_list:
            if '{V}' not in image:
                log.error('Pattern "{image}" doesn\'t contain "{V}" macro. Each test image pattern must contain it.')
                sys.exit(2)

            _dir = os.path.dirname(image)
            _file = os.path.basename(image)
            _ptrn = re.compile(_file.replace('{V}', macro_ptrn))

            for entry in os.listdir(_dir):
                if _ptrn.match(entry):
                    images.append(os.path.join(_dir, entry))

        images.sort()

    else:
        log.info('Reading mode')
        images = opts.image_list

    for image in images:
        reader = GaugeReader(log, image, no_im_show=opts.test_mode)
        value = reader.exec()

        if opts.save_debug_image:
            reader.save_debug_image(save_dir=opts.save_dir, suffix='-debug')

        if value is None:
            log.warning(f'The needle is not found')
        else:
            log.info(f'value: {value}')

        if opts.test_mode:
            expected = macro_re.search(image).group(1)
            measured = value
            test_results[image] = (expected, measured)
            log.info('')

        #if opts.save_dir:
        #    reader.save_image(opts.save_dir, value)

        if csv_fs:
            ts = datetime.fromtimestamp(reader.img_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            csv_fs.write(f'{ts},{value}\n')

    if opts.test_mode:
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
