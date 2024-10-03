# gauge-reader

A script to read a value from the analog gauge.

## Project Goal

I need to remotely monitor the coolant pressure in the heating system 
in my house when I'm away and woild like to integrate the monitoring 
into [Tuya](https://www.tuya.com/) and/or
[Home Assistant](https://www.home-assistant.io/).

It turned out that existing ready-made pressure measuring devices are
industrial, so they cannot be connected to existing smart home solutions.

However, I do not want to make a DIY sensor because it seems quite
complicated (current loop, ZigBee/WiFi module, firmware etc.) and
expensive. Instead, I decided to monitor the analog pressure gauge with
a Wi-Fi camera.

Obviosly, such a setup is enough for visual control of the pressure,
but I was curious if it is possible to convert the needle position to the
number with a kind of computer vision and pass it to my
[Home Assistant](https://www.home-assistant.io/) server.

All of the above led to the emergence of this project. 

## Hardware

### Pressure Gauge

After a lot of experiments with the built-in manometer (by Baxi) [picture will follow] that has a bunch
of colorful sectors, no marks and a low-contrast grey needle I realized that
a gauge must have as simple and contrast dial as possible. Finally, I settled on
an ICMA 244 manometer that is widely available, relatively cheap and has
a clean minimalistic design. 

### Camera

After some investigation, I settled on
[Ifeel Vega IFS-CI004](https://www.amazon.co.uk/Ifeel-Surveillance-IFS-CI004-Bi-directional-Compatible/dp/B0B9XWWRKW)
Wi-Fi camera. The main reason was that its diaeter is very close to the diameter
of the manometer. Then, it supports ONVIF and RTSP protocols.


### Assembly

[will follow soon]

## Software

The script is intended to run in a [**Home Assistant**](https://www.home-assistant.io/)
server with the [**pyscript**](https://github.com/custom-components/pyscript)
integration. It can also be run as a standalone script (i. e. in
JetBains [**Pycharm**](https://www.jetbrains.com/pycharm/)).

Python 3.12 is required.

### Installation in HA

[will follow soon]

### Running "in the wild"

[will follow soon]


## Example

The original camera snapshot:

![gauge.png](docs/gauge.png)

```sh
$ python3.12 run.py --debug images/gauge.png
INFO: Processing images/gauge.png
INFO: Gauge Reader 2.0.2-2-gf03f15d
DEBUG: Median of box areas is 531.92
DEBUG: Average angle between marks: 6.92
DEBUG: Added a zero mark at angle 43.19745304001033
DEBUG: Added a missing mark at angle 160.93
DEBUG: measured needle angle: 110.908380°
DEBUG: needle angle: absolute 110.908380°, relative 115.894167°
DEBUG: calculated value: 1.7737
DEBUG: Debug image saved as "images/gauge-debug.png"
INFO: value: 1.77
```

The image with some debugging drawings:

![gauge-debug.png](docs/gauge-debug.png)

## Issues

* The script's would give a wrong result or crash when
  the needle is around zero.
* Some readings seem to be incorrect in the second decimal digit after comma.

## TODO

* Collect testing images and implement a testing mode.
